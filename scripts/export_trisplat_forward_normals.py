"""Run native TriSplat forward on one fixed DL3DV scene and export normals/depth."""

from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from PIL import Image

from src.checkpoint_utils import extract_state_dict, load_checkpoint_file
from src.config import load_typed_root_config
from src.dataset.data_module import DataModule, get_data_shim
from src.misc.image_io import prep_image
from src.misc.step_tracker import StepTracker
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder


REPO_ROOT = Path(__file__).resolve().parents[1]


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move(item, device) for item in value)
    return value


def build_cfg(data_root: Path, index_path: Path, scene: str, num_context_views: int):
    overrides = [
        "+experiment=trisplat_dl3dv_triangle_refiner_unet_10m_224x448_test",
        "loss=[mse,lpips]",
        "mode=test",
        "dataset/view_sampler@dataset.dl3dv.view_sampler=evaluation",
        f"dataset.dl3dv.roots=[{data_root}]",
        f"dataset.dl3dv.test_roots=[{data_root}]",
        f"dataset.dl3dv.view_sampler.index_path={index_path}",
        f"dataset.dl3dv.view_sampler.num_context_views={num_context_views}",
        f"dataset.dl3dv.overfit_to_scene={scene}",
        "dataset.dl3dv.input_image_shape=[224,448]",
        "dataset.dl3dv.original_image_shape=[540,960]",
        "data_loader.train.batch_size=1",
        "data_loader.train.num_workers=0",
        "data_loader.val.batch_size=1",
        "data_loader.val.num_workers=0",
        "train.use_mono_normal_teacher=false",
        "train.normal_bootstrap.enabled=false",
    ]
    with initialize_config_dir(config_dir=str(REPO_ROOT / "config"), version_base=None):
        return load_typed_root_config(compose(config_name="main", overrides=overrides))


def load_encoder(cfg, checkpoint: Path, device: torch.device):
    encoder, _ = get_encoder(cfg.model.encoder)
    state = extract_state_dict(load_checkpoint_file(checkpoint))
    state = {key[8:]: value for key, value in state.items() if key.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    print(f"loaded encoder weights: {len(state)} tensors; missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print("missing keys (first 20):", missing[:20], flush=True)
    return encoder.to(device).eval()


def save_jpeg(image: torch.Tensor, path: Path, max_bytes: int = 500_000) -> None:
    array = prep_image(image.detach().float().clamp(0, 1)).astype(np.uint8)
    pil = Image.fromarray(array).convert("RGB")
    quality = 92
    while True:
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=True)
        payload = buffer.getvalue()
        if len(payload) <= max_bytes or quality <= 40:
            path.write_bytes(payload)
            return
        quality -= 8


def save_normal(normal: torch.Tensor, path: Path) -> None:
    # CUDA renderer normals are world-space vectors in [-1, 1].
    save_jpeg((normal + 1.0) * 0.5, path)


def save_depth(depth: torch.Tensor, path: Path) -> None:
    values = depth.detach().float().squeeze(0)
    finite = torch.isfinite(values) & (values > 0)
    if finite.any():
        lo = torch.quantile(values[finite], 0.01)
        hi = torch.quantile(values[finite], 0.99).clamp_min(lo + 1e-6)
        image = ((values - lo) / (hi - lo)).clamp(0, 1)
    else:
        image = torch.zeros_like(values)
    # Invert so nearer surfaces are brighter for inspection.
    save_jpeg(1.0 - image, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/root/data/lhmd/dl3dv_torch_960/10K"))
    parser.add_argument("--index", type=Path, default=REPO_ROOT / "assets/dl3dv_step7000_scene_eval.json")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints/dl3dv_trisplat.ckpt")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/trisplat_forward_scene_step7000_normals")
    parser.add_argument("--global-step", type=int, default=7000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    entries = json.loads(args.index.read_text())
    if len(entries) != 1:
        raise ValueError(f"Expected one scene in {args.index}, found {len(entries)}")
    scene, entry = next(iter(entries.items()))
    num_context_views = len(entry["context"])
    cfg = build_cfg(args.data_root, args.index, scene, num_context_views)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} scene={scene} context={entry['context']} target={entry['target']}", flush=True)

    # The evaluation sampler is fixed by the JSON index. Only one batch is read.
    loader = DataModule(cfg.dataset, cfg.data_loader, StepTracker(), global_rank=0).train_dataloader()
    raw_batch = next(iter(loader))
    raw_batch = move(raw_batch, device)
    encoder = load_encoder(cfg, args.checkpoint, device)
    decoder = get_decoder(cfg.model.decoder).to(device).eval()
    batch = get_data_shim(encoder)(copy.deepcopy(raw_batch))
    target = batch["target"]

    with torch.inference_mode():
        primitives = encoder(batch["context"], global_step=args.global_step)
        print(f"encoder triangles={primitives.vertices.shape[1]}", flush=True)
        output = decoder(
            primitives,
            target["extrinsics"],
            target["intrinsics"],
            target["near"],
            target["far"],
            tuple(target["image"].shape[-2:]),
            global_step=args.global_step,
            return_triangle_visibility_mask=True,
        )

    target_indices = raw_batch["target"]["index"][0].detach().cpu().tolist()
    payload = {
        "scene": scene,
        "checkpoint": str(args.checkpoint),
        "global_step": args.global_step,
        "context_indices": raw_batch["context"]["index"][0].detach().cpu().tolist(),
        "target_indices": target_indices,
        "num_triangles": int(primitives.vertices.shape[1]),
        "output_shapes": {
            "color": list(output.color.shape),
            "rend_normal": list(output.rend_normal.shape),
            "surf_normal": list(output.surf_normal.shape),
            "depth": list(output.depth.shape),
        },
    }
    visibility = output.triangle_visibility_mask
    if visibility is not None:
        payload["visible_triangles_per_target"] = [
            int(visibility[0, i].sum().item()) for i in range(visibility.shape[1])
        ]
    (args.out / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n")

    # Preserve exact renderer outputs for later numeric inspection.
    np.save(args.out / "rend_normal.npy", output.rend_normal[0].detach().float().cpu().numpy())
    np.save(args.out / "surf_normal.npy", output.surf_normal[0].detach().float().cpu().numpy())
    np.save(args.out / "depth.npy", output.depth[0].detach().float().cpu().numpy())
    np.save(args.out / "color.npy", output.color[0].detach().float().cpu().numpy())

    for slot, frame_index in enumerate(target_indices):
        save_normal(output.rend_normal[0, slot], args.out / f"target_{frame_index:04d}_rend_normal.jpg")
        save_normal(output.surf_normal[0, slot], args.out / f"target_{frame_index:04d}_surf_normal.jpg")
        save_depth(output.depth[0, slot], args.out / f"target_{frame_index:04d}_depth.jpg")
        save_jpeg(output.color[0, slot], args.out / f"target_{frame_index:04d}_render.jpg")

    # Four-row contact sheets remain convenient, but each is independently capped.
    def sheet(tensor: torch.Tensor, normal: bool = False) -> torch.Tensor:
        images = [((item + 1) * 0.5 if normal else item).clamp(0, 1) for item in tensor]
        return torch.cat(images, dim=-1)

    save_jpeg(sheet(output.rend_normal[0], normal=True), args.out / "rend_normal_all.jpg")
    save_jpeg(sheet(output.surf_normal[0], normal=True), args.out / "surf_normal_all.jpg")
    save_jpeg(sheet(output.color[0]), args.out / "render_all.jpg")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
