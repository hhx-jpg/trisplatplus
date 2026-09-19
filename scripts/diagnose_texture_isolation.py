"""Check that precomputed texture colors do not change triangle geometry outputs."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from src.checkpoint_utils import extract_state_dict, load_checkpoint_file
from src.config import load_typed_root_config
from src.dataset.data_module import DataModule
from src.misc.step_tracker import StepTracker
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder
from src.model.types import Triangles


ROOT = Path(__file__).resolve().parents[1]


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    return value


def max_abs(a, b):
    return float((a.float() - b.float()).abs().max().item())


def main() -> None:
    data_root = Path("/root/data/lhmd/dl3dv_torch_960/10K")
    checkpoint = Path(
        "/root/data/haoxuan/TriSplat/outputs/exp_tsdpt_da3_dl3dv_lpips20_bookshelf_9c_resume7300_1k/"
        "2026-08-28_05-09-42/checkpoints/render_step_008300.ckpt"
    )
    index_path = ROOT / "assets/dl3dv_bookshelf_eval.json"
    experiment = "trisplat_dl3dv_tsdpt_lgtm_stage_a_bookshelf"
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        cfg = load_typed_root_config(
            compose(
                config_name="main",
                overrides=[
                    f"+experiment={experiment}",
                    f"dataset.dl3dv.roots=[{data_root}]",
                    f"dataset.dl3dv.test_roots=[{data_root}]",
                    f"dataset.dl3dv.view_sampler.index_path={index_path}",
                    "dataset.dl3dv.input_image_shape=[224,448]",
                    "dataset.dl3dv.original_image_shape=[540,960]",
                    "data_loader.train.batch_size=1",
                    "data_loader.train.num_workers=0",
                    "train.use_mono_normal_teacher=false",
                    "train.normal_bootstrap.enabled=false",
                ],
            )
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = move(
        next(iter(DataModule(cfg.dataset, cfg.data_loader, StepTracker(), global_rank=0).train_dataloader())),
        device,
    )
    encoder, _ = get_encoder(cfg.model.encoder)
    state = extract_state_dict(load_checkpoint_file(checkpoint))
    encoder.load_state_dict(
        {key[8:]: value for key, value in state.items() if key.startswith("encoder.")},
        strict=False,
    )
    encoder = encoder.to(device).eval()
    decoder = get_decoder(cfg.model.decoder).to(device).eval()
    with torch.inference_mode():
        primitives = encoder(batch["context"], global_step=8300)
        if not isinstance(primitives, Triangles):
            raise TypeError(type(primitives).__name__)
        textured = decoder(
            primitives,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            tuple(batch["target"]["image"].shape[-2:]),
            global_step=8300,
            return_triangle_visibility_mask=True,
        )
        untextured_primitives = copy.copy(primitives)
        untextured_primitives.texture_colors = None
        untextured_primitives.texture_alphas = None
        untextured = decoder(
            untextured_primitives,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            tuple(batch["target"]["image"].shape[-2:]),
            global_step=8300,
            return_triangle_visibility_mask=True,
        )

    print(json.dumps({
        "texture_shape": None if primitives.texture_colors is None else list(primitives.texture_colors.shape),
        "primitive_attrs": {
            name: list(getattr(primitives, name).shape)
            for name in ("vertices", "centers", "normals", "scales", "sigma", "opacity")
            if getattr(primitives, name) is not None
        },
        "output_max_abs_diff_texture_on_minus_off": {
            name: max_abs(getattr(textured, name), getattr(untextured, name))
            for name in ("depth", "opacity", "rend_normal", "surf_normal")
        },
        "visibility_max_diff": max_abs(
            textured.triangle_visibility_mask, untextured.triangle_visibility_mask
        ),
        "normal_stats": {
            "texture_on_absmean": float(textured.rend_normal.abs().mean()),
            "texture_off_absmean": float(untextured.rend_normal.abs().mean()),
            "texture_on_nonzero": float((textured.rend_normal.abs().sum(dim=2) > 1e-6).float().mean()),
            "texture_off_nonzero": float((untextured.rend_normal.abs().sum(dim=2) > 1e-6).float().mean()),
        },
        "color_max_abs_diff": max_abs(textured.color, untextured.color),
    }, indent=2))


if __name__ == "__main__":
    main()
