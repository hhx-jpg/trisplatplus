#!/usr/bin/env python3
"""Render six DL3DV context views as GT / normal / render triptychs.

The geometry branch is always fed 224x448 crops, while the texture branch
receives the untouched 540x960 context pixels and their native intrinsics. A
scene is rejected before model construction if its selected frames do not all
share the expected native resolution.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (
    REPO_ROOT,
    REPO_ROOT / "submodules" / "diff-triangle-rasterization",
    REPO_ROOT.parent / "TriSplat" / "submodules" / "diff-triangle-rasterization",
):
    if str(path) not in sys.path and path.exists():
        sys.path.insert(0, str(path))

from src.model.decoder.decoder_triangle_splatting_cuda import (
    DecoderTriangleSplattingCUDA,
    DecoderTriangleSplattingCUDACfg,
)
from src.model.encoder.encoder_da3_tsdpt import EncoderDA3TSDPT, EncoderDA3TSDPTCfg


class SceneSkipped(RuntimeError):
    pass


def decode_image(encoded: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    with Image.open(io.BytesIO(encoded.numpy().tobytes())) as image:
        rgb = image.convert("RGB")
        return transforms.ToTensor()(rgb), (rgb.height, rgb.width)


def convert_cameras(cameras: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    intrinsics = torch.eye(3, dtype=torch.float32).repeat(cameras.shape[0], 1, 1)
    intrinsics[:, 0, 0] = cameras[:, 0]
    intrinsics[:, 1, 1] = cameras[:, 1]
    intrinsics[:, 0, 2] = cameras[:, 2]
    intrinsics[:, 1, 2] = cameras[:, 3]
    w2c = torch.eye(4, dtype=torch.float32).repeat(cameras.shape[0], 1, 1)
    w2c[:, :3] = cameras[:, 6:].reshape(-1, 3, 4)
    return w2c.inverse(), intrinsics


def normalize_context_poses(c2w: torch.Tensor) -> tuple[torch.Tensor, float]:
    centers = c2w[:, :3, 3]
    pairwise = torch.cdist(centers, centers)
    scale = float(pairwise.max().clamp_min(1e-6))
    normalized = c2w.clone()
    normalized[:, :3, 3] /= scale
    normalized = torch.linalg.inv(normalized[:1]) @ normalized
    return normalized, scale


def resize_and_crop(
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    output_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    out_h, out_w = output_shape
    _, in_h, in_w = images.shape[-3:]
    factor = max(out_h / in_h, out_w / in_w)
    scaled_h, scaled_w = round(in_h * factor), round(in_w * factor)
    images = torch.stack(
        [
            transforms.functional.resize(
                image,
                [scaled_h, scaled_w],
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            )
            for image in images
        ]
    )
    row = (scaled_h - out_h) // 2
    col = (scaled_w - out_w) // 2
    images = images[:, :, row : row + out_h, col : col + out_w]
    intrinsics = intrinsics.clone()
    intrinsics[:, 0, 0] *= scaled_w / out_w
    intrinsics[:, 1, 1] *= scaled_h / out_h
    intrinsics[:, 0, 2] = (intrinsics[:, 0, 2] * scaled_w - col) / out_w
    intrinsics[:, 1, 2] = (intrinsics[:, 1, 2] * scaled_h - row) / out_h
    return images, intrinsics


def save_tensor(path: Path, image: torch.Tensor) -> None:
    image = image.detach().float().cpu().clamp(0, 1)
    if image.ndim == 2:
        image = image.unsqueeze(0).expand(3, -1, -1)
    if image.shape[0] == 1:
        image = image.expand(3, -1, -1)
    array = (image.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().float().cpu().clamp(0, 1)
    if image.ndim == 2:
        image = image.unsqueeze(0).expand(3, -1, -1)
    if image.shape[0] == 1:
        image = image.expand(3, -1, -1)
    array = (image.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(array)


def load_encoder_weights(encoder: EncoderDA3TSDPT, checkpoint: Path) -> int:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    selected = {
        (key[8:] if key.startswith("encoder.") else key): value
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    missing, unexpected = encoder.load_state_dict(selected, strict=False)
    print(
        f"checkpoint={checkpoint} loaded={len(selected)} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if missing:
        print("missing keys (first 12):", missing[:12], flush=True)
    return int(payload.get("global_step", 2700))


def scene_triptych(
    scene: dict,
    scene_index: int,
    chunk_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    context_indices = [int(i) for i in args.context_indices]
    if max(context_indices) >= len(scene["images"]):
        raise SceneSkipped("scene has fewer frames than the requested context indices")

    decoded = [decode_image(scene["images"][i]) for i in context_indices]
    native_shapes = [shape for _, shape in decoded]
    expected_native = (args.native_height, args.native_width)
    if any(shape != expected_native for shape in native_shapes):
        raise SceneSkipped(
            f"native frame shapes {native_shapes} do not match expected {expected_native}"
        )
    if len(set(native_shapes)) != 1:
        raise SceneSkipped(f"selected frames have mixed native shapes: {native_shapes}")

    native_images = torch.stack([image for image, _ in decoded])
    raw_c2w, native_intrinsics = convert_cameras(scene["cameras"])
    context_c2w, pose_scale = normalize_context_poses(raw_c2w[context_indices])
    geometry_images, geometry_intrinsics = resize_and_crop(
        native_images, native_intrinsics[context_indices], (args.height, args.width)
    )
    if tuple(geometry_images.shape[-2:]) != (args.height, args.width):
        raise SceneSkipped(f"geometry shape is {tuple(geometry_images.shape[-2:])}")

    context = {
        "image": geometry_images.unsqueeze(0).to(device),
        "image_highres": native_images.unsqueeze(0).to(device),
        "extrinsics": context_c2w.unsqueeze(0).to(device),
        "intrinsics": geometry_intrinsics.unsqueeze(0).to(device),
        "intrinsics_highres": native_intrinsics[context_indices].unsqueeze(0).to(device),
    }
    near = torch.full((1, len(context_indices)), 0.01, device=device)
    far = torch.full((1, len(context_indices)), 1000.0, device=device)

    encoder_cfg = EncoderDA3TSDPTCfg(
        checkpoint=str(args.da3_checkpoint),
        triangle_scale_min=1.0,
        triangle_scale_max=18.0,
        triangle_scale_init_fraction=0.25,
        sigma_scale_initial=4.6,
        sigma_scale_final=4.6,
        sigma_warmup_steps=1,
        opacity_global_scale=0.62,
        sh_degree=0,
        align_to_context_pose=True,
        texture_enabled=True,
        texture_size=4,
        texture_project_as_base=True,
        texture_color_sigma=1.0,
        texture_only=True,
        use_input_normalization=False,
    )
    encoder = EncoderDA3TSDPT(encoder_cfg)
    schedule_step = load_encoder_weights(encoder, args.checkpoint)
    encoder = encoder.to(device).eval()

    decoder = DecoderTriangleSplattingCUDA(
        DecoderTriangleSplattingCUDACfg(
            name="triangle_splatting_cuda",
            background_color=[0.0, 0.0, 0.0],
            opacity_temp_initial=1.0,
            opacity_temp_final=5.0,
            opacity_temp_warmup_steps=1,
            alpha_floor_min=0.0,
            alpha_floor_warmup_steps=0,
            sh_degree=0,
            texture_size=4,
            texture_color_sigma=1.0,
            cull_out_of_bounds=True,
            center_depth_culling=True,
            near_plane=0.2,
        )
    ).to(device).eval()

    visualization: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        triangles = encoder(context, global_step=schedule_step, visualization_dump=visualization)
        rendered = decoder(
            triangles,
            context["extrinsics"],
            context["intrinsics"],
            near,
            far,
            image_shape=(args.height, args.width),
            global_step=schedule_step,
            return_triangle_visibility_mask=True,
        )

    normal = visualization["geom_normal_cam_forward"][0]
    gt = geometry_images.to(device)
    rgb = rendered.color[0]
    normal_rgb = ((normal + 1.0) * 0.5).clamp(0, 1)
    out_dir = args.out / scene["key"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for row, frame_index in enumerate(context_indices):
        save_tensor(out_dir / f"view_{frame_index:04d}_gt.png", gt[row])
        save_tensor(out_dir / f"view_{frame_index:04d}_normal.png", normal_rgb[row])
        save_tensor(out_dir / f"view_{frame_index:04d}_render.png", rgb[row])

    panel_w, panel_h = args.width, args.height
    label_h, gap = 28, 8
    sheet = Image.new(
        "RGB",
        (panel_w * 3 + gap * 4, (panel_h + label_h) * len(context_indices) + gap),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    labels = ("GT", "NORMAL", "RENDER")
    panels = (gt, normal_rgb, rgb)
    for row, frame_index in enumerate(context_indices):
        y = gap + row * (panel_h + label_h)
        for col, (label, tensor) in enumerate(zip(labels, panels)):
            x = gap + col * (panel_w + gap)
            draw.text((x + 5, y + 5), f"{label}  view {frame_index}", fill="black")
            sheet.paste(tensor_to_pil(tensor[row]), (x, y + label_h))
    sheet.save(out_dir / "triptych_6view.png")

    metadata = {
        "scene": scene["key"],
        "scene_index": scene_index,
        "chunk": str(chunk_path),
        "context_indices": context_indices,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "global_step": schedule_step,
        "native_input_shape_hw": [args.native_height, args.native_width],
        "geometry_input_shape_hw": [args.height, args.width],
        "texture_size": 4,
        "pose_norm_scale": pose_scale,
        "triangle_count": int(triangles.vertices.shape[1]),
        "render_shape": list(rendered.color.shape),
        "finite_render": bool(torch.isfinite(rendered.color).all()),
        "output": str(out_dir),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/root/data/lhmd/dl3dv_torch_960/10K"))
    parser.add_argument("--chunk", type=Path, default=None)
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--da3-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-indices", type=int, nargs="+", default=[0, 20, 40, 60, 80, 100])
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--native-height", type=int, default=540)
    parser.add_argument("--native-width", type=int, default=960)
    args = parser.parse_args()
    if len(args.context_indices) != 6:
        raise ValueError("exactly six --context-indices are required")
    if (args.height, args.width) != (224, 448):
        raise ValueError("DA3/TriSplat++ geometry input must be exactly 224x448")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.da3_checkpoint.is_dir():
        raise FileNotFoundError(args.da3_checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DA3 GIANT and the triangle renderer")
    device = torch.device(args.device)
    chunk = args.chunk
    if chunk is None:
        chunks = sorted((args.data_root / "train").glob("*.torch"))
        if not chunks:
            raise FileNotFoundError(f"No chunks under {args.data_root / 'train'}")
        chunk = chunks[0]
    scenes = torch.load(chunk, map_location="cpu")
    if args.scene_index >= len(scenes):
        raise IndexError(f"scene index {args.scene_index} >= {len(scenes)} in {chunk}")
    scene = scenes[args.scene_index]
    try:
        metadata = scene_triptych(scene, args.scene_index, chunk, args, device)
    except SceneSkipped as exc:
        print(f"SKIP scene={args.scene_index} reason={exc}", flush=True)
        (args.out / "skipped.json").parent.mkdir(parents=True, exist_ok=True)
        (args.out / "skipped.json").write_text(json.dumps({"scene_index": args.scene_index, "reason": str(exc)}, indent=2) + "\n")
        raise SystemExit(2)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
