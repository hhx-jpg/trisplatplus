#!/usr/bin/env python3
"""Export native DA3/TSDPT triangles with the original forward pipeline.

By default this script reads the same DL3DV scene, camera normalization,
resize/crop settings, checkpoint, and schedule step as
``infer_da3_tsdpt_forward.py``.  The output is a native dump for
``mesh-splatting/tools/da3_to_mesh.py``.  ``--source-dump`` remains available
only as an explicit legacy compatibility mode.
"""

from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DA3_SRC = REPO_ROOT.parent / "Depth-Anything-3" / "src"
for path in (REPO_ROOT, DA3_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.dataset.norm_scale import compute_pose_norm_scale
from src.misc.cam_utils import camera_normalization
from src.model.encoder.encoder_da3_tsdpt import EncoderDA3TSDPT, EncoderDA3TSDPTCfg
from src.model.types import Triangles


def _load_init(
    encoder: EncoderDA3TSDPT,
    checkpoint: Path,
    preserve_zero_scale_head: bool = False,
) -> dict[str, object]:
    """Use the same checkpoint loading contract as the original forward."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no state dict: {checkpoint}")
    selected: dict[str, torch.Tensor] = {}
    skipped_scale: list[str] = []
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key.startswith("encoder."):
            key = key[len("encoder."):]
        if preserve_zero_scale_head and "scale_output_conv" in key:
            skipped_scale.append(key)
            continue
        selected[key] = value
    missing, unexpected = encoder.load_state_dict(selected, strict=False)
    if preserve_zero_scale_head:
        projection = encoder.da3.model.gs_head.scale_output_conv.net[2]
        with torch.no_grad():
            projection.weight.zero_()
            projection.bias.zero_()
        max_abs = max(
            float(projection.weight.float().abs().max()),
            float(projection.bias.float().abs().max()),
        )
        if max_abs != 0.0:
            raise RuntimeError(f"scale head was not zeroed: max_abs={max_abs}")
    step = int(payload.get("global_step", 0))
    print(
        f"initialized encoder from {checkpoint} loaded={len(selected)} "
        f"missing={len(missing)} unexpected={len(unexpected)} "
        f"skipped_scale={len(skipped_scale)} global_step={step}",
        flush=True,
    )
    return {
        "path": str(checkpoint),
        "loaded_keys": len(selected),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "skipped_scale_keys": len(skipped_scale),
        "global_step": step,
    }


def _cpu(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach().cpu()


def _decode_image(encoded: torch.Tensor) -> torch.Tensor:
    with Image.open(BytesIO(encoded.numpy().tobytes())) as image:
        return transforms.ToTensor()(image.convert("RGB"))


def _convert_cameras(cameras: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    intrinsics = torch.eye(3, dtype=torch.float32).repeat(cameras.shape[0], 1, 1)
    intrinsics[:, 0, 0] = cameras[:, 0]
    intrinsics[:, 1, 1] = cameras[:, 1]
    intrinsics[:, 0, 2] = cameras[:, 2]
    intrinsics[:, 1, 2] = cameras[:, 3]
    w2c = torch.eye(4, dtype=torch.float32).repeat(cameras.shape[0], 1, 1)
    w2c[:, :3] = cameras[:, 6:].reshape(-1, 3, 4)
    return w2c.inverse(), intrinsics


def _resize_and_crop(
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    output_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy the exact resize/crop and normalized-K update from original forward."""
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


def _normalize_scene_poses(
    c2w: torch.Tensor,
    context_indices: list[int],
    target_index: int,
    pose_reference_indices: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Copy the original forward's target-anchored pose normalization."""
    context_raw = c2w[context_indices]
    target_raw = c2w[[target_index]]
    reference_raw = c2w[pose_reference_indices] if pose_reference_indices is not None else context_raw
    scale = compute_pose_norm_scale(reference_raw, "max_pairwise_d")
    all_poses = torch.cat([context_raw, target_raw], dim=0).clone()
    all_poses[:, :3, 3] /= scale
    anchor = c2w[[pose_reference_indices[0]]] if pose_reference_indices is not None else all_poses[:1]
    if pose_reference_indices is not None:
        anchor = anchor.clone()
        anchor[:, :3, 3] /= scale
    all_poses = camera_normalization(anchor, all_poses)
    return all_poses[: len(context_indices)], all_poses[len(context_indices) :], float(scale)


def _load_context_from_chunk(args: argparse.Namespace) -> dict[str, object]:
    chunk_path = args.chunk
    if chunk_path is None:
        train_chunks = sorted((args.data_root / "10K" / "train").glob("*.torch"))
        if not train_chunks:
            raise FileNotFoundError(f"No DL3DV chunks under {args.data_root / '10K' / 'train'}")
        chunk_path = train_chunks[0]
    chunk = torch.load(chunk_path, map_location="cpu")
    scene = chunk[args.scene_index]
    if args.context_indices is None:
        context_indices = [i * args.context_stride for i in range(args.num_context)]
    else:
        context_indices = [int(i) for i in args.context_indices]
        if len(context_indices) != int(args.num_context):
            raise ValueError(
                "--context-indices must contain exactly --num-context entries: "
                f"got {len(context_indices)} for {args.num_context}"
            )
    selected = context_indices + [args.target_index]
    if max(selected) >= len(scene["images"]):
        raise ValueError(f"requested frame exceeds scene length: {selected}")
    raw_c2w, raw_intrinsics = _convert_cameras(scene["cameras"])
    context_c2w, target_c2w, pose_scale = _normalize_scene_poses(
        raw_c2w, context_indices, args.target_index, args.pose_reference_indices
    )
    images = torch.stack([_decode_image(scene["images"][i]) for i in selected])
    images, intrinsics = _resize_and_crop(
        images, raw_intrinsics[selected], (args.height, args.width)
    )
    return {
        "images": images[: args.num_context].unsqueeze(0),
        "extrinsics": context_c2w.unsqueeze(0),
        "intrinsics": intrinsics[: args.num_context].unsqueeze(0),
        "context_indices": context_indices,
        "scene": scene["key"],
        "chunk": str(chunk_path),
        "scene_index": int(args.scene_index),
        "target_index": int(args.target_index),
        "pose_norm_scale_raw": pose_scale,
        "source_mode": "dl3dv_forward_pipeline",
    }


def _load_context_from_dump(path: Path) -> dict[str, object]:
    source = torch.load(path, map_location="cpu", weights_only=False)
    required = ("context_image", "context_extrinsics", "context_intrinsics")
    missing = [key for key in required if key not in source]
    if missing:
        raise KeyError(f"source dump missing required fields: {missing}")
    return {
        "images": source["context_image"].float(),
        "extrinsics": source["context_extrinsics"].float(),
        "intrinsics": source["context_intrinsics"].float(),
        "context_indices": source.get("context_indices"),
        "scene": source.get("scene"),
        "chunk": None,
        "scene_index": None,
        "target_index": None,
        "pose_norm_scale_raw": None,
        "source_mode": "legacy_source_dump",
        "source_dump": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/root/data/lhmd/dl3dv_torch_960"))
    parser.add_argument("--chunk", type=Path, default=Path("/root/data/lhmd/dl3dv_torch_960/10K/train/000000.torch"))
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--num-context", type=int, default=6)
    parser.add_argument("--context-stride", type=int, default=2)
    parser.add_argument(
        "--context-indices", type=int, nargs="+", default=None,
        help="explicit source frame indices; overrides context-stride",
    )
    parser.add_argument(
        "--pose-reference-indices", type=int, nargs="+", default=None,
        help=(
            "optional frame indices used only for pose normalization scale/anchor; "
            "use the training sampler's context set when evaluating a view subset"
        ),
    )
    parser.add_argument("--target-index", type=int, default=12)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument(
        "--source-dump", type=Path, default=None,
        help="legacy preprocessed context dump; bypasses the original forward data pipeline",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("/root/data/haoxuan/indoor_970_da3_native_forward_epoch131_step2700.pt"),
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("/root/data/haoxuan/Depth-Anything-3/checkpoints/DA3-GIANT-1.1"),
    )
    parser.add_argument(
        "--init-weights", type=Path,
        default=Path(
            "/root/data/haoxuan/TriSplat/outputs/exp_lgtm10k_train/"
            "2026-08-28_11-58-26/checkpoints/epoch_131-step_2700.ckpt"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--global-step", type=int, default=None)
    parser.add_argument("--triangle-scale-min", type=float, default=1.0)
    parser.add_argument("--triangle-scale-max", type=float, default=18.0)
    parser.add_argument("--triangle-scale-init-fraction", type=float, default=0.25)
    parser.add_argument("--sigma-scale", type=float, default=4.6)
    parser.add_argument("--opacity-global-scale", type=float, default=0.62)
    parser.add_argument("--max-triangle-edge-px", type=float, default=0.0)
    parser.add_argument("--no-align-to-context-pose", action="store_true")
    parser.add_argument("--center-only-alignment", action="store_true")
    parser.add_argument("--normalize-input", action="store_true")
    parser.add_argument("--preserve-zero-scale-head", action="store_true")
    args = parser.parse_args()

    if args.pose_reference_indices is not None and args.source_dump is not None:
        raise ValueError("--pose-reference-indices requires the raw DL3DV chunk path")
    if args.source_dump is not None:
        context_state = _load_context_from_dump(args.source_dump)
    else:
        context_state = _load_context_from_chunk(args)
    images = context_state["images"].float()
    extrinsics = context_state["extrinsics"].float()
    intrinsics = context_state["intrinsics"].float()
    if images.ndim != 5 or images.shape[0] != 1:
        raise ValueError(f"context images must be [1,V,3,H,W], got {tuple(images.shape)}")
    views = int(images.shape[1])
    if views != args.num_context:
        raise ValueError(f"expected {args.num_context} context views, got {views}")
    if extrinsics.shape[:2] != (1, views) or intrinsics.shape[:2] != (1, views):
        raise ValueError("camera tensors do not match context image view count")
    if tuple(images.shape[-2:]) != (args.height, args.width):
        raise ValueError(
            f"context shape {tuple(images.shape[-2:])} does not match "
            f"requested {(args.height, args.width)}"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the DA3 GIANT encoder")

    cfg = EncoderDA3TSDPTCfg(
        checkpoint=str(args.checkpoint),
        triangle_scale_min=args.triangle_scale_min,
        triangle_scale_max=args.triangle_scale_max,
        triangle_scale_init_fraction=args.triangle_scale_init_fraction,
        max_triangle_edge_px=args.max_triangle_edge_px,
        sigma_scale_initial=args.sigma_scale,
        sigma_scale_final=args.sigma_scale,
        sigma_warmup_steps=1,
        opacity_global_scale=args.opacity_global_scale,
        sh_degree=0,
        align_to_context_pose=not args.no_align_to_context_pose,
        align_to_context_rotation=not args.center_only_alignment,
        use_input_normalization=args.normalize_input,
    )
    encoder = EncoderDA3TSDPT(cfg)
    init_meta = (
        _load_init(encoder, args.init_weights, args.preserve_zero_scale_head)
        if args.init_weights is not None
        else None
    )
    encoder = encoder.to(device).eval()
    context = {
        "image": images.to(device),
        "extrinsics": extrinsics.to(device),
        "intrinsics": intrinsics.to(device),
    }
    context = encoder.get_data_shim()({"context": context})["context"]
    step = int(args.global_step if args.global_step is not None else (init_meta or {}).get("global_step", 0))
    visualization: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        triangles = encoder(context, global_step=step, visualization_dump=visualization)
    if not isinstance(triangles, Triangles):
        raise TypeError(f"encoder returned {type(triangles).__name__}, expected Triangles")
    if triangles.centers is None or triangles.normals is None:
        raise RuntimeError("EncoderDA3TSDPT did not return native centers/normals")
    count = int(triangles.vertices.shape[1])
    expected = views * int(images.shape[-2]) * int(images.shape[-1])
    if count != expected:
        raise ValueError(f"primitive count {count} != context image grid {expected}")

    out = {
        "vertices": _cpu(triangles.vertices),
        "centers": _cpu(triangles.centers),
        "normals": _cpu(triangles.normals),
        "features": _cpu(triangles.features),
        "opacity": _cpu(triangles.opacity),
        "sigma": _cpu(triangles.sigma),
        "scales": _cpu(triangles.scales),
        "mapped_scales": _cpu(triangles.mapped_scales),
        "primitive_valid_mask": _cpu(triangles.primitive_valid_mask),
        "native_depth_conf": _cpu(visualization.get("da3_depth_conf")),
        "native_raw_gs_conf": _cpu(visualization.get("da3_raw_gs_conf")),
        "context_image": images,
        "context_extrinsics": extrinsics,
        "context_intrinsics": intrinsics,
        "context_indices": context_state["context_indices"],
        "scene": context_state["scene"],
        "source_dump": context_state.get("source_dump"),
        "export_config": {
            "pipeline": context_state["source_mode"],
            "data_root": str(args.data_root) if args.source_dump is None else None,
            "chunk": context_state["chunk"],
            "scene_index": context_state["scene_index"],
            "target_index": context_state["target_index"],
            "pose_reference_indices": args.pose_reference_indices,
            "pose_norm_scale_raw": context_state["pose_norm_scale_raw"],
            "checkpoint": str(args.checkpoint),
            "init_weights": init_meta,
            "global_step": step,
            "triangle_scale_min": args.triangle_scale_min,
            "triangle_scale_max": args.triangle_scale_max,
            "triangle_scale_init_fraction": args.triangle_scale_init_fraction,
            "max_triangle_edge_px": args.max_triangle_edge_px,
            "sigma_scale": args.sigma_scale,
            "opacity_global_scale": args.opacity_global_scale,
            "align_to_context_pose": not args.no_align_to_context_pose,
            "align_to_context_rotation": not args.center_only_alignment,
            "use_context_camera_tokens": True,
            "input_mean": [0.485, 0.456, 0.406],
            "input_std": [0.229, 0.224, 0.225],
            "use_input_normalization": args.normalize_input,
            "input_shape": [int(images.shape[-2]), int(images.shape[-1])],
            "views": views,
        },
    }
    out["visualization"] = {
        key: _cpu(value) for key, value in visualization.items() if isinstance(value, torch.Tensor)
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    summary = {
        "out": str(args.out),
        "pipeline": context_state["source_mode"],
        "scene": context_state["scene"],
        "context_indices": context_state["context_indices"],
        "input_shape": [int(images.shape[-2]), int(images.shape[-1])],
        "triangles": count,
        "centers": list(out["centers"].shape),
        "normals": list(out["normals"].shape),
        "finite_vertices": bool(torch.isfinite(out["vertices"]).all()),
        "finite_centers": bool(torch.isfinite(out["centers"]).all()),
        "finite_normals": bool(torch.isfinite(out["normals"]).all()),
        "checkpoint": init_meta,
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
