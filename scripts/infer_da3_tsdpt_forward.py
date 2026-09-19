#!/usr/bin/env python3
"""Run one DA3-TSDPT forward pass and render a DL3DV target view.

This is intentionally a small inspection script rather than a training entry
point. It uses the native TriSplat triangle/sigma/decoder ranges and writes
all intermediate images needed to diagnose geometry and visibility.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DA3_SRC = REPO_ROOT.parent / "Depth-Anything-3" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DA3_SRC) not in sys.path:
    sys.path.insert(0, str(DA3_SRC))

# The inspection forward only needs ``dataset.types``, the normalization shim,
# and ``norm_scale``.  Importing ``src.dataset`` normally executes the full
# dataset registry, which pulls in optional Lightning training dependencies.
# Keep this inference tool usable in the lean renderer environment by loading
# those three small modules through a lightweight package stub when Lightning
# is not installed.
try:
    from src.dataset.norm_scale import compute_pose_norm_scale
except ModuleNotFoundError as exc:
    if "lightning" not in str(exc):
        raise
    import importlib.util
    import types

    dataset_root = REPO_ROOT / "src" / "dataset"
    dataset_pkg = types.ModuleType("src.dataset")
    dataset_pkg.__path__ = [str(dataset_root)]
    sys.modules["src.dataset"] = dataset_pkg

    def _load_lightweight_module(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {name} from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    _load_lightweight_module("src.dataset.types", dataset_root / "types.py")
    shims_pkg = types.ModuleType("src.dataset.shims")
    shims_pkg.__path__ = [str(dataset_root / "shims")]
    sys.modules["src.dataset.shims"] = shims_pkg
    _load_lightweight_module(
        "src.dataset.shims.normalize_shim",
        dataset_root / "shims" / "normalize_shim.py",
    )
    norm_scale_module = _load_lightweight_module(
        "src.dataset.norm_scale", dataset_root / "norm_scale.py"
    )
    compute_pose_norm_scale = norm_scale_module.compute_pose_norm_scale

# Depth-Anything-3 imports its optional video-export helper at module import
# time.  Forward inference never calls that path, so provide an empty module
# when moviepy is not installed rather than requiring the training/export
# extras just to change the context stride.
try:
    import moviepy.editor  # type: ignore  # noqa: F401
except ModuleNotFoundError as exc:
    if "moviepy" not in str(exc):
        raise
    moviepy_stub = types.ModuleType("moviepy")
    moviepy_editor_stub = types.ModuleType("moviepy.editor")
    moviepy_stub.editor = moviepy_editor_stub
    sys.modules["moviepy"] = moviepy_stub
    sys.modules["moviepy.editor"] = moviepy_editor_stub
from src.misc.cam_utils import camera_normalization
from src.model.decoder.decoder_triangle_splatting_cuda import (
    DecoderTriangleSplattingCUDA,
    DecoderTriangleSplattingCUDACfg,
)
from src.model.encoder.encoder_da3_tsdpt import EncoderDA3TSDPT, EncoderDA3TSDPTCfg
from src.model.types import Triangles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/root/data/lhmd/dl3dv_torch_960"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/root/data/haoxuan/Depth-Anything-3/checkpoints/DA3-GIANT-1.1"),
    )
    parser.add_argument("--chunk", type=Path, default=None)
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--num-context", type=int, default=6)
    parser.add_argument("--context-stride", type=int, default=20)
    parser.add_argument("--target-index", type=int, default=140)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=Path("outputs/da3_tsdpt_forward"))
    parser.add_argument(
        "--opacity-init",
        type=float,
        default=None,
        help="Optional opacity-bias override for an untrained head; disabled for checkpoint evaluation.",
    )
    parser.add_argument("--triangle-scale-min", type=float, default=1.0)
    parser.add_argument("--triangle-scale-max", type=float, default=18.0)
    parser.add_argument("--triangle-scale-init-fraction", type=float, default=0.25)
    parser.add_argument("--max-triangle-edge-px", type=float, default=0.0)
    parser.add_argument(
        "--init-weights",
        type=Path,
        default=Path(
            "/root/data/haoxuan/TriSplat/outputs/"
            "exp_tsdpt_da3_dl3dv_scale_schedule_resume200_v2/2026-08-25_12-12-52/"
            "checkpoints/epoch_1-step_1000.ckpt"
        ),
        help="TriSplat checkpoint to evaluate. All matching output-head weights are loaded.",
    )
    parser.add_argument(
        "--preserve-zero-scale-head",
        action="store_true",
        help="Skip and zero the scale head, for fresh-head initialization tests only.",
    )
    parser.add_argument(
        "--global-step",
        type=int,
        default=None,
        help="Schedule step for the forward pass; defaults to the checkpoint global_step.",
    )
    return parser.parse_args()


def load_encoder_initialization(
    encoder: EncoderDA3TSDPT,
    path: Path,
    preserve_zero_scale_head: bool = False,
) -> dict[str, object]:
    """Load matching encoder weights, optionally preserving a fresh scale head."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported initialization checkpoint: {path}")

    encoder_state: dict[str, torch.Tensor] = {}
    skipped_scale: list[str] = []
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key.startswith("encoder."):
            key = key[len("encoder."):]
        if preserve_zero_scale_head and "scale_output_conv" in key:
            skipped_scale.append(key)
            continue
        encoder_state[key] = value

    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if preserve_zero_scale_head:
        # The hidden scale layer is intentionally Kaiming-initialized; only the
        # final projection must be zero so fresh logits start at zero.
        scale_projection = encoder.da3.model.gs_head.scale_output_conv.net[2]
        with torch.no_grad():
            scale_projection.weight.zero_()
            scale_projection.bias.zero_()
        scale_max_abs = max(
            float(scale_projection.weight.float().abs().max().item()),
            float(scale_projection.bias.float().abs().max().item()),
        )
        if scale_max_abs != 0.0:
            raise RuntimeError(
                "scale_output_conv was expected to remain zero-initialized, "
                f"but max abs parameter value is {scale_max_abs}"
            )
    checkpoint_step = int(checkpoint.get("global_step", 0))
    print(
        f"initialized encoder from {path} (loaded={len(encoder_state)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}, "
        f"skipped_scale={len(skipped_scale)}, global_step={checkpoint_step})",
        flush=True,
    )
    return {
        "path": str(path),
        "loaded_keys": len(encoder_state),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "skipped_scale_keys": len(skipped_scale),
        "global_step": checkpoint_step,
    }


def decode_image(encoded: torch.Tensor) -> torch.Tensor:
    with Image.open(BytesIO(encoded.numpy().tobytes())) as image:
        return transforms.ToTensor()(image.convert("RGB"))


def convert_cameras(cameras: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    intrinsics = torch.eye(3, dtype=torch.float32).repeat(cameras.shape[0], 1, 1)
    intrinsics[:, 0, 0] = cameras[:, 0]
    intrinsics[:, 1, 1] = cameras[:, 1]
    intrinsics[:, 0, 2] = cameras[:, 2]
    intrinsics[:, 1, 2] = cameras[:, 3]

    w2c = torch.eye(4, dtype=torch.float32).repeat(cameras.shape[0], 1, 1)
    w2c[:, :3] = cameras[:, 6:].reshape(-1, 3, 4)
    return w2c.inverse(), intrinsics


def resize_and_crop(
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    output_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match TriSplat's dataset resize/crop, including the normalized-K update.

    The previous inspection script scaled normalized ``fx/fy`` by the *input*
    dimensions (``in_w/out_w`` and ``in_h/out_h``) even though the image was
    first resized by ``factor``.  For a 960x540 frame going to 448x224 this
    doubled the horizontal focal length and produced the apparent view-change
    warp.  Intrinsics must be updated using the actual resized dimensions and
    the crop offset, exactly as the dataset shim does.
    """
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


def save_tensor_image(path: Path, image: torch.Tensor) -> None:
    image = image.detach().float().cpu().clamp(0, 1)
    if image.ndim == 2:
        image = image.unsqueeze(0).expand(3, -1, -1)
    if image.shape[0] == 1:
        image = image.expand(3, -1, -1)
    array = (image.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def save_normal(path: Path, normal: torch.Tensor, valid: torch.Tensor | None = None) -> None:
    normal = normal.detach().float().cpu()
    normal = (normal + 1.0) * 0.5
    if valid is not None:
        valid = valid.detach().to(device=normal.device)
        normal = torch.where(valid.bool(), normal, torch.zeros_like(normal))
    save_tensor_image(path, normal)


def project_world_points(
    points_world: torch.Tensor,
    c2w: torch.Tensor,
    intrinsics_normalized: torch.Tensor,
    image_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project world-space points with the exact camera convention used by TriSplat.

    ``c2w`` is the dataset camera-to-world pose and ``intrinsics_normalized``
    stores ``fx/fy/cx/cy`` normalized by the output width/height.  Keeping this
    conversion in one function is important: the old diagnostic exporter used
    one transform for context points and a second, subtly different transform
    for target points, which hid the pose-origin bug in the encoder.
    """
    height, width = image_shape
    rotation = c2w[..., :3, :3]
    translation = c2w[..., :3, 3]
    # c2w is a rigid transform, so its inverse rotation is its transpose.
    points_cam = torch.einsum(
        "...ji,...hwj->...hwi",
        rotation,
        points_world - translation[..., None, None, :],
    )
    intrinsics_pixel = intrinsics_normalized.clone()
    intrinsics_pixel[..., 0, :] *= float(width)
    intrinsics_pixel[..., 1, :] *= float(height)
    z = points_cam[..., 2]
    xy = points_cam[..., :2] / z.clamp_min(1e-6)[..., None]
    pixels = torch.stack(
        (
            xy[..., 0] * intrinsics_pixel[..., 0, 0][..., None, None]
            + intrinsics_pixel[..., 0, 2][..., None, None],
            xy[..., 1] * intrinsics_pixel[..., 1, 1][..., None, None]
            + intrinsics_pixel[..., 1, 2][..., None, None],
        ),
        dim=-1,
    )
    finite = torch.isfinite(points_cam).all(dim=-1) & torch.isfinite(pixels).all(dim=-1)
    valid = finite & (z > 0) & (pixels[..., 0] >= 0) & (pixels[..., 0] < width)
    valid = valid & (pixels[..., 1] >= 0) & (pixels[..., 1] < height)
    return points_cam, pixels, valid


def zbuffer_normals(
    pixels: torch.Tensor,
    depths: torch.Tensor,
    normals: torch.Tensor,
    valid: torch.Tensor,
    image_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-point splat of normals/depth into an image, without Python loops."""
    height, width = image_shape
    normal_image = torch.zeros((height, width, 3), device=normals.device, dtype=normals.dtype)
    depth_image = torch.zeros((height, width), device=depths.device, dtype=depths.dtype)
    flat_valid = valid.reshape(-1)
    if not flat_valid.any():
        return normal_image, depth_image
    flat_pixels = pixels.reshape(-1, 2)[flat_valid]
    flat_depths = depths.reshape(-1)[flat_valid]
    flat_normals = normals.reshape(-1, 3)[flat_valid]
    x = flat_pixels[:, 0].floor().long().clamp(0, width - 1)
    y = flat_pixels[:, 1].floor().long().clamp(0, height - 1)
    pixel_index = y * width + x
    pixel_count = height * width
    zbuffer = torch.full(
        (pixel_count,), float("inf"), device=depths.device, dtype=depths.dtype
    )
    zbuffer.scatter_reduce_(0, pixel_index, flat_depths, reduce="amin", include_self=True)
    # Select the lowest source index among depth ties for deterministic output.
    source_index = torch.arange(flat_depths.numel(), device=depths.device)
    winner = flat_depths <= zbuffer[pixel_index] + 1e-6
    selected = torch.full(
        (pixel_count,), flat_depths.numel(), device=depths.device, dtype=torch.long
    )
    selected.scatter_reduce_(
        0, pixel_index[winner], source_index[winner], reduce="amin", include_self=True
    )
    covered = selected < flat_depths.numel()
    normal_flat = torch.zeros((pixel_count, 3), device=normals.device, dtype=normals.dtype)
    normal_flat[covered] = flat_normals[selected[covered]]
    depth_flat = torch.zeros((pixel_count,), device=depths.device, dtype=depths.dtype)
    depth_flat[covered] = flat_depths[selected[covered]]
    return normal_flat.reshape(height, width, 3), depth_flat.reshape(height, width)


def write_ply(
    path: Path,
    points: torch.Tensor,
    normals: torch.Tensor,
) -> None:
    """Write an ASCII PLY with explicit point/normal columns."""
    points_np = points.detach().float().cpu().reshape(-1, 3).numpy()
    normals_np = normals.detach().float().cpu().reshape(-1, 3).numpy()
    finite = np.isfinite(points_np).all(axis=1) & np.isfinite(normals_np).all(axis=1)
    points_np, normals_np = points_np[finite], normals_np[finite]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points_np)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property float nx\nproperty float ny\nproperty float nz\nend_header\n")
        np.savetxt(handle, np.concatenate((points_np, normals_np), axis=1), fmt="%.7g")


def rotation_angle_deg(relative_rotation: torch.Tensor) -> torch.Tensor:
    """Compute a stable SO(3) angle after removing float round-off."""
    u, _, vh = torch.linalg.svd(relative_rotation)
    projected = u @ vh
    det = torch.linalg.det(projected)
    if (det < 0).any():
        u = u.clone()
        u[..., :, -1] *= torch.where(
            det < 0,
            -torch.ones_like(det),
            torch.ones_like(det),
        )[..., None]
        projected = u @ vh
    cosine = ((projected.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def pose_comparison(
    predicted_c2w: torch.Tensor,
    target_c2w: torch.Tensor,
    align_rotation: torch.Tensor,
    align_translation: torch.Tensor,
    align_scale: torch.Tensor,
    per_view_rotation: torch.Tensor | None = None,
) -> dict:
    """Compare DA3 poses before/after center and per-view alignment."""
    pred_r = predicted_c2w[:, :3, :3]
    pred_t = predicted_c2w[:, :3, 3]
    target_r = target_c2w[:, :3, :3]
    target_t = target_c2w[:, :3, 3]
    scale = align_scale.reshape(1, 1)
    aligned_t = scale * torch.einsum("ij,vj->vi", align_rotation, pred_t) + align_translation
    aligned_r = torch.einsum("ij,vjk->vik", align_rotation, pred_r)
    relative_r = torch.einsum("vji,vjk->vik", target_r, aligned_r)
    rotation_error_deg = rotation_angle_deg(relative_r)
    center_before = (pred_t - target_t).norm(dim=-1)
    center_after = (aligned_t - target_t).norm(dim=-1)
    stats = {
        "da3_intrinsics": None,
        "center_error_before": center_before.detach().cpu().tolist(),
        "center_error_after_center_sim3": center_after.detach().cpu().tolist(),
        "center_error_before_mean": float(center_before.mean()),
        "center_error_after_center_sim3_mean": float(center_after.mean()),
        "rotation_error_after_center_sim3_deg": rotation_error_deg.detach().cpu().tolist(),
        "rotation_error_after_center_sim3_deg_mean": float(rotation_error_deg.mean()),
        "predicted_camera_centers": pred_t.detach().cpu().tolist(),
        "target_camera_centers": target_t.detach().cpu().tolist(),
        "aligned_camera_centers": aligned_t.detach().cpu().tolist(),
        "sim3_scale": float(align_scale.item()),
    }
    if per_view_rotation is not None:
        corrected_r = torch.matmul(per_view_rotation, aligned_r)
        corrected_relative = torch.einsum("vji,vjk->vik", target_r, corrected_r)
        corrected_error = rotation_angle_deg(corrected_relative)
        stats["rotation_error_after_per_view_alignment_deg"] = corrected_error.detach().cpu().tolist()
        stats["rotation_error_after_per_view_alignment_deg_mean"] = float(corrected_error.mean())
    return stats


def normalize_scene_poses(
    c2w: torch.Tensor,
    context_indices: list[int],
    target_index: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    context_raw = c2w[context_indices]
    target_raw = c2w[[target_index]]
    scale = compute_pose_norm_scale(context_raw, "max_pairwise_d")
    all_poses = torch.cat([context_raw, target_raw], dim=0).clone()
    all_poses[:, :3, 3] /= scale
    all_poses = camera_normalization(all_poses[:1], all_poses)
    return all_poses[: len(context_indices)], all_poses[len(context_indices) :], float(scale)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the DA3 GIANT forward and triangle renderer.")
    device = torch.device(args.device)
    output_dir = args.out.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_path = args.chunk
    if chunk_path is None:
        train_chunks = sorted((args.data_root / "10K" / "train").glob("*.torch"))
        if not train_chunks:
            raise FileNotFoundError(f"No DL3DV chunks under {args.data_root / '10K' / 'train'}")
        chunk_path = train_chunks[0]
    chunk = torch.load(chunk_path, map_location="cpu")
    scene = chunk[args.scene_index]
    num_frames = len(scene["images"])
    context_indices = [i * args.context_stride for i in range(args.num_context)]
    if max(context_indices + [args.target_index]) >= num_frames:
        raise ValueError(f"Requested frame exceeds scene length {num_frames}: {context_indices}")

    raw_c2w, raw_intrinsics = convert_cameras(scene["cameras"])
    context_c2w, target_c2w, pose_scale = normalize_scene_poses(
        raw_c2w, context_indices, args.target_index
    )
    selected = context_indices + [args.target_index]
    images = torch.stack([decode_image(scene["images"][i]) for i in selected])
    images, intrinsics = resize_and_crop(images, raw_intrinsics[selected], (args.height, args.width))
    context_images = images[: args.num_context].unsqueeze(0).to(device)
    target_image = images[args.num_context :].unsqueeze(0).to(device)
    context = {
        "image": context_images,
        "extrinsics": context_c2w.unsqueeze(0).to(device),
        "intrinsics": intrinsics[: args.num_context].unsqueeze(0).to(device),
    }
    target = {
        "image": target_image,
        "extrinsics": target_c2w.unsqueeze(0).to(device),
        "intrinsics": intrinsics[args.num_context :].unsqueeze(0).to(device),
        "near": torch.full((1, 1), 0.01, device=device),
        "far": torch.full((1, 1), 1000.0, device=device),
    }

    # Native TriSplat settings from the DL3DV triangle-refiner experiment.
    encoder_cfg = EncoderDA3TSDPTCfg(
        checkpoint=str(args.checkpoint),
        triangle_scale_min=args.triangle_scale_min,
        triangle_scale_max=args.triangle_scale_max,
        triangle_scale_init_fraction=args.triangle_scale_init_fraction,
        max_triangle_edge_px=args.max_triangle_edge_px,
        sigma_scale_initial=4.6,
        sigma_scale_final=4.6,
        sigma_warmup_steps=1,
        opacity_global_scale=0.62,
        sh_degree=0,
        align_to_context_pose=True,
    )
    encoder = EncoderDA3TSDPT(encoder_cfg)
    initialization = None
    if args.init_weights is not None:
        if not args.init_weights.is_file():
            raise FileNotFoundError(f"Initialization checkpoint not found: {args.init_weights}")
        initialization = load_encoder_initialization(
            encoder,
            args.init_weights,
            preserve_zero_scale_head=args.preserve_zero_scale_head,
        )
    encoder = encoder.to(device).eval()
    if args.opacity_init is not None:
        opacity_logit = math.log(args.opacity_init / (1.0 - args.opacity_init))
        with torch.no_grad():
            encoder.da3.model.gs_head.extra_output_conv.opacity[2].bias[0] = opacity_logit

    schedule_step = (
        int(args.global_step)
        if args.global_step is not None
        else int(initialization.get("global_step", 0) if initialization else 0)
    )

    visualization_dump: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        triangles = encoder(context, global_step=schedule_step, visualization_dump=visualization_dump)
    if not isinstance(triangles, Triangles):
        raise TypeError(f"Expected Triangles, got {type(triangles).__name__}")

    decoder = DecoderTriangleSplattingCUDA(
        DecoderTriangleSplattingCUDACfg(
            name="triangle_splatting_cuda",
            background_color=[0.0, 0.0, 0.0],
            opacity_temp_initial=1.0,
            opacity_temp_final=5.0,
            opacity_temp_warmup_steps=8000,
            alpha_floor_min=0.0,
            alpha_floor_warmup_steps=0,
            sh_degree=0,
            prune_opacity_threshold=0.0,
        )
    ).to(device).eval()
    with torch.inference_mode():
        rendered = decoder(
            triangles,
            target["extrinsics"],
            target["intrinsics"],
            target["near"],
            target["far"],
            image_shape=(args.height, args.width),
            global_step=schedule_step,
            debug_log_interval=1,
            return_triangle_visibility_mask=True,
        )

    context_normal = visualization_dump["geom_normal_cam_forward"][0]
    context_mask = visualization_dump.get("geom_normal_mask")
    if context_mask is None:
        context_mask = torch.ones_like(context_normal[:, :1], dtype=torch.bool)
    else:
        context_mask = context_mask[0]
    for i, frame_index in enumerate(context_indices):
        save_tensor_image(output_dir / f"context_{frame_index:04d}.png", context_images[0, i])
        save_normal(output_dir / f"normal_{frame_index:04d}.png", context_normal[i], context_mask[i])
    save_tensor_image(output_dir / "target.png", target_image[0, 0])

    # Export the raw target-view point cloud from the same canonical world map
    # used to build the triangles.  In particular, do not reconstruct world
    # points from DA3-local points with the pre-alignment pose: that was the
    # source of the old context/target mismatch (the Sim(3) translation was
    # silently lost in the encoder).  Both the diagnostic reprojection below
    # and the CUDA renderer therefore consume exactly the same world points.
    points_world = visualization_dump.get("points_world_aligned")
    if points_world is None:
        # This fallback keeps the inspection script usable with checkpoints or
        # configs that disable pose alignment, while still using one transform
        # implementation for every view.
        local_points = visualization_dump["local_pts"]
        context_pose = visualization_dump["c2w"]
        points_world = torch.einsum(
            "bvij,bvhwj->bvhwi", context_pose[..., :3, :3], local_points
        ) + context_pose[..., :3, 3][..., None, None, :]
    points_world = points_world[0].float()
    if triangles.normals is None:
        raise RuntimeError("Triangle adapter did not return normals for point-cloud export")
    normals_world = triangles.normals[0].reshape(
        args.num_context, args.height, args.width, 3
    ).float()
    target_pose = target["extrinsics"][0, 0].float()
    target_intrinsics = target["intrinsics"][0, 0].float()
    target_points_cam, target_pixels, target_valid = project_world_points(
        points_world, target_pose, target_intrinsics, (args.height, args.width)
    )
    target_normals_cam = torch.einsum(
        "ji,vhwj->vhwi", target_pose[:3, :3], normals_world
    )
    target_normals_cam = F.normalize(target_normals_cam, dim=-1, eps=1e-6)
    target_normal_image, target_depth_image = zbuffer_normals(
        target_pixels,
        target_points_cam[..., 2],
        target_normals_cam,
        target_valid,
        (args.height, args.width),
    )
    save_normal(
        output_dir / "target_reprojected_normal.png",
        target_normal_image.permute(2, 0, 1),
        target_depth_image > 0,
    )
    target_depth_valid = target_depth_image > 0
    target_depth_vis = torch.zeros_like(target_depth_image)
    if target_depth_valid.any():
        target_depth_values = target_depth_image[target_depth_valid]
        target_depth_vis[target_depth_valid] = (
            target_depth_values - target_depth_values.quantile(0.02)
        ) / (
            target_depth_values.quantile(0.98) - target_depth_values.quantile(0.02)
        ).clamp_min(1e-6)
    save_tensor_image(output_dir / "target_reprojected_depth.png", target_depth_vis)
    target_point_mask = target_valid
    write_ply(
        output_dir / "target_pointcloud.ply",
        target_points_cam[target_point_mask],
        target_normals_cam[target_point_mask],
    )

    # The world-point round trip should be exact up to floating-point error: the
    # aligned world point is converted back with the context c2w and must
    # recover the local point map used by TSAdapter.  The resulting pixel
    # residual is reported separately because the Sim(3) center alignment does
    # not force DA3's per-view orientation to equal the dataset orientation;
    # that residual is expected and is not a coordinate-convention failure.
    aligned_local = visualization_dump["local_pts"][0].float()
    context_pose = context["extrinsics"][0].float()
    reconstructed_world = torch.einsum(
        "vij,vhwj->vhwi", context_pose[..., :3, :3], aligned_local
    ) + context_pose[..., :3, 3][..., None, None, :]
    world_roundtrip_error = (reconstructed_world - points_world).norm(dim=-1)
    context_points_cam, context_pixels, context_front = project_world_points(
        points_world,
        context_pose,
        context["intrinsics"][0].float(),
        (args.height, args.width),
    )
    ys, xs = torch.meshgrid(
        torch.arange(args.height, device=context_pixels.device, dtype=context_pixels.dtype)
        + 0.5,
        torch.arange(args.width, device=context_pixels.device, dtype=context_pixels.dtype)
        + 0.5,
        indexing="ij",
    )
    expected_pixels = torch.stack((xs, ys), dim=-1)
    pixel_roundtrip_error = (context_pixels - expected_pixels).norm(dim=-1)
    pixel_roundtrip_error = pixel_roundtrip_error[context_front]
    reprojection_stats = {
        "world_roundtrip_error_max": float(world_roundtrip_error.max()),
        "world_roundtrip_error_mean": float(world_roundtrip_error.mean()),
        "context_projection_residual_after_pose_alignment_px_max": float(pixel_roundtrip_error.max())
        if pixel_roundtrip_error.numel()
        else None,
        "context_projection_residual_after_pose_alignment_px_mean": float(pixel_roundtrip_error.mean())
        if pixel_roundtrip_error.numel()
        else None,
        "target_point_count_in_bounds": int(target_point_mask.sum()),
        "target_pixel_coverage": float((target_depth_image > 0).float().mean()),
    }
    save_tensor_image(output_dir / "render_rgb.png", rendered.color[0, 0])
    save_normal(output_dir / "render_normal.png", rendered.rend_normal[0, 0])
    depth = rendered.depth[0, 0]
    depth_valid = depth > 0
    depth_vis = torch.zeros_like(depth)
    if depth_valid.any():
        values = depth[depth_valid]
        depth_vis[depth_valid] = (values - values.quantile(0.02)) / (
            values.quantile(0.98) - values.quantile(0.02)
        ).clamp_min(1e-6)
    save_tensor_image(output_dir / "render_depth.png", depth_vis)
    save_tensor_image(output_dir / "render_alpha.png", rendered.opacity[0, 0])

    opacity = triangles.opacity.detach().float().flatten()
    sigma = triangles.sigma.detach().float().flatten()
    primitive_valid_mask = triangles.primitive_valid_mask
    if primitive_valid_mask is None:
        primitive_valid_mask = torch.ones(
            triangles.opacity.shape[:2], device=triangles.opacity.device, dtype=torch.bool
        )
    valid = primitive_valid_mask.detach().bool().flatten()
    visibility = rendered.triangle_visibility_mask.detach().bool()
    pose_stats = pose_comparison(
        visualization_dump["da3_pred_c2w"][0].float(),
        context["extrinsics"][0].float(),
        visualization_dump["pose_alignment_rotation"][0].float(),
        visualization_dump["pose_alignment_translation"][0].float(),
        visualization_dump["pose_alignment_scale"].float(),
        visualization_dump.get("pose_alignment_per_view_rotation", None)[0].float()
        if visualization_dump.get("pose_alignment_per_view_rotation", None) is not None
        else None,
    )
    pose_stats["da3_intrinsics"] = visualization_dump["da3_pred_intrinsics"][0].float().cpu().tolist()
    pose_stats["gt_context_intrinsics_normalized"] = context["intrinsics"][0].float().cpu().tolist()
    gt_intrinsics_pixel = visualization_dump["da3_pred_intrinsics"][0].new_tensor(
        context["intrinsics"][0]
    ).clone()
    gt_intrinsics_pixel[:, 0, :] *= args.width
    gt_intrinsics_pixel[:, 1, :] *= args.height
    pose_stats["gt_context_intrinsics_pixel"] = gt_intrinsics_pixel.cpu().tolist()
    pred_depth = visualization_dump["da3_pred_depth"].float()
    pose_stats["da3_depth_quantiles"] = torch.quantile(
        pred_depth.flatten(), torch.tensor([0.01, 0.5, 0.99], device=pred_depth.device)
    ).cpu().tolist()
    stats = {
        "scene": scene["key"],
        "chunk": str(chunk_path),
        "context_indices": context_indices,
        "target_index": args.target_index,
        "global_step": schedule_step,
        "input_shape": [args.height, args.width],
        "pose_norm_scale_raw": pose_scale,
        "triangle_scale_range": [args.triangle_scale_min, args.triangle_scale_max],
        "triangle_scale_init_fraction": args.triangle_scale_init_fraction,
        "max_triangle_edge_px": args.max_triangle_edge_px,
        "initialization": initialization,
        "native_sigma_schedule": [1.0, 1.0, 1],
        "decoder_opacity_schedule": [1.0, 5.0, 5000],
        "opacity_bias_target": args.opacity_init,
        "triangle_count": int(opacity.numel()),
        "valid_triangle_fraction": float(valid.float().mean()),
        "opacity": {
            "min": float(opacity.min()),
            "mean": float(opacity.mean()),
            "median": float(opacity.median()),
            "max": float(opacity.max()),
        },
        "sigma": {
            "min": float(sigma.min()),
            "mean": float(sigma.mean()),
            "median": float(sigma.median()),
            "max": float(sigma.max()),
        },
        "render_alpha": {
            "mean": float(rendered.opacity.float().mean()),
            "max": float(rendered.opacity.float().max()),
            "nonzero_fraction": float((rendered.opacity > 1e-4).float().mean()),
        },
        "visible_triangle_fraction_target": float(visibility[0, 0].float().mean()),
        "render_finite": bool(torch.isfinite(rendered.color).all()),
        "reprojection": reprojection_stats,
        "pose_comparison": pose_stats,
    }
    torch.save(
        {
            "triangles": triangles,
            "context": context,
            "target": target,
            "visualization": visualization_dump,
            "render": rendered,
        },
        output_dir / "forward_render.pt",
    )
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()
