"""Forward TSDPT and native TriSplat on one evaluation scene and compare primitives."""

from __future__ import annotations

import argparse
import copy
import json
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import torch
from hydra import compose, initialize_config_dir
from PIL import Image

from src.checkpoint_utils import extract_state_dict, load_checkpoint_file
from src.config import load_typed_root_config
from src.dataset.data_module import DataModule, get_data_shim
from src.evaluation.metrics import compute_psnr
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
    return value


def compose_cfg(
    experiment: str,
    data_root: Path,
    index_path: Path,
    baseline: bool,
    scene: str,
    num_context_views: int,
    tsdpt_brightness_gain: float | None = None,
    tsdpt_saturation_gain: float | None = None,
):
    overrides = [
        f"+experiment={experiment}",
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
    if baseline:
        overrides.extend(
            [
                "model.encoder.triangle_adapter.triangle_scale_min=0.5",
                "model.encoder.triangle_adapter.triangle_scale_max=18.0",
            ]
        )
    else:
        if tsdpt_brightness_gain is not None:
            overrides.append(
                f"model.decoder.color_brightness_gain={tsdpt_brightness_gain}"
            )
        if tsdpt_saturation_gain is not None:
            overrides.append(
                f"model.decoder.color_saturation_gain={tsdpt_saturation_gain}"
            )
    with initialize_config_dir(config_dir=str(REPO_ROOT / "config"), version_base=None):
        return load_typed_root_config(compose(config_name="main", overrides=overrides))


def load_encoder(cfg, checkpoint: Path, device: torch.device):
    encoder, _ = get_encoder(cfg.model.encoder)
    state = extract_state_dict(load_checkpoint_file(checkpoint))
    state = {key[8:]: value for key, value in state.items() if key.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing:
        print(f"{checkpoint.name}: missing={len(missing)}", flush=True)
    if unexpected:
        print(f"{checkpoint.name}: unexpected={len(unexpected)}", flush=True)
    return encoder.to(device).eval()


def quantiles(tensor: torch.Tensor) -> list[float]:
    values = tensor.detach().float().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return [float("nan")] * 7
    # torch.quantile has a practical input-size limit on this CUDA build.
    # Keep the sample deterministic while retaining the full tensor for mean
    # and standard deviation in ``tensor_stats``.
    if values.numel() > 2_000_000:
        stride = (values.numel() + 2_000_000 - 1) // 2_000_000
        values = values[::stride]
    return [
        float(values.min()),
        float(torch.quantile(values, 0.01)),
        float(torch.quantile(values, 0.05)),
        float(torch.median(values)),
        float(torch.quantile(values, 0.95)),
        float(torch.quantile(values, 0.99)),
        float(values.max()),
    ]


def tensor_stats(tensor: torch.Tensor, per_axis: bool = False) -> dict:
    values = tensor.detach().float()
    finite = torch.isfinite(values)
    result = {
        "shape": list(values.shape),
        "finite_fraction": float(finite.float().mean()),
        "quantiles_min_p01_p05_median_p95_p99_max": quantiles(values),
        "mean": float(torch.nan_to_num(values).mean()),
        "std": float(torch.nan_to_num(values).std()),
    }
    if per_axis and values.shape[-1] <= 4:
        result["per_axis"] = [tensor_stats(values[..., axis]) for axis in range(values.shape[-1])]
    return result


def primitive_stats(primitives, visible_mask: torch.Tensor | None = None) -> dict:
    vertices = primitives.vertices.float()
    if visible_mask is not None:
        visible_mask = visible_mask.bool()
        vertices = vertices[:, visible_mask]

    def primitive_values(tensor: torch.Tensor) -> torch.Tensor:
        values = tensor.float()
        if visible_mask is not None and values.ndim >= 2:
            values = values[:, visible_mask]
        return values

    edges = torch.stack(
        (
            (vertices[..., 1, :] - vertices[..., 0, :]).norm(dim=-1),
            (vertices[..., 2, :] - vertices[..., 0, :]).norm(dim=-1),
            (vertices[..., 2, :] - vertices[..., 1, :]).norm(dim=-1),
        ),
        dim=-1,
    )
    area = torch.cross(
        vertices[..., 1, :] - vertices[..., 0, :],
        vertices[..., 2, :] - vertices[..., 0, :],
        dim=-1,
    ).norm(dim=-1) * 0.5
    result = {
        "triangle_count": int(vertices.shape[1]),
        "triangle_count_before_visibility_filter": int(primitives.vertices.shape[1]),
        "vertices": tensor_stats(vertices, per_axis=True),
        "centers": tensor_stats(primitive_values(primitives.centers), per_axis=True),
        "scales": tensor_stats(primitive_values(primitives.scales), per_axis=True),
        "mapped_scales": tensor_stats(primitive_values(primitives.mapped_scales), per_axis=True),
        "sigma": tensor_stats(primitive_values(primitives.sigma)),
        "opacity": tensor_stats(primitive_values(primitives.opacity)),
        "normals": tensor_stats(primitive_values(primitives.normals), per_axis=True),
        "edge_lengths": tensor_stats(edges, per_axis=True),
        "triangle_area": tensor_stats(area),
    }
    if visible_mask is not None:
        result["visible_triangle_fraction"] = float(visible_mask.float().mean())
    valid = getattr(primitives, "primitive_valid_mask", None)
    if valid is not None:
        result["primitive_valid_fraction"] = float(valid[:, visible_mask].float().mean()) if visible_mask is not None else float(valid.float().mean())
    return result


def projected_triangle_stats(
    vertices: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    near: torch.Tensor,
    image_shape: tuple[int, int],
    visibility: torch.Tensor | None = None,
    sigma: torch.Tensor | None = None,
    opacity: torch.Tensor | None = None,
) -> dict:
    """Measure triangle size in the target image, in pixels and pixel^2.

    ``intrinsics`` are the dataset's normalized K matrices, matching the
    decoder.  The visibility subset is the CUDA rasterizer's ``radii > 0``
    mask, so it is useful for separating projection size from actual render
    participation.
    """
    h, w = image_shape
    world_vertices = vertices[0].float()
    w2c = torch.linalg.inv(extrinsics[0].float())
    k = intrinsics[0].float().clone()
    k[..., 0, :] *= float(w)
    k[..., 1, :] *= float(h)
    near_values = near[0].float()
    projected = []
    per_target = []
    for target_idx in range(extrinsics.shape[1]):
        # Use the same OpenCV camera convention as the decoder.
        camera = torch.einsum("ij,nvj->nvi", w2c[target_idx, :3, :3], world_vertices - extrinsics[0, target_idx, :3, 3])
        z = camera[..., 2]
        safe_z = z.clamp_min(1e-4)
        xy = torch.stack(
            (
                camera[..., 0] / safe_z * k[target_idx, 0, 0] + k[target_idx, 0, 2],
                camera[..., 1] / safe_z * k[target_idx, 1, 1] + k[target_idx, 1, 2],
            ),
            dim=-1,
        )
        edges = torch.stack(
            (
                (xy[:, 1] - xy[:, 0]).norm(dim=-1),
                (xy[:, 2] - xy[:, 0]).norm(dim=-1),
                (xy[:, 2] - xy[:, 1]).norm(dim=-1),
            ),
            dim=-1,
        )
        area = (
            (
                xy[:, 0, 0] * (xy[:, 1, 1] - xy[:, 2, 1])
                + xy[:, 1, 0] * (xy[:, 2, 1] - xy[:, 0, 1])
                + xy[:, 2, 0] * (xy[:, 0, 1] - xy[:, 1, 1])
            ).abs()
            * 0.5
        )
        perimeter = edges.sum(dim=-1)
        incircle_radius = (2.0 * area / perimeter.clamp_min(1e-8)).nan_to_num(0.0)
        valid = torch.isfinite(xy).all(dim=-1).all(dim=-1) & (z > near_values[target_idx]).all(dim=-1)
        in_bounds = valid & (
            (xy[..., 0] >= 0.0)
            & (xy[..., 0] < float(w))
            & (xy[..., 1] >= 0.0)
            & (xy[..., 1] < float(h))
        ).all(dim=-1)
        target = {
            "target_index": target_idx,
            "projected_valid_fraction": float(valid.float().mean()),
            "all_vertices_in_bounds_fraction": float(in_bounds.float().mean()),
            "projected_edge_lengths_px": tensor_stats(edges, per_axis=True),
            "projected_area_px2": tensor_stats(area),
            "projected_incircle_radius_px": tensor_stats(incircle_radius),
        }
        if visibility is not None:
            visible = visibility[0, target_idx].bool() & valid
            target["raster_visible_fraction"] = float(visibility[0, target_idx].float().mean())
            target["visible_projected_edge_lengths_px"] = tensor_stats(edges[visible], per_axis=True)
            target["visible_projected_area_px2"] = tensor_stats(area[visible])
            target["visible_projected_incircle_radius_px"] = tensor_stats(incircle_radius[visible])
            if sigma is not None:
                target["visible_sigma"] = tensor_stats(sigma[0, visible])
            if opacity is not None:
                target["visible_opacity"] = tensor_stats(opacity[0, visible])
            target["visible_count"] = int(visible.sum())
        per_target.append(target)
        projected.append((edges, area, incircle_radius, valid, in_bounds))

    if visibility is not None:
        selected_masks = [visibility[0, i].bool() & item[3] for i, item in enumerate(projected)]
    else:
        selected_masks = [item[3] for item in projected]
    all_edges = torch.cat([item[0][mask] for item, mask in zip(projected, selected_masks)])
    all_area = torch.cat([item[1][mask] for item, mask in zip(projected, selected_masks)])
    all_incircle = torch.cat([item[2][mask] for item, mask in zip(projected, selected_masks)])
    aggregate_prefix = "visible_target" if visibility is not None else "valid_target"
    return {
        "image_shape": [h, w],
        f"{aggregate_prefix}_projected_edge_lengths_px": tensor_stats(all_edges, per_axis=True),
        f"{aggregate_prefix}_projected_area_px2": tensor_stats(all_area),
        f"{aggregate_prefix}_projected_incircle_radius_px": tensor_stats(all_incircle),
        "per_target": per_target,
    }


def forward_one(
    name: str,
    cfg,
    checkpoint: Path,
    raw_batch: dict,
    device: torch.device,
    global_step: int,
):
    encoder = load_encoder(cfg, checkpoint, device)
    decoder = get_decoder(cfg.model.decoder).to(device).eval()
    batch = copy.deepcopy(raw_batch)
    batch = get_data_shim(encoder)(batch)
    dump: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        primitives = encoder(batch["context"], global_step=global_step, visualization_dump=dump)
        target = batch["target"]
        output = decoder.forward(
            primitives,
            target["extrinsics"],
            target["intrinsics"],
            target["near"],
            target["far"],
            tuple(target["image"].shape[-2:]),
            global_step=global_step,
            return_triangle_visibility_mask=True,
        )
    pred = output.color[0].float().clamp(0, 1)
    gt = raw_batch["target"]["image"][0].float().to(device)
    psnr = compute_psnr(gt, pred)
    visibility = output.triangle_visibility_mask.float() if output.triangle_visibility_mask is not None else None
    visible_union = None
    if visibility is not None:
        visible_union = visibility[0].bool().any(dim=0)
    stats = primitive_stats(primitives, visible_mask=visible_union)
    if visible_union is not None:
        stats["raster_visible_union_count"] = int(visible_union.sum())
        stats["raster_visible_union_fraction"] = float(visible_union.float().mean())
    stats["projected_target"] = projected_triangle_stats(
        primitives.vertices,
        target["extrinsics"],
        target["intrinsics"],
        target["near"],
        tuple(target["image"].shape[-2:]),
        visibility=visibility,
        sigma=primitives.sigma,
        opacity=primitives.opacity,
    )
    stats.update(
        {
            "checkpoint": str(checkpoint),
            "global_step": global_step,
            "psnr_per_target": [float(value) for value in psnr],
            "psnr_mean": float(psnr.mean()),
            "rendered_opacity": tensor_stats(output.opacity),
            "rendered_depth": tensor_stats(output.depth),
            "rendered_color": tensor_stats(output.color),
            "rendered_normal": tensor_stats(output.rend_normal),
            "rendered_visible_fraction": None
            if visibility is None
            else [float(value) for value in visibility[0].mean(dim=-1)],
        }
    )
    result = {
        "name": name,
        "stats": stats,
        "pred": pred.detach().cpu(),
        "gt": gt.detach().cpu(),
        "normal": output.rend_normal[0].detach().cpu(),
    }
    del encoder, decoder, primitives, output
    torch.cuda.empty_cache()
    return result


def save_compressed_image(image: torch.Tensor, output: Path, max_bytes: int = 500_000) -> None:
    """Save one RGB view without a contact sheet, capped for quick inspection."""
    array = prep_image(image.clamp(0, 1))
    pil_image = Image.fromarray(array).convert("RGB")
    quality = 92
    while True:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=True)
        payload = buffer.getvalue()
        if len(payload) <= max_bytes or quality <= 40:
            output.with_suffix(".jpg").write_bytes(payload)
            return
        quality -= 8


def color_summary(image: torch.Tensor) -> dict[str, object]:
    values = image.detach().float().clamp(0, 1)
    flat = values.permute(0, 2, 3, 1).reshape(-1, 3)
    return {
        "mean_rgb": [float(x) for x in flat.mean(dim=0)],
        "std_rgb": [float(x) for x in flat.std(dim=0)],
        "p01_rgb": [float(x) for x in torch.quantile(flat, 0.01, dim=0)],
        "median_rgb": [float(x) for x in torch.median(flat, dim=0).values],
        "p99_rgb": [float(x) for x in torch.quantile(flat, 0.99, dim=0)],
        "mean_luma": float((flat * flat.new_tensor([0.299, 0.587, 0.114])).sum(dim=-1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=REPO_ROOT / "assets/dl3dv_single_scene_eval.json")
    parser.add_argument("--baseline-ckpt", type=Path, required=True)
    parser.add_argument("--tsdpt-ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--global-step", type=int, default=600)
    parser.add_argument(
        "--tsdpt-experiment",
        default="trisplat_dl3dv_tsdpt_5k_noschedule",
        help="TSDPT experiment whose decoder settings match the checkpoint.",
    )
    parser.add_argument(
        "--target-index",
        type=int,
        default=None,
        help="Save only this target view; forward still computes all targets for comparison.",
    )
    parser.add_argument("--tsdpt-brightness-gain", type=float, default=None)
    parser.add_argument("--tsdpt-saturation-gain", type=float, default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(111123)

    index_entries = json.loads(args.index.read_text())
    if not index_entries:
        raise ValueError(f"Evaluation index is empty: {args.index}")
    scene = next(iter(index_entries))
    scene_entry = index_entries[scene]
    num_context_views = len(scene_entry.get("context", []))
    if num_context_views <= 0:
        raise ValueError(f"Evaluation index has no context views for scene {scene}")
    tsdpt_cfg = compose_cfg(
        args.tsdpt_experiment,
        args.data_root,
        args.index,
        baseline=False,
        scene=scene,
        num_context_views=num_context_views,
        tsdpt_brightness_gain=args.tsdpt_brightness_gain,
        tsdpt_saturation_gain=args.tsdpt_saturation_gain,
    )
    baseline_cfg = compose_cfg(
        "trisplat_dl3dv_triangle_refiner_unet_10m_224x448_test",
        args.data_root,
        args.index,
        baseline=True,
        scene=scene,
        num_context_views=num_context_views,
    )
    # The local DL3DV export contains a train split only.  overfit_to_scene
    # makes DatasetRE10k use that packed chunk while keeping the evaluation
    # sampler's fixed context/target indices.
    loader = DataModule(tsdpt_cfg.dataset, tsdpt_cfg.data_loader, StepTracker(), global_rank=0).train_dataloader()
    raw_batch = move(next(iter(loader)), torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    scene = raw_batch["scene"][0] if isinstance(raw_batch["scene"], list) else str(raw_batch["scene"])
    device = raw_batch["context"]["image"].device

    baseline = forward_one("trisplat_baseline", baseline_cfg, args.baseline_ckpt, raw_batch, device, args.global_step)
    tsdpt = forward_one("tsdpt_step_600", tsdpt_cfg, args.tsdpt_ckpt, raw_batch, device, args.global_step)
    results = [baseline, tsdpt]
    payload = {
        "scene": scene,
        "context_indices": raw_batch["context"]["index"][0].tolist(),
        "target_indices": raw_batch["target"]["index"][0].tolist(),
        "global_step": args.global_step,
        "baseline": baseline["stats"],
        "tsdpt": tsdpt["stats"],
    }
    payload["baseline"]["rendered_color_summary"] = color_summary(baseline["pred"])
    payload["tsdpt"]["rendered_color_summary"] = color_summary(tsdpt["pred"])
    payload["ground_truth_color_summary"] = color_summary(baseline["gt"])
    (args.out / "primitive_stats.json").write_text(json.dumps(payload, indent=2) + "\n")

    # Keep each image at the native render size.  A single-view file is much
    # easier to inspect for color bias and avoids the large contact sheet.
    target_indices = (
        [args.target_index]
        if args.target_index is not None
        else list(range(results[0]["gt"].shape[0]))
    )
    for target_idx in target_indices:
        if not 0 <= target_idx < results[0]["gt"].shape[0]:
            raise ValueError(
                f"--target-index must be in [0, {results[0]['gt'].shape[0]}), got {target_idx}"
            )
        save_compressed_image(
            baseline["gt"][target_idx], args.out / f"gt_target_{target_idx:02d}.jpg"
        )
        save_compressed_image(
            baseline["pred"][target_idx], args.out / f"trisplat_target_{target_idx:02d}.jpg"
        )
        save_compressed_image(
            tsdpt["pred"][target_idx], args.out / f"tsdpt_target_{target_idx:02d}.jpg"
        )
    print(json.dumps({name: item["stats"]["psnr_mean"] for name, item in [("baseline", baseline), ("tsdpt", tsdpt)]}, indent=2))


if __name__ == "__main__":
    main()
