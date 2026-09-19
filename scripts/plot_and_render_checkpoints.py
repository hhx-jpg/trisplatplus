"""Export training curves and fixed-batch checkpoint render comparisons."""
from pathlib import Path
import argparse
import io
import json

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from hydra import compose, initialize_config_dir

from src.checkpoint_utils import extract_state_dict, load_checkpoint_file
from src.config import load_typed_root_config
from src.dataset.data_module import DataModule
from src.misc.step_tracker import StepTracker
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder
from src.evaluation.metrics import compute_psnr
from src.misc.image_io import prep_image


def save_compressed_sheet(sheet: Image.Image, path: Path, max_bytes: int = 500_000) -> Path:
    """Save a compact contact sheet while preserving the three-column layout."""
    image = sheet.convert("RGB")
    max_dimension = 960
    if max(image.size) > max_dimension:
        scale = max_dimension / max(image.size)
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )

    output_path = path.with_suffix(".jpg")
    quality = 88
    while True:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=True)
        payload = buffer.getvalue()
        if len(payload) <= max_bytes:
            output_path.write_bytes(payload)
            return output_path
        if quality <= 35:
            image = image.resize(
                (max(1, round(image.width * 0.8)), max(1, round(image.height * 0.8))),
                Image.Resampling.LANCZOS,
            )
            quality = 82
            continue
        quality -= 8


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    return value


def find_checkpoint(ckpt_dir: Path, step: int) -> Path:
    """Find a render snapshot, falling back to a regular Lightning ckpt."""
    if step == 0:
        candidates = [
            ckpt_dir / "render_initial.ckpt",
            ckpt_dir / "initial.ckpt",
        ]
    else:
        candidates = [ckpt_dir / f"render_step_{step:06d}.ckpt"]
    candidates = [path for path in candidates if path.is_file()]
    if not candidates and step > 0:
        candidates = sorted(ckpt_dir.glob(f"epoch_*-step_{step}.ckpt"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found for step {step} in {ckpt_dir}"
        )
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--experiment",
        default="trisplat_dl3dv_tsdpt_5k_directdiff",
        help="Hydra experiment matching the checkpoint architecture.",
    )
    parser.add_argument(
        "--step",
        dest="steps",
        action="append",
        type=int,
        help="Render only this checkpoint step (0 means the initial snapshot). May be repeated.",
    )
    parser.add_argument("--debug", action="store_true", help="Print geometry/raster visibility diagnostics.")
    parser.add_argument(
        "--target-context-index",
        type=int,
        default=None,
        help="Use this input context view as the sole target view for a reconstruction check.",
    )
    parser.add_argument(
        "--scene",
        default=None,
        help="Restrict the analysis batch to this dataset scene key.",
    )
    parser.add_argument(
        "--num-context-views",
        type=int,
        default=6,
        help="Number of context views used for the fixed analysis batch.",
    )
    parser.add_argument(
        "--context-indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional fixed context frame indices for a reproducible scene check.",
    )
    parser.add_argument(
        "--target-indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional fixed target frame indices for a reproducible scene check.",
    )
    parser.add_argument(
        "--disable-center-depth-culling",
        action="store_true",
        help="Disable the Python-side center-depth prefilter for the culling ablation.",
    )
    parser.add_argument(
        "--disable-texture",
        action="store_true",
        help="Disable the texture head for a geometry/normal isolation comparison.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in args.metrics.read_text().splitlines() if line.strip()]
    keys = [("loss/total", "Total loss"), ("loss/mse", "MSE"), ("loss/lpips", "LPIPS"), ("train/psnr_probabilistic", "Train PSNR")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, keys):
        # A weights-only continuation resets Lightning's logger step, while
        # info/global_step preserves the actual training position.
        points = [
            (r.get("info/global_step", r["step"]), r[key])
            for r in rows
            if key in r
        ]
        ax.plot([p[0] for p in points], [p[1] for p in points], marker=".", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)
    fig.savefig(args.out / "training_curves.png", dpi=160)
    plt.close(fig)

    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "config"), version_base=None):
        cfg_dict = compose(
            config_name="main",
            overrides=[
                f"+experiment={args.experiment}",
                f"dataset.dl3dv.roots=[{args.data_root}]",
                f"dataset.dl3dv.test_roots=[{args.data_root}]",
                "dataset.dl3dv.original_image_shape=[540,960]",
                "dataset.dl3dv.input_image_shape=[224,448]",
                # Checkpoint comparisons must be deterministic.  The training
                # dataset defaults to random horizontal reflection, which can
                # change DA3's camera/opacity outputs even when view indices
                # are identical and makes texture on/off counts incomparable.
                "dataset.dl3dv.augment=false",
                f"dataset.dl3dv.view_sampler.num_context_views={args.num_context_views}",
                "dataset.dl3dv.view_sampler.num_target_views=4",
                "data_loader.train.batch_size=1",
                "data_loader.train.num_workers=0",
                "data_loader.val.num_workers=0",
                *([f"dataset.dl3dv.overfit_to_scene={args.scene}"] if args.scene else []),
                *(
                    [
                        "+dataset.dl3dv.view_sampler.fixed_context_indices="
                        f"{args.context_indices}"
                    ]
                    if args.context_indices is not None
                    else []
                ),
                *(
                    [
                        "+dataset.dl3dv.view_sampler.fixed_target_indices="
                        f"{args.target_indices}"
                    ]
                    if args.target_indices is not None
                    else []
                ),
                *(
                    ["model.decoder.center_depth_culling=false"]
                    if args.disable_center_depth_culling
                    else []
                ),
                *(
                    [
                        "model.encoder.texture_enabled=false",
                        "model.encoder.texture_only=false",
                    ]
                    if args.disable_texture
                    else []
                ),
            ],
        )
    cfg = load_typed_root_config(cfg_dict)
    loader = DataModule(cfg.dataset, cfg.data_loader, StepTracker(), global_rank=0).train_dataloader()
    batch = next(iter(loader))
    print(
        f"analysis_scene={batch.get('scene')} "
        f"context_indices={batch['context']['index'].detach().cpu().tolist()} "
        f"target_indices={batch['target']['index'].detach().cpu().tolist()}",
        flush=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = move(batch, device)
    target_context_index = args.target_context_index
    if target_context_index is not None:
        context = batch["context"]
        num_context = context["image"].shape[1]
        if not 0 <= target_context_index < num_context:
            raise ValueError(
                f"--target-context-index must be in [0, {num_context}), "
                f"got {target_context_index}"
            )
        # Keep the encoder input unchanged, but replace the target package with
        # one of those same views. This isolates target-view extrapolation from
        # coverage/geometry errors in the trained scene.
        target = dict(batch["target"])
        for key in ("image", "extrinsics", "intrinsics", "near", "far", "index"):
            if key in context:
                target[key] = context[key][:, target_context_index : target_context_index + 1]
        if "valid_mask" in context:
            target["valid_mask"] = context["valid_mask"][:, target_context_index : target_context_index + 1]
        batch["target"] = target
        selected_index = context["index"][:, target_context_index].detach().cpu().tolist()
        print(
            f"target_override=context[{target_context_index}] scene_indices={selected_index}",
            flush=True,
        )
    # Keep all four target views from the training sample. The single encoder
    # and decoder forward below produces all target renders and PSNR values.
    gt = batch["target"]["image"][0]

    if args.steps:
        steps = tuple(dict.fromkeys(args.steps))
    else:
        has_initial = any(
            (args.ckpt_dir / name).is_file()
            for name in ("render_initial.ckpt", "initial.ckpt")
        )
        discovered = {0} if has_initial else set()
        for path in args.ckpt_dir.glob("render_step_*.ckpt"):
            try:
                discovered.add(int(path.stem.rsplit("_", 1)[-1]))
            except ValueError:
                continue
        for path in args.ckpt_dir.glob("epoch_*-step_*.ckpt"):
            try:
                discovered.add(int(path.stem.rsplit("_", 1)[-1]))
            except ValueError:
                continue
        steps = tuple(sorted(discovered))
        if not steps:
            raise FileNotFoundError(f"No checkpoints found in {args.ckpt_dir}")
    for step in steps:
        ckpt = find_checkpoint(args.ckpt_dir, step)
        state = extract_state_dict(load_checkpoint_file(ckpt))
        encoder, _ = get_encoder(cfg.model.encoder)
        decoder = get_decoder(cfg.model.decoder)
        encoder_state = {k[8:]: v for k, v in state.items() if k.startswith("encoder.")}
        missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
        # TSDPT geometry is produced by this head.  Render snapshots are
        # allowed to omit the frozen DA3 backbone, but never the frozen
        # ``gs_head``: texture-only snapshots that omitted it rendered with
        # the base DA3 head and appeared almost entirely black.
        required_geometry = {
            key for key in encoder.state_dict()
            if key.startswith("da3.model.gs_head.") and "pretrained" not in key
        }
        missing_geometry = sorted(required_geometry.intersection(missing))
        if missing_geometry:
            raise RuntimeError(
                f"Checkpoint {ckpt} is missing {len(missing_geometry)} TSDPT gs_head "
                "parameters required for geometry. Refusing to export a misleading render. "
                f"First missing keys: {missing_geometry[:3]}"
            )
        if unexpected:
            print(
                f"checkpoint={ckpt.name} ignored_unexpected_keys={len(unexpected)}",
                flush=True,
            )
        encoder = encoder.to(device).eval()
        decoder = decoder.to(device).eval()
        with torch.no_grad():
            encoder_vis = {} if args.debug else None
            primitives = encoder(batch["context"], step, visualization_dump=encoder_vis)
            if args.debug:
                if primitives.primitive_valid_mask is not None:
                    valid_mask = primitives.primitive_valid_mask.reshape(
                        primitives.primitive_valid_mask.shape[0],
                        batch["context"]["image"].shape[1],
                        -1,
                    )
                    print(
                        "primitive_valid_ratio="
                        f"overall={valid_mask.float().mean().item():.6f} "
                        f"per_context={[round(float(x), 6) for x in valid_mask.float().mean(-1).flatten()]}",
                        flush=True,
                    )
                if encoder_vis is not None and "local_pts" in encoder_vis:
                    # TSDPT point maps are first lifted into one canonical
                    # aligned world frame by the encoder.  Project that same
                    # map with the context pose here; projecting ``local_pts``
                    # directly would silently skip the Sim(3) alignment and
                    # report a misleading reprojection error.
                    if "points_world_aligned" in encoder_vis and "c2w" in encoder_vis:
                        world_pts = encoder_vis["points_world_aligned"]
                        c2w = encoder_vis["c2w"]
                        pts_cam = torch.einsum(
                            "bvji,bvhwj->bvhwi",
                            c2w[..., :3, :3],
                            world_pts - c2w[..., :3, 3][..., None, None, :],
                        )
                        projection_label = "aligned_world_projection_residual_px"
                    else:
                        pts_cam = encoder_vis["local_pts"]
                        projection_label = "local_map_projection_residual_px"
                    geometry_k = encoder_vis["geometry_intrinsics"]
                    _, _, point_h, point_w, _ = pts_cam.shape
                    ys, xs = torch.meshgrid(
                        torch.arange(point_h, device=pts_cam.device, dtype=pts_cam.dtype) + 0.5,
                        torch.arange(point_w, device=pts_cam.device, dtype=pts_cam.dtype) + 0.5,
                        indexing="ij",
                    )
                    projected_x = (
                        pts_cam[..., 0] / pts_cam[..., 2].clamp_min(1e-6)
                        * geometry_k[..., 0, 0, None, None]
                        + geometry_k[..., 0, 2, None, None]
                    )
                    projected_y = (
                        pts_cam[..., 1] / pts_cam[..., 2].clamp_min(1e-6)
                        * geometry_k[..., 1, 1, None, None]
                        + geometry_k[..., 1, 2, None, None]
                    )
                    projection_error = torch.sqrt(
                        (projected_x - xs) ** 2 + (projected_y - ys) ** 2
                    )
                    print(
                        f"{projection_label}="
                        f"mean={projection_error.mean().item():.4f} "
                        f"p95={projection_error.quantile(0.95).item():.4f} "
                        f"per_context={[round(float(x), 4) for x in projection_error.mean(dim=(-1, -2)).flatten()]}",
                        flush=True,
                    )
                if encoder_vis is not None and {
                    "da3_pred_c2w",
                    "c2w",
                    "pose_alignment_rotation",
                }.issubset(encoder_vis):
                    pred_rotation = encoder_vis["da3_pred_c2w"][..., :3, :3]
                    align_rotation = encoder_vis["pose_alignment_rotation"]
                    target_rotation = encoder_vis["c2w"][..., :3, :3]
                    aligned_pred_rotation = torch.matmul(
                        align_rotation[:, None], pred_rotation
                    )
                    relative_rotation = torch.matmul(
                        target_rotation.transpose(-1, -2), aligned_pred_rotation
                    )
                    trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
                    rotation_angle = torch.acos(
                        ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
                    ) * (180.0 / torch.pi)
                    print(
                        "pose_rotation_residual_deg="
                        f"mean={rotation_angle.mean().item():.4f} "
                        f"max={rotation_angle.max().item():.4f} "
                        f"per_context={[round(float(x), 4) for x in rotation_angle.flatten()]}",
                        flush=True,
                    )
                tri = primitives.vertices[0]
                face_normal = torch.nn.functional.normalize(
                    torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1),
                    dim=-1,
                )
                geometry_normal = torch.nn.functional.normalize(primitives.normals[0], dim=-1)
                face_geometry_dot = (face_normal * geometry_normal).sum(dim=-1)
                print(
                    f"step={step} face_geometry_normal_dot="
                    f"mean={face_geometry_dot.mean().item():.6f} "
                    f"p05={face_geometry_dot.quantile(0.05).item():.6f} "
                    f"p95={face_geometry_dot.quantile(0.95).item():.6f}",
                    flush=True,
                )
            if args.debug and step == 500:
                target_c2w = batch["target"]["extrinsics"][0, 0].float()
                target_w2c = torch.linalg.inv(target_c2w)
                target_pts = primitives.vertices[0].reshape(-1, 3)
                target_cam = target_pts @ target_w2c[:3, :3].T + target_w2c[:3, 3]
                target_k = batch["target"]["intrinsics"][0, 0].float().clone()
                target_k[0] *= 448
                target_k[1] *= 224
                target_proj = target_cam[:, :2] / target_cam[:, 2:].clamp_min(1e-6)
                target_proj = target_proj @ target_k[:2, :2].T + target_k[:2, 2]
                print(
                    "geometry_debug "
                    f"vertices={tuple(primitives.vertices.shape)} "
                    f"cam_z=({target_cam[:,2].min().item():.6g},{target_cam[:,2].max().item():.6g}) "
                    f"positive={(target_cam[:,2] > 0).float().mean().item():.6g} "
                    f"px=({target_proj.min().item():.6g},{target_proj.max().item():.6g}) "
                    f"center=({primitives.centers.min().item():.6g},{primitives.centers.max().item():.6g}) "
                    f"scale=({primitives.scales.min().item():.6g},{primitives.scales.max().item():.6g}) "
                    f"align_scale={encoder_vis.get('pose_alignment_scale', torch.tensor(-1.0, device=device)).flatten().tolist()}",
                    flush=True,
                )
            output = decoder.forward(
                primitives,
                batch["target"]["extrinsics"], batch["target"]["intrinsics"],
                batch["target"]["near"], batch["target"]["far"], (224, 448),
                global_step=step, debug_log_interval=10,
                return_triangle_visibility_mask=args.debug,
            )
        if args.debug and output.triangle_visibility_mask is not None:
            visible = output.triangle_visibility_mask.float().mean().item()
            visible_by_context = output.triangle_visibility_mask.float().reshape(
                output.triangle_visibility_mask.shape[0],
                output.triangle_visibility_mask.shape[1],
                batch["context"]["image"].shape[1],
                -1,
            ).mean(-1)
            print(
                f"step={step} color_absmax={output.color.abs().max().item():.6g} "
                f"opacity_mean={output.opacity.mean().item():.6g} "
                f"normal_absmean={output.rend_normal.abs().mean().item():.6g} "
                f"normal_nonzero={(output.rend_normal.abs().sum(dim=2) > 1e-6).float().mean().item():.6g} "
                f"visible_triangles={visible:.6g} "
                f"visible_by_context={visible_by_context.detach().cpu().flatten().tolist()}",
                flush=True,
            )
        pred = output.color[0].clamp(0, 1)
        psnr_per_target = compute_psnr(gt, pred)
        psnr = psnr_per_target.mean().item()
        with (args.out / "checkpoint_metrics.jsonl").open("a", encoding="utf-8") as handle:
            json.dump(
                {
                    "step": step,
                    "checkpoint": ckpt.name,
                    "psnr": psnr,
                    "psnr_per_target": psnr_per_target.detach().cpu().tolist(),
                },
                handle,
            )
            handle.write("\n")
        print(
            f"step={step} psnr_mean={psnr:.6f} dB "
            f"psnr_per_target={[round(float(value), 6) for value in psnr_per_target]}",
            flush=True,
        )
        # One contact sheet per checkpoint: each row is a target view and the
        # columns are GT RGB, that target's rendered triangle normal, and RGB.
        panel_width, panel_height = 448, 224
        header_height, row_gap = 28, 24
        row_height = panel_height + row_gap
        sheet = Image.new(
            "RGB", (panel_width * 3, header_height + row_height * pred.shape[0]), "white"
        )
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), "GT color", fill="black")
        draw.text((panel_width + 8, 8), "target rendered normal", fill="black")
        draw.text((panel_width * 2 + 8, 8), "rendered RGB", fill="black")
        for target_index in range(pred.shape[0]):
            pred_view = pred[target_index]
            gt_view = gt[target_index]
            rendered_normal = (output.rend_normal[0, target_index].clamp(-1, 1) + 1.0) * 0.5
            gt_img = Image.fromarray(prep_image(gt_view))
            pred_img = Image.fromarray(prep_image(pred_view))
            normal_img = Image.fromarray(prep_image(rendered_normal))
            y = header_height + target_index * row_height
            sheet.paste(gt_img, (0, y))
            sheet.paste(normal_img, (panel_width, y))
            sheet.paste(pred_img, (panel_width * 2, y))
            draw.text(
                (panel_width * 2 + 8, y + 5),
                f"target {target_index}  PSNR {psnr_per_target[target_index]:.3f} dB",
                fill="black",
            )
        suffix = (
            f"_context_{target_context_index}"
            if target_context_index is not None
            else ""
        )
        save_compressed_sheet(
            sheet,
            args.out / f"render_grid_step_{step:04d}{suffix}.jpg",
        )
        del encoder, decoder, primitives, output
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
