from dataclasses import dataclass
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from .encoder import Encoder
from ..types import Triangles
from ...dataset.types import DataShim
from ...dataset.shims.normalize_shim import apply_normalize_shim
from ..head.triangle_texture_head import TriangleTextureHead, project_triangle_texture

logger = logging.getLogger(__name__)

_GEOMETRY_DEBUG = os.environ.get("TRISPLAT_GEOMETRY_DEBUG", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _debug_mask_stats(mask: Tensor) -> str:
    return f"{int(mask.sum().item())}/{mask.numel()}"


@dataclass
class EncoderDA3TSDPTCfg:
    name: str = "da3_tsdpt"
    # Keep the common checkpoint-loading path compatible with the native
    # triangle encoder. TSDPT always emits triangle primitives.
    use_triangle: bool = True
    checkpoint: str = os.environ.get(
        "DA3_CHECKPOINT", "../Depth-Anything-3/checkpoints/DA3-GIANT-1.1"
    )
    # Match the final DL3DV TriSplat triangle parameterization.
    triangle_scale_min: float = 1.0
    triangle_scale_max: float = 1.25
    triangle_scale_max_start: float | None = None
    triangle_scale_max_end: float | None = None
    triangle_scale_max_schedule_steps: int = 0
    triangle_scale_init_fraction: float = 0.25
    max_triangle_edge_px: float = 0.0
    # Kept as config fields for checkpoint/config compatibility. The direct
    # TSDPT path does not schedule sigma over training.
    sigma_scale_initial: float = 4.6
    sigma_scale_final: float = 4.6
    sigma_warmup_steps: int = 1
    # Global calibration for the independently trained TSDPT opacity head.
    # This is applied after sigmoid and before the decoder's logit temperature.
    opacity_global_scale: float = 0.62
    sh_degree: int = 0
    sh_degree2_weight: float | None = None
    freeze_backbone: bool = True
    # Freeze every DA3/TSDPT output except the 27 raw second-order SH
    # channels (channels 9:36 of the checkpoint-compatible GS head).
    sh_only: bool = False
    pretrained_weights: str = ""
    gaussian_downsample_ratio: int = 1
    gaussians_per_axis: int = 14
    # The TSDPT geometry path uses the raw DA3 point-map differential. Keep
    # these legacy fields for config/checkpoint compatibility, but default to
    # no smoothing so a generic TSDPT config cannot silently alter the face.
    smooth_kernel: int = 1
    normal_smooth_kernel: int = 1
    normal_smooth_iterations: int = 0
    flip_to_camera: bool = True
    eps: float = 1e-6
    align_to_context_pose: bool = True
    # Match DA3's official inference path when calibrated context cameras are
    # available: encode the known cameras as backbone tokens before predicting
    # depth and pose.  The legacy bridge left this disabled by passing None.
    use_context_camera_tokens: bool = True
    # When exporting point maps for known context cameras, use each supplied
    # c2w rotation as well as its center. The legacy center-only Sim(3) path
    # can leave a large pixel reprojection residual when DA3 rotations drift.
    align_to_context_rotation: bool = True
    # Geometry validity is diagnostic only by default.  Enabling it causes
    # invalid DA3 differential pixels to be hard-culled by the decoder.
    use_geometry_valid_mask: bool = False
    texture_enabled: bool = False
    texture_size: int = 4
    texture_project_as_base: bool = True
    texture_color_sigma: float = 1.0
    texture_only: bool = False
    # DepthAnything3's public input processor uses ImageNet normalization.
    input_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    input_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    # The existing TSDPT checkpoints were trained through the legacy raw
    # [0,1] bridge. Keep that behavior by default; native DA3 API users can
    # opt into ImageNet normalization explicitly.
    use_input_normalization: bool = False


class EncoderDA3TSDPT(Encoder[EncoderDA3TSDPTCfg]):
    """DA3 feature/depth/camera encoder feeding the TriSplat triangle decoder.

    DA3's backbone, camera branch, and legacy GS-DPT projection remain frozen.
    Only TSDPT's newly added opacity/sigma projection is trainable.
    """

    def __init__(self, cfg: EncoderDA3TSDPTCfg) -> None:
        super().__init__(cfg)
        import sys

        # DA3 is intentionally kept as a sibling dependency rather than copied
        # into this repository.  ``DA3_ROOT`` may point either at its checkout
        # or directly at its ``src`` directory; the sibling fallback keeps the
        # original workspace layout working while making cloned checkouts
        # relocatable.
        configured_da3_root = os.environ.get("DA3_ROOT")
        if configured_da3_root:
            da3_root_path = Path(configured_da3_root).expanduser()
            da3_candidates = [
                da3_root_path,
                da3_root_path / "src",
            ]
        else:
            repo_root = Path(__file__).resolve().parents[3]
            da3_candidates = [
                repo_root / "third_party" / "depth-anything-3" / "src",
                repo_root.parent / "Depth-Anything-3" / "src",
            ]
        da3_src_path = next(
            (
                candidate
                for candidate in da3_candidates
                if (candidate / "depth_anything_3").is_dir()
            ),
            da3_candidates[0],
        )
        if not (da3_src_path / "depth_anything_3").is_dir():
            raise FileNotFoundError(
                "Depth-Anything-3 source was not found. Set DA3_ROOT to the "
                "DA3 checkout (or its src directory)."
            )
        da3_root = str(da3_src_path)
        if da3_root not in sys.path:
            sys.path.insert(0, da3_root)
        from depth_anything_3.api import DepthAnything3
        from depth_anything_3.model.ts_adapter import TSAdapter, TSAdapterCfg

        self.da3 = DepthAnything3.from_pretrained(cfg.checkpoint)
        # Frozen DA3 must remain in inference mode as well as out of the
        # autograd graph; train-mode dropout/normalization makes this giant
        # backbone unnecessarily expensive and non-deterministic.
        self.da3.model.eval() if cfg.freeze_backbone else self.da3.model.train()
        for p in self.da3.model.parameters():
            p.requires_grad_(False)
        # The DA3 backbone/camera/depth heads are frozen. By default the GS
        # projection remains trainable for the legacy TSDPT experiments; the
        # SH ablation narrows this to the final projection rows that produce
        # the 27 raw second-order SH coefficients.
        for p in self.da3.model.gs_head.parameters():
            p.requires_grad_(not cfg.sh_only and not cfg.texture_only)
        if cfg.sh_only:
            output_projection = self.da3.model.gs_head.scratch.output_conv2[2]
            output_projection.weight.requires_grad_(True)
            output_projection.bias.requires_grad_(True)
            row_mask = torch.zeros_like(output_projection.weight)
            row_mask[9:36] = 1.0
            bias_mask = torch.zeros_like(output_projection.bias)
            bias_mask[9:36] = 1.0
            # A Parameter cannot be sliced into a separate optimizer leaf.
            # Mask gradients and disable AdamW decay for this group below so
            # non-SH rows stay bitwise fixed while SH rows train normally.
            output_projection.weight.register_hook(
                lambda grad: grad * row_mask.to(device=grad.device, dtype=grad.dtype)
            )
            output_projection.bias.register_hook(
                lambda grad: grad * bias_mask.to(device=grad.device, dtype=grad.dtype)
            )
        self.ts_adapter = TSAdapter(
            TSAdapterCfg(
                triangle_scale_min=cfg.triangle_scale_min,
                triangle_scale_max=cfg.triangle_scale_max,
                triangle_scale_max_start=cfg.triangle_scale_max_start,
                triangle_scale_max_end=cfg.triangle_scale_max_end,
                triangle_scale_max_schedule_steps=cfg.triangle_scale_max_schedule_steps,
                triangle_scale_init_fraction=cfg.triangle_scale_init_fraction,
                max_triangle_edge_px=cfg.max_triangle_edge_px,
                sh_degree=cfg.sh_degree,
                sh_degree2_weight=cfg.sh_degree2_weight,
                sigma_scale_initial=cfg.sigma_scale_initial,
                sigma_scale_final=cfg.sigma_scale_final,
                sigma_warmup_steps=cfg.sigma_warmup_steps,
                eps=cfg.eps,
                use_geometry_valid_mask=cfg.use_geometry_valid_mask,
            )
        )
        self.patch_size = 14
        self.use_triangle = True
        self.pose_free = False
        if cfg.texture_enabled:
            self.texture_head = TriangleTextureHead(
                texture_size=cfg.texture_size,
                project_as_base=cfg.texture_project_as_base,
                sigma=cfg.texture_color_sigma,
            )
        if cfg.texture_only:
            for p in self.da3.model.parameters():
                p.requires_grad_(False)
            for p in self.texture_head.parameters():
                p.requires_grad_(True)

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: dict) -> dict:
            if not self.cfg.use_input_normalization:
                return batch
            return apply_normalize_shim(
                batch,
                mean=self.cfg.input_mean,
                std=self.cfg.input_std,
            )

        return data_shim

    @staticmethod
    def _camera_to_world(extrinsics: Tensor) -> Tensor:
        eye = torch.eye(4, device=extrinsics.device, dtype=extrinsics.dtype)
        c = eye.expand(*extrinsics.shape[:-2], 4, 4).clone()
        c[..., :3, :] = extrinsics
        return torch.linalg.inv(c)

    @staticmethod
    def _sim3_align_camera_centers(
        pred_c2w: Tensor, target_c2w: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return R, t, s mapping predicted world coordinates to target world."""
        pred = pred_c2w[..., :3, 3]
        target = target_c2w[..., :3, 3]
        pred_mean = pred.mean(dim=-2, keepdim=True)
        target_mean = target.mean(dim=-2, keepdim=True)
        pred_centered = pred - pred_mean
        target_centered = target - target_mean
        covariance = torch.matmul(target_centered.transpose(-1, -2), pred_centered)
        u, singular, vh = torch.linalg.svd(covariance)
        rotation = torch.matmul(u, vh)
        det = torch.linalg.det(rotation)
        if (det < 0).any():
            u = u.clone()
            u[..., :, -1] *= torch.where(
                det < 0,
                -torch.ones_like(det),
                torch.ones_like(det),
            )[..., None]
            rotation = torch.matmul(u, vh)
        denom = pred_centered.square().sum(dim=(-2, -1)).clamp_min(1e-8)
        scale = singular.sum(dim=-1) / denom
        translation = target_mean.squeeze(-2) - scale[..., None] * torch.matmul(
            rotation, pred_mean.squeeze(-2)[..., None]
        ).squeeze(-1)
        return rotation, translation, scale

    def forward(self, context: dict, global_step: int = 0, visualization_dump: dict | None = None) -> Triangles:
        images = context["image"]
        b, v, _, h, w = images.shape
        if _GEOMETRY_DEBUG:
            print(
                "[geometry debug][encoder] input "
                f"images={tuple(images.shape)} context_views={v} pixels={b * v * h * w}",
                flush=True,
            )
        # The DA3 backbone is frozen, but calling the monolithic model forward
        # would still build an autograd graph through all 1.4B backbone
        # parameters so gradients can reach gs_head.  Detach the frozen feature
        # path explicitly and keep only the trainable GS projection in the graph.
        da3 = self.da3.model
        if self.cfg.freeze_backbone:
            da3.backbone.eval()
        logger.info("DA3 encoder: backbone start, shape=%s", tuple(images.shape))
        da3_conditioning_extrinsics = None
        da3_conditioning_intrinsics = None
        if (
            self.cfg.use_context_camera_tokens
            and context.get("extrinsics") is not None
            and context.get("intrinsics") is not None
        ):
            # TriSplat stores camera-to-world poses and normalized K.  DA3's
            # CameraEnc receives world-to-camera poses and pixel-space K, just
            # like the public DepthAnything3 API before it normalizes poses.
            context_c2w = context["extrinsics"].to(device=images.device, dtype=torch.float32)
            context_k = context["intrinsics"].to(device=images.device, dtype=torch.float32)
            da3_conditioning_extrinsics = torch.linalg.inv(context_c2w)
            da3_conditioning_intrinsics = context_k.clone()
            da3_conditioning_intrinsics[..., 0, :] *= float(w)
            da3_conditioning_intrinsics[..., 1, :] *= float(h)

            # This is DepthAnything3.api._normalize_extrinsics, reproduced
            # locally so the encoder can retain the tensor-only forward path.
            first_c2w = torch.linalg.inv(da3_conditioning_extrinsics[..., :1, :, :])
            da3_conditioning_extrinsics = da3_conditioning_extrinsics @ first_c2w
            normalized_c2w = torch.linalg.inv(da3_conditioning_extrinsics)
            camera_distances = normalized_c2w[..., :3, 3].norm(dim=-1)
            median_distance = camera_distances.median().clamp_min(1e-1)
            da3_conditioning_extrinsics = da3_conditioning_extrinsics.clone()
            da3_conditioning_extrinsics[..., :3, 3] /= median_distance
            logger.info(
                "DA3 encoder: using context camera tokens (normalized median distance=%.6g)",
                float(median_distance.detach().cpu()),
            )
        with torch.no_grad():
            cam_token = None
            if da3_conditioning_extrinsics is not None:
                with torch.autocast(device_type=images.device.type, enabled=False):
                    cam_token = da3.cam_enc(
                        da3_conditioning_extrinsics,
                        da3_conditioning_intrinsics,
                        (h, w),
                    )
            feats, aux_feats = da3.backbone(
                images,
                cam_token=cam_token,
                export_feat_layers=[],
                ref_view_strategy="saddle_balanced",
            )
            feats = [f.detach() if isinstance(f, torch.Tensor) else f for f in feats]
            out = da3._process_depth_head(feats, h, w)
            out = da3._process_camera_estimation(feats, h, w, out)
        logger.info("DA3 encoder: frozen backbone/depth/camera done")

        # Inject the calibrated DL3DV intrinsics before DA3's GS/TSDPT adapter
        # runs. This makes the point-cloud construction itself use the same K
        # as TriSplat's renderer, instead of constructing with DA3's predicted
        # focal length and correcting it only afterward.
        predicted_intr = out.intrinsics.detach().clone()
        geometry_intr = predicted_intr
        context_intr = context.get("intrinsics")
        if context_intr is not None:
            geometry_intr = context_intr.to(device=predicted_intr.device, dtype=predicted_intr.dtype).clone()
            geometry_intr[..., 0, :] *= float(w)
            geometry_intr[..., 1, :] *= float(h)
            out.intrinsics = geometry_intr

        # TSDPT outputs are needed for triangle attributes, but the native
        # GaussianAdapter result is discarded by this bridge.  Skip that
        # expensive conversion and keep only raw_gs/scale/opacity/sigma.
        out = da3._process_gs_head(
            feats, h, w, out, images, None, None, build_gaussians=False
        )
        logger.info("DA3 encoder: TSDPT gs_head done (native GaussianAdapter skipped)")
        depth = out.depth
        extr = out.extrinsics
        intr = out.intrinsics
        raw = out.raw_gs
        if not hasattr(out, "raw_gs_scale"):
            raise RuntimeError("TSDPT scale_output_conv output is missing from DA3 model output")
        raw_scales = out.raw_gs_scale.movedim(2, -1).contiguous()
        # TSDPT predicts opacity and sigma per pixel. Opacity is mapped here;
        # sigma receives the native warmup schedule inside TSAdapter.
        opacity = (
            torch.sigmoid(out.raw_gs_opacity) * float(self.cfg.opacity_global_scale)
        ).clamp(1e-4, 1 - 1e-4)
        sigma = torch.sigmoid(out.raw_gs_sigma) + 1e-4
        if raw.shape[-1] < 36:
            raise RuntimeError(f"DA3 raw_gs has unexpected dimension {raw.shape[-1]}")

        if _GEOMETRY_DEBUG:
            depth_finite = torch.isfinite(depth)
            depth_positive = depth_finite & (depth > 0)
            print(
                "[geometry debug][encoder] DA3 outputs "
                f"depth={tuple(depth.shape)} finite={_debug_mask_stats(depth_finite)} "
                f"positive_depth={_debug_mask_stats(depth_positive)} "
                f"raw_gs={tuple(raw.shape)} raw_scale={tuple(raw_scales.shape)} "
                f"opacity={tuple(opacity.shape)} sigma={tuple(sigma.shape)}",
                flush=True,
            )

        c2w = self._camera_to_world(extr)
        predicted_c2w = c2w.detach().clone()
        # Back-project DA3's predicted depth with the calibrated geometry
        # intrinsics and predicted pose, without GS offset channels.
        ys, xs = torch.meshgrid(
            torch.arange(h, device=images.device, dtype=images.dtype),
            torch.arange(w, device=images.device, dtype=images.dtype), indexing="ij"
        )
        # Match DA3/GaussianAdapter's sample_image_grid convention: coordinates
        # denote pixel centers rather than the upper-left pixel corners.
        xs = xs + 0.5
        ys = ys + 0.5
        # The DL3DV batch provides calibrated normalized intrinsics. DA3 still
        # predicts a camera branch, but its focal estimate can differ greatly
        # from the dataset camera (e.g. fx~208 vs GT fx~432 at 448x224). Use the
        # known context calibration for point-map unprojection so the geometry
        # and the TriSplat renderer share exactly the same camera model.
        fx, fy = geometry_intr[..., 0, 0][..., None, None], geometry_intr[..., 1, 1][..., None, None]
        cx, cy = geometry_intr[..., 0, 2][..., None, None], geometry_intr[..., 1, 2][..., None, None]
        pts_cam = torch.stack(((xs - cx) * depth / fx, (ys - cy) * depth / fy, depth), dim=-1)
        if _GEOMETRY_DEBUG:
            pts_finite = torch.isfinite(pts_cam).all(dim=-1)
            pts_positive = pts_finite & (pts_cam[..., 2] > 0)
            print(
                "[geometry debug][encoder] backprojected pts_cam "
                f"shape={tuple(pts_cam.shape)} finite={_debug_mask_stats(pts_finite)} "
                f"positive_z={_debug_mask_stats(pts_positive)}",
                flush=True,
            )
        per_view_rotation = None
        # The DA3 camera frame is only defined up to a scene-level Sim(3), while
        # TriSplat's target cameras are in the dataset-normalized frame. Align
        # the frozen DA3 trajectory once per scene before rasterization.
        if self.cfg.align_to_context_pose and context.get("extrinsics") is not None:
            target_c2w = context["extrinsics"].to(c2w)
            with torch.no_grad():
                align_r, align_t, align_s = self._sim3_align_camera_centers(
                    c2w.detach(), target_c2w.detach()
                )
            # Build the canonical aligned world point map first. Center-only
            # Sim(3) alignment preserves DA3's per-view rotations, which is
            # useful for pose-free model coordinates but does not guarantee
            # pixel reprojection under the supplied context cameras. For a
            # native export, retain the DA3 scene scale while using each
            # context c2w rotation and center explicitly.
            if self.cfg.align_to_context_rotation:
                aligned_rotations = target_c2w[..., :3, :3]
                aligned_world = (
                    align_s[:, None, None, None, None]
                    * torch.einsum(
                        "bvij,bvhwj->bvhwi",
                        target_c2w[..., :3, :3],
                        pts_cam,
                    )
                    + target_c2w[..., :3, 3][:, :, None, None, :]
                )
                point_map_rotation = torch.eye(
                    3, device=target_c2w.device, dtype=target_c2w.dtype
                ).expand(*target_c2w.shape[:-2], 3, 3)
            else:
                aligned_rotations = torch.einsum(
                    "bij,bvjk->bvik", align_r, c2w[..., :3, :3]
                )
                world_da = torch.einsum(
                    "bvij,bvhwj->bvhwi", c2w[..., :3, :3], pts_cam
                ) + c2w[..., :3, 3][:, :, None, None, :]
                aligned_world = align_s[:, None, None, None, None] * torch.einsum(
                    "bij,bvhwj->bvhwi", align_r, world_da
                ) + align_t[:, None, None, None, :]
                point_map_rotation = torch.einsum(
                    "bvji,bvjk->bvik",
                    target_c2w[..., :3, :3],
                    aligned_rotations,
                )
            pts_cam = torch.einsum(
                "bvji,bvhwj->bvhwi",
                target_c2w[..., :3, :3],
                aligned_world - target_c2w[..., :3, 3][:, :, None, None, :],
            )
            per_view_rotation = torch.matmul(
                target_c2w[..., :3, :3], aligned_rotations.transpose(-1, -2)
            )
            c2w = target_c2w
            if _GEOMETRY_DEBUG:
                aligned_finite = torch.isfinite(pts_cam).all(dim=-1)
                aligned_positive = aligned_finite & (pts_cam[..., 2] > 0)
                print(
                    "[geometry debug][encoder] pose-aligned pts_cam "
                    f"finite={_debug_mask_stats(aligned_finite)} "
                    f"positive_z={_debug_mask_stats(aligned_positive)} "
                    f"sim3_scale={align_s.detach().flatten().tolist()}",
                    flush=True,
                )
            if visualization_dump is not None:
                visualization_dump["pose_alignment_rotation"] = align_r
                visualization_dump["pose_alignment_translation"] = align_t
                visualization_dump["pose_alignment_scale"] = align_s
                visualization_dump["pose_alignment_per_view_rotation"] = per_view_rotation
                visualization_dump["pose_alignment_point_map_rotation"] = point_map_rotation
                visualization_dump["pose_alignment_per_view_c2w_rotation"] = (
                    target_c2w[..., :3, :3]
                )
                visualization_dump["points_world_aligned"] = aligned_world.detach()
        ts_triangles = self.ts_adapter.from_point_map(
            points_cam=pts_cam,
            c2w=c2w,
            raw_gaussians=raw,
            raw_scales=raw_scales,
            opacity=opacity,
            sigma=sigma,
            image_shape=(h, w),
            intrinsics=geometry_intr,
            global_step=global_step,
            flip_to_camera=self.cfg.flip_to_camera,
        )
        texture_colors = None
        if self.cfg.texture_enabled:
            image_highres = context.get("image_highres")
            intrinsics_highres = context.get("intrinsics_highres")
            if image_highres is None or intrinsics_highres is None:
                raise RuntimeError(
                    "texture_enabled requires context.image_highres and "
                    "context.intrinsics_highres"
                )
            if image_highres.shape[:2] != (b, v):
                raise ValueError(
                    "context.image_highres must have the same batch/view dimensions "
                    f"as geometry images: got {tuple(image_highres.shape)} vs "
                    f"{tuple(images.shape)}"
                )
            if image_highres.shape[-2] < h or image_highres.shape[-1] < w:
                raise ValueError(
                    "context.image_highres must remain at native or higher resolution "
                    f"for texture projection: got {tuple(image_highres.shape[-2:])} "
                    f"for geometry {h}x{w}"
                )
            if intrinsics_highres.shape[:2] != (b, v):
                raise ValueError(
                    "context.intrinsics_highres must match context image batch/views: "
                    f"got {tuple(intrinsics_highres.shape)}"
                )
            if _GEOMETRY_DEBUG:
                logger.info(
                    "DA3 geometry input=%sx%s; texture input=%sx%s (native intrinsics)",
                    h,
                    w,
                    image_highres.shape[-2],
                    image_highres.shape[-1],
                )
            vertices_per_view = ts_triangles.vertices.reshape(b, v, h * w, 3, 3)
            projected = project_triangle_texture(
                vertices_per_view,
                image_highres,
                intrinsics_highres,
                c2w,
                self.cfg.texture_size,
                self.cfg.texture_color_sigma,
            )
            texture_colors = self.texture_head(
                image_highres,
                projected,
                (h, w),
            ).reshape(b, v * h * w, 3, self.cfg.texture_size, self.cfg.texture_size)
        if _GEOMETRY_DEBUG:
            triangle_finite = torch.isfinite(ts_triangles.vertices).reshape(
                *ts_triangles.vertices.shape[:2], -1
            ).all(dim=-1)
            print(
                "[geometry debug][encoder] output to decoder "
                f"vertices={tuple(ts_triangles.vertices.shape)} "
                f"finite_triangles={_debug_mask_stats(triangle_finite)} "
                f"sigma={tuple(ts_triangles.sigma.shape)} "
                f"opacity={tuple(ts_triangles.opacity.shape)} "
                f"scales={tuple(ts_triangles.scales.shape) if ts_triangles.scales is not None else None} "
                f"primitive_mask={'none' if ts_triangles.primitive_valid_mask is None else _debug_mask_stats(ts_triangles.primitive_valid_mask)}",
                flush=True,
            )
        if visualization_dump is not None:
            visualization_dump["da3_pred_c2w"] = predicted_c2w
            visualization_dump["da3_pred_extrinsics_w2c"] = extr.detach()
            visualization_dump["da3_pred_intrinsics"] = predicted_intr
            visualization_dump["geometry_intrinsics"] = geometry_intr.detach()
            visualization_dump["da3_pred_depth"] = depth.detach()
            for name in ("depth_conf", "sky", "raw_gs_conf"):
                value = getattr(out, name, None)
                if isinstance(value, torch.Tensor):
                    visualization_dump[f"da3_{name}"] = value.detach()
            visualization_dump["raw_gs_scale"] = raw_scales.detach()
            normals_world = ts_triangles.normals.reshape(b, v, h, w, 3)
            normals_cam = torch.einsum(
                "bvji,bvhwj->bvhwi", c2w[..., :3, :3], normals_world
            )
            visualization_dump["depth"] = pts_cam[..., 2:3]
            visualization_dump["local_pts"] = pts_cam
            visualization_dump["c2w"] = c2w
            visualization_dump["scales"] = ts_triangles.scales.reshape(b, v, h, w, 3)
            visualization_dump["rotations"] = F.normalize(
                torch.roll(raw[..., 5:9], shifts=1, dims=-1), dim=-1
            )
            normal_panel = normals_cam.permute(0, 1, 4, 2, 3).contiguous()
            visualization_dump["geom_normal_cam_raw"] = normal_panel
            visualization_dump["geom_normal_cam_base"] = normal_panel
            visualization_dump["geom_normal_cam_pred"] = normal_panel
            visualization_dump["geom_normal_cam"] = normal_panel
            visualization_dump["geom_normal_cam_forward"] = normal_panel
            # Do not expose the differential validity test as a normal mask:
            # it is intentionally disabled for rendering and normal losses.
            # Invalid/degenerate pixels remain visible for diagnosis instead
            # of turning into black holes in the context panel.
            visualization_dump["geom_normal_mask"] = torch.ones_like(
                normal_panel[:, :, :1], dtype=torch.bool
            )
        return Triangles(
            vertices=ts_triangles.vertices,
            sigma=ts_triangles.sigma,
            opacity=ts_triangles.opacity,
            features=ts_triangles.features,
            centers=ts_triangles.centers,
            normals=ts_triangles.normals,
            scales=ts_triangles.scales,
            mapped_scales=ts_triangles.mapped_scales,
            primitive_valid_mask=ts_triangles.primitive_valid_mask,
            texture_colors=texture_colors,
        )
