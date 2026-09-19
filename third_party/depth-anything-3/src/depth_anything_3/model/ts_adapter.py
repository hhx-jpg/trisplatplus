"""Triangle-splat geometry adapter.

This module is the geometry counterpart of :mod:`tsdpt`.  ``TSDPT`` predicts
per-pixel parameters; ``TSAdapter`` turns those parameters and a depth/camera
prediction into world-space triangles.  It deliberately has no dependency on
TriSplat, so the DA3 model package owns the complete TSDPT path.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import Tensor, nn

from depth_anything_3.utils.geometry import get_world_rays


_GEOMETRY_DEBUG = os.environ.get("TRISPLAT_GEOMETRY_DEBUG", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _debug_mask_stats(mask: Tensor) -> str:
    return f"{int(mask.sum().item())}/{mask.numel()}"


@dataclass
class TSTriangles:
    """DA3-owned triangle primitive container.

    The fields intentionally mirror TriSplat's ``Triangles`` dataclass.  The
    bridge in TriSplat converts this value at its framework boundary.
    """

    vertices: Tensor
    sigma: Tensor
    opacity: Tensor
    features: Tensor
    centers: Tensor | None = None
    normals: Tensor | None = None
    scales: Tensor | None = None
    mapped_scales: Tensor | None = None
    primitive_valid_mask: Tensor | None = None


@dataclass
class TSAdapterCfg:
    triangle_scale_min: float = 1.0
    triangle_scale_max: float = 1.25
    # Optional schedule for direct-difference triangle plane factors.  None
    # preserves the fixed ``triangle_scale_max`` behavior used by old runs.
    triangle_scale_max_start: float | None = None
    triangle_scale_max_end: float | None = None
    triangle_scale_max_schedule_steps: int = 0
    # Raw scale logits are zero-initialized.  Offset the sigmoid only for the
    # mapping so zero logits start at a smaller fraction of the configured
    # range while nonzero logits can still reach the full range.
    triangle_scale_init_fraction: float = 0.25
    # Cap the local direct-difference triangle footprint before projection.
    # Zero disables this stabilization and preserves the raw spacing.
    max_triangle_edge_px: float = 0.0
    sh_degree: int = 0
    # Optional multiplier for second-order SH coefficients.  The historical
    # Gaussian adapter attenuates degree 2 by 0.1 * 0.25**2; TSDPT ablations
    # can raise this without changing the raw checkpoint channel layout.
    sh_degree2_weight: float | None = None
    sigma_scale_initial: float = 1.0
    sigma_scale_final: float = 0.1
    sigma_warmup_steps: int = 5000
    eps: float = 1e-8
    near_plane: float = 0.2
    # The differential geometry validity test is useful for diagnostics, but
    # must not silently remove source pixels from the renderer.  Keep this
    # opt-in because sparse invalid pixels otherwise become black holes.
    use_geometry_valid_mask: bool = False


def _normalize_vector(vector: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    norm = vector.norm(dim=-1, keepdim=True)
    valid = torch.isfinite(vector).all(dim=-1, keepdim=True) & (norm > eps)
    normalized = torch.where(valid, vector / norm.clamp_min(eps), torch.zeros_like(vector))
    return normalized, valid.squeeze(-1)


def _quaternion_apply(quaternion: Tensor, vector: Tensor) -> Tensor:
    w, x, y, z = quaternion.unbind(dim=-1)
    vx, vy, vz = vector.unbind(dim=-1)
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    return torch.stack(
        (
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        ),
        dim=-1,
    )


class TSAdapter(nn.Module):
    """Convert TSDPT outputs to world-space triangles.

    ``forward`` is the generic parameter adapter used by the original
    TriSplat path. ``from_point_map`` is the DA3 direct-difference path: its
    triangle frame is anchored by central differences of DA3's point map.
    """

    def __init__(self, cfg: TSAdapterCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TSAdapterCfg()
        self.d_sh = (self.cfg.sh_degree + 1) ** 2
        self.sh_dim = 3 * self.d_sh
        self.register_buffer(
            "canonical_triangle",
            torch.tensor(
                [[0.0, 0.57735, 0.0], [-0.5, -0.28868, 0.0], [0.5, -0.28868, 0.0]],
                dtype=torch.float32,
            )
            * 4,
        )
        self.register_buffer("sh_mask", torch.ones(self.d_sh), persistent=False)
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree
        if self.cfg.sh_degree2_weight is not None and self.cfg.sh_degree >= 2:
            self.sh_mask[4:9] = float(self.cfg.sh_degree2_weight)

    @property
    def d_in(self) -> int:
        return 3 + 4 + self.sh_dim + 1

    @staticmethod
    def _ensure_finite(name: str, value: Tensor) -> None:
        if not torch.isfinite(value).all():
            raise FloatingPointError(
                f"[ts adapter] non-finite {name}: shape={tuple(value.shape)}"
            )

    def _scale_max(self, global_step: int) -> float:
        start = self.cfg.triangle_scale_max
        end = self.cfg.triangle_scale_max
        if self.cfg.triangle_scale_max_start is not None:
            start = self.cfg.triangle_scale_max_start
        if self.cfg.triangle_scale_max_end is not None:
            end = self.cfg.triangle_scale_max_end
        steps = self.cfg.triangle_scale_max_schedule_steps
        if steps <= 0:
            return float(end)
        progress = min(max(float(global_step), 0.0) / float(steps), 1.0)
        return float(start + progress * (end - start))

    def _map_scale_logits(self, raw_scales: Tensor, scale_max: float) -> Tensor:
        fraction = float(self.cfg.triangle_scale_init_fraction)
        if not 0.0 < fraction < 1.0:
            raise ValueError(
                "triangle_scale_init_fraction must be strictly between 0 and 1, "
                f"got {fraction}"
            )
        # logit(fraction) is a fixed mapping offset, not a parameter of the
        # output head. Therefore a zero-initialized head remains zero while
        # its mapped scale starts at the requested fraction of the range.
        mapping_bias = raw_scales.new_tensor(fraction).logit()
        return self.cfg.triangle_scale_min + (
            scale_max - self.cfg.triangle_scale_min
        ) * (raw_scales + mapping_bias).sigmoid()

    def forward(
        self,
        extrinsics: Tensor,
        intrinsics: Tensor,
        coordinates: Tensor,
        depths: Tensor,
        opacities: Tensor,
        raw_triangles: Tensor,
        image_shape: tuple[int, int],
        global_step: int = 0,
    ) -> TSTriangles:
        """Adapt normalized pixel coordinates and raw triangle parameters."""
        h, w = image_shape
        self._ensure_finite("depths", depths)
        self._ensure_finite("opacities", opacities)
        self._ensure_finite("raw_triangles", raw_triangles)
        scales, rotations, sh, sigma = raw_triangles.split(
            (3, 4, self.sh_dim, 1), dim=-1
        )
        origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
        centers = origins + directions * depths[..., None]

        scale_max = self._scale_max(global_step)
        mapped_scales = self._map_scale_logits(scales, scale_max)
        pixel_size = 1 / torch.tensor((w, h), dtype=depths.dtype, device=depths.device)
        multiplier = self.get_scale_multiplier(intrinsics, pixel_size)
        scales = mapped_scales * depths[..., None] * multiplier[..., None]
        rotations = F.normalize(rotations, p=2, dim=-1)
        c2w_rotations = extrinsics[..., :3, :3]
        sh = rearrange(sh, "... (d_sh xyz) -> ... d_sh xyz", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, self.d_sh, 3)) * self.sh_mask

        if global_step >= self.cfg.sigma_warmup_steps:
            sigma_scale = self.cfg.sigma_scale_final
        else:
            progress = global_step / max(self.cfg.sigma_warmup_steps, 1)
            sigma_scale = self.cfg.sigma_scale_initial + progress * (
                self.cfg.sigma_scale_final - self.cfg.sigma_scale_initial
            )
        sigma = sigma.sigmoid() * sigma_scale + self.cfg.eps

        batch_shape = centers.shape[:-1]
        canonical = self.canonical_triangle.view(*([1] * len(batch_shape)), 3, 3)
        vertices_local = canonical * scales.unsqueeze(-2)
        vertices_rotated = _quaternion_apply(
            rotations.unsqueeze(-2).expand(*batch_shape, 3, 4),
            vertices_local,
        )
        vertices_world = torch.einsum(
            "...ij,...kj->...ki", c2w_rotations, vertices_rotated
        )
        vertices = vertices_world + centers.unsqueeze(-2)
        return TSTriangles(
            vertices=vertices,
            sigma=sigma,
            opacity=opacities.unsqueeze(-1),
            features=sh.flatten(start_dim=-2),
            centers=centers,
            scales=scales,
            mapped_scales=mapped_scales,
        )

    def from_point_map(
        self,
        points_cam: Tensor,
        c2w: Tensor,
        raw_gaussians: Tensor,
        opacity: Tensor,
        sigma: Tensor,
        image_shape: tuple[int, int] | None = None,
        intrinsics: Tensor | None = None,
        global_step: int = 0,
        flip_to_camera: bool = True,
        raw_scales: Tensor | None = None,
    ) -> TSTriangles:
        """Build triangles from DA3 camera-space points and GS head outputs.

        ``raw_gaussians`` uses the DA3 GIANT layout: XY offset, three scales,
        quaternion, SH coefficients, and optional depth offset.  XY/depth
        offsets and the legacy quaternion are intentionally ignored because
        the point map is the source of truth for this direct-difference path.
        ``raw_scales`` can provide the independent TSDPT scale head output.
        """
        b, v, h, w, _ = points_cam.shape
        eps = self.cfg.eps
        if _GEOMETRY_DEBUG:
            point_finite = torch.isfinite(points_cam).all(dim=-1)
            point_positive_z = point_finite & (points_cam[..., 2] > 0)
            print(
                "[geometry debug][adapter] input point_map "
                f"shape={tuple(points_cam.shape)} total={point_finite.numel()} "
                f"finite={_debug_mask_stats(point_finite)} "
                f"positive_z={_debug_mask_stats(point_positive_z)}",
                flush=True,
            )
        point_map = points_cam.permute(0, 1, 4, 2, 3).reshape(b * v, 3, h, w)
        padded = F.pad(point_map, (1, 1, 1, 1), mode="replicate")
        dx = padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]
        dy = padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]
        points_hw = point_map.permute(0, 2, 3, 1).reshape(b, v, h, w, 3)
        dx = dx.permute(0, 2, 3, 1).reshape(b, v, h, w, 3)
        dy = dy.permute(0, 2, 3, 1).reshape(b, v, h, w, 3)

        # The point-map differential is the sole normal source for this path.
        # Project dx back onto that plane before using it as the tangent so the
        # frame is orthonormal even when finite-difference noise is present.
        normal, normal_valid = _normalize_vector(torch.cross(dx, dy, dim=-1), eps)
        if flip_to_camera:
            view_dot = (normal * points_hw).sum(dim=-1, keepdim=True)
            normal = torch.where(view_dot > 0, -normal, normal)

        tangent = dx - (dx * normal).sum(dim=-1, keepdim=True) * normal
        tangent, tangent_valid = _normalize_vector(tangent, eps)
        bitangent, bitangent_valid = _normalize_vector(
            torch.cross(normal, tangent, dim=-1), eps
        )
        # Keep the deterministic tangent roll from the point-map y direction.
        same_orientation = (bitangent * dy).sum(dim=-1, keepdim=True) >= 0
        bitangent = torch.where(same_orientation, bitangent, -bitangent)
        tangent, tangent_reproj_valid = _normalize_vector(
            torch.cross(bitangent, normal, dim=-1), eps
        )

        valid = normal_valid & tangent_valid & bitangent_valid & tangent_reproj_valid
        if _GEOMETRY_DEBUG:
            print(
                "[geometry debug][adapter] differential frame "
                f"normal_valid={_debug_mask_stats(normal_valid)} "
                f"tangent_valid={_debug_mask_stats(tangent_valid)} "
                f"bitangent_valid={_debug_mask_stats(bitangent_valid)} "
                f"frame_valid={_debug_mask_stats(valid)}",
                flush=True,
            )
        if raw_scales is None:
            raw_scales = raw_gaussians[..., 2:5]
        self._ensure_finite("raw_scales", raw_scales)
        scale_max = self._scale_max(global_step)
        mapped_scales = self._map_scale_logits(raw_scales, scale_max)

        # The three scale channels keep the same meaning as the native
        # TriSplat/GaussianAdapter output: tangent, bitangent, and normal
        # extents.  ``dx``/``dy`` are used only to estimate the local frame.
        # Their *length* must not be used here: a central difference crossing
        # a depth discontinuity can be orders of magnitude larger than one
        # pixel and would create a giant triangle even when the predicted
        # scale factor is within its configured range.
        if image_shape is not None and intrinsics is not None:
            image_h, image_w = image_shape
            # ``intrinsics`` arrives here as a pixel-space K from the bridge.
            # GaussianAdapter's multiplier expects normalized K, so convert it
            # before applying the same depth-times-pixel-angle parameterization.
            intr_normed = intrinsics.clone()
            intr_normed[..., 0, :] = intr_normed[..., 0, :] / float(image_w)
            intr_normed[..., 1, :] = intr_normed[..., 1, :] / float(image_h)
            pixel_size = points_cam.new_tensor((1.0 / image_w, 1.0 / image_h))
            multiplier = self.get_scale_multiplier(intr_normed, pixel_size)
            depth_scale = (
                points_hw[..., 2:3].abs().clamp_min(eps)
                * multiplier[..., None, None, None]
            )
            scales = mapped_scales * depth_scale
        else:
            # Keep the adapter usable for small unit tests that do not supply
            # camera metadata.  This fallback is intentionally conservative.
            depth_scale = points_hw[..., 2:3].abs().clamp_min(eps)
            scales = mapped_scales * depth_scale

        # Optional projected-size guard for experiments that explicitly enable
        # it.  This limits the generated primitive dimensions; it does not
        # discard the source pixel or depend on a renderer-side image-bound
        # test.
        if (
            image_shape is not None
            and intrinsics is not None
            and self.cfg.max_triangle_edge_px > 0
        ):
            image_h, image_w = image_shape
            focal = torch.stack(
                (
                    intrinsics[..., 0, 0][..., None, None],
                    intrinsics[..., 1, 1][..., None, None],
                ),
                dim=-1,
            ).clamp_min(eps)
            # The canonical triangle has radius approximately 2.31.  Bound
            # the corresponding projected radius using the local depth.
            projected_radius = (
                scales[..., :2]
                * focal
                / points_hw[..., 2:3].abs().clamp_min(eps)
            )
            limit = float(self.cfg.max_triangle_edge_px) / 2.31
            shrink = (
                (limit / projected_radius.clamp_min(eps))
                .clamp_max(1.0)
                .amin(dim=-1, keepdim=True)
            )
            scales = scales * shrink

        centers = torch.einsum("bvij,bvhwj->bvhwi", c2w[..., :3, :3], points_cam)
        centers = centers + c2w[..., :3, 3][..., None, None, :]
        rotation = c2w[..., :3, :3]
        normals_world = F.normalize(
            torch.einsum("bvij,bvhwj->bvhwi", rotation, normal), dim=-1, eps=eps
        )
        tangent_world = torch.einsum("bvij,bvhwj->bvhwi", rotation, tangent)
        bitangent_world = torch.einsum("bvij,bvhwj->bvhwi", rotation, bitangent)
        canonical_xy = points_cam.new_tensor(
            [[0.0, 0.57735], [-0.5, -0.28868], [0.5, -0.28868]]
        ) * 4.0
        sx = scales[..., 0][..., None, None]
        sy = scales[..., 1][..., None, None]
        # Keep the direct-difference construction in the same camera convention
        # as GaussianAdapter, but validate the primitive before transforming it
        # to world space. A single large depth discontinuity can otherwise make
        # one vertex cross the camera plane and produce an enormous projected
        # triangle in the CUDA rasterizer.
        vertices_cam = (
            points_hw[..., None, :]
            + canonical_xy[..., 0][None, None, None, None, :, None]
            * tangent[..., None, :]
            * sx
            + canonical_xy[..., 1][None, None, None, None, :, None]
            * bitangent[..., None, :]
            * sy
        )
        # Keep the same conservative criterion as the native rasterizer: use
        # the triangle-center depth, rather than rejecting a triangle because
        # one of its perspective-expanded vertices is close to the plane.
        source_near = max(float(self.cfg.near_plane), 0.2, float(eps))
        source_center_z = vertices_cam[..., 2].mean(dim=-1)
        # Preserve a face rather than letting one vertex cross the near plane.
        # This is a geometric stabilization step: scale both in-plane axes
        # just enough to keep the whole source triangle in front of the
        # rasterizer plane, while retaining the primitive and its gradients.
        z_offset = vertices_cam[..., 2] - source_center_z[..., None]
        z_radius = z_offset.abs().amax(dim=-1)
        near_margin = (source_center_z - source_near).clamp_min(0.0)
        near_shrink = torch.where(
            z_radius > eps,
            (near_margin / z_radius.clamp_min(eps)).clamp_max(1.0),
            torch.ones_like(z_radius),
        )
        scales = scales * near_shrink[..., None]
        sx = scales[..., 0][..., None, None]
        sy = scales[..., 1][..., None, None]
        vertices_cam = (
            points_hw[..., None, :]
            + canonical_xy[..., 0][None, None, None, None, :, None]
            * tangent[..., None, :]
            * sx
            + canonical_xy[..., 1][None, None, None, None, :, None]
            * bitangent[..., None, :]
            * sy
        )
        source_center_z = vertices_cam[..., 2].mean(dim=-1)
        source_vertices_valid = (
            torch.isfinite(vertices_cam).all(dim=-1).all(dim=-1)
            & torch.isfinite(source_center_z)
            & (source_center_z > source_near)
        )
        valid = valid & source_vertices_valid

        if _GEOMETRY_DEBUG:
            near_shrunk = (near_shrink < 0.999999).sum()
            print(
                "[geometry debug][adapter] source triangles "
                f"total={valid.numel()} finite_vertices={_debug_mask_stats(source_vertices_valid)} "
                f"near_center_valid={_debug_mask_stats(source_center_z > source_near)} "
                f"near_shrunk={int(near_shrunk.item())} "
                f"final_valid={_debug_mask_stats(valid)}",
                flush=True,
            )

        vertices = (
            centers[..., None, :]
            + canonical_xy[..., 0][None, None, None, None, :, None]
            * tangent_world[..., None, :]
            * sx
            + canonical_xy[..., 1][None, None, None, None, :, None]
            * bitangent_world[..., None, :]
            * sy
        )

        if global_step >= self.cfg.sigma_warmup_steps:
            sigma_scale = self.cfg.sigma_scale_final
        else:
            progress = global_step / max(self.cfg.sigma_warmup_steps, 1)
            sigma_scale = self.cfg.sigma_scale_initial + progress * (
                self.cfg.sigma_scale_final - self.cfg.sigma_scale_initial
            )
        sigma = sigma * sigma_scale

        # DA3/GaussianAdapter stores SH as xyz-major (all coefficients for R,
        # then G, then B), while the TriSplat rasterizer consumes coefficient-
        # major [d_sh, xyz] rows. Preserve the original adapter's channel
        # contract before flattening; this is a no-op for degree zero.
        sh = raw_gaussians[..., 9 : 9 + self.sh_dim]
        sh = sh.reshape(b, v, h, w, 3, self.d_sh).transpose(-2, -1).contiguous()
        if self.d_sh > 1:
            # ``sh`` is [B, V, H, W, d_sh, 3]; keep the coefficient mask
            # aligned with the SH dimension rather than the RGB dimension.
            sh = sh * self.sh_mask.view(1, 1, 1, 1, self.d_sh, 1)
            features = sh.flatten(start_dim=-2)
        else:
            features = sh.flatten(start_dim=-2)
        output = TSTriangles(
            vertices=vertices.reshape(b, v * h * w, 3, 3),
            sigma=sigma.reshape(b, v * h * w, 1),
            opacity=opacity.reshape(b, v * h * w, 1),
            features=features.reshape(b, v * h * w, -1),
            centers=centers.reshape(b, v * h * w, 3),
            normals=normals_world.reshape(b, v * h * w, 3),
            scales=scales.reshape(b, v * h * w, 3),
            mapped_scales=mapped_scales.reshape(b, v * h * w, 3),
            primitive_valid_mask=(
                valid.reshape(b, v * h * w)
                if self.cfg.use_geometry_valid_mask
                else None
            ),
        )
        if _GEOMETRY_DEBUG:
            output_vertex_finite = torch.isfinite(output.vertices).reshape(
                *output.vertices.shape[:2], -1
            ).all(dim=-1)
            print(
                "[geometry debug][adapter] output "
                f"vertices={tuple(output.vertices.shape)} "
                f"finite_triangles={_debug_mask_stats(output_vertex_finite)} "
                f"sigma={output.sigma.shape[1]} opacity={output.opacity.shape[1]} "
                f"primitive_mask={'none' if output.primitive_valid_mask is None else _debug_mask_stats(output.primitive_valid_mask)}",
                flush=True,
            )
        return output

    def get_scale_multiplier(self, intrinsics: Tensor, pixel_size: Tensor) -> Tensor:
        xy_multipliers = 0.1 * einsum(
            intrinsics[..., :2, :2].float().inverse().to(intrinsics),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)


__all__ = ["TSAdapter", "TSAdapterCfg", "TSTriangles"]
