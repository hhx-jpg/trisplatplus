import logging
import os
from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

from ..types import Triangles
from .cuda_triangle_splatting import render_triangle_cuda
from .decoder import Decoder, DecoderOutput

logger = logging.getLogger(__name__)

_GEOMETRY_DEBUG = os.environ.get("TRISPLAT_GEOMETRY_DEBUG", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@dataclass
class DecoderTriangleSplattingCUDACfg:
    name: Literal["triangle_splatting_cuda"]
    background_color: list[float] | None = None
    opacity_temp_initial: float = 1.0
    opacity_temp_final: float = 25.0
    opacity_temp_warmup_steps: int = 5000
    alpha_floor_min: float = 0.0
    alpha_floor_warmup_steps: int = 0
    sh_degree: int = 0
    prune_opacity_threshold: float = 0.0
    # Native forward compatibility: Python-side face filtering is opt-in for
    # relocated indexed meshes and disabled for the original decoder path.
    cull_out_of_bounds: bool = False
    center_depth_culling: bool = True
    near_plane: float = 0.2
    max_projected_edge_px: float = 0.0
    # Optional post-raster color correction.  Defaults preserve the raw
    # rasterizer output; experiments can raise brightness and chroma without
    # changing triangle geometry or the learned feature head.
    color_brightness_gain: float = 1.0
    color_saturation_gain: float = 1.0
    texture_size: int = 1
    texture_color_sigma: float = 1.0

    def __post_init__(self):
        if self.background_color is None:
            self.background_color = [0.0, 0.0, 0.0]


class DecoderTriangleSplattingCUDA(Decoder[DecoderTriangleSplattingCUDACfg]):

    def __init__(
        self,
        cfg: DecoderTriangleSplattingCUDACfg,
    ) -> None:
        super().__init__(cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        triangles: Triangles,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode=None,
        global_step: int = 0,
        debug_log_interval: int | None = None,
        **kwargs,
    ) -> DecoderOutput:
        return_triangle_visibility_mask = bool(kwargs.pop("return_triangle_visibility_mask", False))
        extrinsics = extrinsics.float()
        intrinsics = intrinsics.float()
        triangles = Triangles(
            vertices=triangles.vertices.float(),
            sigma=triangles.sigma.float(),
            opacity=triangles.opacity.float(),
            features=triangles.features.float(),
            centers=None if triangles.centers is None else triangles.centers.float(),
            normals=None if triangles.normals is None else triangles.normals.float(),
            scales=None if triangles.scales is None else triangles.scales.float(),
            mapped_scales=None if triangles.mapped_scales is None else triangles.mapped_scales.float(),
            primitive_valid_mask=triangles.primitive_valid_mask,
            texture_colors=None if triangles.texture_colors is None else triangles.texture_colors.float(),
            texture_alphas=None if triangles.texture_alphas is None else triangles.texture_alphas.float(),
        )

        if global_step >= self.cfg.opacity_temp_warmup_steps:
            current_temperature = self.cfg.opacity_temp_final
        else:
            progress = global_step / self.cfg.opacity_temp_warmup_steps
            current_temperature = self.cfg.opacity_temp_initial + (
                self.cfg.opacity_temp_final - self.cfg.opacity_temp_initial
            ) * progress

        render_opacity = triangles.opacity.float().clamp(min=1e-6, max=1 - 1e-6)
        render_opacity = torch.sigmoid(
            torch.logit(render_opacity) * current_temperature
        ).float()
        alpha_floor_active = (
            self.cfg.alpha_floor_min > 0
            and global_step < self.cfg.alpha_floor_warmup_steps
        )
        if alpha_floor_active:
            render_opacity = render_opacity.clamp_min(self.cfg.alpha_floor_min)

        should_log_debug = self.should_log_debug_stats(global_step, debug_log_interval)
        if should_log_debug:
            logger.info(
                "[decoder debug] step=%s triangle opacity_temperature=%.6g alpha_floor=%s",
                global_step,
                current_temperature,
                self.cfg.alpha_floor_min if alpha_floor_active else 0.0,
            )
            self.log_debug_tensor_stats("triangle/input_scales", triangles.mapped_scales, global_step)
            self.log_debug_tensor_stats("triangle/render_scales", triangles.scales, global_step)
            self.log_debug_tensor_stats("triangle/input_sigma", triangles.sigma, global_step)
            self.log_debug_tensor_stats("triangle/input_opacity", triangles.opacity, global_step)
            self.log_debug_tensor_stats(
                "triangle/render_input_opacity",
                render_opacity,
                global_step,
            )
        if _GEOMETRY_DEBUG and triangles.scales is not None:
            render_scales = triangles.scales.detach().float()
            print(
                "[geometry debug][decoder wrapper] scales "
                f"range=({render_scales.amin().item():.6g},{render_scales.amax().item():.6g}) "
                f"median={render_scales.median().item():.6g}",
                flush=True,
            )

        output = render_triangle_cuda(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image_shape=image_shape,
            vertices=triangles.vertices,
            opacity=triangles.opacity,
            sigma=triangles.sigma,
            features=triangles.features,
            texture_colors=triangles.texture_colors,
            texture_alphas=triangles.texture_alphas,
            background_color=self.background_color,
            global_step=global_step,
            opacity_temp_initial=self.cfg.opacity_temp_initial,
            opacity_temp_final=self.cfg.opacity_temp_final,
            opacity_temp_warmup_steps=self.cfg.opacity_temp_warmup_steps,
            alpha_floor_min=self.cfg.alpha_floor_min,
            alpha_floor_warmup_steps=self.cfg.alpha_floor_warmup_steps,
            sh_degree=self.cfg.sh_degree,
            near=near,
            primitive_valid_mask=triangles.primitive_valid_mask,
            cull_out_of_bounds=self.cfg.cull_out_of_bounds,
            center_depth_culling=self.cfg.center_depth_culling,
            near_plane=self.cfg.near_plane,
            max_projected_edge_px=self.cfg.max_projected_edge_px,
            log_render_stats=should_log_debug,
            return_triangle_visibility_mask=return_triangle_visibility_mask,
            texture_size=self.cfg.texture_size,
            texture_color_sigma=self.cfg.texture_color_sigma,
        )
        brightness_gain = float(self.cfg.color_brightness_gain)
        saturation_gain = float(self.cfg.color_saturation_gain)
        if brightness_gain <= 0.0 or saturation_gain < 0.0:
            raise ValueError(
                "color_brightness_gain must be > 0 and color_saturation_gain must be >= 0"
            )
        if brightness_gain != 1.0 or saturation_gain != 1.0:
            # Apply correction after rasterization so all losses see the same
            # enhanced image while geometry, alpha, depth and normals remain
            # untouched.  Luma is used as the neutral axis to preserve hue.
            rgb = output.color
            luma = (
                rgb[:, :, 0:1] * 0.299
                + rgb[:, :, 1:2] * 0.587
                + rgb[:, :, 2:3] * 0.114
            )
            output.color = (
                luma * brightness_gain
                + (rgb - luma) * saturation_gain
            ).clamp(0.0, 1.0)
        return output
