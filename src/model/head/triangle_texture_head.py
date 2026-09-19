"""LGTM-style projected texture residual head for TriSplat triangles."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def project_triangle_texture(
    vertices: torch.Tensor,
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    texture_size: int,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Project each source-view triangle plane into an RGB texture tile.

    ``vertices`` are grouped by source view and geometry pixel.  The square
    tile uses the affine frame p0 + u(p1-p0) + v(p2-p0); sampling is border
    clamped so no validity mask can create black holes.
    """
    b, v, n, _, _ = vertices.shape
    _, _, _, h, w = images.shape
    device = vertices.device
    uv = (torch.arange(texture_size, device=device, dtype=vertices.dtype) + 0.5) / texture_size
    vv, uu = torch.meshgrid(uv, uv, indexing="ij")
    uv_grid = torch.stack((uu, vv), dim=-1).reshape(1, 1, 1, texture_size * texture_size, 2)
    uv_grid = (uv_grid * 2.0 - 1.0) * float(sigma)
    uv_grid = (uv_grid + 1.0) * 0.5

    p0 = vertices[..., 0, :].unsqueeze(-2)
    p1 = vertices[..., 1, :].unsqueeze(-2)
    p2 = vertices[..., 2, :].unsqueeze(-2)
    points = p0 + uv_grid[..., 0:1] * (p1 - p0) + uv_grid[..., 1:2] * (p2 - p0)
    points = points.reshape(b, v, n * texture_size * texture_size, 3)

    w2c = torch.linalg.inv(extrinsics.float())
    cam = torch.einsum("bvij,bvnj->bvni", w2c[..., :3, :3], points) + w2c[..., :3, 3].unsqueeze(-2)
    z = cam[..., 2:3].clamp_min(1e-5)
    k = intrinsics.float().clone()
    k[..., 0, :] *= float(w)
    k[..., 1, :] *= float(h)
    xy = cam[..., :2] / z
    xy = torch.einsum("bvij,bvnj->bvni", k[..., :2, :2], xy) + k[..., :2, 2].unsqueeze(-2)
    grid = torch.stack((2.0 * xy[..., 0] / max(w, 1) - 1.0, 2.0 * xy[..., 1] / max(h, 1) - 1.0), dim=-1)
    grid = grid.reshape(b * v, n * texture_size * texture_size, 1, 2)
    source = images.reshape(b * v, 3, h, w)
    sampled = F.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=False)
    sampled = sampled.squeeze(-1).transpose(1, 2)
    return sampled.reshape(b, v, n, 3, texture_size, texture_size)


class TriangleTextureHead(nn.Module):
    """Context-image projected tile plus a zero-initialized learned residual."""

    def __init__(self, texture_size: int = 4, project_as_base: bool = True, sigma: float = 1.0):
        super().__init__()
        self.texture_size = int(texture_size)
        self.project_as_base = bool(project_as_base)
        self.sigma = float(sigma)
        channels = 3 * self.texture_size * self.texture_size
        self.patchify = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=self.texture_size, stride=self.texture_size),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.projected_processor = nn.Sequential(
            nn.Conv2d(channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.output = nn.Conv2d(256, channels, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        images: torch.Tensor,
        projected_tiles: torch.Tensor,
        output_hw: tuple[int, int],
    ) -> torch.Tensor:
        b, v, _, _, _ = images.shape
        h, w = output_hw
        flat_images = images.reshape(b * v, 3, images.shape[-2], images.shape[-1])
        feat_img = self.patchify(flat_images)
        feat_img = F.interpolate(feat_img, size=(h, w), mode="bilinear", align_corners=False)
        # Each source view contributes one tile per output pixel/triangle.
        # Make the pixel grid explicit so einops validates that the tile count
        # matches the requested decoder resolution.
        projected = rearrange(
            projected_tiles,
            "b v (h w) c th tw -> (b v) (c th tw) h w",
            h=h,
            w=w,
        )
        feat_projected = self.projected_processor(projected)
        residual = self.output(self.fusion(torch.cat((feat_img, feat_projected), dim=1)))
        if self.project_as_base:
            residual = residual + projected
        return rearrange(residual, "(b v) (c th tw) h w -> b v (h w) c th tw", b=b, v=v, c=3, th=self.texture_size, tw=self.texture_size)
