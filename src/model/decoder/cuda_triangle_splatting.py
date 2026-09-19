import logging
import math
import os
import torch
from torch import Tensor
from jaxtyping import Float
from diff_triangle_rasterization import TriangleRasterizationSettings, TriangleRasterizer
import torch.nn.functional as F
from .decoder import DecoderOutput

logger = logging.getLogger(__name__)

_GEOMETRY_DEBUG = os.environ.get("TRISPLAT_GEOMETRY_DEBUG", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _debug_mask_stats(mask: Tensor) -> str:
    return f"{int(mask.sum().item())}/{mask.numel()}"


VISIBLE_OPACITY_THRESHOLDS = (1e-3, 1e-2, 1e-1)


def depth_to_normal(depth, K):
    B, _, H, W = depth.shape

    y, x = torch.meshgrid(
        torch.arange(H, device=depth.device),
        torch.arange(W, device=depth.device),
        indexing='ij'
    )
    x = x.unsqueeze(0).expand(B, -1, -1).float()
    y = y.unsqueeze(0).expand(B, -1, -1).float()

    fx = K[:, 0, 0].view(B, 1, 1)
    fy = K[:, 1, 1].view(B, 1, 1)
    cx = K[:, 0, 2].view(B, 1, 1)
    cy = K[:, 1, 2].view(B, 1, 1)

    X = (x - cx) * depth.squeeze(1) / fx
    Y = -(y - cy) * depth.squeeze(1) / fy
    Z = depth.squeeze(1)

    XYZ = torch.stack([X, Y, Z], dim=1)

    padded_XYZ = F.pad(XYZ, (1, 1, 1, 1), mode='replicate')
    dX = padded_XYZ[:, :, 1:-1, 2:] - padded_XYZ[:, :, 1:-1, :-2]
    dY = padded_XYZ[:, :, 2:, 1:-1] - padded_XYZ[:, :, :-2, 1:-1]

    normal = torch.cross(dX, dY, dim=1)

    norm = torch.norm(normal, dim=1, keepdim=True)
    normal = normal / (norm + 1e-8)

    return normal


def get_projection_matrix(
    znear: float,
    zfar: float,
    fovX: float,
    fovY: float,
    K: torch.Tensor,
    H: int,
    W: int,
    device: torch.device
) -> torch.Tensor:
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    P = torch.zeros(4, 4, device=device, dtype=torch.float32)
    z_sign = 1.0

    P[0, 0] = 2.0 * fx / W
    P[1, 1] = 2.0 * fy / H

    P[0, 2] = (2.0 * cx / W) - 1.0
    P[1, 2] = (2.0 * cy / H) - 1.0

    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P


def render_triangle_cuda(
    extrinsics: Float[Tensor, "batch view 4 4"],
    intrinsics: Float[Tensor, "batch view 3 3"],
    image_shape: tuple[int, int],
    vertices: Float[Tensor, "batch num_tris 3 3"],
    opacity: Float[Tensor, "batch num_tris 1"],
    sigma: Float[Tensor, "batch num_tris 1"],
    features: Float[Tensor, "batch num_tris c"],
    background_color: Float[Tensor, "3"],
    global_step: int = 0,
    opacity_temp_initial: float = 1.0,
    opacity_temp_final: float = 25.0,
    opacity_temp_warmup_steps: int = 5000,
    alpha_floor_min: float = 0.0,
    alpha_floor_warmup_steps: int = 0,
    sh_degree: int = 0,
    near: Float[Tensor, "batch view"] | None = None,
    primitive_valid_mask: Tensor | None = None,
    # Keep the native decoder behavior by default: the CUDA rasterizer does
    # its original center-depth test, without a Python-side projected-face
    # prefilter. Indexed mesh callers opt into the stricter filter explicitly.
    cull_out_of_bounds: bool = False,
    center_depth_culling: bool = True,
    reject_near_crossing: bool = False,
    near_plane: float = 0.2,
    max_projected_edge_px: float = 0.0,
    log_render_stats: bool = False,
    return_triangle_visibility_mask: bool = False,
    texture_colors: Tensor | None = None,
    texture_alphas: Tensor | None = None,
    texture_size: int = 1,
    texture_color_sigma: float = 1.0,
    mesh_coverage: bool = False,
) -> DecoderOutput:
    B, V, _, _ = extrinsics.shape
    H, W = image_shape
    device = vertices.device
    # Indexed DA3/MeshSplatting faces are true pixel-grid cells, not splats
    # with a subpixel Gaussian footprint.  The CUDA extension uses a negative
    # texture sigma as an internal marker for its mesh coverage path.
    raster_texture_sigma = -1.0 if mesh_coverage else float(texture_color_sigma)

    all_images_list = []
    all_rend_normals_list = []
    all_surf_normals_list = []
    all_depths_list = []
    all_opacities_list = []
    all_proj_matrices_list = []
    all_visibility_masks_list = []
    rendered_triangles_per_view = []
    rendered_triangles_unique_per_batch = []
    rendered_visible_opacity_per_view = {
        threshold: [] for threshold in VISIBLE_OPACITY_THRESHOLDS
    }
    culled_reason_counts: dict[int, int] = {}

    w2c = torch.linalg.inv(extrinsics.float())

    fx = intrinsics[..., 0, 0] * W
    fy = intrinsics[..., 1, 1] * H
    fov_x = 2 * torch.atan(W / (2 * fx))
    fov_y = 2 * torch.atan(H / (2 * fy))

    for b in range(B):
        batch_images = []
        batch_rend_normals = []
        batch_surf_normals = []
        batch_depths = []
        batch_opacities = []
        batch_proj_matrices = []
        batch_visible_masks = []

        world_vertices = vertices[b].float()
        cur_vertices = world_vertices.contiguous().view(-1)

        if global_step >= opacity_temp_warmup_steps:
            current_temperature = opacity_temp_final
        else:
            progress = global_step / opacity_temp_warmup_steps
            current_temperature = opacity_temp_initial + (opacity_temp_final - opacity_temp_initial) * progress

        opacity_clamped = opacity[b].clamp(min=1e-6, max=1-1e-6)
        opacity_logits = torch.logit(opacity_clamped)
        cur_opacity = torch.sigmoid(opacity_logits * current_temperature).float()
        if alpha_floor_min > 0 and global_step < alpha_floor_warmup_steps:
            cur_opacity = cur_opacity.clamp_min(alpha_floor_min)
        cur_sigma = sigma[b].float()

        d_sh = (sh_degree + 1) ** 2
        cur_features = features[b].float()
        cur_texture = None if texture_colors is None else texture_colors[b].float()

        if cur_features.shape[-1] == d_sh * 3:
            cur_shs = cur_features.reshape(-1, d_sh, 3).contiguous()
        else:
            cur_shs = None
            cur_colors = torch.sigmoid(cur_features[:, :3]).float() if cur_features.shape[-1] >= 3 else torch.sigmoid(cur_features).float()

        num_triangles = cur_opacity.shape[0]

        if _GEOMETRY_DEBUG:
            input_finite = torch.isfinite(world_vertices).reshape(
                world_vertices.shape[0], -1
            ).all(dim=-1)
            print(
                "[geometry debug][decoder] input "
                f"batch={b} triangles={num_triangles} "
                f"finite_vertices={_debug_mask_stats(input_finite)} "
                f"opacity_gt_1e-2={_debug_mask_stats(cur_opacity.squeeze(-1) > 1e-2)}",
                flush=True,
            )

        scaling = torch.zeros((num_triangles), device=device, dtype=torch.float32)
        density = torch.zeros((num_triangles), device=device, dtype=torch.float32)
        means2D = torch.zeros((num_triangles, 2), device=device, dtype=torch.float32)

        num_points_per_triangle = torch.full((num_triangles,), 3, dtype=torch.int32, device=device)
        cumsum_of_points_per_triangle = torch.arange(0, num_triangles * 3, 3, dtype=torch.int32, device=device)
        primitive_count = num_triangles

        for v in range(V):
            view_mat = w2c[b, v].transpose(0, 1).float()

            cur_fovx = fov_x[b, v].item()
            cur_fovy = fov_y[b, v].item()

            cur_K = intrinsics[b, v].clone()
            cur_K[0, :] *= W
            cur_K[1, :] *= H

            proj_mat = get_projection_matrix(
                znear=0.01, zfar=1000.0,
                fovX=cur_fovx, fovY=cur_fovy,
                K=cur_K, H=H, W=W, device=device
            )

            full_proj_mat = (proj_mat @ w2c[b, v]).transpose(0, 1).float()

            campos = extrinsics[b, v, :3, 3].float()
            bg_color_f32 = background_color.float()

            cur_tanfovx = math.tan(cur_fovx * 0.5)
            cur_tanfovy = math.tan(cur_fovy * 0.5)

            # Match TriSplat's native ``in_frustum_triangle`` policy: cull by
            # triangle-center depth only. The CUDA extension uses z <= 0.2 as
            # its near test; checking every vertex (or image bounds) here
            # over-culls triangles that legitimately cover a target edge.
            view_keep = torch.ones((num_triangles,), dtype=torch.bool, device=device)
            primitive_keep = view_keep.clone()
            # The original decoder did not consume this optional diagnostic
            # mask. Keep it out of the native path; indexed mesh callers turn
            # on the Python culling path explicitly and may use the mask.
            if primitive_valid_mask is not None and cull_out_of_bounds:
                view_keep &= primitive_valid_mask[b].to(device=device, dtype=torch.bool)
                primitive_keep = view_keep.clone()
            projected_finite = torch.ones_like(view_keep)
            center_depth_keep = torch.ones_like(view_keep)
            edge_keep = torch.ones_like(view_keep)
            cuda_candidate = None
            if cull_out_of_bounds:
                vertex_h = torch.cat(
                    (
                        world_vertices.reshape(-1, 3),
                        torch.ones((num_triangles * 3, 1), device=device),
                    ),
                    dim=-1,
                )
                camera_vertices = (vertex_h @ w2c[b, v].transpose(0, 1))[:, :3]
                camera_vertices = camera_vertices.reshape(num_triangles, 3, 3)
                znear = float(near[b, v].item()) if near is not None else float(near_plane)
                znear = max(znear, float(near_plane), 0.2)
                z = camera_vertices[..., 2]
                safe_z = z.clamp_min(znear)
                projected_x = camera_vertices[..., 0] / safe_z * cur_K[0, 0] + cur_K[0, 2]
                projected_y = camera_vertices[..., 1] / safe_z * cur_K[1, 1] + cur_K[1, 2]
                center_z = z.mean(dim=-1)
                projected_finite = (
                    torch.isfinite(camera_vertices).all(dim=-1)
                    & torch.isfinite(projected_x)
                    & torch.isfinite(projected_y)
                ).all(dim=-1) & torch.isfinite(center_z)
                projected_valid = projected_finite.clone()
                if center_depth_culling:
                    center_depth_keep = center_z > znear
                    projected_valid &= center_depth_keep
                # The CUDA rasterizer projects all three vertices directly and
                # has no homogeneous near-plane clipping.  A triangle whose
                # center is in front but one vertex is behind the near plane
                # can therefore acquire an enormous projected edge and cover
                # the whole image.  Callers that need stable mesh rendering
                # can reject these triangles before they reach CUDA.
                near_crossing_keep = torch.ones_like(view_keep)
                if reject_near_crossing:
                    near_crossing_keep = z.min(dim=-1).values > znear
                    projected_valid &= near_crossing_keep
                if max_projected_edge_px > 0:
                    edge_lengths = (
                        torch.roll(
                            torch.stack((projected_x, projected_y), dim=-1),
                            shifts=-1,
                            dims=1,
                        )
                        - torch.stack((projected_x, projected_y), dim=-1)
                        ).norm(dim=-1)
                    edge_keep = edge_lengths.max(dim=-1).values <= float(max_projected_edge_px)
                    projected_valid &= edge_keep
                view_keep &= projected_valid

                if _GEOMETRY_DEBUG:
                    # Mirror the CUDA preprocessor checks to identify why a
                    # triangle receives radii=0. This is diagnostic only; the
                    # actual rasterizer remains the source of truth.
                    p2d_cuda = torch.stack(
                        (
                            camera_vertices[..., 0]
                            / (camera_vertices[..., 2] + 1e-7)
                            * cur_K[0, 0]
                            + cur_K[0, 2],
                            camera_vertices[..., 1]
                            / (camera_vertices[..., 2] + 1e-7)
                            * cur_K[1, 1]
                            + cur_K[1, 2],
                        ),
                        dim=-1,
                    )
                    center_cam = camera_vertices.mean(dim=1)
                    center_2d_cuda = torch.stack(
                        (
                            center_cam[:, 0] / (center_cam[:, 2] + 1e-7) * cur_K[0, 0]
                            + cur_K[0, 2],
                            center_cam[:, 1] / (center_cam[:, 2] + 1e-7) * cur_K[1, 1]
                            + cur_K[1, 2],
                        ),
                        dim=-1,
                    )
                    distance_points = (p2d_cuda - center_2d_cuda[:, None]).norm(dim=-1).amax(dim=-1)
                    edge_a = torch.roll(p2d_cuda, shifts=-1, dims=1) - p2d_cuda
                    side_lengths = edge_a.norm(dim=-1)
                    # Incenter uses side opposite each vertex.
                    opposite = torch.stack(
                        (
                            (p2d_cuda[:, 1] - p2d_cuda[:, 2]).norm(dim=-1),
                            (p2d_cuda[:, 0] - p2d_cuda[:, 2]).norm(dim=-1),
                            (p2d_cuda[:, 0] - p2d_cuda[:, 1]).norm(dim=-1),
                        ),
                        dim=-1,
                    )
                    incenter = (opposite[..., None] * p2d_cuda).sum(dim=1) / opposite.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    edge_distances = []
                    for edge_index in range(3):
                        p1_edge = p2d_cuda[:, edge_index]
                        p2_edge = p2d_cuda[:, (edge_index + 1) % 3]
                        edge_normal = torch.stack(
                            (p2_edge[:, 1] - p1_edge[:, 1], -(p2_edge[:, 0] - p1_edge[:, 0])),
                            dim=-1,
                        )
                        edge_norm = edge_normal.norm(dim=-1).clamp_min(1e-8)
                        edge_normal = edge_normal / edge_norm[:, None]
                        edge_offset = -(edge_normal * p1_edge).sum(dim=-1)
                        edge_dist = (edge_normal * incenter).sum(dim=-1) + edge_offset
                        flip = edge_dist > 0
                        edge_distances.append(torch.where(flip, -edge_dist, edge_dist))
                    edge_distances = torch.stack(edge_distances, dim=-1)
                    final_dist = edge_distances[:, 2]
                    cuda_opacity_keep = cur_opacity.squeeze(-1) >= 0.01
                    cuda_center_keep = center_z > 0.2
                    face_cross = torch.cross(
                        camera_vertices[:, 1] - camera_vertices[:, 0],
                        camera_vertices[:, 2] - camera_vertices[:, 0],
                        dim=-1,
                    )
                    face_norm = F.normalize(face_cross, dim=-1, eps=1e-8)
                    view_dir = F.normalize(center_cam, dim=-1, eps=1e-8)
                    grazing_keep = face_norm.mul(view_dir).sum(dim=-1).abs() >= 0.001
                    # Both CUDA paths now keep only a numerical zero-area
                    # guard (1e-4 px); ordinary native triangles no longer
                    # impose the former one-pixel inradius gate.
                    distance_keep = (distance_points <= 1600.0) & (distance_points >= 1e-4)
                    dist_keep = final_dist <= -1e-6
                    cuda_finite = (
                        torch.isfinite(p2d_cuda).reshape(p2d_cuda.shape[0], -1).all(dim=-1)
                        & torch.isfinite(distance_points)
                        & torch.isfinite(final_dist)
                    )
                    cuda_candidate = (
                        cuda_opacity_keep
                        & cuda_center_keep
                        & grazing_keep
                        & distance_keep
                        & dist_keep
                        & cuda_finite
                    )
                    print(
                        "[geometry debug][decoder] CUDA-equivalent culls "
                        f"batch={b} view={v} opacity_keep={_debug_mask_stats(cuda_opacity_keep)} "
                        f"center_keep={_debug_mask_stats(cuda_center_keep)} "
                        f"grazing_keep={_debug_mask_stats(grazing_keep)} "
                        f"distance_points_keep={_debug_mask_stats(distance_keep)} "
                        f"final_dist_keep={_debug_mask_stats(dist_keep)} "
                        f"all_internal_keep={_debug_mask_stats(cuda_candidate)} "
                        f"finite={_debug_mask_stats(cuda_finite)} "
                        f"distance_range=({distance_points.nan_to_num(posinf=1e9).min().item():.6g},{distance_points.nan_to_num(neginf=-1e9).max().item():.6g}) "
                        f"final_dist_range=({final_dist.nan_to_num(posinf=1e9).min().item():.6g},{final_dist.nan_to_num(neginf=-1e9).max().item():.6g})",
                        flush=True,
                    )
                    print(
                        "[geometry debug][decoder] projection "
                        f"batch={b} view={v} znear={znear:.6g} "
                        f"primitive_keep={_debug_mask_stats(primitive_keep)} "
                        f"finite_projected={_debug_mask_stats(projected_finite)} "
                        f"center_depth_keep={_debug_mask_stats(center_depth_keep)} "
                        f"near_crossing_keep={_debug_mask_stats(near_crossing_keep)} "
                        f"edge_keep={_debug_mask_stats(edge_keep)} "
                        f"python_keep={_debug_mask_stats(view_keep)} "
                        f"center_z_range=({center_z.min().item():.6g},{center_z.max().item():.6g})",
                        flush=True,
                    )

            cur_opacity_view = cur_opacity.clone()
            cur_opacity_view[~view_keep] = 0.0

            if _GEOMETRY_DEBUG:
                print(
                    "[geometry debug][decoder] pre-raster "
                    f"batch={b} view={v} python_keep={_debug_mask_stats(view_keep)} "
                    f"opacity_le_1e-2={_debug_mask_stats(cur_opacity.squeeze(-1) <= 1e-2)} "
                    f"opacity_nonzero={_debug_mask_stats(cur_opacity_view.squeeze(-1) > 0)}",
                    flush=True,
                )

            raster_settings = TriangleRasterizationSettings(
                image_height=H,
                image_width=W,
                tanfovx=cur_tanfovx,
                tanfovy=cur_tanfovy,
                bg=bg_color_f32,
                scale_modifier=1.0,
                viewmatrix=view_mat,
                projmatrix=full_proj_mat,
                sh_degree=sh_degree,
                campos=campos,
                prefiltered=False,
                debug=False
            )

            rasterizer = TriangleRasterizer(raster_settings)

            if cur_texture is not None:
                # Texture-aware CUDA path: sample each triangle tile at the
                # pixel's barycentric coordinates.  The primitive alpha,
                # depth, normal, and visibility calculations remain unchanged.
                cur_texture_rgb = cur_texture.mean(dim=(-1, -2))
                render_image, radii, scaling_map, density_map, allmap, max_blending = rasterizer(
                    triangles_points=cur_vertices,
                    sigma=cur_sigma,
                    num_points_per_triangle=num_points_per_triangle,
                    cumsum_of_points_per_triangle=cumsum_of_points_per_triangle,
                    number_of_points=primitive_count,
                    opacities=cur_opacity_view,
                    means2D=means2D,
                    scaling=scaling,
                    density_factor=density,
                    shs=None,
                    colors_precomp=cur_texture_rgb,
                    texture_colors=cur_texture,
                    texture_size=texture_size,
                    texture_color_sigma=raster_texture_sigma,
                )
            elif cur_shs is not None:
                render_image, radii, scaling_map, density_map, allmap, max_blending = rasterizer(
                    triangles_points=cur_vertices,
                    sigma=cur_sigma,
                    num_points_per_triangle=num_points_per_triangle,
                    cumsum_of_points_per_triangle=cumsum_of_points_per_triangle,
                    number_of_points=primitive_count,
                    opacities=cur_opacity_view,
                    means2D=means2D,
                    scaling=scaling,
                    density_factor=density,
                    shs=cur_shs,
                    colors_precomp=None,
                    texture_color_sigma=raster_texture_sigma,
                )
            else:
                render_image, radii, scaling_map, density_map, allmap, max_blending = rasterizer(
                    triangles_points=cur_vertices,
                    sigma=cur_sigma,
                    num_points_per_triangle=num_points_per_triangle,
                    cumsum_of_points_per_triangle=cumsum_of_points_per_triangle,
                    number_of_points=primitive_count,
                    opacities=cur_opacity_view,
                    means2D=means2D,
                    scaling=scaling,
                    density_factor=density,
                    shs=None,
                    colors_precomp=cur_colors,
                    texture_color_sigma=raster_texture_sigma,
                )

            if log_render_stats or return_triangle_visibility_mask:
                visible_mask = (radii > 0) & view_keep
                batch_visible_masks.append(visible_mask)

            if _GEOMETRY_DEBUG:
                raster_visible = radii > 0
                visible_after_python = (
                    visible_mask if (log_render_stats or return_triangle_visibility_mask) else raster_visible
                )
                tile_or_other_cuda_cull = (
                    "unavailable"
                    if cuda_candidate is None
                    else _debug_mask_stats(cuda_candidate & ~raster_visible)
                )
                print(
                    "[geometry debug][decoder] post-raster "
                    f"batch={b} view={v} radii_visible={_debug_mask_stats(raster_visible)} "
                    f"visible_after_python={_debug_mask_stats(visible_after_python)} "
                    f"radii_nonpositive={_debug_mask_stats(~raster_visible)} "
                    f"tile_or_other_cuda_cull={tile_or_other_cuda_cull}",
                    flush=True,
                )

            if log_render_stats:
                rendered_triangles_per_view.append(int(visible_mask.sum().item()))

                cur_opacity_flat = cur_opacity.squeeze(-1)
                for threshold in VISIBLE_OPACITY_THRESHOLDS:
                    rendered_visible_opacity_per_view[threshold].append(
                        int((visible_mask & (cur_opacity_flat > threshold)).sum().item())
                    )

                culled_reasons = radii[radii <= 0]
                if culled_reasons.numel() > 0:
                    unique_reasons, counts = torch.unique(
                        culled_reasons,
                        return_counts=True,
                    )
                    for reason, count in zip(unique_reasons.tolist(), counts.tolist()):
                        culled_reason_counts[int(reason)] = (
                            culled_reason_counts.get(int(reason), 0) + int(count)
                        )

            render_alpha = allmap[1:2]

            render_normal = F.normalize(allmap[2:5], dim=0)
            render_normal = torch.nan_to_num(render_normal, 0.0, 0.0, 0.0)

            render_depth_expected = allmap[0:1]
            render_depth_expected = render_depth_expected / render_alpha
            render_depth_expected = torch.nan_to_num(render_depth_expected, 0.0, 0.0, 0.0)

            surf_depth = render_depth_expected
            surf_normal = depth_to_normal(surf_depth.unsqueeze(0), cur_K.unsqueeze(0)).squeeze(0)
            surf_normal = torch.nan_to_num(surf_normal, 0.0, 0.0, 0.0)

            batch_images.append(render_image)
            batch_rend_normals.append(render_normal)
            batch_surf_normals.append(surf_normal)
            batch_depths.append(surf_depth.squeeze(0))
            batch_opacities.append(render_alpha.squeeze(0))
            batch_proj_matrices.append(proj_mat.T)

        if log_render_stats and batch_visible_masks:
            rendered_triangles_unique_per_batch.append(
                int(torch.stack(batch_visible_masks, dim=0).any(dim=0).sum().item())
            )

        all_images_list.append(torch.stack(batch_images))
        all_rend_normals_list.append(torch.stack(batch_rend_normals))
        all_surf_normals_list.append(torch.stack(batch_surf_normals))
        all_depths_list.append(torch.stack(batch_depths))
        all_opacities_list.append(torch.stack(batch_opacities))
        all_proj_matrices_list.append(torch.stack(batch_proj_matrices))
        if return_triangle_visibility_mask:
            all_visibility_masks_list.append(torch.stack(batch_visible_masks))

    if log_render_stats and rendered_triangles_per_view:
        per_view = torch.tensor(rendered_triangles_per_view, dtype=torch.float32)
        logger.info(
            "[decoder debug] step=%s triangle/rendered_triangles_per_view: num_views=%d min=%d max=%d mean=%.3f median=%.3f total=%d ratio_mean=%.6f",
            global_step,
            len(rendered_triangles_per_view),
            int(per_view.min().item()),
            int(per_view.max().item()),
            per_view.mean().item(),
            per_view.median().item(),
            int(per_view.sum().item()),
            per_view.mean().item() / max(num_triangles, 1),
        )
        for threshold, counts in rendered_visible_opacity_per_view.items():
            threshold_per_view = torch.tensor(counts, dtype=torch.float32)
            logger.info(
                "[decoder debug] step=%s triangle/rendered_triangles_per_view_visible_opacity_gt_%s: num_views=%d min=%d max=%d mean=%.3f median=%.3f total=%d ratio_vs_total_mean=%.6f ratio_vs_visible_mean=%.6f",
                global_step,
                f"{threshold:.0e}",
                len(counts),
                int(threshold_per_view.min().item()),
                int(threshold_per_view.max().item()),
                threshold_per_view.mean().item(),
                threshold_per_view.median().item(),
                int(threshold_per_view.sum().item()),
                threshold_per_view.mean().item() / max(num_triangles, 1),
                (threshold_per_view / per_view.clamp_min(1.0)).mean().item(),
            )
    if log_render_stats and rendered_triangles_unique_per_batch:
        per_batch_unique = torch.tensor(rendered_triangles_unique_per_batch, dtype=torch.float32)
        logger.info(
            "[decoder debug] step=%s triangle/rendered_triangles_unique_per_batch: num_batches=%d min=%d max=%d mean=%.3f median=%.3f ratio_mean=%.6f",
            global_step,
            len(rendered_triangles_unique_per_batch),
            int(per_batch_unique.min().item()),
            int(per_batch_unique.max().item()),
            per_batch_unique.mean().item(),
            per_batch_unique.median().item(),
            per_batch_unique.mean().item() / max(num_triangles, 1),
        )
    if log_render_stats and culled_reason_counts:
        total_culled = sum(culled_reason_counts.values())
        reason_summary = " ".join(
            f"{reason}:{count}({count / total_culled:.4f})"
            for reason, count in sorted(culled_reason_counts.items())
        )
        logger.info(
            "[decoder debug] step=%s triangle/culled_radii_reasons: total=%d %s",
            global_step,
            total_culled,
            reason_summary,
        )

    return DecoderOutput(
        color=torch.stack(all_images_list),
        depth=torch.stack(all_depths_list),
        opacity=torch.stack(all_opacities_list),
        rend_normal=torch.stack(all_rend_normals_list),
        surf_normal=torch.stack(all_surf_normals_list),
        projection_matrix=torch.stack(all_proj_matrices_list),
        triangle_visibility_mask=(
            torch.stack(all_visibility_masks_list)
            if return_triangle_visibility_mask
            else None
        ),
    )
