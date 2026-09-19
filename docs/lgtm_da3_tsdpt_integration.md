# LGTM Texture Branch for DA3-TSDPT

This note records a concrete integration design based on the Apple LGTM
implementation checked out at `/root/data/haoxuan/ml-lgtm` (commit
`d496237e5ab758f052bb888a2ac5a33b3d1d934e`). It is a design document only;
the existing TSDPT checkpoints and rasterizer are unchanged by this note.

## 1. What LGTM actually adds

LGTM does not recover detail by increasing SH degree or by sharpening the
final RGB image. Its DepthSplat variant adds a learned, per-primitive texture
tile:

```text
frozen DepthSplat/geometry path
    -> one 2D Gaussian per low-resolution pixel
    -> base color (SH/DC), scale, rotation, opacity

high-resolution input + DPT features + projected source color tile
    -> texture head
    -> T x T RGB residual (and optional alpha)

2DGS rasterizer
    -> compute local Gaussian coordinates (u, v)
    -> unbounded bilinear sample of the tile
    -> sampled texture + base color
```

The released DepthSplat configuration uses `texture_size: 4`,
`texture_project_enabled: true`, `texture_project_as_base: true`, and
`texture_alpha_mode: GAUSSIAN`. Its texture head is zero initialized, so a
stage-1 2DGS checkpoint renders identically at the beginning of stage 2.
The texture projection samples each source image through the current
primitive's projected 2DGS frame; it is not a screen-space post-process.

The patched gsplat rasterizer uses an unbounded bilinear sampler with
border-clamped coordinates, a texture radius multiplier (`3.33`), and keeps
Gaussian alpha as the alpha source in `GAUSSIAN` mode. Texture alpha is not
required in this mode.

## 2. Mapping to the current TriSplat path

Current DA3-TSDPT already has the correct split point:

```text
DA3 backbone/depth/camera (frozen)
    -> TSDPT raw_gs + raw scale/opacity/sigma heads
    -> TSAdapter.from_point_map()
    -> Triangles(vertices, sigma, opacity, features=SH/DC)
    -> CUDA triangle rasterizer
```

The proposed fields are:

```python
@dataclass
class Triangles:
    ...
    texture_colors: Tensor | None = None  # [B, N, 3, T, T]
    texture_alphas: Tensor | None = None  # [B, N, T, T], optional
```

`N` remains the existing per-view-flattened triangle count in phase 1. The
texture branch must not change `vertices`, `normals`, `scales`, `sigma`, or
`opacity`; this isolates the experiment to appearance.

### Texture head

Add `src/model/head/triangle_texture_head.py`, structurally equivalent to
LGTM's `GSHeadDepthSplat` but consuming the existing DA3 features:

```text
input image at texture resolution       -> PatchifyBlock(T)       [B*V, 256, H/T, W/T]
DA3 fused DPT feature                   -> 1x1 projection            [B*V, 256, H/T, W/T]
projected source RGB tile               -> 3*T*T -> 256 conv block
concat three branches                   -> 3x3 conv block
zero-init final 1x1 conv                -> 3*T*T residual channels
```

For `T=4`, the head emits 48 channels per low-resolution pixel. With
`texture_project_as_base=true`, its output is:

```text
texture_tile = projected_source_tile + learned_delta_tile
```

The source tile is obtained from the same context-view image using the
triangle's local tangent frame. The projective version should use the three
world-space triangle vertices and target/source camera matrices, not the old
DA3 quaternion. This is important because TSDPT deliberately uses the DA3
point-map normal and `dx`-projected tangent as the frame.

### Local triangle sampling

The triangle rasterizer currently computes barycentric coordinates for every
pixel. Reuse those coordinates to form a stable local texture coordinate:

```text
u = barycentric[1] + 0.5 * barycentric[2]
v = barycentric[2]
uv = 2 * (u, v) - 1
```

For a more symmetric frame, use the projected tangent/bitangent basis and the
triangle center, then divide by the projected in-plane half extents. Both
versions must use border-clamped bilinear sampling, matching LGTM's
`bilinear_texture_sampler_unbounded`; do not add a new texture mask or a
screen-space crop.

The CUDA API should accept:

```text
texture_colors [N, 3, T, T]
texture_size   T
texture_sigma  float (start at 1.0)
```

It should return the sampled texture RGB to the existing alpha compositor,
where the final triangle color is `base_color + texture_rgb`. Alpha remains
the existing calibrated TSDPT opacity/temperature path. This avoids the
black-hole failure mode that was previously caused by validity masks.

## 3. High-resolution data contract

The current TriSplat experiment feeds DA3 only `[224, 448]` images and drops
the original pixels after the dataset crop. A texture model cannot learn
information that is no longer present. Add optional fields before the normal
crop shim:

```python
context["image_highres"]  # [B, V, 3, Hh, Wh]
target["image_highres"]   # [B, V, 3, Hh, Wh], training only
context["intrinsics_highres"]
target["intrinsics_highres"]
```

Recommended first experiment:

* Keep DA3 geometry input at `224x448`.
* Keep one triangle per geometry pixel.
* Retain the native `540x960` image as `image_highres`.
* Train/render at the existing target resolution first, then evaluate the
  same triangles at `540x960`.

This tests texture generalization without changing the proven geometry. A
high-resolution loss can be enabled after the branch is numerically stable;
it should be computed on a memory-bounded crop or at most `540x960`, not on
the uncropped source sequence.

## 4. Training schedule

Use a two-stage checkpoint-compatible schedule, matching LGTM:

### Stage A: appearance-only smoke test

* Load `render_step_008300.ckpt` from the completed bookshelf run.
* Freeze DA3 backbone, TSDPT scale/opacity/sigma, triangle geometry, and base
  SH/DC.
* Train only `texture_head` and its projection/feature blocks.
* `texture_size=4`, `texture_alpha_mode=GAUSSIAN`, `texture_sigma=1.0`.
* `texture_project_as_base=true`; final texture layer all zeros.
* Use `MSE + 0.1..0.2 LPIPS` initially. Add a high-resolution crop loss only
  after the low-resolution smoke test converges.
* Export RGB, base-only RGB, texture-only RGB, and GT for each checkpoint.

This gives a direct answer to whether the appearance bottleneck is solved by
local texture capacity, independent of any geometry changes.

### Stage B: compact textured triangles (optional)

After Stage A works, set `gaussian_downsample_ratio=2` (later 4) for the
texture branch only. Each `2x2` or `4x4` geometry cell becomes one triangle
with a `4x4` or `8x8` texture tile. Do not alter TSAdapter scale semantics in
the first compact run. Compare:

```text
one triangle/pixel, T=4
one triangle/2x2 pixels, T=4
one triangle/4x4 pixels, T=8
```

The compact branch must preserve the triangle's center depth and normal from
the valid source sample. It must not average depth across discontinuities;
choose the center sample and project the texture tile from the source image.

## 5. Checkpoint and optimizer compatibility

Existing checkpoints must load with `strict=False` and retain all existing
keys. New keys should be under a single prefix, for example:

```text
encoder.texture_head.*
```

Loading an old checkpoint therefore creates a zero-output texture branch and
reproduces the old render bit-for-bit (apart from the explicitly selected
texture rasterizer path). Do not reuse the DA3 `scratch.output_conv2` scale
rows for texture channels.

Optimizer groups should be explicit:

```text
texture_head + texture_projection: lr = 2e-4
all DA3/TSDPT/TSAdapter parameters: lr = 0
```

The zero-init last layer needs no special warmup. Save lightweight render
checkpoints every 100 steps and keep the existing compressed three-column
render format; add separate base/texture panels only to the analysis output.

## 6. CUDA and numerical safeguards

The current triangle rasterizer changes (`dist > -eps`, outer projected
radius `< 0.25`, negative-coordinate tile clamping, and sub-pixel tile
fallback) remain unchanged. The texture addition must not reintroduce
primitive or validity masks.

Required checks before a full run:

1. `texture_colors` is finite and has shape `[B, N, 3, T, T]`.
2. Sampling a constant tile returns that constant at every in-bounds and
   border-clamped UV.
3. With the zero-initialized head, `render(texture_enabled=true)` equals the
   current base render within numerical tolerance.
4. Gradient reaches only the texture head in Stage A.
5. A triangle whose projected footprint is below one pixel is still submitted
   to its center tile and receives a valid texture sample.

Do not use `valid_mask`, normal masks, alpha floors, or new near/far culls to
hide texture artifacts. Diagnose those separately because they can create the
same black holes that motivated the rasterizer fix.

## 7. Recommended implementation order

1. Add high-resolution image fields to the DL3DV dataset without changing the
   existing `image` tensors.
2. Add `texture_colors` to `Triangles` and pass it through the decoder wrapper.
3. Implement a PyTorch reference sampler and a one-triangle constant-color
   unit test.
4. Add the zero-init texture head and projected-source tile path.
5. Add CUDA texture sampling and compare it against the reference sampler.
6. Run Stage A on the bookshelf checkpoint for 1k steps, exporting base,
   texture, composite, and GT panels.
7. Only if Stage A improves detail, run the compact `2x2`/`4x4` ablations.

The first stage is deliberately conservative: it answers whether LGTM's
appearance representation fixes the observed texture softness while holding
the already-corrected DA3-TSDPT geometry constant.
