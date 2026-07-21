# TriSplat Code Reading Guide

This guide maps the main paper concepts to code and gives a fast reading path for understanding the project.

## 1. Big Picture

TriSplat is organized around this pipeline:

```text
Hydra config
  -> Lightning ModelWrapper
  -> TriSplat encoder
  -> triangle splatting decoder
  -> direct/TSDF mesh exporter
```

The most important implementation files are:

| Concept | File |
|---|---|
| Main train/test entry | `src/main.py` |
| Training/test data flow | `src/model/model_wrapper.py` |
| TriSplat encoder: pointmap, primitive head, camera head, triangle orientation | `src/model/encoder/encoder_trisplat.py` |
| DINOv2 + local/global backbone | `src/model/encoder/backbone/backbone_local_global.py` |
| DINOv2 ViT implementation | `src/model/encoder/backbone/dinov2/models/vision_transformer.py` |
| Transformer/attention layers | `src/model/encoder/layers/block.py`, `src/model/encoder/layers/attention.py` |
| Point/camera head utilities | `src/model/encoder/layers/transformer_head.py`, `src/model/encoder/layers/camera_head.py` |
| Raw triangle params -> world-space triangles | `src/model/encoder/common/triangle_adapter.py` |
| Public primitive dataclasses | `src/model/types.py` |
| Triangle splatting decoder | `src/model/decoder/decoder_triangle_splatting_cuda.py` |
| CUDA triangle rasterizer wrapper | `src/model/decoder/cuda_triangle_splatting.py` |
| Mesh export | `src/mesh/tsdf_gs2d.py` |
| Custom image inference | `src/scripts/infer_custom_mesh.py` |

## 2. Configuration First

Start from the Hydra configs to understand which modules are active.

Recommended files:

1. `config/main.yaml`
2. `config/model/encoder/trisplat.yaml`
3. `config/model/encoder/backbone/local_global.yaml`
4. `config/model/decoder/triangle_splatting_cuda.yaml`
5. `config/mesh/tsdf_gs2d.yaml`
6. `config/experiment/trisplat_re10k_triangle_refiner_unet_10m_test.yaml`
7. `config/experiment/trisplat_dl3dv_triangle_refiner_unet_10m_test.yaml`

Important config flags:

| Flag | Meaning |
|---|---|
| `model.encoder.pose_free` | Predict camera poses instead of requiring known poses. |
| `model.encoder.use_triangle` | Use triangle primitives instead of Gaussian primitives. |
| `model.encoder.geometry_normal.anchor_rotation` | Anchor triangle orientation from pointmap-derived geometry normals. |
| `model.encoder.geometry_normal.refiner.enabled` | Enable learned normal refinement before orienting triangles. |
| `model.encoder.backbone.predict_intrinsics` | Predict focal/intrinsic information in the backbone. |
| `mesh.tsdf_gs2d.export_mode` | `direct`, `tsdf`, or `both`; custom inference uses `direct`. |

## 3. Hydra in This Project

Hydra is the configuration system. Commands such as:

```bash
python -m src.main \
  +experiment=trisplat_re10k_triangle_refiner_unet_10m_test \
  mode=test \
  checkpointing.load=checkpoints/re10k_trisplat.ckpt \
  mesh.tsdf_gs2d.export_mode=direct
```

mean:

1. Load `config/main.yaml`.
2. Add the selected experiment config.
3. Override individual fields from the command line.

So when reading code, always check both the default yaml and the experiment yaml.

## 4. Main Runtime Flow

### Training / official test entry

Read:

1. `src/main.py`
2. `src/model/model_wrapper.py`

High-level flow:

```text
src/main.py
  -> compose Hydra config
  -> get_encoder(cfg.model.encoder)
  -> get_decoder(cfg.model.decoder)
  -> get_mesh(cfg.mesh)
  -> build ModelWrapper
  -> build DataModule
  -> trainer.fit(...) or trainer.test(...)
```

Inside `ModelWrapper.training_step` / `ModelWrapper.test_step`:

```text
batch
  -> data_shim
  -> _prepare_encoder_context(...)
  -> encoder(context)
  -> primitives
  -> decoder(primitives, target cameras)
  -> RGB/depth/normal render outputs
  -> losses or metrics
  -> optional mesh export
```

### Custom image inference entry

Read:

- `src/scripts/infer_custom_mesh.py`

This path does not require RealEstate10K/DL3DV packed datasets. It loads images from a folder, creates default intrinsics when needed, runs the pose-free encoder, and exports direct triangle meshes.

Key functions:

| Function | Purpose |
|---|---|
| `collect_image_paths` | Find images in the input folder. |
| `load_images` | Resize/crop and convert images to tensors. |
| `make_default_intrinsics` | Create normalized pinhole intrinsics from `--fov-deg`. |
| `build_batch` | Build a minimal batch with identity initial poses. |
| `load_encoder_weights` | Load the encoder weights from a TriSplat checkpoint. |
| `main` | Run encoder, update predicted poses, call mesh exporter. |

## 5. DINOv2 + Local/Global Attention

Read:

1. `src/model/encoder/backbone/backbone_local_global.py`
2. `src/model/encoder/backbone/dinov2/models/vision_transformer.py`
3. `src/model/encoder/layers/block.py`
4. `src/model/encoder/layers/attention.py`

Key points:

- `BackboneLocalGlobal` creates a DINOv2 ViT-L/14 register model.
- The local/global behavior is implemented in `BackboneLocalGlobal.decode()` by alternating token layouts:
  - local/view-wise attention: tokens shaped like `(B * V, HW, C)`;
  - global/cross-view attention: tokens shaped like `(B, V * HW, C)`.
- The decoder blocks are RoPE transformer blocks from `src/model/encoder/layers/block.py`.

The core reading target is `BackboneLocalGlobal.decode()`.

## 6. TriSplat Encoder Main Body

Read:

- `src/model/encoder/encoder_trisplat.py`

Recommended order inside the file:

1. `EncoderTrisplatCfg`
2. `EncoderTrisplat.__init__`
3. `EncoderTrisplat.forward`
4. pointmap prediction branch
5. primitive/triangle parameter branch
6. camera prediction branch
7. `_build_triangle_geometry_rotation`
8. `_forward_triangle`

### Pointmap head

Important code:

- `self.point_decoder`
- `self.point_head`
- `LinearPts3d` in `src/model/encoder/layers/transformer_head.py`

Conceptual flow:

```text
backbone tokens
  -> point decoder
  -> point head predicts xy and z
  -> z = exp(z)
  -> local pointmap = [x * z, y * z, z]
```

### Primitive / triangle head

The code still uses historical names such as `gaussian_decoder` and `gaussian_head`, but when `use_triangle: true`, these predict triangle primitive parameters.

Conceptual flow:

```text
backbone tokens
  -> primitive decoder/head
  -> raw density + raw triangle params
  -> optional geometry-anchored rotation override
  -> TriangleAdapter
  -> world-space Triangles
```

### Camera head

Read:

- `src/model/encoder/layers/camera_head.py`

The camera head predicts translation and a 9D rotation representation. The rotation is projected to SO(3) using SVD orthogonalization. In pose-free mode, predicted camera poses are used for downstream triangle construction and mesh export.

## 7. Geometry-Anchored Triangle Orientation

This is one of the key paper ideas. The main function is:

- `EncoderTrisplat._build_triangle_geometry_rotation` in `src/model/encoder/encoder_trisplat.py`

Conceptual flow:

```text
pointmap
  -> finite-difference surface normals
  -> optional smoothing
  -> optional learned normal refiner
  -> optional monocular-normal teacher blend during training
  -> construct tangent / bitangent / normal frame
  -> convert rotation matrix to quaternion
  -> overwrite triangle raw rotation
```

Then `_forward_triangle` sends the modified raw triangle parameters into `TriangleAdapter`.

## 8. TriangleAdapter: Raw Params to World Triangles

Read:

- `src/model/encoder/common/triangle_adapter.py`
- `src/model/types.py`

`TriangleAdapter` takes raw triangle parameters and camera/ray information, then computes:

- triangle centers from rays and predicted depths;
- triangle scale mapping;
- quaternion rotation;
- canonical triangle vertices;
- world-space vertices;
- opacity/density;
- color / spherical harmonics coefficients.

The final public primitive object is the `Triangles` dataclass in `src/model/types.py`.

## 9. Triangle Splatting Decoder

Read:

1. `src/model/decoder/decoder_triangle_splatting_cuda.py`
2. `src/model/decoder/cuda_triangle_splatting.py`
3. `submodules/diff-triangle-rasterization/`

Conceptual flow:

```text
Triangles
  -> opacity/temperature scheduling
  -> TriangleRasterizer
  -> RGB, depth, opacity, rendered normals, surface normals
```

This path is used for differentiable rendering during training/evaluation.

## 10. Mesh Export: Direct Mesh and TSDF/GS2D

Read:

- `src/mesh/tsdf_gs2d.py`

Despite the `tsdf_gs2d` name, custom inference and current mesh evaluation commonly use:

```yaml
mesh.tsdf_gs2d.export_mode: direct
```

In direct mode, the exporter writes triangles directly as an ordinary mesh:

```text
Triangles
  -> opacity/visibility filtering
  -> optional winding correction
  -> duplicate-vertex merge
  -> optional post-process cleanup
  -> .ply/.off mesh
```

Important functions:

| Function | Purpose |
|---|---|
| `TsdfGs2d.main` | Main mesh export entry. |
| `_build_direct_mesh_from_triangles` | Build direct mesh from predicted triangle primitives. |

Outputs usually include:

```text
DIRECT_triangle_mesh.ply
DIRECT_triangle_mesh.off
DIRECT_triangle_mesh_post.ply
DIRECT_triangle_mesh_post.off
```

The `_post` mesh is usually the preferred visualization/simulation output.

## 11. Mesh and Gaussian Splatting Relationship

Gaussian Splatting represents a scene with Gaussian primitives and normally does not directly produce an ordinary triangle mesh. Mesh extraction from Gaussian Splatting is usually a separate post-processing step.

TriSplat instead predicts oriented triangle primitives. Therefore:

```text
TriSplat primitives are already triangles
  -> direct mesh export is natural
  -> no heavy GS-to-mesh conversion is required in direct mode
```

Some code names still contain `gaussian_*` for historical reasons, but in triangle mode they are used to predict triangle parameters.

## 12. Practical Reading Path

If you want the shortest path from paper to code, read in this order:

1. `config/model/encoder/trisplat.yaml`
2. `config/model/encoder/backbone/local_global.yaml`
3. `src/model/encoder/backbone/backbone_local_global.py`
4. `src/model/encoder/encoder_trisplat.py`
5. `src/model/encoder/common/triangle_adapter.py`
6. `src/model/types.py`
7. `src/model/decoder/decoder_triangle_splatting_cuda.py`
8. `src/model/decoder/cuda_triangle_splatting.py`
9. `src/mesh/tsdf_gs2d.py`
10. `src/scripts/infer_custom_mesh.py`

## 13. Useful Commands

Custom 6-view inference example:

```bash
cd /home/v-hanhaoxuan/TriSplat

CUDA_VISIBLE_DEVICES=0 python -m src.scripts.infer_custom_mesh \
  --image-dir /home/v-hanhaoxuan/TriSplat/my_scene_images \
  --ckpt checkpoints/re10k_trisplat.ckpt \
  --out-dir outputs/custom_mesh/my_scene_6view \
  --num-views 6 \
  --force
```

Main mesh output:

```text
outputs/custom_mesh/my_scene_6view/mesh/DIRECT_triangle_mesh_post.ply
```

If the environment cannot find CUDA libraries, use the explicit environment form:

```bash
env CUDA_VISIBLE_DEVICES=0 \
PATH="/usr/local/cuda-12.8/bin:/home/v-hanhaoxuan/miniconda3/envs/trisplat/bin:$PATH" \
LD_LIBRARY_PATH="/home/v-hanhaoxuan/miniconda3/envs/trisplat/lib/python3.10/site-packages/torch/lib:/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH" \
/home/v-hanhaoxuan/miniconda3/envs/trisplat/bin/python -m src.scripts.infer_custom_mesh \
  --image-dir /home/v-hanhaoxuan/TriSplat/my_scene_images \
  --ckpt checkpoints/re10k_trisplat.ckpt \
  --out-dir outputs/custom_mesh/my_scene_6view \
  --num-views 6 \
  --force
```
