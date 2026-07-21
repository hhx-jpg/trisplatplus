# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment and native extensions

The documented baseline is Python 3.10, PyTorch 2.1.2, and CUDA 11.8:

```bash
conda create -y -n trisplat python=3.10
conda activate trisplat
pip install --upgrade pip
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt --no-build-isolation
bash scripts/env/rebuild_extensions.sh
```

`rebuild_extensions.sh` builds the triangle rasterizer, `simple-knn`, CroCo RoPE (`curope`), and an externally downloaded Gaussian rasterizer. It requires a CUDA-enabled PyTorch build, `nvcc`, and a compatible compiler. Rebuild after changing PyTorch, CUDA, or target GPU architecture; use `TORCH_CUDA_ARCH_LIST` when building for a GPU other than the visible one. The downloaded Gaussian rasterizer tracks an unpinned `main` branch.

Expected pretrained files are:

- `pretrained_weights/pi3.safetensors` for encoder initialization in training experiments.
- `pretrained_weights/omnidata_dpt_normal_v2.ckpt` for the monocular-normal teacher; override with `MONO_NORMAL_WEIGHTS` or `train.mono_normal_weights_path=...`.
- Released model checkpoints under `checkpoints/`.

## Main commands

Dataset roots default to `data/re10k` and `data/dl3dv`, or can be set explicitly:

```bash
export RE10K_ROOT="$PWD/data/re10k"
export DL3DV_ROOT="$PWD/data/dl3dv"
```

Packed datasets contain `train/` and `test/` directories with `.torch` chunks plus `index.json`; see `data/README.md`.

Train through the wrappers (arguments after `--` are forwarded as Hydra overrides):

```bash
bash scripts/train/train_re10k.sh --gpus 0,1,2,3,4,5,6,7 --wandb-mode offline
bash scripts/train/train_dl3dv.sh --gpus 0,1,2,3,4,5,6,7 --wandb-mode offline
# Example override:
bash scripts/train/train_re10k.sh --gpus 0 --wandb-mode disabled -- trainer.max_steps=10
```

Evaluate checkpoints and export/render meshes:

```bash
bash scripts/eval/eval_re10k_mesh.sh \
  --ckpt checkpoints/re10k_trisplat.ckpt --data-root "$RE10K_ROOT"
bash scripts/eval/eval_dl3dv_mesh.sh \
  --ckpt checkpoints/dl3dv_trisplat.ckpt --data-root "$DL3DV_ROOT"

# One-scene smoke run; wrappers also accept --skip-export and --skip-render.
bash scripts/eval/eval_re10k_mesh.sh \
  --ckpt checkpoints/re10k_trisplat.ckpt --data-root "$RE10K_ROOT" \
  --max-scenes 1

# Render meshes that were already exported.
bash scripts/eval/render_mesh.sh --test-output PATH --data-root PATH
```

Run custom-image inference:

```bash
python -m src.scripts.infer_custom_mesh \
  --image-dir /path/to/images \
  --ckpt checkpoints/re10k_trisplat.ckpt \
  --out-dir outputs/custom_mesh/demo \
  --num-views 2 \
  --force
```

Use `--view-indices 0,5` for explicit input frames. Without supplied intrinsics this path assumes a normalized camera with a 60-degree horizontal field of view. Repeated `--override key=value` flags pass Hydra overrides.

There is no first-party unit-test suite, pytest configuration, lint command, or formatting command in this repository. Consequently, there is no first-party single-test command; use the one-scene evaluation above as the product smoke test for model/rendering changes. The only CTest suite belongs to vendored GLM, not TriSplat.

## Configuration and runtime architecture

- `src/main.py` is the Hydra/Lightning entry point. `config/main.yaml` supplies defaults; `config/experiment/*.yaml` composes dataset, encoder, decoder, losses, and mesh behavior. Hydra configuration is converted to typed dataclasses in `src/config.py` before components are constructed.
- Component selection is registry-driven (`src/model/encoder/__init__.py`, `src/model/encoder/backbone/__init__.py`, `src/model/decoder/__init__.py`, `src/loss/__init__.py`, and `src/mesh/__init__.py`). Adding a variant normally requires a config dataclass, a matching registry discriminator, and YAML that satisfies the typed union.
- `src/main.py` builds the encoder, decoder, loss modules, mesh exporter, `DataModule`, and a single Lightning `ModelWrapper`. `mode: train` calls `Trainer.fit`; every other mode calls `Trainer.test`. Multi-GPU runs use DDP with `find_unused_parameters=True`.
- Hydra run outputs go to `outputs/exp_${wandb.name}/<timestamp>`. W&B can be disabled, offline, or online. `checkpointing.load` accepts a local path or W&B-style reference. Compatible full checkpoints resume trainer state; incompatible or weights-only checkpoints are loaded non-strictly and start with fresh trainer state.

## Data flow

- Dataset/view-sampler wiring starts in `src/dataset/__init__.py`. RE10K, DL3DV, and ScanNet++ share the packed `DatasetRE10k` implementation; ScanNet has separate implementations. View samplers decide context and target frame selection and can consume fixed evaluation-index JSON files from `assets/`.
- `src/dataset/data_module.py` builds stage-specific loaders and supports multiple datasets. Map-style training datasets use `MixedBatchSampler`; iterable datasets are returned as separate loaders. Variable context-view counts force iterable loaders to batch size 1 to avoid incompatible shapes.
- The central batch contract is in `src/dataset/types.py`: a scene contains `context` and `target` views. Images are `[B,V,C,H,W]`, camera-to-world extrinsics `[B,V,4,4]`, normalized intrinsics `[B,V,3,3]`, and near/far bounds `[B,V]`. Preserve these names and shapes across dataset, model, loss, and export code.

## Model and rendering flow

The core path is:

```text
context images/cameras
  -> EncoderTrisplat + local/global transformer backbone
  -> point maps, predicted cameras/intrinsics, triangle parameters
  -> triangle adapter and normal/orientation refinement
  -> triangle primitives
  -> CUDA triangle decoder at target cameras
  -> RGB/depth/alpha/normals/visibility
  -> training losses or evaluation/mesh export
```

- `src/model/encoder/encoder_trisplat.py` coordinates the backbone and three prediction branches: dense local points, primitive parameters, and camera poses. Released experiments initialize the backbone from Pi3 weights. Predicted poses may be scheduled-sampled with ground truth during training.
- Triangle construction is in `src/model/encoder/common/triangle_adapter.py`. It unprojects predicted depth to world-space centers, maps bounded scales relative to depth/intrinsics, rotates canonical equilateral triangles, and emits vertices. The normal refiner mostly changes the orientation frame rather than directly displacing vertices; the refined frame can replace the raw predicted quaternion before adaptation.
- Triangle primitives follow `src/model/types.py`: vertices `[B,N,3,3]`, sigma/opacity `[B,N,1]`, and flattened color or spherical-harmonic features `[B,N,C]`, with optional centers, normals, scales, and masks.
- `src/model/decoder/decoder_triangle_splatting_cuda.py` applies decoder policy (including opacity scheduling); `src/model/decoder/cuda_triangle_splatting.py` handles camera projection, tensor marshaling, and output-map interpretation. It forces camera and primitive inputs to FP32 and calls the `diff_triangle_rasterization` extension in `submodules/diff-triangle-rasterization/`.
- Decoder outputs are defined in `src/model/decoder/decoder.py`: RGB plus optional depth, alpha, raster normals, depth-derived normals, projection matrices, and triangle visibility. Intrinsics are normalized in datasets and converted to pixel units by the renderer.
- `src/model/model_wrapper.py` owns training, validation, and test orchestration. Training renders target views and passes decoder output, the original batch, primitives, step, and auxiliary pose/normal/intrinsic predictions to every configured loss. Test mode also controls metrics, image/video outputs, pose alignment, and mesh export.
- `src/mesh/tsdf_gs2d.py` contains both direct triangle export and TSDF-style paths. Direct export consumes encoder triangles, filters invalid/transparent primitives, fixes winding from normals, converts features to colors, optionally merges vertices, and writes PLY without needing target-view rendering.

## CUDA boundary

The triangle rasterizer's Python package wraps a custom C++/CUDA autograd operator. Its forward pass returns raster buffers retained for backward gradients to vertices, sigma, opacity, and color/SH inputs. Missing or ABI-incompatible `_C` imports, CPU tensors passed to kernels, and mismatched CUDA architectures are build/runtime failures rather than Python-only bugs. When debugging renderer changes, follow the boundary from:

```text
src/model/decoder/cuda_triangle_splatting.py
  -> submodules/diff-triangle-rasterization/diff_triangle_rasterization/__init__.py
  -> submodules/diff-triangle-rasterization/ext.cpp
  -> CUDA sources under submodules/diff-triangle-rasterization/
```
