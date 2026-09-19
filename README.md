<h1 align="center">TriSplat++: DA3 Geometry with LGTM Texture</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2605.26115"><img src="https://img.shields.io/badge/Paper-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"></a>
  <a href="https://lhmd.top/trisplat"><img src="https://img.shields.io/badge/Project%20Page-000000?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="https://github.com/ziplab/TriSplat"><img src="https://img.shields.io/badge/Code-181717?style=for-the-badge&logo=github&logoColor=white" alt="Code"></a>
  <a href="https://huggingface.co/lhmd/TriSplat"><img src="https://img.shields.io/badge/Models-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Models"></a>
</p>

<p align="center">
  <a href="https://lhmd.top/">Weijie Wang</a><sup>1,*</sup> &nbsp;
  <a href="https://github.com/puLangMu">Zimu Li</a><sup>1,*</sup> &nbsp;
  <a href="https://chuan-10.github.io/">Jinchuan Shi</a><sup>1</sup> &nbsp;
  <a href="https://steve-zeyu-zhang.github.io/">Zeyu Zhang</a><sup>1</sup> &nbsp;
  <a href="https://botaoye.github.io/">Botao Ye</a><sup>2,3</sup> <br />
  <a href="https://people.inf.ethz.ch/~pomarc/">Marc Pollefeys</a><sup>2,4</sup> &nbsp;
  <a href="https://donydchen.github.io/">Donny Y. Chen</a><sup>5</sup> &nbsp;
  <a href="https://bohanzhuang.github.io/">Bohan Zhuang</a><sup>1</sup> &nbsp;
</p>

<p align="center">
  <sup>1</sup>Zhejiang University &nbsp; &nbsp;
  <sup>2</sup>ETH Zurich &nbsp; &nbsp;
  <sup>3</sup>ETH AI Center &nbsp; &nbsp;
  <sup>4</sup>Microsoft &nbsp; &nbsp;
  <sup>5</sup>Monash University
</p>

<p align="center">
  <img src="https://lhmd.top/trisplat/assets/images/teaser.jpg" alt="TriSplat teaser" width="100%">
</p>

TriSplat++ is the focused DA3/LGTM variant of TriSplat. It replaces the
original geometry trunk with the Depth-Anything-3 (DA3) GIANT multi-view
backbone and its DPT-style feature/depth fusion head. The resulting TSDPT
projection predicts a calibrated point map together with the triangle
attributes required by TriSplat (scale, orientation, opacity, sigma, and
appearance coefficients). A dedicated LGTM-style texture-remapping head then
projects native context pixels into each triangle's local 3D frame and learns a
compact residual texture tile. The existing CUDA triangle rasterizer samples
those tiles during rasterization, so high-frequency appearance is improved
without changing the DA3 geometry, depth, normal, or mesh-export paths.

## TriSplat++ architecture and improvements

The model separates geometry from appearance while keeping both branches in
the same differentiable triangle-splatting pipeline:

```text
context images (224x448) + calibrated cameras
        |
        v
DA3-GIANT backbone + camera conditioning (frozen)
        |
        v
DPT-style depth/feature fusion + TSDPT GS projection head
        |  point map, depth, camera frame, scale/rotation/opacity/sigma
        v
TSAdapter + calibrated Sim(3)/pose alignment
        |
        v
TriSplat triangles ------------------------------+
                                                  |
native context images (540x960) + native K       |
        |                                         |
        v                                         |
project each triangle into its source view       |
        |                                         |
        v                                         |
4x4 projected RGB tile + image patch features   |
        |                                         |
        v                                         |
LGTM texture-remapping head                     |
  projected-tile processor + patchify/fusion     |
  zero-initialized residual, base + delta tile   |
        |                                         |
        +------------> CUDA triangle rasterizer -+
                           barycentric tile sampling -> RGB/depth/normal
```

The main changes over the original TriSplat path are:

1. **DA3/DPT geometry trunk.** DA3 supplies multi-view features, camera
   conditioning, depth, and a DPT-like fused feature map. The TSDPT head keeps
   the checkpoint-compatible GS parameterization and adds independent opacity
   and sigma outputs. The frozen DA3 prediction is lifted with the calibrated
   DL3DV intrinsics and aligned to the supplied context poses before triangle
   construction.
2. **Dedicated texture remapping.** Instead of relying only on per-triangle
   SH/DC color, each triangle receives a source-image tile obtained by
   projective remapping through its three world-space vertices. A `4x4` tile is
   processed together with native-image patch features; a zero-initialized
   residual makes the projected color a stable base while the learned head
   restores local details. This is a geometry-aware texture operation, not a
   screen-space sharpening/post-processing step.
3. **Appearance-only optimization.** `texture_only=true` freezes DA3, TSDPT,
   the triangle geometry, and the calibrated opacity/sigma path. Only the
   texture head and its projection/fusion blocks are optimized, allowing
   texture quality to improve without moving the reconstructed surface.
4. **Explicit resolution contract.** The DA3 geometry branch consumes
   `224x448` crops, while the texture branch retains native `540x960` pixels
   and intrinsics. The six-view renderer checks this contract before model
   construction and writes a skip record for incompatible scenes.

## What is in this repository

The independent checkout keeps the training and rendering path together:

```text
src/model/encoder/encoder_da3_tsdpt.py   DA3/TSDPT bridge and camera alignment
src/model/head/triangle_texture_head.py   LGTM projected texture + residual head
src/model/decoder/cuda_triangle_splatting.py
                                         textured CUDA triangle renderer
src/model/decoder/decoder_triangle_splatting_cuda.py
                                         decoder wrapper and alpha schedule
src/model/types.py                        triangle/texture tensor contracts
config/model/encoder/da3_tsdpt.yaml       DA3 encoder defaults
config/experiment/trisplat_dl3dv_tsdpt_lgtm10k_train.yaml
                                         reference 10K LGTM training setup
scripts/render_6view_triptych.py          DL3DV six-view GT/normal/render export
```

The required DA3 Python bridge is vendored under
`third_party/depth-anything-3/`; no DA3 checkout is needed for the source
code. The TriSplat++ checkpoint in `weights/` is output-head-only: it does not
contain the DA3 backbone or the DA3 camera/depth feature branches. Every
forward or training run must therefore provide the complete
`DA3-GIANT-1.1` checkpoint separately. Set `DA3_ROOT` when you want to use a
separately managed Depth-Anything-3 checkout (or its `src` directory), and set
`DA3_CHECKPOINT` to that full model directory. See
[weights/README.md](weights/README.md) for the exact checkpoint contents and
loading contract.

The validated output-head checkpoint is recorded in
[weights/README.md](weights/README.md). By default the reference experiment
loads `weights/trisplatpp_lgtm_step2700.ckpt`; set
`TRISPLATPP_CHECKPOINT` to the actual artifact path when the checkpoint is kept
outside the repository.

## Quick start

```bash
conda create -y -n trisplatpp python=3.10
conda activate trisplatpp
pip install torch torchvision torchaudio
pip install -r requirements.txt --no-build-isolation
bash scripts/env/rebuild_extensions.sh

export DA3_ROOT=/path/to/Depth-Anything-3
export DA3_CHECKPOINT="$DA3_ROOT/checkpoints/DA3-GIANT-1.1"
export DL3DV_ROOT=/path/to/dl3dv_torch_960/10K
export TRISPLATPP_CHECKPOINT=/path/to/render_step_002700.ckpt
```

For the provided DL3DV forward renderer, use the trained TriSplat++ checkpoint
and the DA3-GIANT weights together. The geometry input must stay at `224x448`
and the texture input at the native `540x960`; scenes with another native
resolution are skipped before the model is built:

```bash
/opt/conda/envs/trisplat/bin/python \
  scripts/render_6view_triptych.py \
  --data-root /path/to/dl3dv_torch_960/10K \
  --chunk /path/to/dl3dv_torch_960/10K/train/000000.torch \
  --scene-index 0 \
  --checkpoint weights/trisplatpp_lgtm_step2700.ckpt \
  --da3-checkpoint "$DA3_CHECKPOINT" \
  --out outputs/render_6view_triptych_scene0 \
  --device cuda:0
```

The renderer writes one six-row, three-column `GT / NORMAL / RENDER` sheet,
per-view PNGs, and `metadata.json` under the selected scene directory.

Run the reference configuration (the default launcher uses the normal
Lightning/Hydra entry point):

```bash
python -m src.main \
  +experiment=trisplat_dl3dv_tsdpt_lgtm10k_train \
  trainer.max_steps=10000 \
  wandb.mode=disabled
```

For a source-only smoke check that does not initialize CUDA or DA3 weights:

```bash
python -m py_compile \
  src/model/encoder/encoder_da3_tsdpt.py \
  src/model/head/triangle_texture_head.py \
  src/model/decoder/cuda_triangle_splatting.py \
  src/model/decoder/decoder_triangle_splatting_cuda.py
git diff --check
```

## Method

<p align="center">
  <img src="https://lhmd.top/trisplat/assets/figures/pipeline2.png" alt="TriSplat pipeline" width="100%">
</p>

Given sparse calibrated input views, TriSplat++ first runs DA3-GIANT at
`224x448`. DA3's camera tokens and DPT-style depth/feature fusion provide a
scene-consistent point map; TSDPT converts the map into triangle attributes,
and the calibrated context cameras replace the predicted intrinsics when
lifting points for DL3DV. `TSAdapter.from_point_map` then constructs local
triangle frames and the CUDA decoder renders RGB, depth, and normals.

The appearance branch keeps the original `540x960` context images. For every
geometry pixel, `project_triangle_texture` maps the corresponding world-space
triangle back into its source camera and samples a border-clamped `4x4` RGB
tile. `TriangleTextureHead` combines that tile with patchified native-image
features and emits a zero-initialized residual. The decoder samples the final
tile in barycentric triangle coordinates, preserving sharp local appearance
without modifying vertices, normals, scale, opacity, or sigma. Mesh export and
the existing geometry diagnostics therefore remain compatible with the base
TriSplat pipeline.

## Installation

Create the environment:

```bash
conda create -y -n trisplat python=3.10
conda activate trisplat
pip install --upgrade pip
```

Install PyTorch and Python dependencies:

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt --no-build-isolation
```

Build CUDA extensions:

```bash
bash scripts/env/rebuild_extensions.sh
```

Download initialization weights used by the model:

```bash
mkdir -p pretrained_weights
wget -O pretrained_weights/pi3.safetensors \
  https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors
wget -O pretrained_weights/omnidata_dpt_normal_v2.ckpt \
  'https://zenodo.org/records/10447888/files/omnidata_dpt_normal_v2.ckpt?download=1'
```

## Models

Download released TriSplat checkpoints from [lhmd/TriSplat](https://huggingface.co/lhmd/TriSplat):

```bash
mkdir -p checkpoints
wget -O checkpoints/re10k_trisplat.ckpt \
  https://huggingface.co/lhmd/TriSplat/resolve/main/re10k_trisplat.ckpt
wget -O checkpoints/dl3dv_trisplat.ckpt \
  https://huggingface.co/lhmd/TriSplat/resolve/main/dl3dv_trisplat.ckpt
```

## Datasets

Packed `.torch` datasets default to:

```text
data/re10k
data/dl3dv
```

You can also set:

```bash
export RE10K_ROOT="$PWD/data/re10k"
export DL3DV_ROOT="$PWD/data/dl3dv"
```

See [data/README.md](data/README.md) for dataset layout and conversion notes.

## Training

Train on RealEstate10K:

```bash
bash scripts/train/train_re10k.sh --gpus 0,1,2,3,4,5,6,7 --wandb-mode offline
```

Train on DL3DV:

```bash
bash scripts/train/train_dl3dv.sh --gpus 0,1,2,3,4,5,6,7 --wandb-mode offline
```

Extra arguments after `--` are passed to Hydra. Use `--ckpt` to resume or initialize from a checkpoint.

## Evaluation

Evaluate and render RealEstate10K meshes:

```bash
bash scripts/eval/eval_re10k_mesh.sh \
  --ckpt checkpoints/re10k_trisplat.ckpt \
  --data-root "$RE10K_ROOT"
```

Evaluate and render DL3DV meshes:

```bash
bash scripts/eval/eval_dl3dv_mesh.sh \
  --ckpt checkpoints/dl3dv_trisplat.ckpt \
  --data-root "$DL3DV_ROOT"
```

## Custom Image Inference

For raw custom images, use the plain torch inference script. It does not use the
Lightning `Trainer` or dataset `DataModule`; it loads images from a folder, runs
the pose-free encoder directly, and exports a direct triangle mesh plus predicted
camera poses:

```bash
python -m src.scripts.infer_custom_mesh \
  --image-dir /path/to/images \
  --ckpt checkpoints/re10k_trisplat.ckpt \
  --out-dir outputs/custom_mesh/demo \
  --num-views 2 \
  --force
```

By default the script uses the lightweight RE10K 2-view experiment and selects
views uniformly from the folder. Use `--view-indices 0,5` to choose frames
explicitly. If camera intrinsics are unavailable, the script creates a normalized
pinhole camera from a 60 degree horizontal FOV; override this with `--fov-deg` or
`--intrinsics-json`.

The main outputs are:

```text
outputs/custom_mesh/demo/mesh/DIRECT_triangle_mesh.ply
outputs/custom_mesh/demo/mesh/DIRECT_triangle_mesh_post.ply
outputs/custom_mesh/demo/predicted_c2w.json
outputs/custom_mesh/demo/inference_summary.json
```

Pass extra Hydra options with repeated `--override` flags, for example
`--override mesh.tsdf_gs2d.direct_post_process=false` to save only the raw direct
mesh.

## Simulation

TriSplat exports ordinary triangle meshes, so the output can be opened directly by common graphics and simulation tools. The evaluation scripts above write per-scene meshes under:

```text
outputs/<eval_root>/<run_name>/<scene>/mesh/DIRECT_triangle_mesh.ply
outputs/<eval_root>/<run_name>/<scene>/mesh/DIRECT_triangle_mesh.off
outputs/<eval_root>/<run_name>/<scene>/mesh/DIRECT_triangle_mesh_post.ply
outputs/<eval_root>/<run_name>/<scene>/mesh/DIRECT_triangle_mesh_post.off
```

The `_post` mesh is the default rendering and simulation output. Direct triangle
meshes use quantile geometry cleanup to remove non-finite, degenerate, very large,
or distant triangle outliers before compacting referenced vertices. TSDF meshes
still use connected-component cleanup. For example, after running
`scripts/eval/eval_re10k_mesh.sh`, use:

```bash
ls outputs/re10k_mesh_eval/re10k_mesh_eval/*/mesh/DIRECT_triangle_mesh_post.ply
```

The exported `_post.ply` mesh is vertex-colored and can be imported into [Blender](https://www.blender.org/), [Open3D](https://www.open3d.org/), [Isaac Sim](https://developer.nvidia.com/isaac/sim), [Unity](https://unity.com/), or [PyBullet](https://pybullet.org/) as a static triangle mesh. For simulation, use the `.ply` mesh for visual geometry and generate a collision mesh in your simulator if needed; for example, simplify or convex-decompose it before rigid-body simulation when the raw mesh is too dense.

## Citation

If you find this repository useful, please cite:

```bibtex
@article{wang2026trisplat,
  title={TriSplat: Simulation-Ready Feed-Forward 3D Scene Reconstruction},
  author={Wang, Weijie and Li, Zimu and Shi, Jinchuan and Zhang, Zeyu and Ye, Botao and Pollefeys, Marc and Chen, Donny Y. and Zhuang, Bohan},
  journal={arXiv preprint arXiv:2605.26115},
  year={2026}
}
```

## Acknowledgements

This codebase builds on open-source work including [YoNoSplat](https://github.com/justimyhxu/YoNoSplat), [MVSplat](https://github.com/donydchen/mvsplat), [pixelSplat](https://github.com/dcharatan/pixelsplat), [CroCo](https://github.com/naver/croco), [DINOv2](https://github.com/facebookresearch/dinov2), [Omnidata](https://github.com/EPFL-VILAB/omnidata), [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), and [Triangle Splatting](https://github.com/trianglesplatting/triangle-splatting).
