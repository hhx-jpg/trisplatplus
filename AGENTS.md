# TriSplat++ / DA3 + LGTM Agent Notes

本文档是 TriSplat++ 及相邻 DA3 仓库的快速上手说明。默认工作目录是
当前仓库，默认 Python 环境是 conda `trisplatpp`。

## 环境

```bash
conda activate trisplatpp
cd /path/to/TriSplatPlusPlus
export PYTHONPATH="$PWD/submodules/diff-triangle-rasterization:${PYTHONPATH:-}"
```

DA3 源码已随仓库放在 `third_party/depth-anything-3/`；也可以通过
`DA3_ROOT` 指向单独的 DA3 checkout（或其 `src` 子目录）。TriSplat++ 的
DA3 encoder 会在运行时把选中的目录加入 `sys.path`。模型权重仍需单独提供：

```text
DA3_ROOT=/path/to/Depth-Anything-3
DA3_CHECKPOINT=/path/to/Depth-Anything-3/checkpoints/DA3-GIANT-1.1
```

数据集由 `DL3DV_ROOT`、`RE10K_ROOT` 等环境变量指定，具体默认值见
`config/dataset/`。

## TriSplat 架构

- `src/main.py`：Hydra 入口，构造 `ModelWrapper`、数据模块、Lightning trainer
  和 checkpoint/callback。
- `src/config.py`：把 Hydra 配置解析成 encoder、decoder、loss 和数据集对象。
- `src/model/model_wrapper.py`：训练/验证主循环、loss、原语 mask、渲染和 mesh 导出。
- `src/model/encoder/`：从多视图图像预测几何和外观原语。
  - `encoder_trisplat.py` 是原生 Pi3/CroCo TriSplat encoder。
  - `encoder_da3_tsdpt.py` 是 DA3-TSDPT 桥接 encoder，只负责调用 DA3、做
    数据集相机尺度对齐，并把 DA3 输出转换为 TriSplat 的 `src/model/types.py`
    中的 `Triangles`。
- `src/model/decoder/`：可微 Gaussian/triangle splatting；TSDPT 训练使用
  `decoder_triangle_splatting_cuda.py`。
- `src/model/head/triangle_texture_head.py`：LGTM 风格的 context-image 投影
  纹理 tile 与零初始化 residual head；只改变 appearance，不改变几何。
- `src/model/types.py`：TriSplat 内部的 `Gaussians`、`Triangles` 原语类型。
- `src/mesh/`：直接三角 mesh、TSDF mesh 和后处理。
- `config/experiment/`：完整实验组合；TSDPT 实验名以 `tsdpt` 开头。

## DA3-TSDPT 数据流

```text
Hydra da3_tsdpt config
  -> EncoderDA3TSDPT
  -> DepthAnything3 / DepthAnything3Net
  -> DINOv2 backbone
  -> depth head + camera encoder/decoder
  -> TSDPT (GS-DPT 参数 + opacity/sigma 两个额外通道)
  -> TSAdapter.from_point_map
  -> TriSplat Triangles
  -> CUDA triangle decoder
  -> RGB/depth/normal loss and mesh export
```

DA3 中 `TSDPT` 是 `GSDPT` 的子类。它保留 checkpoint 兼容的 `raw_gs` 和
`raw_gs_conf` 输出，并额外输出 `raw_gs_opacity`、`raw_gs_sigma`。主分支的
`output_dim` 仍是 38；额外的 opacity/sigma 使用独立的 `extra_output_conv`，
不会改变原有 GS-DPT 权重布局。

TSDPT 对应的几何 adapter 是：

```text
/root/data/haoxuan/Depth-Anything-3/src/depth_anything_3/model/ts_adapter.py
```

其中：

- `TSAdapter`：把深度、相机和 raw 参数转换为世界坐标三角形。
- `TSAdapter.from_point_map`：TSDPT 训练使用的 DA3 点图中心差分路径，法线为
  `normalize(cross(dx, dy))`，并生成边界有效 mask。
- `TSTriangles`：DA3 侧的原语容器；TriSplat encoder 在边界处转换成自己的
  `Triangles` dataclass，避免 DA3 反向依赖 TriSplat。

旧的
`src/model/encoder/common/triangle_adapter.py` 是原生 TriSplat encoder 的
通用兼容 adapter，不是 DA3 TSDPT 的 canonical adapter。修改 TSDPT 几何时应
优先改 DA3 的 `model/ts_adapter.py`。

## 关键配置和权重约定

- DA3 TSDPT head 配置：`Depth-Anything-3/src/depth_anything_3/configs/da3-giant.yaml`。
- TriSplat bridge 配置：`config/model/encoder/da3_tsdpt.yaml`。
- 推荐 smoke 实验：`config/experiment/trisplat_dl3dv_tsdpt_5k_directdiff.yaml`。
- DA3 的 backbone、depth、camera 路径默认冻结；训练参数由
  `EncoderDA3TSDPT` 的 `requires_grad` 设置决定，修改冻结策略后要同步更新
  注释和训练文档。
- DA3 extrinsics 是 w2c；adapter 使用 c2w。输入 context pose 已知时，用
  camera center 拟合完整 Sim(3) 对齐 DA3 世界点（旋转、缩放、平移），再
  转入 GT context/target camera frame。不要只缩放 camera-space 点后替换 GT pose。
- 像素坐标采用像素中心 `(x + 0.5) / W`、`(y + 0.5) / H`。
- DL3DV context 的归一化 GT 内参会在 `EncoderDA3TSDPT` 的点图反投影处转成
  像素内参使用；DA3 预测内参仍保留在诊断 dump 中，但不覆盖已知标定。
- direct-difference 的三角形尺度使用 `0.5~18.0`。CUDA rasterizer 会丢弃
  投影内切半径小于 1px 的面片，`0.75~1.25` 在 `224x448` 输入下会让几乎
  所有面片的 `radii` 为 0，训练无法得到有效 RGB 梯度。
- 当前 directdiff 5k 配置固定 sigma 映射（`1.0 -> 1.0`，warmup 1），只把
  opacity temperature 从 `1.0 -> 5.0` 在 5000 步内升高。TSDPT 的 opacity
  和 sigma 现在是独立随机初始化的输出分支；优化器用 `opacity_lr=1e-4`
  和 `tsdpt_lr=1e-6`，全局 `optimizer.lr=2e-4` 保持不变。

## 常用命令

只做静态检查：

```bash
conda activate trisplat
python -m py_compile \
  src/model/encoder/encoder_da3_tsdpt.py \
  third_party/depth-anything-3/src/depth_anything_3/model/tsdpt.py \
  third_party/depth-anything-3/src/depth_anything_3/model/ts_adapter.py
git diff --check
```

短训入口：

```bash
bash scripts/train/train_tsdpt_da3.sh --gpus 0 --steps 1000
```

运行前确认 `DL3DV_ROOT`、CUDA triangle rasterizer 和 DA3 checkpoint 存在。
完整训练/评估命令见 `README.md`、`docs/TSDPT_DDT.md` 和 `scripts/`。

## 修改原则

1. 先阅读相邻模块和配置，再修改接口；不要把 TriSplat 类型导入 DA3。
2. DA3 侧的纯模型/几何逻辑放在 `Depth-Anything-3/src/depth_anything_3/model/`。
3. TriSplat 侧只保留训练框架桥接、数据集相机对齐和本地原语类型转换。
4. 保留用户已有的未提交修改；提交前分别在两个仓库检查 `git status`。
5. 不能运行完整 CUDA 训练时，至少执行 `py_compile`、`git diff --check`，并用
   小张量验证 adapter 的形状、有限性和边界 mask。
