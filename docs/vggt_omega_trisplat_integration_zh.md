# TriSplat 集成 VGGT-Omega（Feature-only 首版）

## 1. 目标与许可边界

本实现为 TriSplat 增加独立的 `vggt_omega` backbone 路径，用于研究更长的 context-view 序列。首版只复用官方 VGGT-Omega `Aggregator` 的多视图 token；TriSplat 原有 point、primitive、camera heads、`TriangleAdapter`、CUDA renderer 和 mesh exporter 保持不变。

VGGT-Omega 源码仍位于外部仓库 `/home/v-hanhaoxuan/vggt-omega`，固定于 commit `39a0cb8af88554f15ddcb5354cd52bde588fa014`，不复制进 TriSplat。该项目使用 FAIR Noncommercial Research License；本路径只适用于其许可覆盖的非商业研究。

## 2. Token contract

| 项目 | local_global | VGGT-Omega |
|---|---:|---:|
| patch size | 14 | 16 |
| TriSplat 可见 prefix | 5 | 17（1 camera + 16 register） |
| 输出宽度 | 2048 | 2048 |
| 输出 | 最后两个 local/global block 拼接 | 同层 frame/inter-frame 特征拼接 |

Omega adapter 输出：

```text
images:         [B,V,3,H,W], RGB [0,1]
tokens:         [B*V,17+(H/16)*(W/16),2048]
positions:      [B*V,17+(H/16)*(W/16),2]
patch_start:    17
intrinsic_pred: None
```

`H`、`W` 必须能被 16 整除，否则官方卷积 patch embed 会静默丢弃右/下边缘，adapter 会主动报错。

## 3. 数据流和双 RoPE 边界

输入不经过 TriSplat adapter 额外的 ImageNet normalization，因为官方 `Aggregator` 内部已经执行一次。Omega Aggregator 使用其原生二维 RoPE；输出 token 进入 TriSplat 三个 Transformer heads 后，继续使用 TriSplat 的 `RoPE2D(freq=100)` 与 integer `PositionGetter`。两个 RoPE 实现及其接口不互换。

Omega 对首帧使用独立的 camera/register token 模板，后续帧共享另一模板，因此 context view 顺序应保持稳定。

## 4. Patch 16 与全分辨率 primitive grid

Omega 实验配置固定：

```yaml
gaussians_per_axis: 16
upscale_token_ratio: 2
```

patch token 先从 16 网格上采样到 8 网格，每 token 再输出 `8×8` 空间结果，因此 224×224 和 224×448 的最终 primitive grid 与输入分辨率一致。若仍使用 14，会降为 196×196 / 196×392。

从 released TriSplat checkpoint warm-start 时，loader 仅加载非 `backbone.*` heads，并显式转换：

- point head：`3×7×7 → 3×8×8`
- triangle/primitive head：`C×7×7 → C×8×8`
- `rgb_embed` kernel：`7×7 → 8×8`

转换使用 bicubic 二维插值，其他 shape-compatible head 权重直接加载。

## 5. 两源 checkpoint

Omega 配置默认使用：

```text
pretrained_weights/vggt_omega_1b_512.pt
SHA-256 c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934
```

加载顺序：

1. 可选 `trisplat_head_checkpoint_path`：只加载非 backbone 的 TriSplat heads，并做 7→8 转换。
2. `checkpoint_path`：只提取 `aggregator.*`，strip prefix 后对官方 `Aggregator` strict-load。
3. 正常训练产生的新架构 checkpoint 才通过 `checkpointing.load` resume；旧 TriSplat checkpoint 和 Omega raw `.pt` 不充当 trainer resume。

loader 会记录 matched、converted、excluded、missing、unexpected 和 shape mismatch；Omega 未匹配或 strict-load 不完整时立即失败。

## 6. Feature-only 首版

当前相机仍由 TriSplat camera branch 预测，并归一到首帧坐标；geometry 仍消费数据集 normalized intrinsics。Omega adapter 返回 `intrinsic_pred=None`，所以 Omega experiments 不启用 `intrinsic` loss。默认冻结完整 Aggregator，只训练 TriSplat heads。

## 7. 复杂度与真实边界

Omega 并非 streaming：24 个外层 block 中有 19 个对全部 `view × token` 做 global inter-frame attention。TriSplat primitive 数还会随 view 数线性增长：224×448、50 views 会产生约 502 万个 triangles。因此 backbone 成功并不代表 normal refinement、CUDA render 或 mesh export 同样可扩展。

backbone-only 基准：

```bash
python scripts/benchmark_vggt_omega.py \
  --checkpoint pretrained_weights/vggt_omega_1b_512.pt \
  --checkpoint-sha256 c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934 \
  --height 224 --width 448 --views 1,2,6,12,24,50
```

脚本记录耗时、peak allocated/reserved memory、dtype、token shape、finite 状态和 OOM 上限。

## 8. 运行

先安装固定外部依赖：

```bash
/home/v-hanhaoxuan/miniconda3/envs/trisplat/bin/pip install -e /home/v-hanhaoxuan/vggt-omega
```

组合配置：

```bash
python -m src.main +experiment=trisplat_re10k_vggt_omega_triangle_224x224
python -m src.main +experiment=trisplat_dl3dv_vggt_omega_triangle_224x448
```

如需从 released TriSplat heads warm-start：

```bash
python -m src.main +experiment=trisplat_re10k_vggt_omega_triangle_224x224 \
  model.encoder.backbone.trisplat_head_checkpoint_path=checkpoints/re10k_trisplat.ckpt
```

## 9. 第二阶段

后续可先旁路运行 Omega CameraHead，再评估 DenseHead depth：

- Omega pose 是 `[t, quat_xyzw, fov_h, fov_w]`，解码为 OpenCV world-to-camera，不能直接替代 TriSplat C2W。
- Omega pixel K 需转换为 normalized K。
- Omega depth 是 camera-Z；TriSplat TriangleAdapter 使用 unit-ray distance，需做 `ray_distance = z / ray_z`。
- `depth_conf = 1 + exp(logit)` 不是 opacity。

只有通过 pose round-trip、重投影、depth convention 和显存 gate 后，才进入 Camera/Dense 替换阶段。
