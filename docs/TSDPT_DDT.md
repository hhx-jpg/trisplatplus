# TSDPT + DA3 + Triangle Splatting DDT

## 目标

以 Depth Anything 3 GIANT 主干提取多视图场景特征，使用新的 `TSDPT` 输出头预测 TriSplat 所需的三角形属性，并通过 TriSplat 的可微 triangle splatting 管线训练 DL3DV 渲染质量。第一阶段只训练 TSDPT，DA3 主干、相机解码器和已有的尺度/SH 输出网络全部冻结。

## 固定配置

- 权重：`/root/data/haoxuan/Depth-Anything-3/checkpoints/DA3-GIANT-1.1`。
- 数据：`/root/data/lhmd/dl3dv_torch_960`，由 `DL3DV_ROOT` 指向该目录。
- context views：12（训练和验证均保持 12）。
- 三角形渲染：TriSplat `triangle_splatting_cuda` / `diff-triangle-rasterization`。
- TSDPT 的几何适配器位于 DA3 `src/depth_anything_3/model/ts_adapter.py`；
  TriSplat 只在 `src/model/encoder/encoder_da3_tsdpt.py` 做框架桥接。
- 不预测 `xyoffset`：`pred_offset_xy=false`，像素中心坐标直接由 DA3 相机内参反投影。
- 不使用 TriSplat normal refinement：关闭 `geometry_normal.refiner.enabled`，法线只由点云相邻像素差分得到。
- TSDPT 在 GSDPT 的 `scale/rotation/SH` 输出基础上新增两个标量通道：`opacity` 和 `sigma`；当前 bridge 对二者都使用有界 sigmoid 映射，并保留正值下限。
- `TSAdapter.from_point_map` 只用 DA3 点图差分的方向构造局部切平面：`normal = normalize(dx × dy)`，`tangent` 为 `dx` 在该平面上的投影。三维 scale 仍按原生 TriSplat 的深度乘像素角尺寸映射；不能把 `||dx||/||dy||` 直接当作尺寸，否则深度边界会生成跨近平面的巨面片。

## 坐标与尺度约定

DA3 深度在 DA3 相机坐标系中定义。构造三角形时必须使用 DA3 预测的 `extrinsics/intrinsics`，先将深度和相机平移按同一个 Umeyama/GT pose scale 对齐，再调用 DA3 `TSAdapter.from_point_map`。禁止直接混用数据集 GT 相机尺度。内参归一化使用当前输入 `(H, W)`，像素中心为 `((x+0.5)/W, (y+0.5)/H)`。

## 差分法线

将局部点图 reshape 为 `[B,V,H,W,3]`，用中心差分 `dx=P[x+1]-P[x-1]`、`dy=P[y+1]-P[y-1]`，法线为 `normalize(cross(dx,dy))`；无效点和边界不参与三角形旋转锚定。根据相机视线翻转法线，使其朝向相机。不要引入学习式 normal refiner 或额外单目法线 teacher。

## 权重加载与冻结

从 DA3-GIANT-1.1 加载 backbone、camera、原有 DPT/GS head 的匹配参数。新 TSDPT 的 opacity/sigma 行随机初始化（建议零 bias、稳定正值初始化），允许旧输出行按名称/前缀部分复制。启动后打印 missing/unexpected keys，并断言只有 TSDPT 参数 `requires_grad=True`。

## 训练验收

建议 `precision=32-true`、batch size 1、输入 224x448、AdamW `lr=2e-4`、warmup 1000 步；先运行 1000 步 smoke/短训。记录验证 PSNR、RGB loss、opacity/sigma 均值和有效三角形比例。目标是 1k steps 时 PSNR > 16 dB；未达到时优先检查 context=12、相机/triangle scale 一致、sigma 为正、xyoffset/refiner 关闭、主干冻结且 TSDPT 有梯度。

## 运行

```bash
export DL3DV_ROOT=/root/data/lhmd/dl3dv_torch_960
bash scripts/train/train_tsdpt_da3.sh --gpus 0 --steps 1000
```

训练日志写入 `outputs/tsdpt_da3_*/train.log`，tmux 会话名为 `tsdpt-ddt`。每次纠错后从最新可用 checkpoint 重新启动，并在日志中保留 PSNR 对比。
