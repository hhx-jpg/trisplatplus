# TriSplat 项目架构说明

本文档面向当前代码库中实际使用的 TriSplat 三角形重建主线，结合论文《TriSplat: Simulation-Ready Feed-Forward 3D Scene Reconstruction》的方法设计说明项目结构。文档重点解释从稀疏无位姿图像到可直接用于仿真和渲染的三角网格的完整链路，不展开未参与当前 TriSplat 主线的历史分支和旁路实验模块。

## 1. 项目目标与论文核心思想

TriSplat 的目标是从少量输入图像中，以一次前向推理的方式重建一个可直接导出的三角网格场景。论文强调的关键区别是：传统 feed-forward splatting 方法大多输出 Gaussian primitives，虽然适合新视角合成，但 Gaussian 本身不是显式曲面；如果要得到 mesh，通常还要经过 TSDF fusion、marching cubes 或其他后处理。TriSplat 则直接把场景表示为 oriented triangle primitives，训练和渲染使用的 primitive 本身就是三角面片，因此 mesh export 不再是额外重建问题，而是对预测出的三角形做筛选、朝向修正、顶点合并和文件写出。

当前项目主线可以概括为：配置系统选择 TriSplat triangle experiment，数据模块提供 context/target 多视角样本，EncoderTrisplat 从 context 图像预测 point maps、camera poses、intrinsics 和 triangle attributes，triangle splatting decoder 负责可微渲染，训练阶段用图像、位姿和法线相关损失监督，测试或自定义推理阶段通过 direct mesh export 直接写出 PLY/OFF 三角网格。

## 2. 配置和运行入口

项目使用 Hydra 组织实验。主配置 `config/main.yaml` 默认挂载 encoder、decoder、loss 和 mesh 配置；具体实验配置位于 `config/experiment/`，例如 RE10K 和 DL3DV 的 triangle refiner 实验会覆盖 encoder 为 `trisplat`、decoder 为 `triangle_splatting_cuda`、mesh 为 direct export，并打开 `use_triangle`、pose-free 和 geometry-normal anchoring。

训练和官方测试的 Python 入口是 `src/main.py`。它负责读取 Hydra 配置，构造数据模块、编码器、解码器、损失函数、mesh exporter 和 Lightning `ModelWrapper`，然后进入训练或测试流程。项目根目录下的 shell 脚本只是对这个入口的实验封装：`scripts/train/train_re10k.sh` 和 `scripts/train/train_dl3dv.sh` 对应论文中的 RE10K/DL3DV 训练；`scripts/eval/eval_re10k_mesh.sh` 和 `scripts/eval/eval_dl3dv_mesh.sh` 对应 mesh 导出与评估流程，并会把 `test.export_mesh` 与 `mesh.tsdf_gs2d.export_mode=direct` 这类关键开关传给 Hydra。

自定义图片推理入口是 `src/scripts/infer_custom_mesh.py`。它绕过 Lightning DataModule，直接从一个图片文件夹构造最小 batch，加载 TriSplat checkpoint，运行 pose-free encoder，得到 predicted camera poses 和 triangle primitives，然后调用 direct mesh exporter 写出网格。这条路径是项目中最接近“用户拿一组图片直接生成 mesh”的使用方式。

## 3. 数据流：context views 与 target views

训练和评估数据由 `src/dataset/` 体系提供。RE10K、DL3DV 等数据被组织成 packed `.torch` chunks，dataset 读取每个 scene 的图像、相机内外参和近远裁剪范围。view sampler 根据实验配置选择 context views 和 target views：context views 是模型输入，target views 用于渲染监督和指标评估。

进入模型之前，batch 的基本结构围绕 `context` 和 `target` 两部分展开。context 包含输入图像、相机内参、相机外参、near/far 和帧索引；target 包含待渲染视角的同类信息以及真实图像。Encoder 自身提供 data shim，对图像做归一化，使输入适配 DINOv2 backbone。训练时，ModelWrapper 还可能给 context 增加 monocular normal teacher，用于论文中的 normal bootstrap 和 normal teacher loss。

自定义推理脚本会模拟同样的 context batch 结构。若没有真实内参，它会根据 FOV 生成归一化 pinhole intrinsics；若没有真实位姿，encoder 会以 pose-free 模式预测相机位姿。

## 4. 总体模型架构

当前主线模型由四个连续阶段构成。

第一阶段是图像特征提取和跨视角融合。`src/model/encoder/backbone/backbone_local_global.py` 使用 DINOv2 ViT-L/14 提取 patch tokens，再经过 local/global attention decoder 交替处理单视角局部信息和跨视角全局对应关系。这个模块对应论文图 2 中的 DINOv2 和 Local-Global Attention。

第二阶段是 `src/model/encoder/encoder_trisplat.py` 中的三个预测头：pointmap head、primitive head 和 camera head。pointmap head 预测每个像素的局部 3D point map；primitive head 预测 density、scale、quaternion、SH appearance 和 blur/sigma；camera head 在 pose-free 设置下预测每个输入视角的 camera-to-world pose，并把所有位姿归一到第一视角坐标系。

第三阶段是 triangle primitive 构造。Encoder 会先用 point map 推导 geometry normal，并把 normal 转成稳定的局部 tangent frame，再用这个 frame 覆盖 primitive head 预测的旋转。随后 `src/model/encoder/common/triangle_adapter.py` 根据 canonical triangle、scale、rotation、depth、intrinsics 和 extrinsics 生成世界坐标系下的三角形顶点，形成公共 primitive 类型 `src/model/types.py` 中的 `Triangles`。

第四阶段分成两条用途：训练和新视角合成使用 `src/model/decoder/decoder_triangle_splatting_cuda.py` 与 `src/model/decoder/cuda_triangle_splatting.py` 调用 CUDA triangle rasterizer 可微渲染 RGB、depth、opacity 和 normal；mesh 输出使用 `src/mesh/tsdf_gs2d.py` 的 direct export 分支，把 `Triangles` 直接转成普通 mesh 文件。

## 5. Backbone：DINOv2 与 Local-Global Attention

Backbone 负责从多张输入图像中提取既有单图语义又有跨图几何对应的 token 表示。它先将每个 view 独立送入 DINOv2 ViT-L/14，得到 patch tokens；再加入 register tokens 和位置编码，进入自定义 decoder。

Local/global 机制体现在 token 排布的交替切换：一层按每个 view 独立处理，强调局部空间结构；下一层把同一 batch 内所有 view 的 tokens 拼到一起处理，强调跨视角匹配和几何一致性。论文中说 decoder blocks alternate between intra-view self-attention and cross-view joint attention，代码中的这个 backbone 就是对应实现。

Backbone 还可以预测 intrinsics，并使用 intrinsic embedding 对 tokens 进行相机条件化。这对应论文中“learnable intrinsic token / optional intrinsics”的设计目标：在 pose-free 和稀疏视角条件下，相机参数本身也是模型需要处理的不确定因素。

## 6. EncoderTrisplat：point map、primitive 和 camera 三条分支

`EncoderTrisplat` 是项目中最核心的模型文件。它接收 context images 和 intrinsics，输出当前主线所需的 `Triangles` primitives。

Pointmap 分支预测每个像素的局部 3D 点。网络输出两个横向坐标和一个 log-depth，depth 通过指数映射保证为正，再把横向坐标乘以 depth 得到局部相机坐标下的 3D point map。这个 point map 在项目中有双重作用：一方面提供每个 primitive 的深度和中心位置，另一方面为论文核心的 triangle orientation anchoring 提供几何法线来源。

Primitive 分支在代码里仍保留了一些历史命名，但在 `use_triangle: true` 的实验中，它预测的是 triangle attributes。每个像素对应一个 triangle primitive，属性包括 density、三维 scale logits、四元数、SH 颜色系数和 blur/sigma。输入 RGB 会通过零初始化 patch embedding 加到 primitive 分支特征里，让外观分支可以直接访问图像颜色信息，这对应论文 Appendix A 对 primitive head 的描述。

Camera 分支预测每个输入视角的 pose。由于论文强调 sparse unposed images，当前实验默认 `pose_free: true`。训练时使用 scheduled sampling，在早期更频繁使用 ground-truth pose 稳定学习，随后逐渐转向 predicted pose，减少训练和测试之间的分布差异。测试和自定义推理时，位姿完全来自 camera head。

## 7. 论文核心一：Anchoring Triangle Orientation to Geometry

Triangle orientation anchoring 是 TriSplat 区别于直接回归三角形姿态的关键设计，对应论文 Sec. 3.2。主要实现集中在 `EncoderTrisplat._build_triangle_geometry_rotation`。

首先，encoder 从 pointmap 分支得到密集局部 3D 点图，对横向和纵向邻域做有限差分，并用叉乘得到 raw geometry normal。由于边界像素和退化差分不可靠，代码会构造 validity mask，屏蔽图像边缘和非有限、近零范数的法线。随后法线会根据相机观察方向翻转，保证朝向一致。

其次，项目对 geometry normal 做平滑，但不是无条件平均。平滑时只聚合与中心法线方向一致的邻居，避免跨越深度断层把不同平面的 normal 混在一起。这对应论文中 orientation-aware box filter 的思想：大平面内部更平滑，边界处尽量保留不连续性。

第三，normal refinement head 对 geometry normal 做学习型修正。它是轻量 CNN/U-Net，输入包括 raw normal、smoothed/base normal、RGB、depth 和 valid mask，输出 residual normal。输出层在 residual 模式下零初始化，因此训练开始时不会随机扰动几何法线，而是从 identity-like 行为逐渐学习修正。这一点与论文中强调的 zero-initialization 稳定性一致。

第四，训练早期可以使用 monocular normal teacher 做 bootstrap。ModelWrapper 根据 normal bootstrap schedule 计算 teacher blend alpha，并通过 `src/model/encoder/mono_estimator/mono_normal.py` 中的 Omnidata normal estimator 生成 context normal。Encoder 收到 teacher normal 后，在有效区域内按时间系数把 teacher normal 与模型 refined normal 混合。这个机制不是单纯额外 loss，而是直接进入 forward representation，影响三角形构造和后续渲染梯度，正是论文中 mono-normal bootstrap 的含义。

最后，encoder 把 forward normal 转成完整 tangent frame。它将 point map 的横向差分投影到法线垂直平面得到 tangent，再通过叉乘得到 bitangent，并重新正交化，形成 `[tangent, bitangent, normal]` 旋转矩阵。旋转矩阵被转换为 quaternion，并在有效像素处覆盖 primitive head 原本预测的 quaternion；无效像素则保留网络预测的 quaternion 作为 fallback。这个覆盖动作就是代码层面真正的“anchor”。

需要注意的是，decoder 中也有从 rendered depth 估计 normal 的辅助函数，但它用于渲染输出、可视化或 normal 相关监督，不是论文中 triangle orientation anchoring 的主实现。真正决定三角形朝向的是 encoder 中 point map normal 到 quaternion 的链路。

## 8. TriangleAdapter：从 raw attributes 到世界空间三角形

`src/model/encoder/common/triangle_adapter.py` 负责把 encoder 准备好的 raw triangle attributes 实例化为世界坐标中的三角形。它不负责推导 geometry normal，而是使用上游已经 anchor 过的 quaternion。

每个 primitive 从一个 canonical equilateral triangle template 出发。Scale logits 先经 sigmoid 映射到配置给定的 `[triangle_scale_min, triangle_scale_max]` 区间，再结合预测 depth 和 intrinsics 推导出的 pixel footprint 转成世界尺度。这样同样的图像像素覆盖会根据深度自动变成合理的世界尺寸。

三角形中心由像素坐标、相机内外参和预测深度确定：每个像素对应一条 world ray，ray origin 加上 ray direction 乘 depth 得到 primitive center。随后 canonical triangle 先按 scale 拉伸，再按 quaternion 在局部相机坐标中旋转，再乘 camera-to-world rotation，最后平移到 center。这个过程对应论文 Eq. (2) 中的三角形顶点公式。

Adapter 还处理 progressive sharpening 中的 blur/sigma。Raw sigma 经过 sigmoid 后乘以随训练步数变化的 scale，从训练早期较软的 triangle footprint 逐步过渡到后期更清晰的 surface element。输出的 `Triangles` 包含 vertices、opacity、sigma、appearance features、centers 和 scales，供 decoder 和 mesh exporter 使用。

## 9. Triangle Splatting Decoder：可微渲染路径

训练和新视角合成通过 triangle splatting decoder 完成。`DecoderTriangleSplattingCUDA` 先根据训练步数对 opacity 做 temperature sharpening，并处理 alpha floor 等稳定性设置，然后调用 `cuda_triangle_splatting.py` 中的 rasterizer wrapper。底层可微 CUDA rasterizer 来自 `submodules/diff-triangle-rasterization/`，Python 侧通过 `diff_triangle_rasterization` 暴露 `TriangleRasterizer`，真正的投影、tile sorting、alpha compositing 和反向传播在扩展中完成。

Rasterizer 对每个 target view 根据 predicted triangles、target camera extrinsics/intrinsics 和 image shape 进行投影、排序和 alpha compositing，输出 RGB、depth、opacity、rendered normal、surface normal，以及可选的 triangle visibility mask。RGB 输出用于 photometric loss 和 PSNR/SSIM/LPIPS 指标；depth 和 normal 输出用于可视化、辅助监督或 mesh export 的可选过滤逻辑。

论文 Sec. 3.3 的 progressive surface sharpening 在项目中分成两处体现：sigma/blur 的 schedule 在 TriangleAdapter 中完成，opacity temperature schedule 在 decoder 中完成。前者控制三角形边缘软硬，后者控制 primitive 透明度从连续分布逐渐变得更接近二值 surface selection。二者共同解决论文所说的早期训练梯度覆盖问题。

## 10. 训练组织与损失

训练由 `ModelWrapper.training_step` 组织。每个 batch 先经过 data shim 和 `_prepare_encoder_context`。如果当前实验启用了 normal teacher 或 normal bootstrap，ModelWrapper 会在这里准备 monocular teacher normal 和 teacher blend alpha。随后 encoder 输出 triangle primitives，decoder 从 target cameras 渲染预测图像和几何输出，最后依次计算配置中的损失。

当前 triangle 实验的主损失包括图像重建损失、LPIPS 感知损失、pose loss、intrinsic loss 和 normal teacher loss。图像损失对应论文中的 photometric term；pose loss 使用所有视角对之间的相对位姿误差，对全局坐标 gauge 更鲁棒；normal teacher loss 将 refined/model normal 与 monocular teacher normal 对齐，是论文中 normal supervision 的实现之一。

训练代码还包含若干稳定性机制。大 loss 样本会在 warm-up 后被过滤或降权，避免坏位姿或异常渲染破坏训练；primitive 和 decoder 输出会做 finite 检查；normal bootstrap schedule 会在早期更多依赖 teacher normal，随后释放给模型自己的 geometry normal。整体训练目标与论文 Sec. 3.4 的 `L_photo + L_cam + L_normal` 对应，只是工程实现中拆分为多个可配置 loss 模块。

## 11. Direct Mesh Export：论文核心二

Direct mesh export 是 TriSplat “simulation-ready”的落地点，主要在 `src/mesh/tsdf_gs2d.py`。Mesh exporter 通过 `src/mesh/__init__.py` 注册，接口定义在 `src/mesh/mesh_exporter.py`，但当前 TriSplat triangle 主线实际落到 `TsdfGs2d` 的 direct 分支。虽然文件名包含 tsdf，但当前主线配置中 `mesh.tsdf_gs2d.export_mode` 为 `direct`，也就是直接使用 predicted triangles 写 mesh，而不是先渲染 depth 再做 TSDF fusion。

Direct export 的输入必须是 `Triangles`。Exporter 首先取出 batch 0 的 triangle vertices、opacity、appearance features 和 normals。然后根据 direct opacity threshold 过滤低透明度三角形；如果启用更多工程过滤，还可以结合 visibility、camera、projected footprint、depth consistency 或 triangle budget 去掉异常面片。这些过滤是为了提升导出 mesh 的紧凑性和鲁棒性，但核心并不改变：mesh faces 来自 encoder 已经预测出的三角形。

过滤之后，exporter 会计算每个 face 的几何 normal，并与 encoder 输出的 primitive normal 比较。如果方向相反，就交换三角形顶点顺序来修正 winding order。颜色来自 SH0 appearance 系数，转换成 vertex color。顶点合并通过量化位置哈希完成，精度约为 `1e-5`，并把 normal octant 纳入 key，避免把方向相反的面错误合并。最后 exporter 打包二进制 PLY，并可选写 OFF 和 post-processed mesh。

这条链路直接对应论文 Sec. 3.4 的 mesh extraction 描述：低 opacity triangles 被丢弃，winding 被修正，近邻重复顶点被合并，输出是标准 triangle mesh。与 Gaussian baseline 需要 TSDF fusion 的路径相比，这里没有从体表示重新提取曲面的过程，因此导出速度和表示一致性是论文强调的优势。

## 12. 测试、评估和自定义推理

官方 test 流程仍由 `ModelWrapper.test_step` 组织。Encoder 先预测 primitives 和 poses，decoder 按 target views 分块渲染，项目计算 PSNR、SSIM、LPIPS 等新视角合成指标。如果 `test.export_mesh` 打开，则调用 mesh exporter。在 direct 模式下，mesh export 不需要额外渲染 context depth；只要有 `Triangles` primitives 就可以写出 mesh。

RE10K、DL3DV 和 ScanNet 相关实验通过 `config/experiment/`、`config/evaluation/` 以及 `src/scripts/export_*_mesh_eval.py` 组织。官方 mesh 评估脚本会先导出 direct mesh，再使用 `scripts/eval/render_mesh_open3d.py` 从数据集相机视角渲染导出的 mesh，计算 mesh-render PSNR、SSIM 和 LPIPS；几何质量则通过离线 mesh metrics 和 ground-truth point cloud/mesh 比较。

自定义推理脚本面向没有 packed dataset 的普通图片文件夹。它选择若干输入图像，构造 context batch，运行 encoder，保存 predicted camera poses、summary 和 direct mesh。默认输出路径中最重要的是 `mesh/DIRECT_triangle_mesh.ply` 与 `mesh/DIRECT_triangle_mesh_post.ply`，后者经过 post-process，更适合作为可视化和仿真导入结果。

## 13. 当前主线关键文件索引

| 模块 | 关键文件 | 作用 |
|---|---|---|
| 配置入口 | `config/main.yaml`、`config/experiment/*triangle*` | 选择 TriSplat triangle encoder、triangle decoder、loss 和 direct mesh export |
| 训练/测试入口 | `src/main.py`、`scripts/train/train_re10k.sh`、`scripts/train/train_dl3dv.sh`、`scripts/eval/eval_re10k_mesh.sh`、`scripts/eval/eval_dl3dv_mesh.sh` | 构造模型、数据、loss、mesh exporter 并启动 train/test；shell 脚本封装论文实验配置 |
| 训练/测试编排 | `src/model/model_wrapper.py` | 组织 batch、normal teacher、encoder、decoder、loss、metric 和 mesh export |
| 数据模块 | `src/dataset/data_module.py`、`src/dataset/dataset_re10k.py` | 读取 packed scenes，采样 context/target views，形成统一 batch |
| Backbone | `src/model/encoder/backbone/backbone_local_global.py` | DINOv2 特征、intrinsics 条件化、local/global attention 融合 |
| 主编码器 | `src/model/encoder/encoder_trisplat.py` | point map、primitive attributes、pose prediction、orientation anchoring |
| 三角形构造 | `src/model/encoder/common/triangle_adapter.py` | canonical triangle 到 world-space vertices 的转换 |
| Primitive 类型 | `src/model/types.py` | 定义 `Triangles` 在 encoder、decoder、exporter 之间传递的字段 |
| 可微渲染 | `src/model/decoder/decoder_triangle_splatting_cuda.py`、`src/model/decoder/cuda_triangle_splatting.py`、`submodules/diff-triangle-rasterization/` | 调用 CUDA triangle rasterizer，输出 RGB/depth/opacity/normal |
| Mesh 导出 | `src/mesh/__init__.py`、`src/mesh/mesh_exporter.py`、`src/mesh/tsdf_gs2d.py` | 注册 mesh exporter，执行 direct triangle mesh export、winding 修正、顶点合并、PLY/OFF 写出 |
| 自定义推理 | `src/scripts/infer_custom_mesh.py` | 从普通图片文件夹运行 pose-free TriSplat 并导出 mesh |
| Mono normal teacher | `src/model/encoder/mono_estimator/mono_normal.py` | 为 early orientation bootstrap 和 normal teacher loss 提供教师法线 |
| 损失 | `src/loss/` 中的 mse、lpips、pose、intrinsic、normal_teacher 等 | 对应 photometric、camera、normal 监督 |

## 14. 端到端执行链路总结

训练时，项目从数据集中采样 sparse context views 和 target views。Context images 进入 DINOv2 + local/global backbone，encoder 的 pointmap head 预测局部 3D 点，primitive head 预测 triangle attributes，camera head 预测 pose。Point map 通过有限差分、normal refiner 和 mono-normal bootstrap 产生稳定法线，再转成 tangent frame 和 quaternion 覆盖 primitive rotation。TriangleAdapter 根据这些属性生成世界空间 triangles。Decoder 将 triangles 渲染到 target views，训练损失约束图像重建、位姿和法线。

测试或推理时，同一 encoder 直接产生 `Triangles`。如果目的是新视角合成，decoder 把 triangles 渲染成目标视角图像；如果目的是仿真可用 mesh，direct exporter 不再做隐式曲面提取，而是直接筛选这些 triangles、修正朝向、合并顶点并写出标准网格文件。

因此，整个项目的核心架构服务于论文的两个主张：一是用 point-map geometry anchor triangle orientation，让三角面片朝向稳定且贴合曲面；二是让 rendering primitives 本身就是 mesh primitives，从而把 feed-forward reconstruction 的输出直接变成 simulation-ready triangle mesh。