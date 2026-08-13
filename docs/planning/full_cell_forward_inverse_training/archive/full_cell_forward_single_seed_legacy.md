<!--
  Superseded single-seed forward-only full-cell training plan.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="full-cell-forward-single-seed-legacy"></a>

# 二值/渐进全胞 Full-Wh 双表征训练计划（历史草案）

| Metadata | Value |
|---|---|
| Status | Superseded historical draft |
| Primary layer | L3 experiment/evidence |
| Superseded by | [全胞双路线训练与混合算力计划](../full_cell_dual_lane_hybrid_compute_plan.md) |
| Archived | 2026-08-06 |

> 本文保留旧版单 seed、确定阈值、仅正向训练方案的原始内容，仅用于决策追溯。它不再是当前执行入口，不得与新版三 seed、混合算力和独立逆向扩展协议混用。

---

## 总结

- Primary layer：L3 实验与证据；L4 仅实现 HDF5 流式训练和新表征接口。
- 目标：分别训练二值、渐进全胞数据，预测 `Wh=[81,10]`，比较连续数组、阈值二值数组及对应二维图像，并做跨表征、跨数据路线测试。
- 二值 10k 已通过 WSL 验收；渐进 10k 必须先取得 `wsl_cross_environment_acceptance_pass`。
- 本轮固定 seed `0`，状态只能是 `single_seed_full_cell_forward_pilot`。

## 数组与图像表征

设 HDF5 中的 `material_weight_grid_64` 为 \(u\in[0,1]^{64\times64}\)。

- 连续数组：
  ```text
  x_continuous = u.astype(float32).reshape(4096)
  ```
- 连续图像：
  ```text
  x_continuous_image = u.astype(float32)[None, :, :]
  ```
- 阈值二值数组：
  ```text
  b_t = (u >= t).astype(float32)
  x_threshold_t = b_t.reshape(4096)
  ```
- 阈值二值图像：
  ```text
  x_threshold_image_t = b_t[None, :, :]
  ```
- 展平固定使用 C-order；不随机采样，不进行 Bernoulli 噪声化。
- 阈值固定为 `t ∈ {0.3, 0.4, 0.5, 0.6, 0.7}`，不得根据测试集调整。
- 二值原始数据严格为 `{0,1}`，上述五个阈值结果相同，因此二值路线只训练一次原生表征。
- 渐进阈值输入继续使用原连续介质的 `Wh` 标签。这是信息损失/表征消融，不代表阈值图自身的真实 MPB 响应；产物必须记录：
  ```text
  label_semantics: original_continuous_medium_Wh
  representation_status: thresholded_proxy_input
  physical_equivalence: false
  ```

## 模型矩阵与训练协议

当前注册模型共 17 个：7 个 Full-Wh、5 个 Query、4 个 Field；`pod_res_mlp` 仍是占位项。本轮使用四个可训练 Full-Wh 模型：

- `mlp`：展平数组。
- `res_mlp_wide`：展平数组。
- `lifted_cnn`：二维图像。
- `fno2d_band_v2`：二维图像。

训练矩阵：

- 二值路线：4 个模型 × 1 个原生表征，共 4 个 `64×64` 训练。
- 渐进路线：4 个模型 × 6 个表征（连续＋五个阈值），共 24 个 `64×64` 训练。
- 主矩阵共 28 个单 seed 训练。
- 每种表征按验证集 bands 2–10 MRE 选择模型；总体渐进阈值也只按验证集选择。
- 分辨率消融：
  - 二值路线胜出模型补训 32/128。
  - 渐进连续表征胜出模型补训 32/128。
  - 渐进阈值表征胜出组合补训 32/128。
  - 共增加 6 个训练，整轮合计 34 个训练任务。

统一训练设置：

- seed `0`，FP32，AdamW，weight decay `1e-5`。
- MLP/ResMLP 学习率 `3e-4`；CNN/FNO 学习率 `1e-3`。
- 最多 100 epochs，early-stopping patience 15。
- 有效 batch size 64；按 `64→32→16→8` 做显存预检，不足时用梯度累积保持有效批量。
- 输入不再标准化；连续值保持 `[0,1]`，阈值值保持 `{0,1}`。
- `Wh` 只使用训练集逐 `(k,band)` 统计量标准化；checkpoint 保存均值、标准差、数据 hash、split hash、阈值和分辨率。
- 损失为标准化 `Wh` 的全矩阵 MSE；checkpoint 选择指标为验证集 bands 2–10 MRE。

## 交叉评估

使用固定测试集，所有模型和阈值先冻结后评估。

- 渐进内部形成每个模型的 `6×6` 表征迁移矩阵：
  - 连续训练→连续/五阈值测试。
  - 阈值 \(t_i\) 训练→连续及所有阈值 \(t_j\) 测试。
  - 共 `4×6×6=144` 个轻量评估单元。
- 跨路线评估：
  - 4 个二值模型分别测试渐进连续及五种阈值输入，共 24 个单元。
  - 24 个渐进模型分别测试真实二值 test split，共 24 个单元。
- 跨路线时始终使用源模型的 `Wh` 归一化统计，不读取目标路线训练或验证标签。
- 分别报告：
  - 原生性能。
  - 阈值量化损失。
  - 阈值漂移敏感性。
  - 连续↔阈值非对称性。
  - binary↔graded 跨路线泛化。
- 阈值模型即使优于连续模型，也只能说明该代理表征更易学习；不能据此声称阈值结构与连续介质物理等价。

## 实现、产物与验收

- 扩展 `DataRepresentationSpec`，加入连续/阈值的 flat/grid 表征以及显式 `threshold`。
- 增加 `phc2d.full_cell.v1` 流式 Dataset：按 worker 延迟打开 HDF5，只读取指定分辨率和样本行，复用冻结的 `8000/1000/1000` split。
- 所有形状从 HDF5 schema 推导，禁止硬编码 `64`、`81` 或 `10` 到模型和指标实现。
- 每个 run 输出 checkpoint、resolved config、results、metrics、history、normalization、模型身份、数据/split hash、训练时间、推理时间和峰值显存。
- 汇总输出模型 leaderboard、阈值 leaderboard、`6×6` 热图、跨路线矩阵、per-band/per-k 指标和学习曲线。
- 主指标为 bands 2–10 MRE；同时报告 MAE、RMSE、band 1 诊断、排序违例率、参数量、吞吐和运行时间。
- 测试覆盖连续/阈值转换、C-order 展平、阈值边界、HDF5 多 worker、安全关闭、split 无泄漏、训练集归一化和四模型 `[B,81,10]` 输出。
- 在 WSL `Ubuntu` + `meep_env` 完成 pytest、compileall、CLI smoke 和每模型 64 样本两 epoch CUDA smoke。
- seeds `1/2`、Query、Field/PINO、inverse、随机 Bernoulli 表征及阈值图重新计算 MPB 均不属于本轮；单 seed 结果通过人工审阅后再决定是否进入三 seed 正式基准。
