<!--
  Deferred analytic-parameter inverse-design extension after full-cell forward training.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="full-cell-inverse-design-extension-plan"></a>

# 全胞解析参数逆向设计扩展计划

| Metadata | Value |
|---|---|
| Status | Deferred Draft pending forward-plan completion |
| Primary layer | L3 experiment/evidence；L4 仅提供必要接口支持 |
| Authority | [生成网络调整研究计划](../fno_generative_network_adjustment_plan_20260617.md) |
| Prerequisite | [全胞双路线训练与混合算力计划](full_cell_dual_lane_hybrid_compute_plan.md) |
| Audience | 逆向设计研究人员、GPU/超算执行人员与项目负责人 |
| Owner | 研究生维护，导师审阅 inverse promotion |
| Last reviewed | Pending human review |
| Reading time | 8 minutes |

本文只描述前向双路线完成后的解析参数 inverse 扩展，不属于当前 102 个 Full-Wh 训练 run。只有前向主计划通过人工审计、三 seed surrogate 冻结并取得新的执行授权后，才能实施本文。

## 一、准入条件

- binary 与 graded continuous 的主、辅 Full-Wh surrogate 已按三 seed validation 冻结；
- graded 10k 已通过 WSL acceptance；
- threshold proxy 模型不进入 inverse ranker；
- test split 未用于 generator 超参数选择；
- GPU 和 M9 预算另行授权。

## 二、解析参数输出

现有 full-cell solver 不接受任意生成 raster，inverse 必须输出可还原的解析结构。五个专用 codec 为：

| 变体 | 原参数宽度 |
|---|---:|
| `binary_polar_holes_1` | 16 |
| `binary_polar_holes_2` | 31 |
| `binary_spectral_threshold` | 580 |
| `graded_rbf_fourier_1` | 106 |
| `graded_rbf_fourier_2` | 112 |

codec 只预测可变字段，常量阶数和 mode coordinates 由模板恢复；输出必须构造 `FullCellStructureSpec`，并记录 `family_id`、`variant_id`、mask、`geometry_hash`、精确 raster 和 provenance。

## 三、方法与训练量

每个变体比较：

- train-only nearest-neighbor retrieval；
- `direct_inverse_mlp`；
- `single_mode_generator`；
- `mixture_conditional_generator`，`mode_count=4`。

retrieval 不训练；三个可训练方法 × 5 变体 × 3 seeds = **45 个 inverse training runs**。生成器仍是 mode-conditioned latent MLP，不得称作 cVAE、flow 或 diffusion。

每次训练包含最多 200 epochs 参数监督预训练和最多 50 epochs 的冻结双 surrogate 微调。直接模型学习率 `3e-4`，生成器 `1e-4`，AdamW、batch 64。

## 四、候选健康与筛选

- 生成方法每目标 `K=128`；binary 变体配额 `43/43/42`，graded 配额 `64/64`；
- `validity_rate=1.0`、`effective_k_mean>=64`、`effective_k_p10>=32`；
- mixture mode coverage 为 1，主辅 surrogate disagreement 均值 `<=0.02`；
- 通过门限后按双 surrogate MRE 均值排序，再以精确 raster 距离去重，选择 Top-5；
- direct 每变体只有一个候选，单独报告 `K=1/variant`；
- surrogate、replay 和 surrogate–replay gap 必须分开。

## 五、8→20→100 real-MPB replay

| 阶段 | split | 每路线目标数 | seeds | 方法 |
|---|---|---:|---|---|
| Smoke | val | 8 | 0 | 全部方法 |
| Pilot | val | 20 | 0 | 全部方法 |
| Formal | test | 100 | 0/1/2 | retrieval、direct、pilot 胜出 generator |

M9 只负责批量 real-MPB；租用 GPU 只负责 inverse training、candidate generation 和 surrogate ranking。不同环境不做 DDP，通过冻结 manifest、SHA-256 和原子 shard 交接。M9 bulk 结果必须经过固定 WSL `Ubuntu` + `meep_env` 子集重放后，才能形成正式物理结论。

正式 promotion 要求生成器相对最强基线的 paired bootstrap 95% CI 完全低于 0，solver success 不下降，cross-surrogate gate 通过；否则保留基线并报告未晋级。

## 六、授权边界

本文保持 Deferred Draft。完成前向计划不自动授权 inverse training、candidate sampling 或 M9 replay；每个计费阶段仍需独立 preflight、预算和明确授权。
