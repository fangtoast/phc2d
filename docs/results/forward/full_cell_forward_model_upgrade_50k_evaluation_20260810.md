<!--
  Draft experimental report for the Full-Wh forward model upgrade campaign.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# Full-Cell 前向升级模型 50K 实验报告（2026-08-10）

> **Status:** Draft experimental report; main evaluation complete; threshold diagnostic pending  
> **Primary layer:** L3 experiment/evidence  
> **Audience:** 项目负责人、模型开发者与后续论文结果整理人员  
> **Training platform:** N40，单任务 1×RTX 4090，最多 8 个独立任务并行  
> **Frozen split:** 40,000 train / 5,000 validation / 5,000 held-out test  
> **Seeds:** 0、1、2  
> **Evidence campaign:** `runs/20260810/full_cell_upgrade_50k_evaluation_20260810085944`  
> **Evaluation jobs:** Slurm array `5401548`；finalizer `5401549`  
> **Threshold campaign:** `pending_not_executed`

## 1. 本轮最重要的实验结论

本轮五个升级模型中，`convnext_band_v1` 是两条 native lane 上共同最优的模型：binary lane（B）测试 MRE 为 **0.3212% ± 0.0079%**，graded lane（G）为 **0.0500% ± 0.0041%**。相对各自直接基线，误差分别降低 **77.74%** 和 **92.86%**；相对当前 `fno2d_band_v2` 的 epoch-200 结果，分别降低 **48.98%** 和 **68.75%**。

三种独立升级模型通过预注册的直接基线 promotion 门：

- `mlp_mixer_band_v1`
- `convnext_band_v1`
- `tfno2d_band_v3`

`res_mlp_gated_v2` 没有通过：它在 B lane 相对直接 ResMLP 仅改善 8.18%，在 G lane 反而退化 27.81%，因此必须作为负结果保留。

`dual_lane_band_operator_v1` 虽然在两条 lane 上都优于当前 FNO，但没有达到“一个联合模型替代两个最佳独立模型”的标准。它相对最佳独立 ConvNeXt 在 B/G lane 分别退化 45.22%/139.00%，且 checkpoint 存储比例为 107.31%，未满足不超过 60% 的存储门。

因此，当前建议是：

1. 将 `convnext_band_v1` 作为 native 64×64 Full-Wh 前向任务的首选升级模型。
2. 将 `mlp_mixer_band_v1` 作为计算开销较低、分辨率迁移表现更稳的候选模型。
3. 保留 `tfno2d_band_v3` 作为频谱算子升级的有效正结果，但不将其描述为本轮最佳模型。
4. 不推广 `res_mlp_gated_v2`，也不以 `dual_lane_band_operator_v1` 替代两个独立模型。

## 2. Native held-out test：核心结果对比

下表均为冻结 checkpoint 后在 5,000 个 held-out test 样本上的三 seed 平均 MRE。数值越低越好。

| 模型 | 身份 | B lane MRE | G lane MRE | 对直接基线结论 | 对 FNO-e200 结论 |
| --- | --- | ---: | ---: | --- | --- |
| `mlp` | 原始直接基线 | 1.8007% | 1.7701% | 参照 | 劣于 FNO |
| `res_mlp_wide` | 原始直接基线 | 0.9575% | 0.3131% | 参照 | 劣于 FNO |
| `lifted_cnn` | 原始直接基线 | 1.4427% | 0.7012% | 参照 | 劣于 FNO |
| `fno2d_band_v2` e200 | 当前强基线 | 0.6296% | 0.1602% | 频谱升级参照 | 参照 |
| `mlp_mixer_band_v1` | MLP 升级 | 0.5927% ± 0.0314% | 0.0646% ± 0.0046% | **通过** | 两 lane 均更好 |
| `res_mlp_gated_v2` | ResMLP 升级 | 0.8792% ± 0.0165% | 0.4001% ± 0.0388% | **不通过/退化** | 两 lane 均更差 |
| `convnext_band_v1` | CNN 升级 | **0.3212% ± 0.0079%** | **0.0500% ± 0.0041%** | **通过** | **两 lane 最优** |
| `tfno2d_band_v3` | FNO 升级 | 0.3950% ± 0.0023% | 0.1019% ± 0.0016% | **通过** | 两 lane 均更好 |
| `dual_lane_band_operator_v1` | 双 lane 联合模型 | 0.4665% ± 0.0076% | 0.1196% ± 0.0030% | 未达到联合替代门 | 两 lane 均更好 |

### 2.1 各升级模型相对直接基线

| 升级模型 | B lane 相对改善 | G lane 相对改善 | 10,000 次 paired bootstrap | Promotion |
| --- | ---: | ---: | --- | --- |
| `mlp_mixer_band_v1` | 67.08% | 96.35% | 两 lane 的差值 CI 上界均小于 0 | `promoted_over_direct_baseline` |
| `res_mlp_gated_v2` | 8.18% | -27.81% | G lane CI 完全位于退化侧 | `regressed` |
| `convnext_band_v1` | 77.74% | 92.86% | 两 lane 的差值 CI 上界均小于 0 | `promoted_over_direct_baseline` |
| `tfno2d_band_v3` | 37.27% | 36.36% | 两 lane 的差值 CI 上界均小于 0 | `promoted_over_direct_baseline` |

这里的 promotion 只表示相对预注册直接基线的 held-out surrogate test 改善，不等价于 real-MPB 物理验证通过。

## 3. 结果为什么值得关注

### 3.1 CNN 路线从弱基线变成本轮最佳

原始 `lifted_cnn` 的 B/G MRE 为 1.4427%/0.7012%，容量与结构都不足以和 FNO 竞争。升级后的 `convnext_band_v1` 降至 0.3212%/0.0500%，说明深层残差卷积、归一化、large-kernel depthwise convolution、layer scale 与 drop path 的组合在当前 Full-Wh 映射上非常有效。

### 3.2 MLP 升级有效，但不同设计差异很大

`mlp_mixer_band_v1` 在两条 lane 上均显著成功，尤其 G lane 从 1.7701% 降至 0.0646%。这说明升级不是简单“把原 MLP 加深”，而是将空间 token mixing 与通道 mixing 分开，并用残差结构稳定优化。

相反，`res_mlp_gated_v2` 在 G lane 出现明确退化。这表明门控与加深并不能自动解决固定展平表示的限制，也说明保留负结果是必要的。

### 3.3 频谱升级有效，但未超过 ConvNeXt

`tfno2d_band_v3` 相对 `fno2d_band_v2` 在 B/G lane 分别改善 37.27%/36.36%，证明更深的因子化频谱块是有效升级。不过其最终误差和训练成本仍高于 ConvNeXt，因此不能仅依据 FNO 的既有优势继续假定频谱路线必然最优。

## 4. Cross-lane：跨结构分布诊断

Cross-lane 将只在一个 lane 上训练的模型直接用于另一个 lane。它用于观察域偏移，不参与 native promotion。

| 模型 | B→G MRE | G→B MRE | 观察 |
| --- | ---: | ---: | --- |
| `convnext_band_v1` | **2.3806% ± 0.1027%** | 2.8664% ± 0.3044% | B→G 最好，但显著差于 native |
| `mlp_mixer_band_v1` | 3.7476% ± 0.3379% | **2.5361% ± 0.2406%** | G→B 最好 |
| `res_mlp_gated_v2` | 5.8125% ± 0.3298% | 5.4645% ± 0.2407% | 两方向均较差 |
| `tfno2d_band_v3` | 7.2218% ± 4.1043% | 2.5860% ± 0.1229% | B→G seed 波动很大 |

所有独立模型的跨 lane 误差都明显高于 native 误差，说明 binary 与 graded 数据分布不能被视为可直接互换。联合模型同时见过两条 lane，因此 cross-lane 对它不适用，不能填成零误差。

## 5. 32/128 零样本输入分辨率诊断

模型在 64×64 上训练，随后直接接收 32×32 或 128×128 输入。该结果只反映输入分辨率变化下的代理稳定性，不构成网格收敛或物理分辨率不变性证明。

| 模型 | B@32 | B@128 | G@32 | G@128 | 主要观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| `convnext_band_v1` | 3.5978% | 3.4842% | 2.9905% | 1.6050% | native 最好，但直接迁移退化明显 |
| `dual_lane_band_operator_v1` | 0.7142% | 0.7032% | 0.2071% | 0.3193% | 比 ConvNeXt 更稳 |
| `mlp_mixer_band_v1` | **0.6627%** | **0.5863%** | **0.0742%** | **0.0639%** | 四个诊断项总体最佳 |
| `tfno2d_band_v3` | 1.0542% | 0.8960% | 0.1960% | 0.5088% | G@128 退化较明显 |

`res_mlp_gated_v2` 因固定展平输入不支持动态分辨率，对应 12 个 seed/lane/resolution 项均记为 `not_applicable_fixed_flatten_shape`，而不是失败或零误差。

## 6. 逐 band 误差观察

逐 band 结果显示，ConvNeXt 在 B lane 从 band 1 的 0.2605% 上升到 band 10 的 0.4457%；在 G lane 除 band 1 为 0.1161% 外，其余 band 约为 0.0355%–0.0636%。TFNO 也呈现随 band 增大而逐步升高的趋势。MLP-Mixer 在 B lane 的逐 band 误差更平坦，约为 0.5450%–0.6384%。

这一结果提示最终论文图表不能只给全矩阵平均值：至少应同时展示逐 band 曲线，避免平均值掩盖高阶 band 的误差增长。

## 7. 训练与资源结果

| 模型 | Lane | 平均训练时间 | 训练峰值显存 | 参数量 | 推理吞吐 |
| --- | --- | ---: | ---: | ---: | ---: |
| `convnext_band_v1` | B/G | 0.44 h / 0.39 h | 821 MiB | 2.30 M | 7,335 / 7,202 samples/s |
| `mlp_mixer_band_v1` | B/G | 0.68 h / 0.77 h | 2,901 / 2,276 MiB | 2.92 M | 7,019 / 8,078 samples/s |
| `res_mlp_gated_v2` | B/G | 0.21 h / 0.19 h | 659 MiB | 6.51 M | 9,821 / 9,623 samples/s |
| `tfno2d_band_v3` | B/G | 4.98 h / 5.12 h | 6,037 / 6,781 MiB | 2.69 M | 1,446 / 1,484 samples/s |
| `dual_lane_band_operator_v1` | joint | 7.33 h | 6,450 MiB | 2.70 M | 1,499 / 1,547 samples/s |

ConvNeXt 同时取得最低 native 误差、较小显存和较高吞吐，是本轮最均衡的模型。TFNO 与联合算子模型的训练耗时约为 ConvNeXt 的一个数量级，当前精度结果不足以抵消这一成本差异。

## 8. 尚未完成的结果

冻结 graded lane 胜者后的 threshold proxy 尚未执行，因此本报告不提供 threshold 数值。该诊断完成后，只会补充“同一冻结模型在阈值化结构上的代理误差”，不会改写本报告的 native promotion 判定，也不会将阈值化结构表述为真实二值结构的物理等价物。

本轮也没有执行 real-MPB replay、inverse-design replay 或物理收敛验证。因此以下表述目前均不成立：

- “模型已经物理验证通过”；
- “32/64/128 已证明分辨率不变”；
- “联合模型已经替代两套独立模型”；
- “前向升级已经自动授权替换 inverse surrogate”。

## 9. 证据文件与复核入口

主评估 campaign 内的关键制品包括：

- `results.json`：正式完成状态与总体结论；
- `native_test_summary.csv`：native 三 seed 汇总；
- `native_per_band_summary.csv`：bands 1–10 结果；
- `paired_bootstrap_comparisons.csv`：10,000 次 paired bootstrap；
- `promotion_decisions.csv`：预注册 promotion 门判定；
- `cross_lane_summary.csv`：跨路线诊断；
- `resolution_transfer_summary.csv`：32/128 诊断；
- `resource_summary.csv`：训练与评估资源汇总；
- `checkpoint_inventory.csv`：51 个冻结 checkpoint 的身份清单；
- `figures/`：native、cross-lane 与 resolution 三组 PNG/PDF 图。

主评估共规划 144 行，其中 126 行可执行、18 行为预注册 N/A；8 个评估 shard 与 finalizer 均以 `COMPLETED/0:0` 结束。评估前未读取 test 指标，checkpoint 由 validation 冻结。

## 10. 完成门与人工理解记录

### 本文档已经完成

- 汇报五个升级模型的 50K native held-out test；
- 与原始 MLP、ResMLP、CNN 和当前 FNO 强基线进行同口径比较；
- 汇报三 seed、paired bootstrap、逐 band、cross-lane、32/128 和资源结果；
- 明确保留 ResMLP 与联合替代门未通过的负结果；
- 将代理诊断、native 结果和物理验证边界分开。

### 本文档尚未完成

- threshold proxy 数值；
- real-MPB 或 inverse replay；
- 人工 owner 将 Draft 接受为 Reviewed/Canonical。

### 人工理解记录

- **本文回答的问题：** 五个升级模型是否在冻结 50K 实验合同下优于各自直接基线，以及哪一个模型在 native Full-Wh 前向预测中最好。
- **最关键结论：** ConvNeXt 是两条 lane 的共同最佳模型；Mixer 与 TFNO 也通过 promotion；Gated ResMLP 退化；联合模型不能替代两个最佳独立模型。
- **最容易误读的边界：** cross-lane、32/128 与 threshold 都是诊断或代理证据，不能提升为物理验证。
- **下一证据门：** 完成冻结 graded 胜者的 threshold campaign，并将其作为独立补充结果写回本报告。

