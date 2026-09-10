---
title: EXP-FWD-004｜全胞升级模型 10k/50k 评估
summary: ConvNeXt 在 B/G 两条 native lane 上取得最低三 seed held-out MRE，但跨路线与输入分辨率迁移仍明显退化。
page_type: evidence
status: draft
audience: 导师、研究生、论文作者与模型维护者
owner: 研究生维护，导师审阅研究主张
last_reviewed: 2026-08-21
reading_time: 3 minutes
primary_layer: L3
experiment_id: EXP-FWD-004
evidence_date: 2026-08-10
evidence_scope: 50k B/G native held-out surrogate test、cross-lane 与输入分辨率诊断；不包含 real-MPB replay。
dataset_scope: full_cell_bz9_wh81x10
evidence_level: surrogate_held_out
evidence_status: completed
validation_backend: surrogate_only
source_artifacts:
  - runs/20260810/full_cell_upgrade_50k_evaluation_20260810085944
  - runs/20260819/full_cell_upgrade_50k_r2_seed0_202608191658
---

<!--
  Human-readable result for EXP-FWD-004.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-FWD-004｜全胞升级模型 10k/50k 评估

## 一句话结论

`convnext_band_v1` 在 B/G 两条 native lane 上均为本轮最优，MRE 分别为 **0.3212%** 和 **0.0500%**；但本域最优没有转化为稳定的跨路线或跨输入分辨率泛化。

## 核心结果

以下是冻结 checkpoint 在各 lane 5,000 个 held-out test 样本上的三 seed 平均 MRE，越低越好。

| 模型 | B lane MRE | G lane MRE | 结论 |
|---|---:|---:|---|
| `fno2d_band_v2` e200 | 0.6296% | 0.1602% | 当前强基线 |
| `mlp_mixer_band_v1` | 0.5927% ± 0.0314% | 0.0646% ± 0.0046% | 两 lane 均改善 |
| `res_mlp_gated_v2` | 0.8792% ± 0.0165% | 0.4001% ± 0.0388% | 退化 |
| `convnext_band_v1` | **0.3212% ± 0.0079%** | **0.0500% ± 0.0041%** | 两 lane 最优 |
| `tfno2d_band_v3` | 0.3950% ± 0.0023% | 0.1019% ± 0.0016% | 有效升级 |
| `dual_lane_band_operator_v1` | 0.4665% ± 0.0076% | 0.1196% ± 0.0030% | 未达到联合替代门 |

![50k native 模型对照](../reports/assets/full_cell_forward_inverse_report_20260810/figure_6_1_native_50k_model_comparison.png)

## 与基线相比

ConvNeXt 相对其直接 CNN 基线 `lifted_cnn`，B/G 误差分别降低 **77.74%/92.86%**；相对强基线 `fno2d_band_v2`，分别降低 **48.98%/68.75%**。Mixer、ConvNeXt 和 TFNO 均通过预注册的直接基线 promotion 门，Gated ResMLP 在 G lane 明确退化。

## 解释与限制

当前 Full-Wh 映射并不存在“频谱模型天然最优”：深层残差卷积在 native 数据上更准确、更省资源。另一方面，ConvNeXt 的 B→G/G→B MRE 上升到 2.3806%/2.8664%，32/128 输入诊断也明显退化，因此不能把 native 排名外推为跨域通用性、物理收敛或 inverse surrogate 替换许可。

## 决策

**继续：** 将 ConvNeXt 保留为 native 64×64 Full-Wh 的首选候选，保留 TFNO 作为有效频谱路线。**停止：** 不继续推广 Gated ResMLP，也不把当前联合模型作为两条最佳独立模型的替代。**需要额外验证：** 进入 inverse 或物理主张前，必须另做冻结协议验证。

## 关键设置

| 项目 | 设置 |
|---|---|
| 数据 | B/G 各 50k；40k/5k/5k 冻结 split |
| 输入/目标 | 64×64 全胞表示 → `Wh[81,10]` |
| 主指标 | held-out bands 2–10 MRE |
| 重复 | 3 seeds；10,000 次 paired bootstrap |
| 证据层级 | surrogate held-out；无 real-MPB replay |

## 证据入口

- experiment spec：[升级实验协议](https://github.com/fangtoast/phc2d/blob/master/experiments/research/full_cell_forward_upgrade_10k.md)
- config：[B 50k](https://github.com/fangtoast/phc2d/blob/master/configs/research/full_cell_forward_upgrade_binary_50k_v1.yaml) · [G 50k](https://github.com/fangtoast/phc2d/blob/master/configs/research/full_cell_forward_upgrade_graded_50k_v1.yaml) · [双路线 50k](https://github.com/fangtoast/phc2d/blob/master/configs/research/full_cell_forward_upgrade_dual_lane_50k_v1.yaml)
- runs：[主 campaign 与产物索引](../results/forward/full_cell_forward_model_upgrade_50k_evaluation_20260810.md)，源目录 `runs/20260810/full_cell_upgrade_50k_evaluation_20260810085944/`
- raw metrics：[指标与 promotion 复核入口](../results/forward/full_cell_forward_model_upgrade_50k_evaluation_20260810.md)，包括 `metrics.json`、`native_test_summary.csv` 与 `paired_bootstrap_comparisons.csv`
- figures：[原始 figures 目录说明](../results/forward/full_cell_forward_model_upgrade_50k_evaluation_20260810.md) · [当前图件 provenance](../reports/assets/full_cell_forward_inverse_report_20260810/figure_index.md)
