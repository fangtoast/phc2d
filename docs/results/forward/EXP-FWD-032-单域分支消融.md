---
title: EXP-FWD-032 单域分支消融
summary: K5与F50四个单域的Full、Global-only、Local-only和容量对照，在六个目标上的三种子代理测试。
page_type: evidence
status: draft
audience: 正向模型研究者与结果审阅者
owner: fangtoast
last_reviewed: 2026-11-02
evidence_date: 2026-11-02
evidence_scope: four_single_domain_branch_ablation_six_target
dataset_scope: k5_f50_binary_graded_plus_dat007_dat008
evidence_level: surrogate_held_out
evidence_status: completed
validation_backend: surrogate_only
reading_time: 7 minutes
primary_layer: L3
source_artifacts:
  - runs/20261102/EXP-FWD-032_202611020900/comparison/comparison_audit.json
  - runs/20261102/EXP-FWD-032_202611020900/comparison/new_test_cells.csv
  - runs/20261102/EXP-FWD-032_202611020900/comparison/paired_full_branch_summary.csv
  - runs/20261102/EXP-FWD-032_202611020900/comparison/mixed_reference_three_seed_summary.csv
---

<!--
  Single-domain global and local branch ablation results.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-FWD-032：单域分支消融

**结论：四个单域的原域测试中，普通 Full 的三 seed 平均 MRE 均低于普通 Global-only 和 Local-only；但这一顺序不适用于容量对照。** 例如 F50-Graded 的 7×7 Local-only 容量对照为 0.0657%，低于 7×7 Full 的 0.1063%；该容量对照有一个 seed 在 1000 轮未达到停止门。结果为 Draft，等待人工审阅，不作统计显著性或物理验收结论。

72 个新增 checkpoint 在四域 held-out、DAT-007 12K 和 DAT-008 24K 上形成 **432/432 个唯一测试 cell**。两批最终均在 N40 执行；原 N56 测试在排队且无 cell 完成时，经用户授权转至 N40，保留训练来源及[接管核验](../../../runs/20261102/EXP-FWD-032_202611020900/comparison/inputs/rehost_audit.json)。每批 216 个 cell 的逐样本、逐 band、指标与预测文件哈希通过测试管线审计；另从预测和标签独立重算 12 个 cell 的全部样本误差，均通过。训练的 F50-Graded Local-only 容量对照有三个 seed 预算耗尽，不能称收敛。

## 四域原域对照

指标为 Band 1–10 MRE（%，越低越好），单元为三个 seed 的均值 ± 样本标准差。Full 采用历史单域 checkpoint，数据、子集、normalizer 和训练合同见[实验协议](../../../experiments/research/EXP-FWD-032-单域全局局部消融.md)。

| 训练及测试域 | Full 3×3 | Full 7×7 | Global-only | Global-only cap |
|---|---:|---:|---:|---:|
| K5 Binary | 0.5199 ± 0.0095 | 0.5229 ± 0.0004 | 0.5970 ± 0.0097 | 0.5326 ± 0.0205 |
| K5 Graded | 0.3699 ± 0.0050 | 0.3676 ± 0.0051 | 0.4312 ± 0.0083 | 0.3638 ± 0.0105 |
| F50 Binary | 0.4824 ± 0.0151 | 0.4750 ± 0.0045 | 0.5546 ± 0.0161 | 0.4683 ± 0.0085 |
| F50 Graded | 0.1080 ± 0.0048 | 0.1063 ± 0.0017 | 0.1143 ± 0.0029 | 0.0764 ± 0.0025 |

| 训练及测试域 | Local-only 3×3 | Local-only 7×7 | Local-only cap 3×3 | Local-only cap 7×7 |
|---|---:|---:|---:|---:|
| K5 Binary | 0.9867 ± 0.0066 | 0.7100 ± 0.0051 | 0.9992 ± 0.0072 | 0.6970 ± 0.0036 |
| K5 Graded | 0.8231 ± 0.0033 | 0.5167 ± 0.0020 | 0.8734 ± 0.0117 | 0.5151 ± 0.0072 |
| F50 Binary | 0.8729 ± 0.0034 | 0.5748 ± 0.0032 | 0.8326 ± 0.0112 | 0.4961 ± 0.0015 |
| F50 Graded | 0.1543 ± 0.0112 | 0.1232 ± 0.0063 | 0.0770 ± 0.0025 | 0.0657 ± 0.0028 |

Global-only 无局部核，不因 3×3/7×7 重复训练。其容量对照改变了输出头宽度，Local-only 容量对照改变了模型宽度；它们是参数量对照，不是结构等价替换。普通 Local-only 的 7×7 在四个原域均低于对应 3×3，但不能将未控制的优化与结构差异排除。逐 seed 的 Full 与单支差值、以及 Full 相对事后最佳单支的描述性差值见[配对结果](../../../runs/20261102/EXP-FWD-032_202611020900/comparison/paired_full_branch_cells.csv)和[最佳单支表](../../../runs/20261102/EXP-FWD-032_202611020900/comparison/full_vs_best_single_seed.csv)；后者不用于选模。

## 与 K5+K20 混合训练对照

下表为 K5 原域，单元为“单域 8k / K5+K20 混合 16k”，各自为三 seed 均值 ± 样本标准差。混合的 Binary 与 Graded 分开训练，每个 lane 中 K5、K20 各 8k；它们与单域训练的样本数和分布不同，不能把差值归因于单一因素。历史 3×3 来自 EXP-FWD-027，7×7 来自 EXP-FWD-030。

| K5 Binary 变体 | 3×3：单域 / 混合 | 7×7：单域 / 混合 |
|---|---:|---:|
| Full | 0.5199 ± 0.0095 / 0.5095 ± 0.0057 | 0.5229 ± 0.0004 / 0.5040 ± 0.0027 |
| Global-only | 0.5970 ± 0.0097 / 0.6098 ± 0.0050 | 0.5970 ± 0.0097 / 0.6098 ± 0.0050 |
| Global-only cap | 0.5326 ± 0.0205 / 0.5354 ± 0.0034 | 0.5326 ± 0.0205 / 0.5354 ± 0.0034 |
| Local-only | 0.9867 ± 0.0066 / 0.9797 ± 0.0086 | 0.7100 ± 0.0051 / 0.7015 ± 0.0170 |
| Local-only cap | 0.9992 ± 0.0072 / 0.9796 ± 0.0052 | 0.6970 ± 0.0036 / 0.6612 ± 0.0048 |

| K5 Graded 变体 | 3×3：单域 / 混合 | 7×7：单域 / 混合 |
|---|---:|---:|
| Full | 0.3699 ± 0.0050 / 0.3664 ± 0.0028 | 0.3676 ± 0.0051 / 0.3744 ± 0.0025 |
| Global-only | 0.4312 ± 0.0083 / 0.4402 ± 0.0077 | 0.4312 ± 0.0083 / 0.4403 ± 0.0078 |
| Global-only cap | 0.3638 ± 0.0105 / 0.3774 ± 0.0040 | 0.3638 ± 0.0105 / 0.3777 ± 0.0044 |
| Local-only | 0.8231 ± 0.0033 / 0.8215 ± 0.0030 | 0.5167 ± 0.0020 / 0.5147 ± 0.0061 |
| Local-only cap | 0.8734 ± 0.0117 / 0.8412 ± 0.0065 | 0.5151 ± 0.0072 / 0.5059 ± 0.0023 |

跨域和两套新几何的每个来源、目标、变体与 seed 分列在[新增 cell](../../../runs/20261102/EXP-FWD-032_202611020900/comparison/new_test_cells.csv)、[三 seed 汇总](../../../runs/20261102/EXP-FWD-032_202611020900/comparison/new_three_seed_summary.csv)和[混合参照汇总](../../../runs/20261102/EXP-FWD-032_202611020900/comparison/mixed_reference_three_seed_summary.csv)。这些场景不合并为总排名。

## 可比性与下一步

历史 7×7 对照逐 cell 核对了 checkpoint、目标数据、split、样本数、输入字段与 k 点哈希。历史 3×3 的逐 seed 派生表已核对 checkpoint、目标数据哈希和样本数，但未记录逐 cell split 哈希；其 split 身份沿用历史审计，当前不称已独立逐项复核。[比较审计](../../../runs/20261102/EXP-FWD-032_202611020900/comparison/comparison_audit.json)记录各输入与输出哈希。

下一步由研究负责人审阅容量对照和未收敛 seed 的解释，并在需要更强的 3×3 配对结论时回查其原始 split 记录。已有测试集并非全新盲测；12K/24K 是代理迁移证据，不等于 WSL MPB 复算、分辨率收敛或物理验收。
