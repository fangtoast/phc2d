---
title: EXP-FWD-024 训练集与新几何迁移测试
summary: 54 个冻结检查点在各自训练子集、原生测试集和两批新几何数据上的 Bands 1–10 MRE。
page_type: evidence
status: draft
audience: 正向模型研究者与结果审阅者
owner: 研究生维护，导师审阅科学主张
last_reviewed: 2026-10-22
reading_time: 5 minutes
primary_layer: L3
evidence_date: 2026-10-21
evidence_scope: frozen_checkpoint_training_overlap_native_heldout_and_external_geometry_transfer
evidence_level: surrogate_held_out
evidence_status: completed
dataset_scope: k5_k20_f50_binary_graded_train_native_test_dat007_12000_dat008_24000
validation_backend: surrogate_only
source_artifacts:
  - runs/20261020/EXP-FWD-024_202610201920
  - runs/20261021/EXP-FWD-024_frozen_eval_module_v3
---

<!--
  EXP-FWD-024 frozen checkpoint evaluation result.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-FWD-024：训练集与新几何迁移测试

**结论：54 个冻结检查点已完成各自 8,000 条训练样本重构、原生 held-out 测试，以及 EXP-DAT-007 的 12,000 条和 EXP-DAT-008 的 24,000 条新几何直接迁移评估。** F50 Binary ConvNeXt 在两批新几何上的三 seed 平均 Bands 1–10 MRE 分别为 **0.217%** 和 **0.354%**，在本次 Binary 来源模型中最低。K20 Binary 虽有较低原生测试误差，在两批新几何上明显退化。新几何标签的 `resolution=64` 收敛和正式 WSL MPB 复算仍待验证，因此这些数值是当前标签合同下的 surrogate 迁移结果，不升级为物理精度结论。

## 评估范围与完整性

| 证据 | 样本与统计单位 | 解释 |
|---|---|---|
| 训练集 | 每 checkpoint 对应来源域的 8,000 条；54 单元 | 与训练样本重叠，用于重构检查，不是泛化测试 |
| 原生 held-out | 每 checkpoint 来源域原 `idx_test` 的 1,000 条；54 单元 | 原训练 campaign 已冻结的测试；与新几何分开比较 |
| EXP-DAT-007 | 六类单孔各 2,000 条，共 12,000 条；54 单元 | 新几何直接迁移，无重新训练或 checkpoint 选择 |
| EXP-DAT-008 | 十二个 PCSEL 几何族各 2,000 条，共 24,000 条；54 单元 | 新几何直接迁移，无重新训练或 checkpoint 选择 |

表中每格为三个 seed 的**均值**，单位 `%`；`CX` 为 `convnext_band_v1`，`GL` 为 `global_local_band_operator_v1`，`TFNO` 为 `tfno2d_band_v3`。主指标是完整 $81\times10$ 频带矩阵的 Bands 1–10 MRE。训练、原生、新几何样本分布不同，不能用其数值差作同分布泛化差。Graded 来源模型面对 Binary 面积分数输入的两批新几何时是**跨路线输入诊断**，不与 Binary 来源模型作公平主排名。逐 seed 值、seed 标准差及 Bands 2–10 辅助指标在原始 CSV 中。

| 来源域 | 模型 | 训练重叠 | 原生 held-out | DAT-007 12k | DAT-008 24k |
|---|---|---:|---:|---:|---:|
| K5 Binary | CX | 0.197 | 0.706 | 1.112 | 0.704 |
| K5 Binary | GL | 0.418 | 0.529 | 0.790 | 0.614 |
| K5 Binary | TFNO | 0.449 | 0.532 | 0.762 | 0.581 |
| K5 Graded* | CX | 0.142 | 0.467 | 2.381 | 2.472 |
| K5 Graded* | GL | 0.297 | 0.377 | 2.255 | 2.961 |
| K5 Graded* | TFNO | 0.318 | 0.386 | 2.827 | 3.497 |
| K20 Binary | CX | 0.254 | 0.500 | 3.808 | 4.066 |
| K20 Binary | GL | 0.172 | 0.205 | 6.712 | 6.011 |
| K20 Binary | TFNO | 0.184 | 0.210 | 5.123 | 4.751 |
| K20 Graded* | CX | 0.296 | 0.463 | 5.596 | 6.058 |
| K20 Graded* | GL | 0.115 | 0.135 | 8.112 | 7.380 |
| K20 Graded* | TFNO | 0.123 | 0.138 | 6.231 | 6.039 |
| F50 Binary | CX | 0.168 | 0.485 | **0.217** | **0.354** |
| F50 Binary | GL | 0.374 | 0.492 | 0.258 | 0.467 |
| F50 Binary | TFNO | 0.421 | 0.502 | 0.266 | 0.485 |
| F50 Graded* | CX | 0.054 | 0.071 | 0.585 | 1.028 |
| F50 Graded* | GL | 0.125 | 0.130 | 0.798 | 1.669 |
| F50 Graded* | TFNO | 0.133 | 0.138 | 0.938 | 1.974 |

\* 新几何两列使用 Binary 面积分数输入，与 Graded 来源训练输入不同；训练和原生列仍是其原生 Graded 数据。来源：N40 原生运行 `/data/run01/scw6g7a/phc2d/runs/20261020/EXP-FWD-024_202610201920/native_test_primary_summary.csv`，冻结评估运行 `/data/run01/scw6g7a/phc2d/runs/20261021/EXP-FWD-024_frozen_eval_module_v3/evaluation_summary.csv`。

逐族结果显示迁移难度并不均匀：F50 Binary CX 在 DAT-007 的 circle/triangle 均值分别为 0.086%/0.350%，在 DAT-008 的 U01/Q01 分别为 0.106%/0.708%。这四项仅用于指出族别差异，完整 18 族 × 54 checkpoint 的逐族记录见同一冻结评估运行的 `family_cells.csv`，逐样本文件保留在各 cell 目录。

## 审计与决策边界

N40 准入作业 `5508228`、8 个互不重叠数组任务 `5508230_0…7` 和汇总作业 `5508231` 均为 `COMPLETED/0:0`。独立审计检查了 **54 个唯一 checkpoint、162 个训练/新几何单元、1,026 条逐族记录和 2,376,000 条逐样本记录**：每单元分别有 8,000/12,000/24,000 条、三 seed 齐全、逐族每族 2,000 条、指标有限且逐样本均值与单元及 seed 汇总相符；checkpoint SHA 与源测试清单一致。源清单 SHA-256 为 `a11c26e1d48ffe2403f33196c3c637214c9ba78463d370dce5af032afe27383f`；DAT-007/008 数据 SHA-256 分别为 `badd2081180406d6eaac451d758c8d17259a23e13a2c00198b2c82059c9b85a8` / `8c3721e03603a01dd5a8e5c8f6b183e8c70259a259e18ac2f53483585e04aa5c`。核查记录见评估运行的 `independent_audit.json`；完整原始证据和每项配置均在该运行目录。

54 项训练虽完成固定 300 epoch，只有 **13/54** 满足既定末段收敛门；其余 41 项保留 `training_complete_convergence_not_met`，因此不能把模型排序写成充分优化后的能力上限。原生 held-out 结果来自既有训练 campaign；此次新几何评估没有用标签重新选择 checkpoint。下一研究门是新几何 `resolution=64→96→128` 数值收敛与正式 WSL MPB 复算验收；在此之前保留本页 Draft，不作模型 promotion、器件级预测或物理验证主张。
