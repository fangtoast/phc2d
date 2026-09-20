---
title: EXP-FWD-023 十二域交叉测试
summary: 固定 seed0 的 63 个模型在 K5/K8/K10/K15/K20/F50、Binary/Graded 十二个 held-out 域上的交叉测试。
page_type: evidence
status: draft
audience: 研究用户与项目负责人
owner: 研究生维护，导师审阅
last_reviewed: 2026-09-20
reading_time: 8 minutes
primary_layer: L3
evidence_date: 2026-09-20
evidence_level: surrogate_held_out
evidence_scope: seed0_exploratory_surrogate_only; K8/K10/K15 pending WSL physical acceptance
dataset_scope: full_cell_k5_k8_k10_k15_k20_f50_binary_graded_wh81x10_held_out
evidence_status: completed
validation_backend: surrogate_only
completion_qualification: fixed_checkpoint_twelve_domain_complete_pending_human_review_and_new_dataset_wsl_acceptance
source_artifacts:
  - runs/EXP-FWD-021/20260915/EXP-FWD-021_20260915194709
  - runs/20260917/EXP-FWD-023_202609171354
---

<!--
  Fixed-checkpoint twelve-domain Draft; numerical evidence remains in runs.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-FWD-023：十二域交叉测试

## 结论

63 个 EXP-FWD-021 validation-best checkpoint 已完成 K5/K8/K10/K15/K20/F50、Binary/Graded 十二域交叉测试。756/756 个 cell、7,560/7,560 条逐 Band 记录均通过唯一性、完整性和有限值审计；这次测试固定已有模型，不做 K-fold 重训，也不根据目标 test 域重新选择 checkpoint 或 normalizer。

十二域等权宏平均的最佳组合是 `S3-BG/tfno2d_band_v3`：Band 1 MRE 为 **0.453154%**，Bands 2–10 MRE 为 **0.505587%**，Bands 1–10 MRE 为 **0.500344%**。两种主统计口径均把它排在第一位，没有冠军口径冲突。`S3-BG/global_local_band_operator_v1` 以 Bands 2–10 **0.509195%** 排第二；两者只差 0.003608 个百分点，而现有 bootstrap 以同制度 FNO 为固定参考，不包含这两个第一、第二名之间的直接配对区间，因此不能宣称 TFNO 对 Global-Local 有统计显著优势。

在 S3-BG 下，TFNO 和 Global-Local 相对同制度 `fno2d_band_v2` 在 12/12 个目标域的样本 bootstrap 区间都低于零。该结果支持“六域 Binary+Graded 混合训练能显著提高固定预算下的跨 K 泛化”这一探索性结论，但证据仍只有 seed 0；它不构成架构普遍优越、模型 promotion 或物理有效性结论。

## 十二域等权结果

每个制度的两个口径冠军一致。

| 训练制度 | Bands 2–10 冠军 | MRE% | Bands 1–10 冠军 | MRE% |
|---|---|---:|---|---:|
| S1-B | `global_local_band_operator_v1` | 1.847650 | `global_local_band_operator_v1` | 1.875588 |
| S1-G | `global_local_band_operator_v1` | 1.094103 | `global_local_band_operator_v1` | 1.089536 |
| S2-K5-BG | `global_local_band_operator_v1` | 1.240104 | `global_local_band_operator_v1` | 1.255732 |
| S2-K20-BG | `tfno2d_band_v3` | 1.431902 | `tfno2d_band_v3` | 1.465542 |
| S3-B | `convnext_band_v1` | 1.843012 | `convnext_band_v1` | 1.860059 |
| S3-G | `global_local_band_operator_v1` | 1.001458 | `global_local_band_operator_v1` | 0.994796 |
| S3-BG | `tfno2d_band_v3` | **0.505587** | `tfno2d_band_v3` | **0.500344** |

S3-BG 的前三名如下。这里的 12 域宏平均对每个目标域等权，不按 F50 的 5,000 条样本放大权重。

| 排名 | 模型 | Band 1 MRE% | Bands 2–10 MRE% | Bands 1–10 MRE% | 训练收敛门 |
|---:|---|---:|---:|---:|---|
| 1 | `tfno2d_band_v3` | 0.453154 | **0.505587** | **0.500344** | 未通过末窗门 |
| 2 | `global_local_band_operator_v1` | 0.475374 | 0.509195 | 0.505812 | 未通过末窗门 |
| 3 | `convnext_band_v1` | 0.800929 | 0.838264 | 0.834530 | 未通过末窗门 |

“未通过末窗门”不表示测试失败：三个 checkpoint 都完成了固定 200 epochs 并产生有效预测，但不能标为收敛成功。全部 63 个训练任务中，23 个为 `formal_training_success`，40 个为 `training_complete_convergence_not_met`。

## 最佳组合的逐域表现

以下为 `S3-BG/tfno2d_band_v3`；每格均来自独立冻结 held-out test。

| 目标域 | Band 1 MRE% | Bands 2–10 MRE% | Bands 1–10 MRE% |
|---|---:|---:|---:|
| K5-B | 0.510225 | 0.748797 | 0.724939 |
| K5-G | 0.426276 | 0.620567 | 0.601138 |
| K8-B | 0.484812 | 0.595220 | 0.584179 |
| K8-G | 0.493610 | 0.547903 | 0.542474 |
| K10-B | 0.472004 | 0.556828 | 0.548346 |
| K10-G | 0.583719 | 0.577018 | 0.577688 |
| K15-B | 0.405372 | 0.420935 | 0.419378 |
| K15-G | 0.460019 | 0.429941 | 0.432949 |
| K20-B | 0.404028 | 0.331812 | 0.339033 |
| K20-G | 0.303752 | 0.230823 | 0.238116 |
| F50-B | 0.554481 | 0.657321 | 0.647037 |
| F50-G | 0.339550 | 0.349879 | 0.348846 |

对该组合，六个训练来源域的 Bands 2–10 等权 MRE 为 0.489866%，六个未见 K8/K10/K15 域为 0.521307%，未见域增加 0.031441 个百分点。配对 K-shift 显示：Binary 的 K8→K10、K8→K15 均改善；Graded 的 K8→K10 恶化 0.029115 个百分点，而 K8→K15 改善 0.117963 个百分点。这说明跨 K 变化并非严格单调，不能只用十二域总平均掩盖单域退化。

## 合同与完整性

| 项目 | 审计结果 |
|---|---|
| checkpoint | 63 个唯一 validation-best 身份，模型 factory、normalizer、来源域和 SHA-256 冻结 |
| 目标域 | K5/K8/K10/K15/K20/F50 × Binary/Graded，共 12 域 |
| cell | 63 × 12 = 756，唯一且齐全 |
| 逐 Band | 7,560 条；每个 cell 恰好包含 Band 1–10 |
| 指标 | Band 1、Bands 2–10、Bands 1–10 与逐 Band MRE 均为有限值 |
| 测试身份 | K5/K20 各路线 1,000 条，F50 各路线 5,000 条；K8/K10/K15 各路线 1,000 条 |
| 选择泄漏 | checkpoint 只由源域 validation 选择；目标 test 不参与选择或调参 |
| 汇总权重 | 12 个目标域等权，不按样本量加权 |

## 证据边界与下一决策

- 本页是 `seed0 exploratory surrogate evidence`。样本 bootstrap 只描述固定 seed、固定 checkpoint 下的样本不确定性，不能替代跨 seed 稳定性。
- K8/K10/K15 六份数据已通过 N56 文件哈希、schema、split、同 lane 配对和 source train/validation 隔离审计，但仍标记为 `cluster_audited_pending_wsl_acceptance`；它们尚未完成 WSL `meep_env` 的独立物理验收。
- 结果没有 real-MPB/Meep replay，不能推出物理带隙或器件性能结论，也不自动触发模型 promotion。
- 如果要在 TFNO 与 Global-Local 之间作正式模型选择，下一门应是直接的两模型逐样本配对区间、至少三个 seed，并先解决 40/63 训练身份未通过末窗收敛门的问题。

## 原始证据

N56 合并证据目录：`/data/run01/scxl892/runs/20260917/EXP-FWD-023_202609171354/evaluation/combined/`。

| 证据 | 用途 |
|---|---|
| `results.json` / `metrics.json` | 756-cell 完整性、父 campaign 哈希、证据边界 |
| `cell_matrix.csv` / `twelve_domain_macro.csv` | 逐 cell 数值与十二域等权汇总 |
| `per_band_metrics.csv` | 7,560 条逐 Band 指标 |
| `paired_model_bootstrap.csv` | 各模型相对同制度 FNO 的逐目标域配对区间 |
| `paired_k_shift_bootstrap.csv` | 同 checkpoint、同 lane 配对样本的 K8→K10/K15 区间 |
| `transfer_groups.csv` / `ranking_conflicts.csv` | 训练来源域、未见域和口径冲突 |
| `training_resources.csv` | 训练状态、时间与峰值显存 |

[返回正向代理结果索引](README.md)
