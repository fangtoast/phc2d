---
title: "EXP-FWD-007：K5训练与六域验证"
summary: "K5 17-task 训练与 85 checkpoint × 6 域 held-out surrogate 评估已完成；Draft 仅报告迁移诊断，不作物理或 promotion 结论。"
page_type: evidence
status: draft
audience: "研究负责人、审阅者与复现执行者"
owner: "研究生维护"
last_reviewed: 2026-08-24
reading_time: 6 minutes
primary_layer: L3
evidence_date: 2026-08-24
evidence_scope: "K5 B/G 17-task training and K5/K20/F50 six-domain held-out surrogate evaluation; 85 checkpoints and 510 cells"
dataset_scope: full_cell_fourier_k5_k20_f50_wh81x10
evidence_level: surrogate_held_out
evidence_status: completed
validation_backend: surrogate_only
source_artifacts:
  - runs/20260824/EXP-FWD-007_K5_20260824080309
  - runs/20260824/EXP-FWD-007_six_domain_20260824162756
---

<!--
  EXP-FWD-007 evidence page.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-FWD-007：K5训练与六域验证

**结论：K5 的 17 个训练任务和六域 510-cell held-out surrogate 重推均已完成，且旧 272-cell 子矩阵数值回归通过；但这不是 real-MPB/物理验证，也尚不能发布模型 promotion。** K5 native 的最低 MRE 来自 ConvNeXt（B `0.7789%`、G `0.5169%`）；跨路线的最低项却分别是 MLP 的 B→G `2.0544%` 和 FNO 的 G→B `1.5178%`，不支持“一个模型跨域普遍最优”的说法。计划要求的 paired-bootstrap 与资源汇总文件未生成，故本页保持 Draft，结果只作方向性迁移诊断。

## 证据状态

| 门 | 观察到的证据 | 状态 |
|---|---|---|
| K5 训练 | 17/17 checkpoint；admission A `15/15`、admission B `2/2`、W1 `8/8`、W2 `8/8`、W3 `1/1` | 通过 |
| 六域预检 | 54/54 去重模型身份在 N40 GPU 预检通过 | 通过 |
| 六域推理 | 85 个 checkpoint × 6 个目标域 = 510/510 cells；10/10 shards | 通过 |
| 完整性 | 无缺失、重复、OOM 或 nonfinite；所有 cell 为 `completed` | 通过 |
| 旧证据回归 | 272/272 cells 通过；最大预测绝对差 `3.5763e-7 ≤ 1e-6` | 通过 |
| 统计与资源制品 | 没有生成计划中的 paired-bootstrap/resource 汇总 | 未闭合；不作 promotion |
| WSL real-MPB / inverse | 未执行 | 不适用；不得由本页升级 |

原始证据位于 N40 的 `runs/20260824/EXP-FWD-007_K5_20260824080309` 与 `runs/20260824/EXP-FWD-007_six_domain_20260824162756`。后者保存 `evaluation_cells.csv`、逐 band/k 指标、dataset/checkpoint/test-cache manifests、分片计划和 `legacy_272_numeric_regression.csv`；正文不复制 510 行原始结果。

## 冻结协议与可比性

| 项目 | 冻结值 |
|---|---|
| 目标 | `Wh[N,81,10]`，64×64 输入，bands 2–10 MRE 为主指标 |
| 域 | K5-B/G、K20-B/G 均为 10,000 样本（test=1,000）；F50-B/G 均为 50,000 样本（test=5,000） |
| 模型来源 | K5 17 + K20 17 + F50 51 = 85 个唯一 checkpoint |
| 评估 | 每个 cell 使用源 checkpoint 的 normalizer；joint 源按目标 lane 选择 B/G normalizer 并传入 `lane_id`；无微调或重拟合 |
| 训练选择 | seed 0、最多 100 epoch、patience 15；validation 冻结前不读取 held-out test |
| 部署链 | K5 training commit `3ea2b3c`；评估逻辑冻结 `82d82ef`；提交入口部署 `2e09433` |

K5 Binary 与 Graded 数据哈希分别为 `7d2c6bef…2ea86b6`、`eb140534…0e6af20`；F50 两个目标域按 `Wh[50000,81,10]` 与 5,000-sample test 合同冻结。K5/K20 是独立随机样本，因而 K5↔K20 差异混合了 cutoff 与分布改变，不能解释为 paired-K 因果效应；F50 checkpoint 的训练预算也不同，跨 family 结果只作迁移诊断。

## K5 训练与曲线

17 个任务均满足 `1 ≤ stop_epoch ≤ 100`、`best_epoch ≤ stop_epoch`。早停在 MLP/Mixer/Gated 路线较早触发（最早 best epoch 3），而 Lifted CNN、ConvNeXt、TFNO 和 joint 的若干路线运行至 epoch 100；完整 `history.csv`、17 条 PNG/PDF 曲线和 `curve_manifest.csv` 已保存在训练 run 中。

| K5 native 汇总 | B MRE | G MRE |
|---|---:|---:|
| 4 个 baseline 平均 | 1.6064% | 1.2990% |
| 4 个单路线升级平均 | 1.1810% | 0.9642% |
| 最低单模型（ConvNeXt） | 0.7789% | 0.5169% |
| joint（单独报告，不进入 B→G/G→B 泛化结论） | 0.5805% | 0.4488% |

该汇总只描述 K5 seed-0 native held-out 表现；joint 使用不同训练路线，不能作为单路线升级的替代排名。

## K5 内路线迁移

| 方向 | 最低模型 / MRE | baseline 平均 | 单路线升级平均 |
|---|---:|---:|---:|
| B→G | MLP / 2.0544% | 2.7012% | 3.2112% |
| G→B | FNO / 1.5178% | 3.1523% | 2.9474% |

路线迁移显著弱于 native，且最佳模型随方向改变。joint 在 K5-B/G 目标上的 MRE 为 `0.5805%/0.4488%`，但它只作为联合源的独立结果，不能用于证明 B→G 或 G→B 泛化。

## K5 到 K20 与 F50 的迁移

下表只汇总 K5 的 8 个单路线来源 checkpoint；joint 另列，避免混入单路线结论。

| K5 目标域与迁移类型 | cells | bands 2–10 MRE 均值 |
|---|---:|---:|
| K20-B，同路线跨生成器 | 8 | 5.1121% |
| K20-B，生成器+路线改变 | 8 | 7.2094% |
| K20-G，同路线跨生成器 | 8 | 4.4654% |
| K20-G，生成器+路线改变 | 8 | 4.1072% |
| F50-B，同路线跨生成器 | 8 | 2.4545% |
| F50-B，生成器+路线改变 | 8 | 3.3371% |
| F50-G，同路线跨生成器 | 8 | 3.4413% |
| F50-G，生成器+路线改变 | 8 | 5.8863% |

在 K5 来源这一切片中，变更路线通常会增加误差，但 K20-G 是例外，提示数据分布与模型交互不能用单一“路线更难”的结论概括。

## 六域总体迁移图景

| 比较类型（seed-0 共同来源集合） | cells | MRE 均值 |
|---|---:|---:|
| native | 48 | 0.8970% |
| 同 family 跨路线 | 48 | 3.6431% |
| 跨生成器、同路线 | 96 | 5.0474% |
| 生成器+路线均改变 | 96 | 6.9221% |
| joint source family（单独报告） | 18 | 2.0284% |

这组均值与具体模型排名不同：它只说明在本次 source/target 组合上，native 最低，而同时改变生成器与路线的平均迁移负担最大。F50 的 seed 1/2 共 204 cells 保留为补充；不伪装为与 K5/K20 配对的 seed-matched 比较。

## 限制与下一决策

1. 本页是 held-out surrogate 证据，不是 WSL real-MPB replay、物理验证或 inverse 成功。
2. 缺少计划中的 10,000 次 paired-bootstrap 和资源汇总制品；在补齐且审计通过前，不发布显著性、总体排名或 promotion 结论。
3. 训练/评估已完成并保留原始证据；任何后续修复必须使用新的 derivative run，不覆盖本次 510-cell 结果或旧 272-cell 证据。

相关协议：[EXP-FWD-007](../../../experiments/research/EXP-FWD-007-K5训练与六域验证.md)。
