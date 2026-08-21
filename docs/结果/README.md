---
title: 当前结果
summary: 当前主要实验的结论、状态、Experiment ID 与证据入口。
page_type: landing
status: draft
audience: 研究使用者、导师与结果审阅者
owner: 研究生维护，导师审阅研究主张
last_reviewed: 2026-08-21
reading_time: 5 minutes
primary_layer: L3
---

<!--
  Human-readable current result overview for phc2d.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 当前结果

本页只保留需要被人快速理解的当前结论。指标明细、运行日志、配置、checkpoint 和求解器产物继续留在[实验与证据库](../results/README.md)与 `runs/`，不复制到这里。

下表“状态”指实验或证据状态；页面本身仍是等待人类审阅的 Draft。

## 主要实验

| Experiment ID（现存） | 证据层级 | 状态 | 当前结论 | 证据 |
|---|---|---|---|---|
| `full_cell_resolution_convergence_v1` | physical validation | completed | 分辨率收敛支持正式全胞数据使用 MPB `resolution=64`，且已完成 WSL 交叉验收 | [收敛验证](../results/data-provenance/全胞MPB分辨率收敛验证_20260817.md) |
| `full_cell_upgrade_50k_evaluation` | surrogate held-out | completed | 三 seed 评估中 ConvNeXt 是 B/G 本域最强候选；不同模型族没有形成统一替代关系 | [50k 升级评估](../results/forward/full_cell_forward_model_upgrade_50k_evaluation_20260810.md) |
| `full_cell_k20_forward_seed0` | surrogate held-out | completed, single seed | B、G 单路线训练与 34 个跨路线单元均完成；本域结果优于跨路线结果 | [K=20 单 seed](../results/forward/full_cell_k20_b_g_seed0_cross_eval_20260820.md) |
| `full_cell_cross_domain_n40_v2` | surrogate cross-domain | completed | 68 个 checkpoint、272 个评估单元全部完成；跨数据域性能普遍下降，不支持“单一模型族普遍最优”的结论 | [K20/50k 跨域验证](../results/forward/全胞K20与50K跨数据域正向验证_20260820.md) |
| `a2_inverse_objective_bridge_seed0_v1` | solver replay | completed, seed 0 | V2 inverse 已获得 real-MPB replay 证据，但只支持当前 seed 与协议，不能外推到 full-cell inverse | [A2 replay](../results/inverse/a2_inverse_objective_bridge_seed0_v1_20260727.md) |
| `field_eigen_te_b4_dense_k_gate` | admission gate | blocked | 历史 Field/adapted PINO 三项准入均未通过，不进入当前正式结果主线 | [准入记录](../results/field-physics/field_eigen_te_b4_dense_k_admission_20260722.md) |

## 当前结论

1. 全胞数据的物理生成链已达到可用于当前研究的证据门槛，正式设置以 `resolution=64` 为准。
2. 全胞正向模型在本域已有稳定候选，但 K 截断、路线与样本分布改变会造成明显域偏移。
3. V2 inverse 的 seed-0 replay 已完成；full-cell inverse 的正式训练、筛选和 replay 仍未完成。
4. Field/adapted PINO 仍是被阻塞的历史方向，不能与当前全胞正向结果并列宣传。

## 证据路径

| 想核查什么 | 入口 |
|---|---|
| V2 数据上的正向、查询与 inverse | [V2 10k 证据集合](../results/V2-10k正式结果.md) |
| 全胞数据验收、正向与物理验证 | [全胞数据证据集合](../results/全胞数据正式结果.md) |
| 完整 evidence 页面与历史兼容入口 | [实验与证据库](../results/README.md) |
| 当前 full-cell inverse 协议 | [全胞逆向索引](../planning/全胞逆向索引.md) |

以上 ID 来自现有运行与报告。历史制品不会在本轮被追溯改名；新正式实验必须按 `EXP-FWD-*`、`EXP-QRY-*`、`EXP-INV-*`、`EXP-PHY-*` 或 `EXP-DAT-*` 建立稳定 ID，并贯穿 spec、config、run 与 result。
