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

本页只保留需要被人快速理解的当前结论。普通结果按[人类可读实验结果规范](实验结果规范.md)控制在 1–3 屏；配置、日志、checkpoint、raw metrics 和求解器产物继续下沉到[实验与证据库](../results/README.md)与 `runs/`。

下表“状态”指实验或证据状态；页面本身仍是等待人类审阅的 Draft。

## 主要实验

| Experiment ID | 证据层级 | 状态 | 当前结论 | 阅读入口 |
|---|---|---|---|---|
| `EXP-PHY-001` | physical validation | completed | r64 满足当前 B/G 分布的预注册误差门并完成 WSL 交叉验收 | [短结果](EXP-PHY-001-MPB分辨率.md) · [详细证据](../results/data-provenance/全胞MPB分辨率收敛验证_20260817.md) |
| `EXP-FWD-004` | surrogate held-out | completed | ConvNeXt 是 B/G native 三 seed 最优候选，但跨路线和分辨率迁移仍退化 | [短结果](EXP-FWD-004-全胞升级模型.md) · [详细证据](../results/forward/full_cell_forward_model_upgrade_50k_evaluation_20260810.md) |
| `EXP-FWD-005` | surrogate held-out | completed, seed 0 | K20 B/G 单路线与跨路线评估完成；本域明显优于跨路线 | [详细证据](../results/forward/full_cell_k20_b_g_seed0_cross_eval_20260820.md) |
| `EXP-FWD-006` | surrogate cross-domain | completed | 272 个四域评估单元完成；不支持单一模型族跨域普遍最优 | [详细证据](../results/forward/全胞K20与50K跨数据域正向验证_20260820.md) |
| `EXP-INV-004` | solver replay | completed, seed 0 | Objective-Bridge 完成 Test625 real-MPB replay，但尚无跨 seed 结论 | [短结果](EXP-INV-004-Objective-Bridge.md) · [详细证据](../results/inverse/a2_inverse_objective_bridge_seed0_v1_20260727.md) |
| `EXP-PHY-003` | admission gate | blocked | Field/adapted PINO 三项准入未通过，不进入当前正式 inverse 主线 | [详细证据](../results/field-physics/field_eigen_te_b4_dense_k_admission_20260722.md) |

## 当前结论

1. 当前证据支持推荐全胞 MPB `resolution=64`；新结构分布仍需自己的数据质量门。
2. 全胞正向模型在本域已有稳定候选，但 K 截断、路线与样本分布改变会造成明显域偏移。
3. V2 inverse 的 seed-0 replay 已完成；full-cell inverse 的正式训练、筛选和 replay 仍未完成。
4. Field/adapted PINO 仍是被阻塞的历史方向，不能与当前全胞正向结果并列宣传。

## 证据路径

| 想核查什么 | 入口 |
|---|---|
| 正式 Experiment ID、状态与跨层映射 | [`experiments/registry.yaml`](https://github.com/fangtoast/phc2d/blob/master/experiments/registry.yaml) |
| V2 数据上的正向、查询与 inverse | [V2 10k 证据集合](../results/V2-10k正式结果.md) |
| 全胞数据验收、正向与物理验证 | [全胞数据证据集合](../results/全胞数据正式结果.md) |
| 完整 evidence 页面与历史兼容入口 | [实验与证据库](../results/README.md) |
| 当前 full-cell inverse 协议 | [全胞逆向索引](../planning/全胞逆向索引.md) |

以上 ID 来自统一 Experiment Registry。历史制品不因建立短结果页而追溯改名；短结果负责结论和决策，详细证据页负责可复核边界，raw runs 不进入人类结果正文。
