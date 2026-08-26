---
title: 当前结果
summary: 当前主要实验的结论、状态、Experiment ID 与证据入口。
page_type: landing
status: draft
audience: 研究使用者、导师与结果审阅者
owner: 研究生维护，导师审阅研究主张
last_reviewed: 2026-08-25
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
| `EXP-FWD-007` | surrogate cross-domain | completed, Draft review | K5 17-task 与 K5/K20/F50 六域 510-cell held-out 重推完成，旧 272-cell 回归通过；缺 paired-bootstrap/resource 汇总，尚不作 promotion | [Draft 结果页](../results/forward/EXP-FWD-007-K5训练与六域验证.md) |
| `EXP-FWD-009` | surrogate cross-domain | completed with convergence waiver, Draft review | 五个权重热启动身份收敛，最终 17 个 K5 身份完成六域 510-cell、408/272 回归和 102-cell bootstrap；仅 9/17 满足主收敛门 | [Draft 结果页](../results/forward/EXP-FWD-009-K5权重热启动与六域验证.md) |
| `EXP-FWD-010` | surrogate cross-domain | completed with convergence not met, Draft review | joint 各 4k 从零完成 seed0 固定 200 轮与六域 6-cell；Binary 末窗略未过门，六域均未击败最佳 K5 单路线 | [009/010 汇总](../results/forward/EXP-FWD-009-K5权重热启动与六域验证.md) |
| `EXP-INV-004` | solver replay | completed, seed 0 | Objective-Bridge 完成 Test625 real-MPB replay，但尚无跨 seed 结论 | [短结果](EXP-INV-004-Objective-Bridge.md) · [详细证据](../results/inverse/a2_inverse_objective_bridge_seed0_v1_20260727.md) |
| `EXP-PHY-003` | admission gate | blocked | Field/adapted PINO 三项准入未通过，不进入当前正式 inverse 主线 | [详细证据](../results/field-physics/field_eigen_te_b4_dense_k_admission_20260722.md) |
| `EXP-PHY-004` | physical sensitivity | completed | K5 全量 MRE 较低，但代表样本最大带隙选择失配可显著改变真实 Meep 响应 | [详细证据](../results/field-physics/代理选频与真实响应差异_20260825.md) |

## 当前结论

1. 当前证据支持推荐全胞 MPB `resolution=64`；新结构分布仍需自己的数据质量门。
2. 全胞正向模型在本域已有稳定候选，但 K 截断、路线与样本分布改变会造成明显域偏移。
3. V2 inverse 的 seed-0 replay 已完成；full-cell inverse 的正式训练、筛选和 replay 仍未完成。
4. Field/adapted PINO 仍是被阻塞的历史方向，不能与当前全胞正向结果并列宣传。
5. K5 的整体 held-out MRE 较低，但最大带隙的离散 band/k 选择可能失配；forward 进入 inverse 选频前需要独立的 gap-selection 与 real-solver sensitivity gate。

## 证据路径

| 想核查什么 | 入口 |
|---|---|
| 正式 Experiment ID、状态与跨层映射 | [`experiments/registry.yaml`](https://github.com/fangtoast/phc2d/blob/master/experiments/registry.yaml) |
| V2 数据上的正向、查询与 inverse | [V2 10k 证据集合](../results/V2-10k正式结果.md) |
| 全胞数据验收、正向与物理验证 | [全胞数据证据集合](../results/全胞数据正式结果.md) |
| 完整 evidence 页面与历史兼容入口 | [实验与证据库](../results/README.md) |
| 当前 full-cell inverse 协议 | [全胞逆向索引](../planning/全胞逆向索引.md) |

以上 ID 来自统一 Experiment Registry。历史制品不因建立短结果页而追溯改名；短结果负责结论和决策，详细证据页负责可复核边界，raw runs 不进入人类结果正文。
