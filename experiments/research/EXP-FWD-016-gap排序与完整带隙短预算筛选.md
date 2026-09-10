---
title: "EXP-FWD-016：gap 排序与完整带隙短预算筛选"
summary: "在 N40 上用一个 seed0、20 epoch 任务筛选 ranked-v2 损失，不重复已冻结的 EXP-FWD-014 对照。"
page_type: experiment
plan_state: active
status: draft
audience: "正向代理研究者、N40 执行者与结果审阅者"
owner: "研究生维护，负责人授权远端训练"
last_reviewed: 2026-08-28
primary_layer: L3
---

<!--
  EXP-FWD-016 experiment contract.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-FWD-016：gap 排序与完整带隙短预算筛选

**本实验只需新增一个 N40 训练任务。** baseline 与 gap-aware v1 直接复用 EXP-FWD-014 的冻结 validation 证据；新任务保持同一模型、seed0、K5 Binary/Graded 各 4,000、20 epoch、优化器、初始化和 checkpoint 选择规则。

## 新损失

`equal_lane_gap_ranked_v2` 保留全谱 MSE 和已验证有效的上边带 k-profile，移除 v1 中重复拉动绝对 gap 数值的四项低权重回归，新增：

1. 候选排序间隔：要求真实最大 gap 在预测中相对其余候选保持目标间隔；预测等于目标时为零。
2. complete-gap 召回：只对真实完整带隙的预测宽度不足进行 lane 内条件平均，避免被大量 pseudo-gap 样本稀释。

| 项 | 权重 |
|---|---:|
| 全谱 MSE | 1.00 |
| 上边带 k-profile | 0.05 |
| 候选排序间隔 | 0.05 |
| complete-gap 召回 | 0.10 |

## 通过门

两条 lane 必须同时满足：

- Bands 2–10 MRE 相对 EXP-FWD-014 baseline 恶化不超过 `5%`；
- exact band accuracy 相对 gap-aware v1 至少提高 `0.03`；
- 非近简并 exact joint accuracy 相对 v1 至少提高 `0.03`；
- exact upper-k accuracy 相对 v1 下降不超过 `0.03`；
- complete-gap recall：Binary 至少 `0.25`，Graded 至少 `0.10`。

任一门失败即停止；通过只允许申请 seed0/1/2 的正式训练，不自动读取 test 或提交后续任务。远端失败不自动重试、取消、改参或换资源。

## 任务量

当前待授权任务为 `1` 个 N40 单卡短筛任务，预计产生一个标准 run。若通过且另行授权，下一阶段为 `3` 个 seed 的正式训练任务；它们不属于本轮执行。

相关配置：[`full_cell_k5_gap_ranked_exp_fwd_016_seed0_v2.yaml`](../../configs/research/full_cell_k5_gap_ranked_exp_fwd_016_seed0_v2.yaml)。前置审计见[EXP-FWD-015](EXP-FWD-015-gap决策近简并审计.md)。
