---
title: "EXP-FWD-015：gap 决策近简并审计"
summary: "用 EXP-FWD-014 的冻结检查点和 K5 validation 判断低命中率来自近简并还是模型决策能力。"
page_type: experiment
plan_state: completed
status: draft
audience: "正向代理研究者、实验执行者与结果审阅者"
owner: "研究生维护，导师审阅后续损失合同"
last_reviewed: 2026-08-28
primary_layer: L3
---

<!--
  EXP-FWD-015 experiment contract.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-FWD-015：gap 决策近简并审计

**结论：严格命中率低并非主要由近简并造成，当前瓶颈是模型对候选 gap 与工作频率 k 点的决策能力。** 本实验不训练、不读取 held-out test，也不运行 MPB/Meep。

| 合同 | 冻结值 |
|---|---|
| 数据 | EXP-FWD-014 的 K5 Binary/Graded 各 4,000 train 子集与各 1,000 validation |
| 模型 | EXP-FWD-014 baseline 与 gap-aware 最佳检查点，按 SHA256 冻结 |
| 候选 | lower band 2–9 的 8 个相邻带隙 |
| 近简并阈值 | 每条 lane 只用 train 正间隔的第 10 百分位确定 |
| 判定门 | 容差 joint accuracy 相对严格值最大增益达到 `0.05` 才判为指标近简并主导 |
| 证据 | `validation_only_surrogate_decision_audit` |

审计同时报告严格、容差和排除两类近简并后的 joint accuracy，并单独报告 complete-gap 真实比例、预测比例与召回率。容差只用于诊断，不能替代严格指标或作为调参后的新成功标准。

原始证据：`runs/20260828/EXP-FWD-015_gap_margin_audit_20260828150000_wsl/`。结论见[结果页](../../docs/results/forward/EXP-FWD-015-gap决策近简并审计.md)。
