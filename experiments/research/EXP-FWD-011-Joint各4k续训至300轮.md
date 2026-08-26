---
title: "EXP-FWD-011：Joint 各 4k 续训至 300 轮"
summary: "从 EXP-FWD-010 的 epoch-200 权重状态重置 AdamW，固定续训至总 epoch 300，并复查收敛与六域 held-out 表现。"
page_type: experiment
plan_state: active
status: draft
audience: "模型研究者、复现执行者与结果审阅者"
owner: "研究生维护，导师审阅研究主张"
last_reviewed: 2026-08-26
primary_layer: L3
---

<!--
  EXP-FWD-011 experiment contract.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-FWD-011：Joint 各 4k 续训至 300 轮

**研究问题：**EXP-FWD-010 的公平 4k+4k joint 在固定 200 轮时仅 Binary 曲线略未过收敛门；从其 epoch-200 权重续训至总 epoch 300 后，三条 validation 曲线能否同时通过，并改善冻结六域 held-out 表现？

| 合同 | 固定值 |
|---|---|
| 源状态 | EXP-FWD-010 seed0 `epoch200.pt`；源 history、结果、配置与 checkpoint hash 必须闭合 |
| 续训语义 | 只恢复模型权重；AdamW 与数据顺序状态不恢复 |
| 数据 | 与 EXP-FWD-010 相同的 Binary 4,000 + Graded 4,000 子集和 lane-specific normalizer |
| 不变项 | 模型结构、dense head、seed0、学习率、损失、32/32 平衡 batch、FP32 |
| 预算 | 强制训练到总 epoch 300；不得因中途达到门槛提前停止 |
| 选模 | 在合并的 epoch 1–300 validation history 上选择等权 bands 2–10 MRE 最优 checkpoint |
| 收敛门 | epoch 281–290 与 291–300 的 Binary、Graded、等权三条曲线均满足 `r≤0.005` |
| 六域 | K5/K20/F50 × Binary/Graded；与 EXP-FWD-010 epoch-200 checkpoint 做逐样本 paired bootstrap |
| 证据边界 | `surrogate_held_out_cross_domain`；无 real-MPB、Meep 或物理验收 |

本实验是一条新的权重热启动优化轨迹，不得写成 EXP-FWD-010 原 AdamW 状态的无损延长。若第 300 轮仍有任一曲线未过门，状态继续保持 `training_complete_convergence_not_met`。若通过，也只说明该 seed0 续训轨迹达到预登记曲线门，不自动支持 joint promotion。
