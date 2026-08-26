---
title: "EXP-FWD-010：Joint 各 4k 公平训练与六域验证"
summary: "将 dual-lane joint 的总训练样本固定为 8,000，与单一路线模型的总训练样本量一致，并在冻结六域测试集上比较。"
page_type: experiment
plan_state: active
status: draft
audience: "模型研究者、复现执行者与结果审阅者"
owner: "研究生维护，导师审阅研究主张"
last_reviewed: 2026-08-26
primary_layer: L3
---

<!--
  EXP-FWD-010 experiment contract.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-FWD-010：Joint 各 4k 公平训练与六域验证

**研究问题：**当 `dual_lane_band_operator_v1` 从零训练时仅使用 K5 Binary 4,000 条与 K5 Graded 4,000 条，总训练样本量与每个单一路线模型的 8,000 条一致，它在六个冻结 held-out 域上的表现是否仍优于 EXP-FWD-009 的单路线候选？

| 合同 | 固定值 |
|---|---|
| 模型身份 | `dual_lane_band_operator_v1`，dense head，seed 0 |
| 训练数据 | K5 Binary 4,000 + K5 Graded 4,000；由冻结 `idx_train` 按稳定 sample-id hash 子集规则选择 |
| Normalizer | 每个 lane 仅由本 lane 被选中的 4,000 条训练样本拟合 |
| 总数据量 | 8,000，与 EXP-FWD-008/009 的每个单一路线模型一致 |
| 初始化 | 从零训练；不得加载 EXP-FWD-008/009 的 8k-per-lane joint 权重 |
| 训练预算 | 固定 200 epoch，AdamW，FP32，Binary/Graded 32/32 平衡 batch |
| checkpoint | 等权 Binary/Graded validation bands 2–10 MRE 最优；另保留 epoch 200 终态 |
| 收敛门 | epoch 181–190 与 191–200 的 Binary、Graded、等权三条 validation 曲线均满足 `r≤0.005` |
| 六域 | K5/K20/F50 × Binary/Graded，复用 EXP-FWD-009 冻结 test split，不微调、不重拟合 normalizer |
| 证据边界 | `surrogate_held_out_cross_domain`；无 real-MPB、Meep 或物理验收 |

本实验只修正 joint 的训练数据量公平性。它不重写 EXP-FWD-009 的原始 8k+8k joint 结果，也不改变 16 个单路线 checkpoint。若固定 200 epoch 未通过收敛门，结果仍可作为固定预算对照报告，但必须保持 `training_complete_convergence_not_met`，不得写成收敛通过。
