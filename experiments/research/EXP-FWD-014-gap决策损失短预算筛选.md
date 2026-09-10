---
title: "EXP-FWD-014：gap 决策损失短预算筛选"
summary: "在 N40 上以公平 K5 4k+4k、seed0、20 epoch 配对训练检验新损失能否改善工作频率、gap 与对应 k 点。"
page_type: experiment
plan_state: completed
status: draft
audience: "正向代理研究者、实验执行者与结果审阅者"
owner: "研究生维护，导师审阅正式扩展"
last_reviewed: 2026-08-28
primary_layer: L3
---

<!--
  EXP-FWD-014 experiment contract.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-FWD-014：gap 决策损失短预算筛选

**结论边界：本实验只回答新损失是否值得进入多 seed 正式训练，不回答最终模型是否优于基线。** 它在 N40 上运行两个配对任务，禁止读取 held-out test，也不运行 MPB/Meep。

| 合同 | 冻结值 |
|---|---|
| 数据 | K5 Binary 4,000 + Graded 4,000；沿用 EXP-FWD-010 稳定子集与各 lane validation 1,000 |
| 模型 | `dual_lane_band_operator_v1`，两个任务架构、初始化 seed、优化器和预算完全相同 |
| 对照 | `equal_lane_normalized_full_matrix_mse` |
| 新损失 | 全谱 MSE + gap profile + 选中 gap 边缘/宽度/中心 + 选中上边带相对 k-profile |
| 训练 | seed0、20 epoch、AdamW、FP32、每步 Binary/Graded 各 32 |
| checkpoint | 两个任务都按 equal-lane validation Bands 2–10 MRE 选择，避免改变选择规则 |
| gap 范围 | lower band 2–9；由目标谱选最大 complete/pseudo gap |
| 平台 | N40 `gpu_4090` 单卡任务；两个 arm 可独立并行 |
| 证据 | `preliminary_single_seed_validation_screen_no_test` |

## 为什么新增 k-profile 项

原 gap-aware 实现约束了上边缘频率值，却没有直接要求该最小值出现在正确 k 坐标。新增项选择真实最大 gap 的上边带，以真实工作频率 k 点为参考，比较整条上边带相对该点的频率差：

$$
L_{k\text{-profile}}=\operatorname{MSE}\left(
\frac{(\hat\omega_u(k)-\hat\omega_u(k^*))-(\omega_u(k)-\omega_u(k^*))}{s_\omega}
\right).
$$

它在预测等于目标时严格为零，并同时约束工作频率的数值与其在 k 路径上的相对极小位置。

## 通过门

Binary 与 Graded validation 分别比较新损失和基线。在新损失的 Bands 2–10 MRE 不比基线恶化超过 `5%` 的前提下，以下四项至少三项通过，且 `joint_band_k_accuracy` 必须通过：

1. `selected_frequency_mre_pct` 相对降低至少 `5%`；
2. `gap_width_mae` 相对降低至少 `5%`；
3. `gap_center_mae` 相对降低至少 `5%`；
4. `joint_band_k_accuracy` 绝对提高至少 `0.02`。

任一 lane 失败即停止自动扩展。通过只允许申请三 seed 正式训练，不允许读取 test、promotion 或物理外推。

## 任务与产物

训练任务共 `2` 个：baseline seed0 与 gap-aware seed0。每个 run 保存标准配置、日志、历史、指标和 checkpoint；比较只读取两个 run 的 validation 指标。远端失败不自动重试、取消或改参。

相关计划：[工作频率与带隙损失探索计划](../../docs/planning/工作频率与带隙损失探索计划.md)。

## 执行结论

2026-08-28 的两个 N40 任务均正常完成，但比较状态为 `gap_screen_fail`。新损失显著提高两条 lane 的 joint band-k accuracy，并改善多数 gap 指标；两条 lane 的 Bands 2–10 MRE 均恶化超过 `12%`，且 complete-gap 子集状态命中仍为 `0`，因此停止自动三 seed 扩展。详见[结果页](../../docs/results/forward/EXP-FWD-014-gap决策损失短预算筛选.md)。
