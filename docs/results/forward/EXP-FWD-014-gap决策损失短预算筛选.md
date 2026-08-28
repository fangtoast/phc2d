---
title: "EXP-FWD-014：gap 决策损失短预算筛选"
summary: "N40 seed0 配对训练表明新损失显著改善对应 k 点与多数 gap 指标，但全谱误差超出安全门，停止自动扩展。"
page_type: evidence
evidence_level: surrogate_held_out
status: draft
audience: "正向代理研究者、实验执行者与结果审阅者"
owner: "研究生维护，导师审阅后决定是否重构损失"
last_reviewed: 2026-08-28
reading_time: 4 minutes
evidence_date: 2026-08-28
primary_layer: L3
evidence_scope: "EXP-FWD-014 K5 Binary/Graded fair 4k+4k seed0 20-epoch paired training and 1k+1k validation-only gap screen; no test, MPB, or Meep"
dataset_scope: "K5 Binary/Graded Wh[10000,81,10] frozen train and validation subsets"
evidence_status: completed
validation_backend: surrogate_only
source_artifacts:
  - runs/20260828/EXP-FWD-014_gap_screen_20260828101706
---

<!--
  EXP-FWD-014 result.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-FWD-014：gap 决策损失短预算筛选

**结论：新损失确实更容易找到工作频率对应的 k 点和多数 gap 属性，但当前合同未通过整体准入门。** 两条 lane 的 joint band-k accuracy 分别提高 `10.9/15.1` 个百分点；Graded 的工作频率、gap 宽度和中心误差全部改善，Binary 的 gap 宽度改善、工作频率改善 `4.67%`。但两条 lane 的 Bands 2–10 MRE 分别恶化 `13.44%/12.65%`，超过 `5%` 安全上限；Binary gap center 还恶化 `7.27%`。因此结果为 `gap_screen_fail`，不得自动进入三 seed 正式训练。

## 实验合同

| 项目 | 冻结值 |
|---|---|
| N40 作业 | `5433581_0` baseline、`5433581_1` gap-aware；均 `COMPLETED (0:0)` |
| 部署 commit | `0f857097d1be022344faddf519742a914a84cdbb` |
| 数据 | K5 Binary 4,000 + Graded 4,000 train；各 lane validation 1,000 |
| 模型与预算 | 同一 `dual_lane_band_operator_v1`、seed0、20 epoch、AdamW、FP32 |
| 唯一主变量 | normalized full-matrix MSE vs. gap-aware + selected upper-band relative k-profile |
| checkpoint | 两个 arm 都按 equal-lane validation Bands 2–10 MRE 选择 |
| 最佳 epoch | baseline `19`；gap-aware `20` |
| 证据边界 | validation-only single-seed screen；未读取 test，未运行 MPB/Meep |

## Validation 结果

### Binary

| 指标 | MSE baseline | 新损失 | 变化 | 门 |
|---|---:|---:|---:|---|
| Bands 2–10 MRE | 1.2223% | 1.3867% | **+13.44%** | 失败：最多 +5% |
| 最大 gap band accuracy | 36.7% | 37.7% | +1.0 pp | 诊断 |
| 上边缘 k accuracy | 21.5% | 42.4% | **+20.9 pp** | 诊断 |
| joint band-k accuracy | 13.5% | 24.4% | **+10.9 pp** | 通过 |
| 工作频率 MRE | 16.0545% | 15.3047% | -4.67% | 失败：至少 -5% |
| gap width MAE | 0.007775 | 0.007243 | -6.84% | 通过 |
| gap center MAE | 0.088173 | 0.094588 | **+7.27%** | 失败 |

### Graded

| 指标 | MSE baseline | 新损失 | 变化 | 门 |
|---|---:|---:|---:|---|
| Bands 2–10 MRE | 1.0126% | 1.1407% | **+12.65%** | 失败：最多 +5% |
| 最大 gap band accuracy | 34.5% | 40.9% | +6.4 pp | 诊断 |
| 上边缘 k accuracy | 8.9% | 35.7% | **+26.8 pp** | 诊断 |
| joint band-k accuracy | 7.7% | 22.8% | **+15.1 pp** | 通过 |
| 工作频率 MRE | 17.7614% | 15.1620% | -14.63% | 通过 |
| gap width MAE | 0.006699 | 0.006359 | -5.08% | 通过 |
| gap center MAE | 0.091928 | 0.084099 | -8.52% | 通过 |

## 解释与失败模式

事实是 k-profile 项有效地改变了优化方向：两条 lane 的上边缘 k accuracy 和 joint band-k accuracy 都大幅提高，因此不能把结果解释成“损失找不到工作频率点”。失败发生在代价分配上：辅助项总权重使模型在 20 epoch 内牺牲了过多全谱拟合，Binary 的中心误差也没有随 k 点与宽度同步改善。

另一个更严格的限制是 complete-gap 状态：真实 complete-gap 在 Binary/Graded validation 中只占 `20.7%/5.8%`，两个 arm 对该子集的 `status_accuracy` 都为 `0`。新损失虽然提高了 complete-gap 的 band/k 选择，但仍没有可靠地把预测 gap 推到正值。因此它当前更准确地描述为“改善最大 gap 决策位置的损失”，还不能称为“可靠找到 complete gap 的损失”。

## 决策

1. 停止 EXP-FWD-014 的自动三 seed 扩展，不读取 held-out test。
2. 保留 `selected_upper_profile` 实现，因为它对 k 点有明确正向证据。
3. 下一假设应同时解决两个问题：降低辅助项对全谱的竞争，并对稀少 complete-gap 样本加入显式的正 gap/status 约束。任何新权重或新项必须登记新实验并获得新的 N40 训练授权，不能在本 run 上追调。

原始证据位于 N40：`runs/20260828/EXP-FWD-014_gap_screen_20260828101706/`。其中 `comparison/metrics.json` 是本页数值来源，两个训练目录保留冻结配置、历史、日志、指标和 checkpoint。
