---
title: "EXP-FWD-015：gap 决策近简并审计结果"
summary: "严格命中率低主要来自模型决策能力，而不是标签近简并；据此进入排序与完整带隙召回损失修改。"
page_type: evidence
status: draft
audience: "正向代理研究者、实验执行者与结果审阅者"
owner: "研究生维护，导师审阅科学结论"
last_reviewed: 2026-08-28
reading_time: 3 minutes
evidence_date: 2026-08-28
primary_layer: L3
evidence_scope: "EXP-FWD-015 K5 Binary/Graded fair 4k+4k train-derived tolerances and 1k+1k validation audit of two frozen EXP-FWD-014 checkpoints; no test, training, MPB, or Meep"
dataset_scope: "K5 Binary/Graded Wh[10000,81,10] frozen train and validation subsets"
evidence_level: surrogate_held_out
evidence_status: completed
validation_backend: surrogate_only
source_artifacts:
  - runs/20260828/EXP-FWD-015_gap_margin_audit_20260828150000_wsl
---

<!--
  EXP-FWD-015 curated result.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-FWD-015：gap 决策近简并审计结果

**结果：低正确率不是“正常工作频率不存在”，也不是严格指标被近简并大幅压低；主要问题是模型尚未稳定学会候选 gap 排序和 complete-gap 正号。** 最大容差 joint accuracy 增益为 `3.0` 个百分点，低于预注册的 `5` 点门，状态为 `model_decision_limited`。

## 与严格指标相比

| Lane | 模型 | 严格 joint | 容差 joint | 非近简并严格 joint | complete-gap 召回 |
|---|---|---:|---:|---:|---:|
| Binary | baseline | 13.5% | 16.0% | 13.9% | 36.2% |
| Binary | gap-aware v1 | 24.4% | 27.3% | 27.7% | 9.7% |
| Graded | baseline | 7.7% | 10.6% | 7.6% | 0.0% |
| Graded | gap-aware v1 | 22.8% | 25.8% | 24.8% | 0.0% |

Binary/Graded 的 gap 近并列率分别为 `10.5%/9.6%`，上边工作频率 k 近并列率为 `11.3%/10.2%`。排除这些样本后，v1 的严格 joint 仍只有 `27.7%/24.8%`，因此不能把低正确率解释为标签歧义。

## complete gap 暴露出的偏置

Binary 的真实 complete-gap 比例为 `20.7%`，v1 只预测 `3.9%`，召回 `9.7%`；Graded 的真实比例为 `5.8%`，v1 完全没有预测出 complete gap。v1 虽改善 k 点，却倾向把 gap 宽度压到非正值，这解释了为何工作频率方向改善而完整带隙状态仍失败。

## 决策

不修改严格评价口径。后续损失直接加入候选 gap 排序间隔和 complete-gap 条件召回项，并降低会牺牲全谱 MRE 的重复绝对回归权重。该修改只形成 EXP-FWD-016 的一个 N40 短筛任务；尚未产生新的训练性能结论。

原始证据：`runs/20260828/EXP-FWD-015_gap_margin_audit_20260828150000_wsl/`，其中 `metrics.json` 保存完整两条 lane、两个模型的审计指标，`run.meta.json` 保存数据、子集、validation 与 checkpoint 哈希。未读取 held-out test，未训练，未运行 MPB/Meep。
