---
title: "EXP-FWD-012：工作频率先验覆盖审计"
summary: "X/M 一阶 Bragg 尺度能覆盖部分谱坐标，但完全未命中 K5 Binary/Graded 最大 gap center，因此 loss-only 训练不准入。"
page_type: evidence
status: draft
audience: "正向代理研究者、实验执行者与结果审阅者"
owner: "研究生维护，导师审阅负结果"
last_reviewed: 2026-08-27
reading_time: 3 minutes
primary_layer: L3
evidence_date: 2026-08-27
evidence_scope: "EXP-FWD-012 K5 Binary/Graded fair 4k+4k train and 1k+1k validation prior audit; no test targets, training, MPB, or Meep"
---

<!--
  EXP-FWD-012 evidence summary.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-FWD-012：工作频率先验覆盖审计

**结论：预注册的工作频率先验门失败，不进入 EXP-FWD-013 loss-only 训练。** $X$ 方向一阶 Bragg 窗口在 Binary/Graded validation 的谱坐标覆盖率为 `90.9%/91.4%`，但 $X$ 与 $M$ 两个先验对最大 gap center 的命中率均为 `0%`。同宽 train-only 固定中心对照仍命中 `1.4%/2.4%`，因此结构条件先验相对对照分别退化 `1.4/2.4` 个百分点，远低于预注册的 `+5` 个百分点门。

| Validation 指标 | Binary | Graded | Gate |
|---|---:|---:|---:|
| $X$ 窗口谱坐标覆盖率 | 90.9% | 91.4% | ≥85%：通过 |
| $X$ 最大 gap-center 命中率 | 0.0% | 0.0% | — |
| $M$ 最大 gap-center 命中率 | 0.0% | 0.0% | 敏感性，不参与主门 |
| 固定中心最大 gap-center 命中率 | 1.4% | 2.4% | 对照 |
| $X$ 相对固定中心改善 | -1.4 pp | -2.4 pp | ≥+5 pp：失败 |

## 为什么失败

train-only 审计得到 $\Delta_f=0.003125$。validation 中 $X$ 先验中位数约为 Binary `0.208`、Graded `0.207`，而最大 gap center 中位数约为 `0.654/0.654`；两者中位绝对距离为 `0.455/0.443`。这不是轻微窗口宽度偏差，而是低阶第一 Bragg 尺度与当前 lower-band 2–9 最大 gap 目标位于不同频率区间。

complete gap 占比仅为 Binary `20.7%`、Graded `5.8%`，但 complete/pseudo 均使用同一最大-gap 选择合同；该比例不改变两个先验零命中的判定。

## 学到什么与下一决策

- 平均折射率的一阶 Bragg 尺度可以靠近完整 `Wh` 中的某些坐标，但不能据此声称它对齐最大带隙或下游选频。
- 不应通过扩大 $\Delta_f$、移动中心或读取 test 来挽救本假设；那会改变已经冻结的研究问题。
- EXP-FWD-013 的四个训练任务保持未启动。若继续工作频率路线，需要人工提出新的、独立的 L1 假设和新的 Experiment ID，而不是修改 EXP-FWD-012 gate。

原始证据：`runs/20260827/EXP-FWD-012_202608271707/`。该 run 保存 resolved config、数据与子集 hash、`work_frequency_prior.json`、`work_frequency_metrics.json`、标准 metrics/results/meta；`test_targets_loaded=false`，未启动训练或 real solver。

实验合同：[EXP-FWD-012 协议](../../../experiments/research/EXP-FWD-012-工作频率先验覆盖审计.md)。
