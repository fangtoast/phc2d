---
title: Field 与 PINO 历史结果
summary: 64/1k/9216 规模、dense-k、物理残差和度规修正的历史证据索引。
page_type: landing
status: draft
audience: 物理建模研究者、复现执行者与证据审阅者
owner: 研究生维护
last_reviewed: 2026-08-14
reading_time: 4 minutes
primary_layer: L3
---

<!--
  Field and physics result landing page.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# Field 与 PINO 历史结果

!!! warning "历史证据"
    下列 Field/PINO 页面不进入当前正式结果白名单。它们保留失败、修正与准入过程；全胞 Meep 结果请从[新全胞数据正式结果](../全胞数据正式结果.md)进入。

| 结果页面 | 状态与边界 |
|---|---|
| [B1：64 条真实场数据](field_eigen_te_b1_64_20260720.md) | real-MPB field-data pilot |
| [B2：1k 数据与 residual gate](field_eigen_te_b2_b1_1000_20260720.md) | 数据扩展与准入检查 |
| [监督基线](field_eigen_te_supervised_baseline_20260720.md) | 首轮 data-only 负结果 |
| [B3 与 adapted PINO](field_eigen_te_b3_v2_pino_20260720.md) | 度规修正前的历史证据 |
| [B4 dense-k 准入](field_eigen_te_b4_dense_k_admission_20260722.md) | frequency、dense-k 与 inverse 准入 |
| [P6 target-residual pilot](field_eigen_te_p6_seed0_target_residual_pilot_20260729.md) | 单 seed pilot；准入未通过 |
| [电场度规勘误](field_electric_metric_correction_20260803.md) | 当前介电加权指标合同；旧指标需要重算 |

较低 Maxwell residual 不能单独证明 frequency、inverse reranking 或 field inverse 收益。

[查看正式 Meep 结果](full_cell_dual_lane_meep_scattering_20260806.md) · [返回历史档案](../../历史档案.md)
