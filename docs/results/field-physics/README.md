---
title: Field 与物理响应结果
summary: 当前 real-MPB/Meep 响应证据与历史 Field/PINO 准入记录的分层索引。
page_type: landing
status: draft
audience: 物理建模研究者、复现执行者与证据审阅者
owner: 研究生维护
last_reviewed: 2026-08-29
reading_time: 4 minutes
primary_layer: L3
---

<!--
  Field and physics result landing page.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# Field 与物理响应结果

!!! warning "证据分层"
    下列 Field/PINO 页面不进入当前正式结果白名单。它们保留失败、修正与准入过程；全胞 Meep 与代理选频响应差异属于独立的正式物理证据。

## 当前物理证据

| 结果页面 | 状态与边界 |
|---|---|
| [validation 带隙标签门](EXP-PHY-007-validation带隙标签门.md) | EXP-PHY-007；64 个 validation 结构全部完成，但仅 18 个结构同时满足 edge 与 resolution 稳定，标签门失败并停止 loss 训练 |
| [全 BZ 带隙边缘加密试点](EXP-PHY-006-全BZ带隙边缘加密试点.md) | EXP-PHY-006；4 个 validation 结构 real-MPB pilot，发现 $9\times9$ gap 高估与一次 complete→pseudo 翻转，不等同于全局 BZ 认证 |
| [TE 有效折射率与折叠频率校准](EXP-PHY-005-TE有效折射率与折叠频率校准.md) | EXP-PHY-005；192 结构 near-Γ real-MPB 正式负结果，单一标量中心未通过 |
| [K5 代理选频与真实响应差异](代理选频与真实响应差异_20260825.md) | EXP-PHY-004；1000 条 test 频谱审计 + 6 条代表结构 real-MPB/Meep sensitivity |
| [双路线有限阵列 Meep 近远场](full_cell_dual_lane_meep_scattering_20260806.md) | EXP-PHY-002；冻结 32 结构的真实受激响应 |

## 历史 Field / PINO 证据

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

[返回实验与证据库](../README.md) · [查看历史档案](../../历史档案.md)
