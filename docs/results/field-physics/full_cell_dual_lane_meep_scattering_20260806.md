---
title: "双路线全胞有限阵列 Meep 近远场结果"
summary: "日期化实验与验证证据；正文保持原始记录。"
page_type: evidence
status: draft
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 4 minutes
primary_layer: L3
evidence_date: 2026-08-06
evidence_scope: "日期化证据或历史记录；正文未因站点重构而改写。"
dataset_scope: full_cell_bz9_wh81x10
evidence_level: physical_validation
evidence_status: completed
validation_backend: wsl_meep
source_artifacts:
  - runs/20260806/full_cell_dual_lane_meep_scattering_v1_202608061700
---

<!--
  Real-Meep finite-array near/far-field response result report.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 双路线全胞有限阵列 Meep 近远场结果

| Metadata | Value |
|---|---|
| Status | Draft experimental technical report |
| Primary layer | L3 experiment/evidence |
| Audience | 二维光子晶体研究者、复核者与图件使用者 |
| Owner | 研究生维护；散射合同与论文主张由导师确认 |
| Last reviewed | 2026-08-06 |
| Reading time | 6 minutes |

## 结论

在冻结的 5×5 有限阵列、从左向右正入射 TE 脉冲、四周 PML 和逐样本带隙上边缘中心频率条件下，渐变介质与二值空气孔两套数据各 16 个样本均完成了真实 Meep 时域计算。全部 32 个样本都在最大仿真时间之前达到 DFT 衰减阈值；近场电能密度、时间平均 Poynting 矢量和 near-to-far 径向通量均为有限值。

结果说明不同全胞结构会形成明显不同的局域热点、能流偏折和角向辐射瓣。但这些图是冻结有限阵列与入射条件下的受激响应，不是无限周期晶体的固有“唯一近场/远场”，也不能替代渐变数据集尚未完成的正式跨环境验收。

## 计算合同

- 结构：每个解析全胞沿 x/y 各复制 5 次，阵列外为空气。
- 激励：从左向右（+x）传播的二维 TE Gaussian `Hz` 线源。
- 频率：每个样本完整 BZ 最大正带隙的上边缘频率；无正带隙时沿用已冻结的最大伪带隙回退规则。
- 网格与边界：`resolution=24`，四周 1.0 个晶格常数厚的 PML。
- 近场：`ε(|Ex|²+|Ey|²)/4`，逐样本最大值归一化；白色箭头为时间平均 Poynting 矢量，青色箭头为入射方向。
- 远场：闭合空气监测框的 Meep near-to-far 变换，在 100 个波长半径、181 个角度上计算向外径向 Poynting 通量，并逐样本归一化。
- 收敛：DFT 衰减阈值 `1e-5`，最大 Meep 时间 2000；未提前收敛即 fail closed。

## 正式结果与数值审计

正式运行位于 `runs/20260806/full_cell_dual_lane_meep_scattering_v1_202608061700/`。

| 检查项 | 结果 |
|---|---:|
| 样本数 | 32（渐变 16，二值 16） |
| 响应分片 | 32 |
| 正式图文件 | 15（5 张图版 × PNG/SVG/PDF） |
| 累计 Meep 求解器耗时 | 915.850 s |
| 单样本平均 / 最大耗时 | 28.620 s / 59.858 s |
| DFT 提前达到衰减阈值 | 32/32 |
| Meep 结束时间范围 | 281.833–1632.333 |
| 近场、远场与 Poynting 数组有限 | 通过 |
| 逐样本近场、远场最大值 | 均为 1.0 |
| 裁剪前负径向通量比例 | 32 个样本均为 0 |

最后一次仅排版调用复用了全部 32 个已验证响应分片，没有重复执行 Meep；因此 `metrics.json` 同时记录累计求解耗时和本次 `response_shard_reuse_count=32`，避免把快速重绘误认为重新完成了物理计算。

## 图件说明

- `graded_medium_meep_near_field` 与 `binary_air_holes_meep_near_field`：两套 4×4 近场图版，亮度表示归一化电能密度，青色线表示介质边界，白色箭头表示能流方向。
- `graded_medium_meep_far_field` 与 `binary_air_holes_meep_far_field`：两套 4×4 极坐标远场图版，+x 为前向、−x 为后向。
- `full_cell_dual_lane_meep_response_overview`：把同编号样本的渐变近场、渐变远场、二值近场和二值远场集成为一张大图。

所有图均提供 600 dpi PNG、保留文字对象的 SVG 和 PDF。逐样本归一化只适合比较空间形貌或瓣形，不能用于比较样本间绝对场强、透射率、反射率或散射截面。

## 人类理解记录

| 问题 | 记录 |
|---|---|
| 解决了什么 | 把先前无源 Bloch 本征场图与真实受激响应区分开，并为冻结样本生成带能流箭头的近场和角向远场。 |
| 为什么采用此方案 | 远场必须先明确有限结构、源、开放边界和监测面；5×5 阵列与闭合 near-to-far 面构成了可复现的最小物理合同。 |
| 改动位于哪里 | L3 求解、绘图、研究配置、实验说明、结果报告与对应测试；未修改源 HDF5 和 MPB 频带。 |
| 哪些证据支持结论 | 32 个响应分片、HDF5 原始数组、CSV lineage、收敛元数据、指标文件、图件 manifest 和 WSL `meep_env` 测试。 |
| 最大剩余风险 | 当前未做空场功率基准，因此远场仅是归一化角向分布；若论文需要 R/T 或绝对散射截面，需另做参考仿真与功率归一化。 |

## 证据边界

二值数据保留既有 `wsl_cross_environment_acceptance_pass`；渐变数据继续保持 `cluster_generated_pending_wsl_acceptance`。本次 32 个真实 Meep 响应证明的是该散射合同可执行且结果收敛，不提升任何源数据集的验收状态。

[← 返回实验合同](https://github.com/fangtoast/phc2d/blob/main/experiments/research/full_cell_dual_lane_meep_scattering.md)
