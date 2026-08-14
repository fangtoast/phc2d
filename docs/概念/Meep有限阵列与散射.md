---
title: Meep 有限阵列与散射
summary: 解释周期本征问题与有限结构近远场计算的差异和证据边界。
page_type: explanation
status: draft
audience: 物理建模研究者、数据生产者与结果审阅者
owner: 研究生维护，导师审阅物理表述
last_reviewed: 2026-08-14
reading_time: 6 minutes
primary_layer: L2
---

<!--
  Meep finite-array scattering explanation for phc2d.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="meep-finite-array-scattering"></a>

# Meep 有限阵列与散射

MPB 的周期本征问题和 Meep 的有限阵列时域散射回答不同问题。本页解释两者如何连接，不把近远场图自动解释为带隙或 inverse 成功。

## 计算对象

有限阵列散射需要明确：阵列尺寸、源类型和频带、PML、空间分辨率、监测面、运行时间以及 near-to-far 变换。结果通常包括场快照、能量密度、能流和角向远场响应。

## 与 MPB 的关系

- MPB 给出无限周期介质在 Bloch 条件下的本征频率和场。
- Meep 给出有限结构在指定激励和开放边界下的时域响应。
- 两者可以相互解释，但一个计算的通过不能自动替代另一个计算的 gate。

## 当前证据

[双路线全胞有限阵列 Meep 近远场结果](../results/field-physics/full_cell_dual_lane_meep_scattering_20260806.md)记录了固定散射合同下的 32 个真实 Meep 响应。该证据证明合同可执行且结果收敛，不升级 full-cell inverse 或源数据集的其他验收状态。

执行环境必须是 WSL `Ubuntu` + conda `meep_env`；操作入口见[超算物理任务](../指南/超算物理任务.md)。
