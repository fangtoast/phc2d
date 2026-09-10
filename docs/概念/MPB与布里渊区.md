---
title: Bloch 条件、布里渊区与 MPB
summary: 解释周期边界、k 点采样、V2 路径和 full-cell 全 BZ 网格。
page_type: explanation
status: draft
audience: 数据生产者、研究使用者与模型开发者
owner: 研究生维护，导师审阅物理表述
last_reviewed: 2026-08-14
reading_time: 8 minutes
primary_layer: L2
---

<!--
  Bloch, Brillouin-zone, and MPB concepts for phc2d.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="bloch-bz-mpb"></a>

# Bloch 条件、布里渊区与 MPB

本页解释 k 点为什么属于数据合同，以及 V2 路径和 full-cell 全 BZ 网格为什么不能直接混表。

## Bloch 周期条件

周期介质中的本征场满足

$$
\mathbf H_{n\mathbf k}(\mathbf r+\mathbf R)
=e^{i\mathbf k\cdot\mathbf R}\mathbf H_{n\mathbf k}(\mathbf r).
$$

因此同一结构需要在一组 $\mathbf k$ 上求解，才能得到 `Wh[N_k,N_b]`。k 坐标、顺序、重复端点和路径累计弧长都必须进入 metadata。

## 两种 k 采样

| 合同 | 采样 | 目标 shape | 作用 |
|---|---|---|---|
| V2 | Γ–X–M–Γ，31 个索引 | `[N,31,10]` | 路径 band 与 Query 基线 |
| full-cell | $9\times9$ 完整第一 BZ | `[N,81,10]` | 全胞、多形态和全 BZ 学习 |

V2 的路径端点可能包含重复 Γ；full-cell 则保存二维 `k_points` 和 `k_grid_index`。模型必须读取数据合同，不能假设 `31` 或 `81` 是通用常量。

## MPB 在项目中的职责

MPB 负责从解析介质和 k 点生成本征频率或本征场。数据页需要记录 solver 参数、resolution、band 数、极化、结构 hash、求解失败和输出排序。

关键实现入口见[核心代码导航](../参考/核心代码导航.md)，实际执行步骤见[数据生成](../指南/数据生成.md)。
