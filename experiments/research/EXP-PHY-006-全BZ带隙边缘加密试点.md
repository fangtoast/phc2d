---
title: "EXP-PHY-006：全 BZ 带隙边缘加密试点"
summary: "用四个冻结 validation 结构初步检验 9×9 BZ 网格是否高估 Bands 1–10 的相邻带隙。"
page_type: experiment
plan_state: active
status: draft
audience: "光子晶体物理建模研究者与正向代理研究者"
owner: "研究生维护，导师审阅科学结论"
last_reviewed: 2026-08-29
primary_layer: L3
---

<!--
  EXP-PHY-006 experiment contract.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-PHY-006：全 BZ 带隙边缘加密试点

**本实验只判断现有 $9\times9$ BZ 标签是否存在足以改变 gap 解释的局部离散误差。** 它使用 EXP-PHY-005 冻结 validation 清单中的 Binary/Graded 各两个结构，不读取 test、不训练、不使用 N40，也不把局部极值加密写成整个连续 BZ 的数学认证。

## 实验合同

| 项目 | 冻结值 |
|---|---|
| 样本 | Binary validation 2 + Graded validation 2；显式 dataset index 与 geometry hash |
| 基线 | HDF5 中现有 $9\times9$、Bands 1–10 real-MPB `Wh` |
| 加密 | 对 B1–B2 至 B9–B10 的 lower maximum 与 upper minimum 分别做两轮周期 $3\times3$ 局部 stencil |
| 步长 | $1/18$、$1/36$ reduced reciprocal coordinate |
| 求解 | TE，resolution 64，mesh 3，tolerance $10^{-7}$，10 bands + 4 guards |
| 分辨率复核 | 最终边缘 k 点使用 resolution 96 重算 |
| 输出 | 全部九组 pair 的 coarse/refined edge、center、signed width、status 与局部不确定度 |

## 判定边界

- 任一 pair 的 complete/pseudo 状态翻转，或 signed width 变化超过 $5\times10^{-4}\,a/\lambda$，记为 `coarse_grid_risk_detected`。
- 两轮末变化与 resolution 64→96 边缘变化都不超过 $5\times10^{-4}\,a/\lambda$，只记为 `pilot_local_refinement_stable`。
- 后一种状态仍不能证明全局极值已经找到；它只允许提出 64 结构全 BZ 自适应认证，不允许直接解锁新的 loss 训练。
- 任一结构 solver 失败则为 `blocked`，不自动重试或改变资源。

正式配置为 [`exp_phy_006_full_bz_gap_refinement_pilot_v1.yaml`](../../configs/research/exp_phy_006_full_bz_gap_refinement_pilot_v1.yaml)。
