---
title: "EXP-PHY-007：validation 带隙标签门"
summary: "对 K5 Binary/Graded 全部 64 个冻结 validation 结构执行自适应全 BZ band-edge 加密，决定是否解锁一个 seed0 loss 短筛。"
page_type: experiment
plan_state: active
status: draft
audience: "光子晶体物理建模研究者、正向代理研究者与结果审阅者"
owner: "研究生维护，导师审阅标签门与训练准入"
last_reviewed: 2026-09-02
primary_layer: L3
---

<!--
  EXP-PHY-007 experiment contract.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-PHY-007：validation 带隙标签门

**本实验只判断精细带隙标签能否在冻结 validation 集上稳定生成。** 它不读取 test、不训练模型，并且不把有限层局部自适应搜索表述为连续 BZ 的数学全局证明。

## 冻结合同

| 项目 | 值 |
|---|---|
| 样本 | EXP-PHY-005 冻结 manifest 中全部 Binary 32 + Graded 32 validation 结构 |
| 基线 | HDF5 现有 $9\times9$、Bands 1--10 real-MPB `Wh` |
| 自适应搜索 | 对 B1--B2 至 B9--B10 的 lower maximum 与 upper minimum 周围反复放置周期 $3\times3$ stencil |
| 层数 | 至少 2 层，最多 4 层；连续 2 层所有 edge frequency shift 不超过 $5\times10^{-4}\,a/\lambda$ 才提前停止 |
| 分辨率复核 | 最终上下边缘在 resolution 96 重算；相对 resolution 64 的 shift 不超过 $5\times10^{-4}\,a/\lambda$ |
| 完整输出 | 九个相邻 pair 的上下边、中心、signed width、complete/pseudo 状态、粗到精翻转与逐层历史 |
| 禁止项 | test、训练、自动重试、改阈值、改样本、把局部搜索升级为连续全局认证 |

## 标签门

只有 64/64 结构求解成功，且每个 pair 同时满足自适应 edge 收敛与 resolution 收敛，终态才是 `adaptive_label_gate_pass`。否则为 `adaptive_label_gate_fail` 或 `blocked`，并停止 loss 训练。

通过仅解锁一个已预注册的 EXP-FWD-016 seed0 配对短筛；不自动解锁扩 seed、held-out test 或六域评估。

正式配置为 [`exp_phy_007_validation_gap_refinement_v1.yaml`](../../configs/research/exp_phy_007_validation_gap_refinement_v1.yaml)，上游 pilot 见 [EXP-PHY-006](EXP-PHY-006-全BZ带隙边缘加密试点.md)。
