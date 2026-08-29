---
title: "EXP-PHY-006：全 BZ 带隙边缘加密试点结果"
summary: "四个 validation 结构的 real-MPB 局部加密表明 9×9 BZ 标签会实质高估相邻带隙，并出现一次 complete-to-pseudo 翻转。"
page_type: evidence
status: draft
audience: "光子晶体物理建模研究者、正向代理研究者与结果审阅者"
owner: "研究生维护，导师审阅科学结论"
last_reviewed: 2026-08-29
reading_time: 5 minutes
evidence_date: 2026-08-29
primary_layer: L3
evidence_scope: "EXP-PHY-006 K5 Binary/Graded validation-only 2+2 structure local band-edge refinement pilot; no test, training, or N40"
dataset_scope: "K5 Binary/Graded Wh[10000,81,10], four samples frozen from the EXP-PHY-005 validation manifest"
evidence_level: physical_validation
evidence_status: completed
validation_backend: wsl_mpb
source_artifacts:
  - runs/20260829/EXP-PHY-006_202608291923
---

<!--
  EXP-PHY-006 curated result.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-PHY-006：全 BZ 带隙边缘加密试点结果

**结论：现有 $9\times9$ BZ 标签不足以承担“精准 complete gap”监督。** 四个冻结 validation 结构中，32/36 个相邻 band pair 的 signed width 变化超过预注册的 $5\times10^{-4}\,a/\lambda$ 阈值；Binary index 96 的 B8–B9 从粗网格 complete gap 变为加密后的 pseudo-gap。终态为 `coarse_grid_risk_detected`，相关 gap-prior 训练继续锁定。

这不是“所有现有 gap 都错误”的结论。该试点只在粗网格所指向的局部极值附近做两轮加密；它足以证伪“$9\times9$ 标签天然精确”，但不足以证明已经找到连续 BZ 的全局极值。

## 实验与制品完整性

| 项目 | 正式证据 |
|---|---|
| 正式 run | `runs/20260829/EXP-PHY-006_202608291923/` |
| 代码身份 | `38e8c8c09e8852015bb55cf6e1f957107db61c00` |
| 样本 | Binary validation index 89、96；Graded validation index 62、337 |
| 求解合同 | TE；resolution 64；mesh 3；tolerance $10^{-7}$；10 bands + 4 guards |
| 加密合同 | B1–B2 至 B9–B10；两轮周期 $3\times3$ 局部 stencil；末端 resolution 96 复核 |
| 完整性 | 4/4 样本、36/36 pair、零 solver failure；NPZ 对 CSV 独立复算最大差异为 0 |
| 禁止项 | `test_targets_loaded=false`、`training_started=false`、`n40_used=false` |

首次求解制品 `runs/20260829/EXP-PHY-006_202608291913/` 暴露了 solver adapter 对方形 `all_freqs` 数组的误转置问题，因此不作为科学结论来源。修复加入方阵方向回归后，以上述 `1923` run 重新求解；旧 run 保留为可追溯的污染制品。

## 总体结果

| 指标 | 结果 | 解释 |
|---|---:|---|
| pair 总数 | 36 | 4 个结构 × 9 个相邻 pair |
| materially changed pair | 32/36 | signed width 变化超过 $5\times10^{-4}\,a/\lambda$ |
| complete/pseudo 翻转 | 1 | Binary 96 的 B8–B9：complete→pseudo |
| 最大粗网格 gap 高估 | 0.02980 | 发生在 Graded 337 的 B1–B2 |
| 最大中心频率移动 | 0.00806 | 粗网格中心也不是稳定的精确中心 |
| 末轮局部加密稳定 | 23/36 | 其余 pair 仍需继续加密 |
| resolution 64→96 稳定 | 30/36 | 6 个 pair 的边缘变化仍超过阈值 |
| 两项同时稳定 | 21/36 | 不能把本 pilot 升级为全局认证 |

Binary 96 的 B8–B9 粗网格 signed width 为 `+0.0041515`，加密后为 `-0.0017374`，变化为 `-0.0058889`。但其估计局部不确定度为 `0.0023873`，大于加密后 overlap 的绝对值，因此保守分类是 `numerically_ambiguous`。准确表述是：粗网格 complete 标签不可靠；尚不能用当前两轮加密宣称该 pair 已完成严格的全局 pseudo-gap 认证。

## B1–B2 与折叠频率先验

| Lane / index | $9\times9$ signed width | 加密后 signed width | 局部判定 |
|---|---:|---:|---|
| Binary 89 | -0.04233 | -0.06946 | certified no-gap |
| Binary 96 | -0.04127 | -0.05739 | certified no-gap |
| Graded 62 | -0.02302 | -0.05147 | certified no-gap |
| Graded 337 | -0.03858 | -0.06838 | certified no-gap |

四个样本的 B1–B2 都不是 complete gap，而且局部加密后 band overlap 进一步增大。这进一步澄清 EXP-PHY-005 中 `x=1/(2n_eff)` 的高 pair 一致率：它可以作为“关注低阶 B1–B2”的弱 band-pair prior，却没有提供“该处存在 complete gap”或“已找到精确 gap 中心”的证据。

因此，若损失只想提高 B1–B2 能带预测精度，可以把它登记为 band-focused weighting 并独立验证；若目标是预测 complete gap、gap center 或可工作的频率窗口，就必须以加密后的全 BZ band edges 为监督，不能把 pair identity 当作 gap 命中。

## 决策与下一门

- 不使用现有 $9\times9$ complete/pseudo 标签直接解锁 gap-prior 或 resonance-prior 训练。
- 不把这 4 个结构外推成 Binary/Graded 全体分布结论，也不声称已完成连续全 BZ 认证。
- 下一物理门应是冻结 64 个 validation 结构的全局自适应 BZ 搜索：每轮围绕所有候选极值加密，直到 edge shift 与 resolution shift 同时收敛；随后再比较粗/精标签的状态、中心和 band-pair 分布。
- 只有扩大验证表明精标签可稳定生成后，才讨论训练标签重建与 N40 配对任务；该扩大求解需要新的人工授权。

原始 k 点与能带见 `refined_bands.npz`，逐 pair 指标见 `gap_refinement_samples.csv`，总体统计见 `metrics.json`。本结果不改写 EXP-PHY-005，也不产生 surrogate 性能结论。
