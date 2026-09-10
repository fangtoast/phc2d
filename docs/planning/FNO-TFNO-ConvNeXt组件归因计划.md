---
title: FNO、TFNO 与 ConvNeXt 组件归因计划
summary: 用冻结诊断、同家族消融和最小桥接实验，判断三个较优模型究竟由哪些组件受益。
page_type: plan
status: draft
audience: 正向代理研究者、模型实现者与结果审阅者
owner: 研究生维护，导师审阅组件假设与训练准入
last_reviewed: 2026-09-08
reading_time: 5 minutes
primary_layer: L1
plan_state: active
effective_date: 2026-09-01
superseded_by: null
related_results:
  - docs/results/forward/EXP-FWD-009-K5权重热启动与六域验证.md
  - docs/results/forward/full_cell_forward_model_upgrade_50k_evaluation_20260810.md
  - docs/results/forward/全胞K20与50K跨数据域正向验证_20260820.md
  - docs/results/forward/EXP-FWD-019-组件归因三种子确认.md
---

<!--
  Focused component-attribution plan for FNO, TFNO, and ConvNeXt.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# FNO、TFNO 与 ConvNeXt 组件归因计划

## 结论

现有结果只能说明 FNO、TFNO、ConvNeXt 在若干域上表现较好，不能说明“Fourier 一定更符合物理”或“局部卷积就是原因”。三个模型还同时改变了输出头、容量、归一化、下采样和优化正则化。

本计划要回答：

> 频谱全局混合、Tucker 参数化、周期局部卷积、层级下采样和输出头中，哪些组件真正带来稳定收益？

组件路线固定使用原 `normalized_full_matrix_mse`，不与 loss 改进混在一起。用户已于 2026-09-02 授权按本文漏斗门自主提交 N40 训练与过门后的 seeds 1/2；不得扩展到计划外改参、资源变化、loss 路线或物理 replay。

## 要验证的猜想

| 编号 | 猜想 | 当前信号 | 如何证伪 |
|---|---|---|---|
| C1 | Fourier 全局混合有利于 K5/native 全谱建模 | TFNO v3 是 K5 native 主指标候选，FNO v2 在部分迁移方向较稳 | 删除 spectral 后没有稳定退化，或收益被输出头解释 |
| C2 | 周期局部大核与层级特征有利于复杂材料边界 | ConvNeXt 在 50K native 两条 lane 最优，并在 Graded Band 1 有优势 | 大核、circular padding 或层级下采样消融后无稳定变化 |
| C3 | Tucker 的主要价值可能是参数效率，不一定是额外精度 | TFNO 同时改变 factorization、modes、width、depth 和局部分支 | rank 对照在参数/计算控制后没有差异 |
| C4 | 一部分排行榜差异来自 head、容量和优化，而不是 backbone | FNO v2 使用单层 Linear head，TFNO/ConvNeXt 使用共同 dense head | 共同 head 与资源控制后排序完全不变 |
| C5 | spectral/local/channel 的最佳比例随结构或域变化 | Joint 已有三支路，但 gate 是每层固定标量 | 支路贡献在样本、lane 和域之间没有稳定差异 |

K5 与 K20 是独立数据，F50 训练预算也不同。因此这些结果只能提出猜想，不能把数据域差异直接解释成 Fourier cutoff 或组件因果。

## 当前组件差异

| 模型 | 主要组件 | 需要控制的混杂 |
|---|---|---|
| FNO v2 | Fourier spectral、pointwise、额外小 modes 权重、band gate、低秩全局通道摘要 | 单层 Linear head，无局部 $k>1$ 卷积 |
| TFNO v3 | 双侧 Tucker spectral、circular $3\times3$ local、channel branch、GroupNorm/DropPath | spectral、local、factorization 和容量同时变化 |
| ConvNeXt | circular depthwise $7\times7$ block、三 stage、stem 与两次下采样、channel expansion | 只有 block 卷积显式 circular；stem 和层级压缩需单独检验 |

## 要做什么

### C0：冻结模型诊断，不训练

先复用已有 checkpoint：

- 按输入高频能量、边界密度、灰度比例和 lane 分层统计逐样本误差；
- 对输入做整数像素周期平移，检查预测是否保持不变；
- 对输入低/高 Fourier 模态做预注册遮罩，只记录 prediction delta；
- 对 FNO、TFNO、Joint 做冻结支路屏蔽，定位值得训练消融的组件。

冻结屏蔽会造成分布偏移，只用于筛候选，不作为组件因果结论。

### C1：先控制公平性

1. 给 FNO v2 建立与 TFNO/ConvNeXt 同语义的 dense-head 对照，新身份从零训练，不覆盖旧 checkpoint。
2. 同时报告原配置结果和资源控制结果，包括参数量、有效训练 step、峰值显存、训练时间和推理吞吐。
3. 每个消融只与同家族直接基线比较；跨家族结论还需相同优化器搜索预算。

### C2：同家族最小消融

| 家族 | 第一波候选 | 回答的问题 |
|---|---|---|
| FNO v2 | 去额外小 modes 权重；去 band gate；去 spectral、仅留 pointwise | 双频谱、门控和 Fourier 混合分别是否必要 |
| TFNO v3 | no-spectral；no-local；no-channel；rank=1.0 对照 | 三支路和 Tucker rank 各自作用 |
| ConvNeXt | kernel $7\to1$；circular $\to$ zero padding；去层级下采样的 bridge | 大核、周期边界和层级压缩各自作用 |
| Joint | no-spectral；no-local；no-channel | 在同一个 block 内复核三类组件方向 |

每个家族最多先选两个最有信息量的 seed0 候选，不一次运行完整矩阵。参数减少时，同时报告 raw ablation 和 parameter-matched control。

### C3：只确认稳定组件

seed0 通过后，每个家族最多保留一个候选补 seeds 1/2。只有“冻结诊断、同家族消融、最小 bridge”三类证据方向一致，才能称为组件贡献，并进入六域冻结评估。

## 看哪些指标

- Band 1、Bands 2--10、Bands 1--10 MRE/MAE；
- band、边缘 $k$、gap width/center 与 complete-gap 状态；
- K5/K20/F50 × Binary/Graded 六域逐域变化；
- 参数量、显存、训练/推理时间和收敛状态；
- 周期平移误差及不同结构属性分层结果。

## 通过与停止条件

- 三 seeds 方向一致，且参数/计算控制后仍有收益，才通过。
- Binary/Graded 任一 lane 出现实质退化，不给出通用组件结论。
- 同家族消融与 bridge 方向不一致，停止该猜想。
- 若共同 head 已解释主要差距，优先以“head 混杂”结题，不继续扩大 backbone。
- 若支路贡献不随样本或域变化，不开发自适应 gate。

## 可能的后续模型

只有 C5 通过，才把 Joint 的每层固定 spectral/local/channel gate 改为轻量样本条件 gate。不能因为 TFNO 和 ConvNeXt 都较好，就直接拼接两个完整 backbone。

## 与 loss 计划的关系

[loss 探索计划](工作频率与带隙损失探索计划.md)与本计划并行。本计划所有组件训练使用原始 loss；两条路线分别通过后，才运行“原/新 loss × 原/新组件”的 $2\times2$ 组合确认。

## 当前执行决定

正式执行合同登记为 [`EXP-FWD-019`](https://github.com/fangtoast/phc2d/blob/master/experiments/research/EXP-FWD-019-FNO-TFNO-ConvNeXt组件归因.md)。C0、每家族最多两个 seed0 候选、过门候选 seeds 1/2 与六域冻结评估均已获 N40 提交授权；仍按停止条件逐门推进。

截至2026-09-08，C0、21项seed0接受谱系和42项三种子validation-only确认均已完成，详见[三种子确认结果](../results/forward/EXP-FWD-019-组件归因三种子确认.md)。计划仍为`active`：23/42项终态未满足收敛门，TFNO `no_channel`未通过Graded三种子方向门，六域冻结评估和held-out test尚未执行。
