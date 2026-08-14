---
title: "P6 seed0 target-residual 条件准入诊断（2026-07-29）"
summary: "日期化实验与验证证据；正文保持原始记录。"
page_type: archive
status: historical
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 4 minutes
primary_layer: L3
evidence_date: 2026-07-29
evidence_scope: "日期化证据或历史记录；正文未因站点重构而改写。"
---

<!--
  P6 seed-0 target-residual conditional inverse-admission evidence.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# P6 seed0 target-residual 条件准入诊断（2026-07-29）

!!! warning "历史证据"
    本页是单 seed residual 诊断 pilot，未通过的准入不得解释为正式模型结果。

## 结论

P6A 和 P6B 的 seed0 均未通过固定 64×64 的 target-residual 校准，因此当前不能进入 PINO inverse adapter preflight。该结论不依赖三-seed门槛：按照“seed0 通过即可条件准入”的规则，两个 seed0 自身仍然失败。

剩余 P6 seed 继续训练，只用于检查失败是否稳定以及形成完整消融证据；它们不会改变本次 seed0 结论。

## 实验边界

| 项目 | 设置 |
| --- | --- |
| 数据划分 | validation only |
| test / MPB | 均未访问 |
| 分辨率 | 64×64 |
| 结构数 | 32 |
| 物理 k 点 | 30 个唯一坐标，不重复计算闭合 Γ 点 |
| bands | 2–10 |
| 条件记录 | 每候选 8,640 条，共 17,280 条 |
| 聚类单位 | `source_sample_id` |
| bootstrap | 1,000 次 |
| 候选 | P6A seed0、P6B seed0 |

配置为 [`field_eigen_te_p6_seed0_target_residual_pilot_64.yaml`](https://github.com/fangtoast/phc2d/blob/master/configs/research/field_eigen_te_p6_seed0_target_residual_pilot_64.yaml)，正式 artifacts 位于本地 `runs/20260729/field_eigen_te_p6_seed0_target_residual_pilot_64_202607291023`。

## 准入结果

准入条件仍为：

1. target residual 与目标频率误差的 Spearman bootstrap 95% CI 下界大于 0；
2. 低 residual 四分位相对高 residual 四分位的频率 MRE 至少降低 10%；
3. 所有统计量有限。

| 候选 | Spearman | 95% CI | 低 residual MRE | 高 residual MRE | 四分位改善 | seed0 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| P6A | -0.4014 | [-0.6901, -0.0402] | 2.7912% | 2.2873% | -22.03% | blocked |
| P6B | -0.3424 | [-0.6801, 0.0495] | 2.8170% | 2.2476% | -25.33% | blocked |

两组数据均有限，无 NaN/Inf。失败原因不是数值异常，而是 residual 的排序方向错误：选择低 residual 结构会使频率误差恶化约 22%–25%。

本地 `runs/20260729/field_eigen_te_p6_seed0_target_residual_pilot_64_202607291023/candidate_decision.json` 因此保持：

- `physics_target_residual_64_ready=blocked`
- `pino_inverse_adapter_preflight_ready=blocked`
- `pino_inverse_rerank_64_ready=blocked`

## 为什么 P6 没有扭转排序

### 1. 替换 residual 频率来源几乎没有改变结构排序

将旧 P4 与 P6A 的 32 个 validation 结构逐一对齐后：

- target-residual 结构排序的 Spearman 为 1.0000；
- 频率误差排序的 Spearman 为 1.0000；
- 平均 target-residual score 仅变化 +0.024%；
- 平均频率 MRE 仅变化 +0.008%。

P5 到 P6B 也只有很小变化：

- residual 排序 Spearman 为 0.9927；
- 频率误差排序 Spearman 为 0.9901；
- 平均 residual score 变化 -0.224%；
- 平均频率 MRE 变化 -0.091%。

这说明把 PDE 中的预测频率换成真实目标频率虽然切断了频率头的自洽梯度，但没有实质改变场分支在结构层面的难度排序。P4/P5 的预测频率误差本来只有约 2.5%，其频率替换不足以改变主导训练轨迹。

### 2. 训练 physics 覆盖与准入诊断的 dense-k 覆盖不一致

冻结的 field HDF5 每个结构只有 36 条场记录，即 9 个 band × 4 个 k 点；整个数据集共出现 8 个场监督 k 坐标。准入诊断则在每个结构上检查 30 个唯一物理 k 坐标。

P6 的 dense-frequency loader 覆盖完整频率路径，但只监督频率头；target-PDE 仍主要作用于稀疏场记录。因此模型没有在与准入诊断一致的 30-k 分布上学习 target-aware 场 residual。

### 3. 场分支与频率误差仍缺少显式排序耦合

P6 target-PDE 能训练场解码器，五个梯度审计点均显示频率头 PDE 梯度为 0、场解码器梯度非零。但“场 residual 变小”和“频率头误差变小”仍是两个独立目标；共享主干并不能保证二者在结构层面单调相关。

P6B 的 Rayleigh 项将相关性从约 -0.401 改善到 -0.342，但低 residual 四分位仍更差，因此改善不足以支持 rerank。

## 后续判断

当前不应执行 64×64 inverse adapter，也不应把 P6 residual 加入 FNO 候选排序。剩余 seed 完成后应先确认失败是否跨 seed 稳定。

若继续研究 PINO inverse，下一轮应优先处理训练与准入分布不一致：

1. 在完整 30-k dense-frequency 条件上同时计算 target-PDE，而不只训练频率头；
2. 若仍无正相关，再显式输入 `f_target`，训练
   \((\epsilon,k,\mathrm{band},f_{\mathrm{target}})\rightarrow\hat E\)；
3. 任何新方案仍先做 validation residual 校准，再决定是否进入 adapter，不直接调用 inverse 或 MPB。
