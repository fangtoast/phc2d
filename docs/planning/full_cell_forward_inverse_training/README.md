<!--
  Index for full-cell forward training, hybrid compute, and inverse-design plans.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="full-cell-training-plan-index"></a>

# 全胞双路线训练与混合算力规划索引

| Metadata | Value |
|---|---|
| Status | Draft planning index |
| Primary layer | L3 experiment/evidence |
| Authority | [生成网络调整研究计划](../fno_generative_network_adjustment_plan_20260617.md) |
| Audience | 项目负责人、研究人员、GPU/超算执行人员与 AI 协作者 |
| Owner | 研究生维护，导师审阅科学主张与 promotion gate |
| Last reviewed | Pending human review |
| Reading time | 3 minutes |

本目录把 **当前 Full-Wh 前向双路线训练** 与 **后续解析参数逆向设计** 拆成独立文档，避免把 102 个前向训练 run、阈值物理验证和 inverse replay 混为同一执行批次。所有文档均为待人工审计的计划，不表示配置已实现、资源已租用或实验已启动。

## 一、文档入口

| 优先级 | 文档 | 状态 | 负责内容 |
|---|---|---|---|
| 当前主计划 | [全胞双路线训练与混合算力计划](full_cell_dual_lane_hybrid_compute_plan.md) | Draft | 连续/五阈值表征、三 seed、102 个训练 run、三张租用 GPU、阈值 real-MPB 验证 |
| 后续扩展 | [全胞解析参数逆向设计扩展计划](full_cell_inverse_design_extension_plan.md) | Deferred Draft | 解析参数生成、候选健康门、双 surrogate 筛选与 8→20→100 replay |
| 历史草案 | [旧版单 seed 正向计划](archive/full_cell_forward_single_seed_legacy.md) | Superseded | 仅用于决策追溯，不得作为执行入口 |

## 二、执行顺序

1. 人工审计并冻结前向主计划。
2. 完成渐进数据 WSL acceptance 和单卡租用 preflight。
3. 单独授权三张云 GPU，执行 84 个主训练 run。
4. 冻结三个胜出配置后，执行 18 个分辨率消融 run。
5. 若阈值代理值得继续，单独授权 M9 进行阈值真实物理验证。
6. 前向证据完整后，重新审计并决定是否启动 inverse 扩展计划。

任何租卡、M9 `sbatch`、正式训练或 real-MPB replay 都需要新的明确授权。本次文档固化不构成执行授权。

## 三、决策覆盖关系

- 最新前向主计划采用固定阈值 `0.3/0.4/0.5/0.6/0.7`，**不使用 Bernoulli 随机采样**。
- 前向训练总数以主计划的 **102 runs** 为准，不再使用旧版单 seed 的 34 runs 或此前 Bernoulli 草案的 54 runs。
- inverse 扩展只能使用前向计划冻结的原生 binary 与 graded continuous surrogate；阈值代理不能直接充当物理等价的 inverse ranker。
- 运行制品、结果报告和 solver 输出高于规划文档；计划不得被引用为实验完成证据。
