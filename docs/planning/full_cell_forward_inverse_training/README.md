---
title: "全胞双路线训练与混合算力规划索引"
summary: "研究计划索引；按任务族路由到现行协议与历史记录。"
page_type: landing
status: draft
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 5 minutes
primary_layer: L0
---

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

本目录把 **Full-Wh 网络代码实现、10k/50k 实验验证、已完成的双路线训练证据合同与后续解析参数逆向设计** 拆成独立文档，避免把架构规格、代码测试、神经网络训练、阈值物理验证和 inverse replay 混为同一执行批次。计划仍是待人工审计的 Draft；实际完成状态以代码、测试、run 制品和结果报告为准。

## 一、文档入口

| 优先级 | 文档 | 状态 | 负责内容 |
|---|---|---|---|
| 1. 代码实现记录 | [全胞正向模型升级代码实现规格](full_cell_forward_model_upgrade_implementation_plan.md) | Draft implementation record / implemented / no_result | 五个新增模型、dense/k-conditioned 输出头、registry/runner 接口、参数预算与 WSL 代码测试；不包含性能结论 |
| 2. 实验验证方案 | [全胞正向升级模型 10k/50k 训练与验证方案](full_cell_forward_model_upgrade_10k_50k_training_validation_plan.md) | Draft experiment plan / not authorized | 54 个 10k jobs、27 个 50k jobs、validation-only head 选择、N40 8×4090 独立单卡调度与 promotion |
| 10k 历史证据合同 | [全胞双路线训练与混合算力计划](full_cell_dual_lane_hybrid_compute_plan.md) | Historical Draft / experiment completed | 连续/五阈值表征、三 seed、102 个训练 run、12-job GPU 波次及其原始验收合同 |
| 50k 历史证据合同 | [全胞双路线 50k 训练、迁移评估与混合算力计划](full_cell_dual_lane_50000_training_hybrid_compute_plan.md) | Historical Draft / experiment completed | 17 波、204 个独立训练 run、旧 10k checkpoint→50k test、40k 的 20/100/200、50k final-fit、72-cell binary↔continuous 与零样本输入分辨率测试；结果见 [50k Forward 报告](../../results/forward/full_cell_dual_lane_50000_forward_training_20260808.md) |
| 后续扩展 | [全胞解析参数逆向设计扩展计划](full_cell_inverse_design_extension_plan.md) | Deferred Draft | 解析参数 codec、45 个 inverse GPU run、候选健康门与 8-target smoke→full validation→test-100 replay |
| 历史草案 | [旧版单 seed 正向计划](archive/full_cell_forward_single_seed_legacy.md) | Superseded | 仅用于决策追溯，不得作为执行入口 |

## 二、执行顺序

1. 人工审阅 [模型升级代码实现记录](full_cell_forward_model_upgrade_implementation_plan.md)，确认五个 canonical name、逐层结构、输出头、接口、参数预算和测试边界。
2. 五个模型的 L4 实现与代码测试已经完成；`implemented` 只表示源码可构建、可前后向和可接入 runner，不代表性能提升。
3. 人工审阅 [10k/50k 训练验证方案](full_cell_forward_model_upgrade_10k_50k_training_validation_plan.md)，检查 54/27 任务、validation-only 选择、N40 `%8` 调度和 promotion。
4. 在新的明确授权下先执行 10k admission 与三 seed/head 矩阵；未通过 10k 门的模型不得进入 50k。
5. 保留已完成的旧 10k/50k Forward 结果和 checkpoint inventory，作为新模型比较基线，不重新解释旧模型身份。
6. threshold real-MPB、inverse surrogate 准入与 replay 仍是独立任务，不能由新 forward 指标自动触发。

任何租卡、M9 `sbatch`、正式训练或 real-MPB replay 都需要新的明确授权。本次文档固化不构成执行授权。

## 三、决策覆盖关系

- 10k 参考计划和 50k 当前计划均采用固定阈值 `0.3/0.4/0.5/0.6/0.7`，**不使用 Bernoulli 随机采样**。
- 10k 已完成训练总数为 **102 runs**；50k 当前冻结矩阵为 **17 波、204 个独立 runs**。8k/20k 与 threshold 20/200 epoch 已被负责人移除，不得恢复旧 63 波/756 观察矩阵。
- 新升级 campaign 与历史 102/204-run 矩阵相互独立；文档命名、代码实现、admission、正式训练和 held-out test 是五个不同状态。
- 五个升级模型目前均为 `implemented / no_result`；它们已进入实时 registry 和模型 inventory，但尚不进入当前结果排行榜。
- inverse 扩展只能使用前向计划冻结的原生 binary 与 graded continuous surrogate；阈值代理不能直接充当物理等价的 inverse ranker。
- 运行制品、结果报告和 solver 输出高于规划文档；计划不得被引用为实验完成证据。
