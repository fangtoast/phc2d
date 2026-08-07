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

本目录把已完成的 **10k Full-Wh 前向双路线训练**、待执行的 **50k 数据规模/epoch/分辨率实验** 与后续解析参数逆向设计拆成独立文档，避免把神经网络训练、阈值物理验证和 inverse replay 混为同一执行批次。新增计划均为待人工审计的 Draft，不表示配置已实现、资源已租用或实验已启动。

## 一、文档入口

| 优先级 | 文档 | 状态 | 负责内容 |
|---|---|---|---|
| 10k 参考计划 | [全胞双路线训练与混合算力计划](full_cell_dual_lane_hybrid_compute_plan.md) | Historical Draft / experiment completed | 连续/五阈值表征、三 seed、102 个训练 run、12-job GPU 波次及其原始验收合同 |
| 50k 当前计划 | [全胞双路线 50k 训练、迁移评估与混合算力计划](full_cell_dual_lane_50000_training_hybrid_compute_plan.md) | Draft | 旧 10k checkpoint→50k test、8k/20k/40k nested、20/100/200 milestone、252 条训练轨迹、50k 自身 72-cell binary↔continuous 与总计 2,448-cell 跨路线/跨表征评估、零样本分辨率测试及 N40/M9 分工 |
| 后续扩展 | [全胞解析参数逆向设计扩展计划](full_cell_inverse_design_extension_plan.md) | Deferred Draft | 解析参数 codec、45 个 inverse GPU run、候选健康门与 8-target smoke→full validation→test-100 replay |
| 历史草案 | [旧版单 seed 正向计划](archive/full_cell_forward_single_seed_legacy.md) | Superseded | 仅用于决策追溯，不得作为执行入口 |

## 二、执行顺序

1. 保留已完成 10k Forward 结果和 checkpoint inventory，不重新选择旧模型。
2. 完成 binary/graded 50k HDF5 的下载、哈希、内容与 WSL acceptance。
3. 实现 50k 计划所需的 milestone、nested split、完整 resume 和跨 dataset/resolution 推理接口。
4. 完成 N40 单卡 benchmark 与 12-job 50k HDF5 admission。
5. 经单独授权后执行旧 10k checkpoint 推理和 21 个 12-job 训练波次。
6. 完成共同 test、epoch/data-size 对照、2,448-cell 双向跨路线/跨表征评估与 CNN/FNO 零样本分辨率测试。
7. 若阈值代理值得继续，单独授权 M9 进行阈值真实物理验证。
8. Forward 证据完整后，重新审计并决定是否更新 inverse 扩展入口。

任何租卡、M9 `sbatch`、正式训练或 real-MPB replay 都需要新的明确授权。本次文档固化不构成执行授权。

## 三、决策覆盖关系

- 10k 参考计划和 50k 当前计划均采用固定阈值 `0.3/0.4/0.5/0.6/0.7`，**不使用 Bernoulli 随机采样**。
- 10k 已完成训练总数为 **102 runs**；50k 新计划是 252 条独立训练轨迹并在 20/100/200 形成 756 个 milestone observations，两者不得合并计数。
- inverse 扩展只能使用前向计划冻结的原生 binary 与 graded continuous surrogate；阈值代理不能直接充当物理等价的 inverse ranker。
- 运行制品、结果报告和 solver 输出高于规划文档；计划不得被引用为实验完成证据。
