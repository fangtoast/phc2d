<!--
  Authority-aware planning index for the synchronized phc2d workspace.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 研究规划与协议索引

本目录记录“计划做什么、为什么这样安排、通过什么 gate”，不证明实验已经完成。当前完成状态从 [`../results/README.md`](../results/README.md)、运行制品和 [`../../PROJECT_MAP.md`](../../PROJECT_MAP.md) 判断。

## 权威路由

本索引是规划入口，不用单一旧计划覆盖所有后续专题。先按任务主题选择当前协议；若专题协议与早期总计划冲突，以实际制品和当前实现为证据事实，以更具体、更新且尚未被取代的协议约束后续执行。

| 角色 | 文档 | 使用边界 |
|---|---|---|
| 早期总研究计划 | [`fno_generative_network_adjustment_plan_20260617.md`](fno_generative_network_adjustment_plan_20260617.md) | FNO、生成式网络及其既定任务家族、方法路线、gate 与失败判据；不覆盖后来新增专题的全部执行细节 |
| TODO / promotion gate | [`research_todo_20260617.md`](research_todo_20260617.md) | 勾选可能滞后，必须用更强证据复核 |
| A2 单 seed 协议 | [`a2_inverse_single_seed_protocol_20260726.md`](a2_inverse_single_seed_protocol_20260726.md) | A2 固定公平比较与 replay 契约 |

## 当前阶段协议

| 主题 | 文档 |
|---|---|
| 物理模型路线与当前修正 | [`0718-0801/02_physics_model_reproduction_plan.md`](0718-0801/02_physics_model_reproduction_plan.md) |
| A0/B0 基础协议 | [`0718-0801/09_phase1_a0_b0_protocol.md`](0718-0801/09_phase1_a0_b0_protocol.md) |
| B4 frequency / dense-k 准入 | [`0718-0801/10_b4_frequency_fidelity_dense_k_admission.md`](0718-0801/10_b4_frequency_fidelity_dense_k_admission.md) |
| 全胞双路线 Forward、升级模型与 inverse | [`full_cell_forward_inverse_training/README.md`](full_cell_forward_inverse_training/README.md) |

## 参考层

这些文件提供早期审计、文献和验证矩阵，不高于当前计划或实际制品：

- [`非 p4m 全布里渊区双路线数据生成方案.md`](非%20p4m%20全布里渊区双路线数据生成方案.md)：2026-08-03 的 `full_cell_v1` 双路线历史设计 Draft；当前完成状态以 `PROJECT_MAP.md` 和结果报告为准。
- [`reference/experimental_validation_plan_20260617.md`](reference/experimental_validation_plan_20260617.md)
- [`reference/network_literature_survey_20260617.md`](reference/network_literature_survey_20260617.md)
- [`reference/repo_data_audit_20260617.md`](reference/repo_data_audit_20260617.md)

规划任务也必须遵守 [文档与协作治理](../collaboration.md)：计划、配置就绪和实际完成保持不同状态。
