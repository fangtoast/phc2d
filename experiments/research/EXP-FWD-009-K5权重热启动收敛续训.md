---
title: "EXP-FWD-009：K5 epoch-200 权重热启动收敛续训"
summary: "从 EXP-FWD-008 的五个 epoch200 权重状态建立独立 derivative，以重置 AdamW 的方式续训并按滚动双十轮窗口停止。"
page_type: experiment
plan_state: completed
status: draft
audience: "研究执行者与结果审阅者"
owner: "研究生维护"
last_reviewed: 2026-08-25
primary_layer: L3
---

# EXP-FWD-009：K5 epoch-200 权重热启动收敛续训

本实验不覆盖 EXP-FWD-008。源 checkpoint 仅提供 epoch-200 模型权重，没有 AdamW 或数据顺序状态；因此本实验明确属于 `weight_only_warm_start_optimizer_reset`，不能解释为原优化器轨迹的无损续接。

| 合同 | 固定值 |
|---|---|
| 任务 | ConvNeXt Binary/Graded、TFNO Binary/Graded、dual-lane joint，均 seed0 |
| 源状态 | EXP-FWD-008 对应任务的 `epoch200.pt`，源 history/model/terminal hash 必须闭合 |
| 优化器 | AdamW，沿用各模型原学习率与 weight decay，但 optimizer state 明确重置 |
| 训练范围 | 从总 epoch 201 开始，最大总 epoch 600 |
| 收敛检查 | 每 20 epoch 检查一次；比较结束 epoch 前两段各 10 epoch 的 bands 2–10 validation MRE |
| 阈值 | `r≤0.005`；单路线一条主曲线，joint 要求 Binary、Graded、等权三条同时通过 |
| Band 1 | 同步报告，不阻断 |
| 选模 | 在源 epoch 1–200 与新增 epoch 201–stop 的全部验证历史中选择 bands 2–10 MRE 最优 checkpoint |

## 六域交叉验证

五个续训任务全部通过收敛门后，冻结最终 K5 17-checkpoint inventory：未续训的 12 个身份使用 EXP-FWD-008 的验证最优 checkpoint，ConvNeXt B/G、TFNO B/G 与 joint 使用 EXP-FWD-009 的验证最优 checkpoint。随后严格复用 EXP-FWD-007 的六域测试合同：K5 17 + 原 K20 17 + 原 F50 51 = 85 checkpoint，完整推理 510 cells；先完成 54 个去重身份 GPU preflight，再提交 10 shards（`0-9%8`）。

评估始终使用源 checkpoint normalizer，joint 按目标路线传入 `lane_id`，不微调、不重拟合 normalizer、不修改测试集。K20/F50 408 cells 与 EXP-FWD-007 做 `1e-6` 数值回归，历史 272 cells 单独回归；K5 102 cells 与 EXP-FWD-007 做 10,000 次 paired bootstrap。指标和结论仍只属于 held-out surrogate evidence。

若到总 epoch 600 仍未通过，状态保持 `training_complete_convergence_not_met`，不得宣称收敛。所有结果仍仅属于 held-out surrogate evidence。

## 执行结果

五个续训任务均在总 epoch 600 前通过主收敛门，最终由 12 个 EXP-FWD-008 checkpoint 与 5 个 EXP-FWD-009 checkpoint 组成 17 个 K5 身份，并完成 85 checkpoint × 6 域的 510-cell 评估。由于 12 个保留身份中仍有 8 个未通过 EXP-FWD-008 原收敛门，最终结果属于带人工收敛豁免的完整评估，不等于原合同 17/17 收敛通过。

[阅读 EXP-FWD-009 结果](../../docs/results/forward/EXP-FWD-009-K5权重热启动与六域验证.md)。
