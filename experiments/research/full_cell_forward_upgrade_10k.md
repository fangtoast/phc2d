<!--
  Execution profile for upgraded Full-Wh 10k training on N40.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# Full-Wh 升级模型 10k N40 训练

| Metadata | Value |
|---|---|
| Status | Authorized execution; admission pending |
| Primary layer | L3 experiment/evidence |
| Platform | N40 `gpu_4090`, eight independent single-GPU slots |
| Authorization | User request on 2026-08-08 to start the planned training and inspect every 30 minutes |

本实验执行[10k/50k 训练验证方案](../../docs/planning/full_cell_forward_inverse_training/full_cell_forward_model_upgrade_10k_50k_training_validation_plan.md)中的 `A0-A2` 与 `10K-W1-W7`。本轮授权不绕过 admission、不提前读取 test，也不自动启动 50k。

## 输入与执行入口

- Binary：[配置](../../configs/research/full_cell_forward_upgrade_binary_10k_v1.yaml)
- Graded：[配置](../../configs/research/full_cell_forward_upgrade_graded_10k_v1.yaml)
- Joint：[配置](../../configs/research/full_cell_forward_upgrade_dual_lane_10k_v1.yaml)
- 提交器：[`paracloud_submit_full_cell_upgrade_wave.sh`](../../jobs/paracloud_submit_full_cell_upgrade_wave.sh)
- 训练 worker：[`paracloud_full_cell_upgrade_train.sh`](../../jobs/paracloud_full_cell_upgrade_train.sh)
- 只读状态：[`paracloud_full_cell_upgrade_status.sh`](../../jobs/paracloud_full_cell_upgrade_status.sh)

每波必须先生成 dry-run manifest，再在部署 commit、数据 hash、上一波 artifact 和授权范围一致时提交。单任务使用一张 RTX4090、6 CPU、FP32；联合任务每批固定 32 binary + 32 graded，并使用各 lane train split 的独立 target normalizer。

## 非目标与完成门

- Admission 不是正式训练结果；单 seed 或 validation 改善不构成模型胜出。
- 本轮不提交 50k、threshold、cross-lane test、inverse 或 MPB/Meep。
- 10k 完成必须有54个任务的终态、标准 run 制品和完整性审计；随后先用 validation 冻结输出头，才允许读取 held-out test。
- 监控为每30分钟只读检查；失败任务只报告，不自动取消、改参或重启。
