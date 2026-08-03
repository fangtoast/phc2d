<!--
  Evaluation-only test625 extension for the A2 objective-bridge seed-0 checkpoints.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# A2 Objective-Bridge Seed0 Test625 Extension

## 结论边界

本实验将已经通过 100-target real-MPB replay 的 objective-bridge seed0
`single_mode_generator` 和 `mixture_generator` 冻结扩展到完整 625 个 test targets。
它不重新训练、不重新选择 checkpoint，也不修改原 A1、原 A2 或 100-target bridge 结果。

证据状态固定为 `single_seed_adjustment_test625_replay`。最终完成只支持当前 seed0
checkpoint 在完整 test split 上的物理回放结论，不授权跨 seed 稳定性或单因素因果归因。

## 固定输入

| 项目 | 固定值 |
|---|---|
| 配置 | [`../../configs/research/a2_inverse_objective_bridge_seed0_test625_v1.yaml`](../../configs/research/a2_inverse_objective_bridge_seed0_test625_v1.yaml) |
| 数据 | 10k V2，SHA-256 `991cb381...974ad4` |
| test manifest | [`../../configs/research/target_manifests/inverse_test625_v1.json`](../../configs/research/target_manifests/inverse_test625_v1.json) |
| targets | 完整 test split，dataset indices `9375..9999`，共 625 个 |
| checkpoint | 100-target bridge validation-selected epoch50，seed0，LR `1e-4` |
| ranker | A1 validation-selected forward FNO seed0 |
| diagnostic | A1 query FNO seed0 |
| candidates | `K=128`、阈值 `0.5`、二值去重、最小一位 normalized Hamming |
| replay | single/mixture 各 target top-5，real MPB，8 个隔离 worker |

## 扩展实现

- `training.evaluation_only=true`，统一入口拒绝在本协议下运行训练阶段。
- 候选健康门禁在全部 625 targets 上重新计算，但不访问 MPB 进行筛选。
- candidate manifest 只保存冻结后的 top-5 结构与完整 lineage；effective-K、连续候选
  std、threshold margin 和 cross-surrogate 指标仍基于完整 K 计算。
- replay 使用结构 SHA-256 缓存，并复用已完成的 100-target bridge MPB 缓存；只求解
  缺失结构。
- aggregate 将两个当前 generator 与 2026-07-06 historical objective-finetuned mixture
  的相同 625 targets 做 paired bootstrap 和 sign test。

## 公平性限制

历史 mixture 使用 seed2 和旧 query-FNO ranker，当前 bridge 使用 seed0 和 A1
forward-FNO ranker。因此相同 625 targets 支持结果级配对比较，但不能把差异归因于
mode count、seed、ranker 或某个单独 loss。

retrieval 与 direct inverse 在本扩展中不重新回放，因为本实验的目的，是让本次重训的
两个 generator 与初次 625-target mixture 保持相同目标规模。它们仍保留在原
100-target A2 公平主表中。

## 执行顺序

```powershell
PYTHONPATH=src python -m phc2d.cli inverse benchmark --config configs/research/a2_inverse_objective_bridge_seed0_test625_v1.yaml --stage preflight --run-dir <run_dir>
PYTHONPATH=src python -m phc2d.cli inverse benchmark --config configs/research/a2_inverse_objective_bridge_seed0_test625_v1.yaml --stage sample --run-dir <run_dir>
PYTHONPATH=src python -m phc2d.cli inverse benchmark --config configs/research/a2_inverse_objective_bridge_seed0_test625_v1.yaml --stage replay --run-dir <run_dir>
PYTHONPATH=src python -m phc2d.cli inverse benchmark --config configs/research/a2_inverse_objective_bridge_seed0_test625_v1.yaml --stage aggregate --run-dir <run_dir>
```

正式执行必须位于 WSL `Ubuntu` + conda `meep_env`，训练与 MPB 不得并发。
