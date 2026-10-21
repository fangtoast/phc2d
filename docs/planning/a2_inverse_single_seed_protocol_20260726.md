---
title: "A2 单 Seed 逆向设计派生协议"
summary: "研究计划或执行协议；实施前需与当前代码和证据重新核对。"
page_type: plan
status: draft
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 3 minutes
primary_layer: L0
plan_state: completed
effective_date: 2026-07-26
superseded_by: null
related_results:
  - docs/results/inverse/a2_inverse_single_seed_formal_20260727.md
  - docs/results/inverse/a2_inverse_objective_bridge_seed0_v1_20260727.md
---

<!--
  A2 single-seed fair inverse benchmark derivative protocol.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# A2 单 Seed 逆向设计派生协议

## 目的与证据边界

本协议记录用户批准的成本压缩：A2 全流程固定 `seed=0`，但不改写
[`baseline_protocol_v1.yaml`](https://github.com/fangtoast/phc2d/blob/main/configs/research/baseline_protocol_v1.yaml)
中的原三 seed 历史。完整执行后的唯一正式证据标签为
`single_seed_formal_replay`。

`formal_completed` 仅表示本派生协议完成了 validation 选型、候选冻结和
real-MPB replay。它不支持“跨 seed 稳定”“三 seed 显著”或普遍架构优势。

## 固定输入

| 项目 | 固定值 |
|---|---|
| 数据 | 10k V2，SHA-256 `991cb381…974ad4` |
| split | `6875/2500/625` |
| targets | [`inverse_paired_test100_v1.json`](https://github.com/fangtoast/phc2d/blob/main/configs/research/target_manifests/inverse_paired_test100_v1.json) |
| seed | `0` |
| 方法 | retrieval、direct inverse、single-mode generator、mixture generator |
| LR | `1e-3 / 3e-4 / 1e-4` |
| 候选 | 请求 `K=128`，阈值 `0.5`，二值去重，最小 Hamming `1 / geometry_dof` |
| replay | 每 target/method 最多 top-5；每方法最多 500 次引用 |
| 排名器 | A1 validation-selected `fno2d_band_v2` seed 0 |
| 诊断器 | A1 validation-selected `fno2d_query_v2` seed 0，仅作 cross-surrogate 诊断 |

所有路径与 hash 的机器可执行定义位于
[`a2_inverse_single_seed_v1.yaml`](https://github.com/fangtoast/phc2d/blob/main/configs/research/a2_inverse_single_seed_v1.yaml)。

## 阶段门禁

1. `preflight`：核验 dataset/split/targets/checkpoints/k-path/solver provenance，
   生成 `lineage.json` 与拒绝式 `preflight_decision.json`。
2. `fake-smoke`：四方法、两 target 的候选—排名—缓存—恢复—汇总链路，
   solver 明确标记为 fake。
3. `real-smoke`：两 target、每 record 最多两候选，仅校准真实 MPB 可运行性与速度。
4. `train`：三个可训练方法各执行 `3 LR × seed 0`，逐 checkpoint 仅在 validation
   上选择。
5. `sample`：生成并原子冻结 `4 × 100` 条 target-method records。冻结后 hash
   不匹配会拒绝 replay。
6. `replay`：结构 hash 去重，隔离子进程执行，单候选 shard 原子落盘；恢复时只补缺失项。
7. `aggregate`：固定四行主表，paired bootstrap、sign test 和 mixture 三项 Holm 校正。

## PINO 边界

A2 不读取 B4 checkpoint，不计算 PINO loss/field residual，也不进行
physics-aware rerank。PINO、field inverse 与 physics residual rerank 均保持
`blocked`；后续若修复，属于独立 B4/A3 扩展，不回填 A2 主表。

## 状态纪律

- `preflight_passed`：只表示输入与 lineage 合法。
- `ready_to_execute`：必须同时具备 fake-MPB 与 real-MPB smoke 成功证据。
- `formal_completed`：必须具备 9 个 LR validation cases、400 条冻结 records、
  完整 real-MPB replay 与统计报告。
- run outputs、checkpoint 和缓存只保存在 `runs/`，不进入 Git。
