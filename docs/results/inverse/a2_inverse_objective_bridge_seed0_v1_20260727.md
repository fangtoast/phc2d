---
title: "A2 Objective-Bridge Seed0：100/625-target real-MPB replay"
summary: "汇总 Objective-Bridge seed0 在固定 100-target pilot 与完整 Test625 上已经完成的 real-MPB replay。"
page_type: evidence
status: draft
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-14
reading_time: 8 minutes
primary_layer: L3
evidence_date: 2026-07-27
evidence_scope: "V2 10k 固定数据身份下的 seed0 adjustment；100-target pilot 与完整 Test625 replay，禁止跨 seed 推广。"
dataset_scope: v2_10k_wh31x10
evidence_level: solver_replay
evidence_status: completed
validation_backend: wsl_mpb
source_artifacts:
  - runs/20260727/a2_inverse_objective_bridge_seed0_v1_202607271804
  - runs/20260728/a2_inverse_objective_bridge_seed0_test625_v1_202607280842
---

<!--
  A2 objective-bridge seed-0 adjustment status and result report.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# A2 Objective-Bridge Seed0：100/625-target real-MPB replay

## 结论

Objective-Bridge 的 seed0 调整已完成两个层级的真实 MPB 回放：2026-07-27 的固定 100-target pilot 状态为 `adjustment_pilot_full_100_completed`，2026-07-28 的完整 test split（625 targets）状态为 `adjustment_test625_completed`。旧正文中的“待回填”已被真实 aggregate 制品取代。

在 Test625 上，冻结的 bridge single-mode 和 mixture 的 mean replay MRE 分别为 `2.2614%` 与 `2.3916%`；二者都低于用于诊断的 historical mixture seed2 `3.0775%`。但该历史基线与本轮协议并非完全匹配，且 bridge 仍只有 seed0，因此这不是跨 seed 或普遍优越性证据。

## 协议与固定边界

| 项目 | 固定内容 |
|---|---|
| 数据 | V2 10k，目标 `Wh[31,10]`；Test625 使用完整冻结 test split |
| A1 forward FNO | 冻结，作为 bands 2–10 主 objective 与候选 ranker |
| A1 query FNO | 冻结，只作为 cross-surrogate 诊断 |
| retrieval | 冻结原 A2 结果 |
| direct inverse | 冻结原 A2 结果 |
| bridge single-mode / mixture | 使用 validation 阶段冻结的 epoch 50、learning rate `1e-4` checkpoint |
| 候选与回放 | `K=128`，有效 Top-M 均值为 5；test MPB 不参与 checkpoint 选择 |
| PINO/field inverse | 未包含；不得主张 physics-aware rerank 收益 |

## 100-target pilot

来源：`runs/20260727/a2_inverse_objective_bridge_seed0_v1_202607271804/aggregate/`。

| 方法 | replay mean MRE | surrogate mean MRE | surrogate–replay gap | effective K |
|---|---:|---:|---:|---:|
| retrieval（冻结参考） | `1.5214%` | `1.5170%` | `0.0044 pp` | `128.00` |
| bridge single-mode | `2.3986%` | `2.2772%` | `0.1213 pp` | `111.55` |
| bridge mixture | `2.5105%` | `2.3534%` | `0.1571 pp` | `107.86` |

在这 100 个 paired targets 上，single-mode 相对原 A2 single-mode 的 mean replay MRE 改善 `81.81%`、paired wins 为 `98/100`；mixture 相对原 A2 mixture 改善 `78.28%`、paired wins 为 `95/100`。retrieval 仍优于 bridge mixture，说明“修复生成器”不等于“超过检索基线”。

## 完整 Test625 replay

来源：`runs/20260728/a2_inverse_objective_bridge_seed0_test625_v1_202607280842/aggregate/`。本轮没有再次更新训练参数，`training_reused_without_update=true`。

| 方法 | replay mean MRE | 95% bootstrap CI | median | surrogate–replay gap |
|---|---:|---:|---:|---:|
| historical mixture seed2 | `3.0775%` | `[2.9700%,3.1856%]` | `2.6543%` | `1.4454 pp` |
| bridge single-mode seed0 | **`2.2614%`** | `[2.2263%,2.2965%]` | `2.2196%` | `0.1180 pp` |
| bridge mixture seed0 | `2.3916%` | `[2.3485%,2.4341%]` | `2.3281%` | `0.1440 pp` |

与 historical mixture seed2 的 paired 诊断为：

- bridge single-mode 相对 mean improvement `26.52%`，paired delta 为 `-0.8162 pp`，95% CI `[-0.9220,-0.7131] pp`，paired wins `474/625`；
- bridge mixture 相对 mean improvement `22.29%`，paired delta 为 `-0.6859 pp`，95% CI `[-0.7930,-0.5794] pp`，paired wins `438/625`；
- 同一 bridge 协议内部，single-mode mean replay MRE 比 mixture 低 `0.1302 pp`。

第一、二项比较的 `historical_baseline_protocol_matched=false`，所以只能称历史诊断，不能包装为严格同协议模型胜负。

## 可追溯产物

两个 run 都保留 resolved config、训练/选择记录、candidate manifest、replay shards、per-target/per-band 表、runtime ledger、aggregate `metrics.json` 与 `results.json`。本页数值直接来自各自 `aggregate/` 目录，不从日志或进程状态推断。

## 证据边界

本结果支持“seed0 Objective-Bridge 在固定 100-target pilot 和 Test625 上已经完成 real-MPB replay，并明显缩小自身 surrogate–replay gap”。它不授权“跨 seed 稳定”“普遍显著优势”“严格同协议击败 historical seed2”或 PINO/physics-aware gain。原 A1/A2 历史证据保持不变。
