---
title: "K=20 B/G 单 Seed 正向训练与跨路线评估"
summary: "记录 K=20 binary/graded Fourier 全胞数据上的 17-task seed=0 campaign、N40 分波训练和 34-cell native/cross-lane evaluation。"
page_type: evidence
status: draft
audience: "研究执行者、导师与结果审阅者"
owner: "研究生维护"
last_reviewed: 2026-08-20
evidence_date: 2026-08-20
evidence_scope: "K=20 B/G Fourier full-cell; Wh[10000,81,10]; seed=0; N40 surrogate training"
reading_time: 5 minutes
primary_layer: L3
backend: "surrogate training; WSL Ubuntu + meep_env real-MPB data gate"
---

<!--
  Draft evidence record for the K=20 B/G campaign.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 当前状态

本页是本次执行的本地 Draft 证据记录，不是最终模型结果。B/G 数据、WSL real-MPB data gate 和 17/17 admission 已完成；N40 formal training 正在按 W1/W2/W3 分波执行，34-cell evaluation 尚未开始。

## 已锁定的输入与数据 gate

| 项目 | B binary | G graded |
|---|---|---|
| HDF5 SHA-256 | `2661eb1e9e3b398a7b69c4083daefa761ea9c26a1081c478b968fd6a951d6b56` | `96b127b43f2c31fb7591e9634317a7ed35332ceae1077b2277de15058902fd72` |
| grid field | `material_area_fraction_grid_64` | `material_weight_grid_64` |
| `Wh` shape | `10000×81×10` | `10000×81×10` |
| split | `8000/1000/1000` | `8000/1000/1000` |
| WSL MPB samples | 8 structures × 9 k points | 8 structures × 9 k points |
| WSL result | pass; median `0`, P95 `1.08e-7` | pass; median `0`, P95 `1.02e-7` |

WSL solver profile is Ubuntu 6.6 + conda `meep_env`, Meep `1.33.0-beta`, MPB `1.12.0`, resolution 64, mesh 3, TE bands 2–10. These values clear the data gate only; they do not validate surrogate model performance.

## N40 execution record

- Campaign: `/HOME/scw6g7a/run/phc2d/runs/20260820/full_cell_k20_forward_seed0_202608201430`
- Deployment commit marker: `b675efe` or later control-script commits; the campaign manifest records the deployed commit used at freeze.
- Protocol: 17 tasks, seed 0, dense protocol, FP32, 100-epoch cap, patience 15, effective batch 64, 1 GPU + 6 CPU per task.
- Admission: 17/17 successful after bounded scheduler/running-script corrections; no model hyperparameter changes.
- W1 baseline: job `5422465`, 8/8 `formal_training_success`.
- W2 upgrade: job `5422496`, 6/8 successful at the last local update; TFNO B/G remained in progress.

The N40 `MaxJobsPU=16` and array-parent accounting required admission submission as 15+2 tasks. A runner correction was required for baseline models that do not accept a selectable head override; the successful admission and formal runs use the original model/config/seed protocol.

## Remaining gates

1. Finish W2 and submit W3 dual-lane only after W2 is 8/8 successful.
2. Run `plan-eval`, submit 8 evaluation shards, and require exact 34-cell coverage: 16 single-lane native/cross cells plus dual-lane B/G native cells.
3. Finalize native/cross summaries and 10,000-replicate paired bootstrap deltas, then update this page with actual run paths, checkpoint hashes, metrics, failures, and resource totals.

Until those gates close, no model ranking, upgrade promotion, seed-robust claim, or real-MPB model-success claim is admissible. The final report must keep single-seed held-out surrogate evidence separate from the WSL physical data gate.
