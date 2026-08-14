---
title: "A2 Objective-Bridge Seed0 调整结果"
summary: "日期化实验与验证证据；正文保持原始记录。"
page_type: evidence
status: draft
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 2 minutes
primary_layer: L3
evidence_date: 2026-07-27
evidence_scope: "日期化证据或历史记录；正文未因站点重构而改写。"
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

# A2 Objective-Bridge Seed0 调整结果

## 当前结论

截至 2026-07-27，本报告处于 `implementation_and_validation_in_progress`。新协议仅产生
`single_seed_adjustment_pilot` 派生证据，不改变原 A1/A2 正式结果。任何 validation、
pilot 或 real-MPB 数值必须在对应 run 产物完成后写入，静止进程、超时或仅有 checkpoint
文件均不视为成功。

## 调整内容

| 项目 | 调整 |
|---|---|
| A1 forward FNO | 冻结，作为可微 bands 2–10 主 objective 与候选 ranker |
| A1 query FNO | 冻结，作为 cross-surrogate consistency 与门禁诊断 |
| retrieval | 冻结原 A2 结果 |
| direct inverse | 冻结原 A2 结果 |
| single-mode | 从原 validation-selected checkpoint objective-finetune |
| mixture | 从原 validation-selected checkpoint objective-finetune |
| PINO/field inverse | `blocked`，不进入本轮 |

## 验收门禁

训练 checkpoint 必须同时满足 validity、effective-K、continuous std、
threshold-margin、latent gradient/weight change、mode coverage、cross-surrogate gap
和相对原 validation MRE 改善门禁。门禁先于 MRE 排序，且 test MPB 不参与 checkpoint
选择。

20-target pilot 中，mixture 只有同时满足 mean MRE 相对原 A2 改善至少 30%、
至少 `14/20` paired targets 更优、绝对 surrogate–replay gap 不超过 2 pp 且 solver
failure 为 0，才允许完整 100-target replay。

## 待回填的正式产物

- resolved config 与所有显式 SHA-256；
- training/validation history；
- candidate diagnostics 与冻结 manifest；
- real-MPB smoke、pilot 与 full replay shards；
- per-target/per-band 指标与 runtime ledger；
- paired bootstrap、sign test 和与原 A2 的比较表。

## 证据边界

即使完整 100-target replay 通过，本实验仍只说明 seed0 调整在固定 paired targets
上的效果，不授权“跨 seed 稳定”“普遍显著优势”或 PINO/physics-aware gain 等结论。
