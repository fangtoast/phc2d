---
title: "EXP-FWD-007：K5训练与六域验证"
summary: "Draft result shell for the frozen K5 all-model training and 510-cell six-domain evaluation."
page_type: evidence
status: draft
audience: "研究负责人、审阅者与复现执行者"
owner: "研究生维护"
last_reviewed: 2026-08-24
reading_time: 3 minutes
primary_layer: L3
evidence_date: 2026-08-24
evidence_scope: "K5 B/G 17-task training and K5/K20/F50 six-domain surrogate evaluation; scheduler admission is blocked before training"
dataset_scope: full_cell_fourier_k5_k20_f50_wh81x10
evidence_level: surrogate_held_out
evidence_status: blocked
validation_backend: surrogate_only
source_artifacts:
  - runs/20260823/EXP-FWD-007_K5_20260823220138
---

<!--
  EXP-FWD-007 evidence page.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-FWD-007：K5训练与六域验证

**结论：尚无结果。** 该页已登记冻结协议和证据边界；K5 训练、N40 six-domain 重推、旧 272-cell 数值回归及人工审阅完成前，不发布模型排名、promotion 或物理结论。

| 证据问题 | 当前状态 |
|---|---|
| K5 17-task 训练 | completed；17/17 validation checkpoint 已冻结，所有 admission 与三波正式训练均未读取 held-out test |
| K5 B→G/G→B | pending K5 checkpoint freeze 与 six-domain evaluation |
| K5↔K20、K5↔F50 | pending；评估器已修正为从冻结配置读取 F50 的 50,000-sample tensor shape，待重新 freeze、GPU preflight 与 10-shard evaluation |
| WSL real-MPB / 物理验证 | 未完成；不由本页或训练作业升级 |

完成后本页只汇总 K5 native、B→G/G→B、K5↔K20、K5↔F50、baseline/upgrade 对比及汇总训练曲线。510 行 raw cells、预测数组、per-band/per-k/bootstrap/resource 明细仅链接 run 证据，不复制到正文。

相关协议：[EXP-FWD-007](../../../experiments/research/EXP-FWD-007-K5训练与六域验证.md)。计划 run 根目录为 `runs/{date}/EXP-FWD-007_K5_*` 与 `runs/{date}/EXP-FWD-007_six_domain_*`。
