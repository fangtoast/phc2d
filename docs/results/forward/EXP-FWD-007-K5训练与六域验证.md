---
title: "EXP-FWD-007：K5训练与六域验证"
summary: "Draft result shell for the frozen K5 all-model training and 510-cell six-domain evaluation."
page_type: evidence
status: draft
audience: "研究负责人、审阅者与复现执行者"
owner: "研究生维护"
last_reviewed: 2026-08-23
primary_layer: L3
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
| K5 17-task 训练 | planned；仅允许在用户授权的 surrogate training 门后运行 |
| K5 B→G/G→B | pending K5 checkpoint freeze 与 six-domain evaluation |
| K5↔K20、K5↔F50 | pending；独立随机样本的迁移诊断，不是 paired-K 因果效应或公平训练排名 |
| WSL real-MPB / 物理验证 | 未完成；不由本页或训练作业升级 |

完成后本页只汇总 K5 native、B→G/G→B、K5↔K20、K5↔F50、baseline/upgrade 对比及汇总训练曲线。510 行 raw cells、预测数组、per-band/per-k/bootstrap/resource 明细仅链接 run 证据，不复制到正文。

相关协议：[EXP-FWD-007](../../../experiments/research/EXP-FWD-007-K5训练与六域验证.md)。计划 run 根目录为 `runs/{date}/EXP-FWD-007_K5_*` 与 `runs/{date}/EXP-FWD-007_six_domain_*`。
