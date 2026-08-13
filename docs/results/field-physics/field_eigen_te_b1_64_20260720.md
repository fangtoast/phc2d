---
title: "B1 TE Bloch 本征场：64 条 real-MPB field-data pilot（2026-07-20）"
summary: "日期化实验与验证证据；正文保持原始记录。"
page_type: evidence
status: draft
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 3 minutes
primary_layer: L3
evidence_date: 2026-07-20
evidence_scope: "日期化证据或历史记录；正文未因站点重构而改写。"
---

<!--
  B1 64-record real-MPB field-data pilot evidence.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="field-eigen-te-b1-64-20260720"></a>

# B1 TE Bloch 本征场：64 条 real-MPB field-data pilot（2026-07-20）

本报告仅确认 B1 的 64 条可追溯 field-data pilot 已通过。它不是 field 模型训练、PINO/MaxwellNet、physics-aware inverse 或跨方法性能结论。

## 结果

有效运行目录为本地 `runs/20260720/field_eigen_te_b1_64_202607201021`，在 WSL `Ubuntu` + conda `meep_env` 执行，状态为 `b1_64_field_data_pilot_pass`。

| 项目 | 实际值 |
|---|---:|
| 完成记录 | 64 / 64 |
| 失败记录 | 0 |
| source split | train/val/test=`48/8/8` |
| bands 2–10 | `8/7/7/7/7/7/7/7/7`，最大差 1 |
| physical-k strata | 8 个去重物理坐标 |
| worker | 4 个隔离子进程，`fail_fast=false` |
| wall time | `80.8549 s` |
| MPB 累计求解时间 | `239.3463 s` |
| worker 吞吐 | `0.8020 records/s` |

数据集为本地 `runs/20260720/field_eigen_te_b1_64_202607201021/field_eigen_te_b1_64.h5`，选择 manifest 为同目录的 `field_selection_manifest.json`，其 SHA256 为 `707ce28b86f794ba6f47729684b96d915379d2b3b0f908f68a3ac62b07df7b34`。

## 字段与物理审计

HDF5 含 native `complex64` 的 `efield_complex`/`hfield_complex`、`source_split`、Bloch phase、`dx/dy`、field normalization 与每条 solver provenance。全部 64 条 E/H 场均有非零虚部；E 场 cell-volume L2 范围为 `[0.99999994, 1.0]`，最大相位锚点虚部为 `6.52e-16`。所有冻结 provenance 字段均存在。

10k source 的 k-path sidecar 是 `shards_merged`；本运行保留了它与已验证 1k reference 坐标逐点相同、`k_points_hash` 相同的证明。此证明仅继承 physical-k 坐标验证，不继承其他 dataset 或 replay claim。

首次 64-record 尝试 `runs/20260720/field_eigen_te_b1_64_202607201018` 因 worker 强制转换为 `float32` 而丢失虚部，已明确作废，不能引用为 B1 证据。有效运行已修复为保留 native complex shard。

## 边界与下一门槛

本批次状态为 `pilot_pass`，不进入 `formal_completed`。不启动任何训练；B2/B4 仍 blocked，直到完成 Maxwell residual 单元测试以及 solver reference 相对随机场的区分测试。

[↑ 回顶部](#field-eigen-te-b1-64-20260720)
