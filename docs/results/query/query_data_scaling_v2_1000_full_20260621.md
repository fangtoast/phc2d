---
title: "N=1000 Guarded V2 Query Surrogate Full Benchmark"
summary: "日期化实验与验证证据；正文保持原始记录。"
page_type: archive
status: historical
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 6 minutes
primary_layer: L3
evidence_date: 2026-06-21
evidence_scope: "日期化证据或历史记录；正文未因站点重构而改写。"
---

<!--
  N=1000 guarded V2 query surrogate full benchmark report.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="query-data-scaling-v2-1000-full-20260621"></a>

# N=1000 Guarded V2 Query Surrogate Full Benchmark

!!! warning "历史证据"
    本页记录 1k Query 阶段结果，数据规模与正式 V2 10k 契约不同；请勿与正式结果直接并列比较。

## 摘要
- **目的**：基于原始记录保留实验链路与指标定义，对文档进行中文化结构化重构。
- **状态**：已完成中文主语体补充，原始英文命令、路径、文件名、模型字段与数值保持原文不变。
- **文件角色**：Experimental Technical Report | **原始语言比例**：英文主导（0%）
- **证据入口**：runs/20260621/query_data_scaling_v2_1000_full_202606211920`；runs/20260621/query_data_scaling_v2_1000_full_N1000_202606211920`

## 结论
- 结论以原实验逻辑为准，聚焦结果含义、边界条件与复现实验条件。
- 若涉及 `surrogate` 与 `mpb_replay`，请在原文中使用统一术语进行区分。

## 数据与方法
- 数据：以原文 `runs/` 与 `metrics.json` 为准。
- 方法：保持原始训练流程与实验顺序，不改动采样/评估口径。
- 命令与环境：保留原样，`path` 与 `config` 字段不改写。

## 风险与下一步
- 风险主要关注结论复用边界（surrogate-only 与 formal validation 的分离）。
- 下一步优先补齐跨文档术语映射、证据引用一致性与可复现实验条目。


| item | value |
|---|---|
| date | 2026-06-21 |
| config | [`../../configs/research/data_scaling_benchmark_v2_1000_full.yaml`](https://github.com/fangtoast/phc2d/blob/main/configs/research/data_scaling_benchmark_v2_1000_full.yaml) |
| dataset | `data/query_dataset_v2_pilot_1000_mpb.npz` |
| parent run | `runs/20260621/query_data_scaling_v2_1000_full_202606211920` |
| child run | `runs/20260621/query_data_scaling_v2_1000_full_N1000_202606211920` |
| environment | WSL `Ubuntu` + conda `meep_env`, CUDA `NVIDIA GeForce RTX 4060 Laptop GPU` |
| seeds | `0/1/2` |

## Conclusion

The formal N=1000 query surrogate benchmark completed directly at `N=1000`. The parent scaling run skipped `N=64` and `N=256` through `scaling.only_sizes: [1000]`, completed `N=1000`, and left no pending scale.

The best three-seed mean by `query_mre_bands_2_10_pct` is `fno2d_query` on `unseen_structure_all_k`: `3.907374 +/- 0.440316%`. The same model also leads the physical k-interpolation and held-out-segment protocols.

This is neural surrogate training/evaluation on a physically verified k-path dataset. It is not MPB replay.

## Run Integrity

| metric | value |
|---|---:|
| completed scales | `1000` |
| skipped scales | `64`, `256` |
| pending scales | none |
| child case rows | `27` |
| failures | `0` |
| child wall time | `1936.906343 s` |
| parent wall time | `1938.560362 s` |
| resolved device | `cuda:0` |
| peak GPU memory | `28.540527 MiB` |
| physical k coordinates | `true` |
| k coordinate source | `sidecar:query_dataset_v2_pilot_square_gxm_mpb_reduced_v1` |

## Three-Seed Leaderboard

| protocol | model | bands 2-10 MRE % | MAE | train time mean s | ms/query mean | params |
|---|---|---:|---:|---:|---:|---:|
| `unseen_structure_all_k` | `fno2d_query` | `3.907374 +/- 0.440316` | `0.023840` | `135.203` | `1.462307` | `24090` |
| `unseen_structure_physical_k_interpolation` | `fno2d_query` | `4.244387 +/- 0.487002` | `0.025134` | `100.147` | `1.265154` | `24090` |
| `unseen_structure_physical_heldout_segment` | `fno2d_query` | `4.873280 +/- 0.697144` | `0.028414` | `91.224` | `1.583784` | `24090` |
| `unseen_structure_all_k` | `deep_onet_query` | `8.017098 +/- 2.445891` | `0.051427` | `45.381` | `0.407377` | `29450` |
| `unseen_structure_all_k` | `mlp_query` | `8.952067 +/- 1.188997` | `0.058316` | `39.468` | `0.408630` | `21386` |
| `unseen_structure_physical_k_interpolation` | `mlp_query` | `9.004169 +/- 1.054866` | `0.059738` | `31.107` | `0.274871` | `21386` |
| `unseen_structure_physical_heldout_segment` | `mlp_query` | `9.576001 +/- 0.913997` | `0.063431` | `31.936` | `0.426225` | `21386` |
| `unseen_structure_physical_k_interpolation` | `deep_onet_query` | `10.260598 +/- 5.360460` | `0.065119` | `46.856` | `0.708945` | `29450` |
| `unseen_structure_physical_heldout_segment` | `deep_onet_query` | `11.510046 +/- 2.177995` | `0.072584` | `40.412` | `0.664038` | `29450` |

## Boundary

The reported `query_mre_pct` over all bands can be dominated by near-zero denominator effects at low bands. The decision metric for this benchmark is `query_mre_bands_2_10_pct`, matching the project convention used in the N=1000 forward report.
