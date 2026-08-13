---
title: "10k V2 Inverse Retrain, Candidate Preflight, And Real-MPB Replay Report"
summary: "日期化实验与验证证据；正文保持原始记录。"
page_type: evidence
status: draft
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 12 minutes
primary_layer: L3
evidence_date: 2026-07-06
evidence_scope: "日期化证据或历史记录；正文未因站点重构而改写。"
---

<!--
  Inverse retrain, candidate-preflight, and replay report for the 10k V2 photonic-crystal run.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 10k V2 Inverse Retrain, Candidate Preflight, And Real-MPB Replay Report

## 摘要
- **目的**：基于原始记录保留实验链路与指标定义，对文档进行中文化结构化重构。
- **状态**：已完成中文主语体补充，原始英文命令、路径、文件名、模型字段与数值保持原文不变。
- **文件角色**：Experimental Technical Report | **原始语言比例**：英文主导（0%）
- **证据入口**：runs/20260704/inverse_mixture_generator_v2_10000_pretrain_202607041725`；runs/20260704/inverse_mixture_generator_v2_10000_objective_202607041740`；runs/20260706/inverse_mixture_generator_v2_10000_objective_sample_202607060836`；runs/20260706/inverse_mixture_generator_v2_10000_objective_sample_separation_202607060848`；runs/20260706/inverse_mpb_replay_mixture_v2_10000_40_202607060853`

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


- Report date: 2026-07-06
- Repository: `D:\project\article_reproduction\POD-DeepONET-2d(codex)`
- Dataset: `data/query_dataset_v2_pilot_10000_mpb.npz`
- Dataset SHA256: `991cb3810e881938755495ad0bccfa5b091194c2b74451842a9a50ddce974ad4`
- Formal environment: WSL `Ubuntu`, conda `meep_env`, CUDA `NVIDIA GeForce RTX 5090`
- Replay boundary: `formal_inverse_candidate_mpb_replay`, backend `real_mpb`

## Executive Conclusion

The inverse sequence now passes candidate preflight after a binary candidate-separation repair, and the repaired candidate pool passes 40-target real-MPB replay.

The first 2026-07-06 preflight failed only because `nearest_neighbor_hamming_p10` was below the one-bit gate. The repair adds explicit binary de-duplication and `min_hamming: auto_1_over_dim` before surrogate ranking. With the same generator checkpoint and frozen surrogates, the repaired preflight passes all replay-entry gates.

| Stage | Result | Evidence |
|---|---|---|
| inverse pretrain | completed | `runs/20260704/inverse_mixture_generator_v2_10000_pretrain_202607041725` |
| inverse objective finetune | completed | `runs/20260704/inverse_mixture_generator_v2_10000_objective_202607041740` |
| first candidate preflight | failed p10 gate | `runs/20260706/inverse_mixture_generator_v2_10000_objective_sample_202607060836` |
| repaired candidate preflight | passed | `runs/20260706/inverse_mixture_generator_v2_10000_objective_sample_separation_202607060848` |
| 40-target real-MPB replay | passed | `runs/20260706/inverse_mpb_replay_mixture_v2_10000_40_202607060853` |

## Frozen Surrogate Inputs

Manifest: `runs/20260704/frozen_surrogate_manifest_v2_10000_202607041715/frozen_surrogates.json`

| Role | Checkpoint | Gate metric |
|---|---|---:|
| full-Wh cross surrogate | `runs/20260702/forward_formal_retrain_v2_10000_202607021842/cases/epsilon_grid_fno2d_band_v2_2_lr1em03_seed2/train/model.pt` | bands 2-10 MRE `0.870688796043396%` |
| query surrogate | `runs/20260702/query_formal_retrain_v2_10000_202607022213/cases/unseen_structure_all_k_fno2d_query_v2_seed1_lr1em03.pt` | bands 2-10 MRE `1.078163981437683%` |

The query all-k seed mean/std is `1.0911012093226116% / 0.00959650601397462%`. The harder heldout-segment query protocol remains diagnostic-only and is not counted as a passing gate.

## Inverse Training

Reconstruction pretrain:

| Field | Value |
|---|---|
| run | `runs/20260704/inverse_mixture_generator_v2_10000_pretrain_202607041725` |
| seeds | `0, 1, 2` |
| learning rates | `1e-3, 3e-4, 1e-4` |
| best checkpoint | `runs/20260704/inverse_mixture_generator_v2_10000_pretrain_202607041725/model_seed0_lr3em04.pt` |
| best final train loss | `0.23111790225461679` |
| wall time | `452.13922618999993 s` |

Surrogate-objective finetune:

| Field | Value |
|---|---|
| run | `runs/20260704/inverse_mixture_generator_v2_10000_objective_202607041740` |
| validation boundary | `inverse_objective_training_surrogate_side_no_mpb_meep` |
| best checkpoint | `runs/20260704/inverse_mixture_generator_v2_10000_objective_202607041740/objective_model_seed2_lr1em04.pt` |
| best final objective total loss | `0.03334046037936652` |
| best surrogate objective loss | `0.018862281011900416` |
| best cross-surrogate disagreement, bands 2-10 | `0.01392080010501323` |
| wall time | `72374.02944594799 s` |

These training metrics are surrogate-side only; acceptance depends on preflight plus real-MPB replay below.

## Candidate Preflight Repair

Initial preflight:

| Metric | Threshold | Actual | Result |
|---|---:|---:|---|
| `validity_rate` | `= 1.0` | `1.0` | pass |
| `unique_rate` | `>= 0.45` | `0.89609375` | pass |
| `nearest_neighbor_hamming_p10` | `>= 1 / 36 = 0.02777778` | `0.00138889` | fail |
| `largest_duplicate_cluster_fraction` | `<= 0.10` | `0.02402344` | pass |
| `best_of_k_surrogate_mre_bands_2_10` | `<= 0.025` | `0.01629412` | pass |
| `cross_surrogate_disagreement_bands_2_10` | `<= 0.025` | `0.01385615` | pass |

Repair implementation:

- Code: `src/phc2d/inverse/candidates.py`, `src/phc2d/inverse/config.py`, `src/phc2d/inverse/runner.py`
- Config: `configs/research/inverse_mixture_generator_v2_10000_preflight_separation.yaml`
- Repair mode: `inverse.candidate_separation.enabled=true`, `deduplicate_binary=true`, `min_hamming=auto_1_over_dim`

Repaired preflight:

| Metric | Threshold | Actual | Result |
|---|---:|---:|---|
| `validity_rate` | `= 1.0` | `1.0` | pass |
| `unique_rate` | `>= 0.45` | `1.0` | pass |
| `nearest_neighbor_hamming_p10` | `>= 1 / 36 = 0.02777778` | `0.02777778` | pass |
| `nearest_neighbor_hamming_min` | `>= 1 / 36 = 0.02777778` | `0.02777778` | pass |
| `largest_duplicate_cluster_fraction` | `<= 0.10` | `0.0` | pass |
| `best_of_k_surrogate_mre_bands_2_10` | `<= 0.025` | `0.01630862` | pass |
| `cross_surrogate_disagreement_bands_2_10` | `<= 0.025` | `0.01387544` | pass |

Repaired preflight artifacts:

| Artifact | Path |
|---|---|
| run | `runs/20260706/inverse_mixture_generator_v2_10000_objective_sample_separation_202607060848` |
| metrics | `runs/20260706/inverse_mixture_generator_v2_10000_objective_sample_separation_202607060848/metrics.json` |
| candidates | `runs/20260706/inverse_mixture_generator_v2_10000_objective_sample_separation_202607060848/candidates.json` |
| ranked candidates | `runs/20260706/inverse_mixture_generator_v2_10000_objective_sample_separation_202607060848/ranked_candidates.csv` |

## Real-MPB Replay

- Config: `configs/research/inverse_mpb_replay_mixture_v2_10000_40.yaml`
- Run: `runs/20260706/inverse_mpb_replay_mixture_v2_10000_40_202607060853`
- Candidate source: `runs/20260706/inverse_mixture_generator_v2_10000_objective_sample_separation_202607060848/candidates.json`
- Targets: `40`
- Replay top-M: `5`
- Backend: `real_mpb`
- Validation boundary: `formal_inverse_candidate_mpb_replay`
- Success rate: `1.0`
- Replay time: `908.1292843800004 s`

| Replay metric | Threshold | Actual | Result |
|---|---:|---:|---|
| `best_of_k_replay_mre_bands_2_10_pct` | `<= 10%` | `3.0135754422062084%` | pass |
| `surrogate_replay_gap_bands_2_10_pct` | `<= 6%` | `1.382713755197917%` | pass |
| `mpb_replay_success_rate` | `= 1.0` | `1.0` | pass |

Replay artifacts:

| Artifact | Path |
|---|---|
| metrics | `runs/20260706/inverse_mpb_replay_mixture_v2_10000_40_202607060853/metrics.json` |
| results | `runs/20260706/inverse_mpb_replay_mixture_v2_10000_40_202607060853/results.json` |
| target bands | `runs/20260706/inverse_mpb_replay_mixture_v2_10000_40_202607060853/target_bands.npy` |
| surrogate predictions | `runs/20260706/inverse_mpb_replay_mixture_v2_10000_40_202607060853/surrogate_pred_bands.npy` |
| MPB replay bands | `runs/20260706/inverse_mpb_replay_mixture_v2_10000_40_202607060853/mpb_replay_bands.npy` |
| per-target metrics | `runs/20260706/inverse_mpb_replay_mixture_v2_10000_40_202607060853/per_target_replay_metrics.csv` |
| per-band metrics | `runs/20260706/inverse_mpb_replay_mixture_v2_10000_40_202607060853/per_band_replay_metrics.csv` |
| replay candidate trace | `runs/20260706/inverse_mpb_replay_mixture_v2_10000_40_202607060853/replay_candidates.jsonl` |

## Interpretation

The candidate-separation repair did not materially change the surrogate best-of-k result: `0.01629412` before repair versus `0.01630862` after repair. It did remove binary duplicate neighborhoods and lifted `nearest_neighbor_hamming_p10` to the exact one-bit threshold required by the replay-entry gate.

The real-MPB replay result supports the current inverse candidate chain on this fixed 40-target gate: replay MRE is `3.0136%` for bands 2-10, and the surrogate-replay gap is `1.3827%`. Both are below the formal thresholds in the plan.

The band-1/full-band MRE values remain unsuitable as headline inverse metrics because band 1 has known near-zero denominator sensitivity. The acceptance headline should stay on bands 2-10.