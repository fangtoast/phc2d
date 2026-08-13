---
title: "Inverse MPB Replay Mixture V2 10000 Test625 Report"
summary: "日期化实验与验证证据；正文保持原始记录。"
page_type: evidence
status: draft
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 8 minutes
primary_layer: L3
evidence_date: 2026-07-06
evidence_scope: "日期化证据或历史记录；正文未因站点重构而改写。"
---

# Inverse MPB Replay Mixture V2 10000 Test625 Report

## 摘要
- **目的**：基于原始记录保留实验链路与指标定义，对文档进行中文化结构化重构。
- **状态**：已完成中文主语体补充，原始英文命令、路径、文件名、模型字段与数值保持原文不变。
- **文件角色**：Experimental Technical Report | **原始语言比例**：英文主导（0%）
- **证据入口**：runs/20260706/inverse_mpb_replay_mixture_v2_10000_test625_summary_20260706224424/summary_tables/candidate_preflight_metrics.csv`.；runs/20260706/inverse_mpb_replay_mixture_v2_10000_test625_summary_20260706224424/figures/band_curve_sample100/structure_band_gallery_sample100.png`.；runs/20260706/inverse_mpb_replay_mixture_v2_10000_test625_summary_20260706224424/figures/band_curve_sample100/structure_band_gallery_sample100.svg`.；runs/20260706/inverse_mpb_replay_mixture_v2_10000_test625_summary_20260706224424/figures/band_curve_sample100/band_curve_sample100_overview.png`.；runs/20260706/inverse_mpb_replay_mixture_v2_10000_test625_summary_20260706224424/figures/band_curve_sample100/band_curve_sample100_overview.svg`.

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


## Evidence Boundary

- Status: `complete`.
- No training was run in this replay summary path; candidates come from the fixed inverse generator checkpoint and frozen query/full-Wh surrogates.
- Replay backend is required to be `real_mpb`; Windows-local surrogate or smoke results are not counted as formal validation.
- Target split is the pure dataset `idx_test` split: 625 targets, expected index range 9375..9999.
- Headline metrics use bands 2-10 only; band1/full-band views are retained only as risk diagnostics.

## Candidate Preflight

- Target count: 625 / 625.
- Replay top-M: 5.
- Shard count complete: 13 / 13.
- Primary candidate pool: `surrogate_ranked_candidates`.
- Validity / unique rate: 1.0 / 1.0.
- Nearest-neighbor Hamming min / p10 / mean: 0.02777777798473835 / 0.02777777798473835 / 0.04047156998515129.
- Fill fraction mean: 0.46821752071656547.
- Candidate preflight table: `runs/20260706/inverse_mpb_replay_mixture_v2_10000_test625_summary_20260706224424/summary_tables/candidate_preflight_metrics.csv`.

## Real-MPB Replay

- Mean bands 2-10 replay MRE: 3.0775461763527154.
- Median / p10 / p90: 2.654285651485456 / 1.899344664285724 / 4.810185812267845.
- Min / max: 0.8679596446911464 / 11.347585863113785.
- Bootstrap 95% CI for mean: [2.9658208137984485, 3.1827767486528185].

## Runtime

- Sum MPB replay time: 11100.917903844998 s.
- Sum shard wall time: 11100.931917340997 s.
- MPB replay time per target / candidate: 17.761468646151997 s / 3.5522937292303993 s.

## Gap Analysis

- Mean surrogate-replay gap bands 2-10: 1.4453926733519793.
- Pearson r: 0.13502965089149965.
- Spearman r: 0.1963023347259769.

## Band Curve Samples

- Sample count: 100 targets, uniformly sampled without replacement with seed 20260707.
- Structure + band gallery: `runs/20260706/inverse_mpb_replay_mixture_v2_10000_test625_summary_20260706224424/figures/band_curve_sample100/structure_band_gallery_sample100.png`.
- Structure + band gallery SVG: `runs/20260706/inverse_mpb_replay_mixture_v2_10000_test625_summary_20260706224424/figures/band_curve_sample100/structure_band_gallery_sample100.svg`.
- Band-only overview figure: `runs/20260706/inverse_mpb_replay_mixture_v2_10000_test625_summary_20260706224424/figures/band_curve_sample100/band_curve_sample100_overview.png`.
- Band-only overview SVG: `runs/20260706/inverse_mpb_replay_mixture_v2_10000_test625_summary_20260706224424/figures/band_curve_sample100/band_curve_sample100_overview.svg`.
- Individual full-band overlays: `runs/20260706/inverse_mpb_replay_mixture_v2_10000_test625_summary_20260706224424/figures/band_curve_sample100/individual/`.
- Sample index: `runs/20260706/inverse_mpb_replay_mixture_v2_10000_test625_summary_20260706224424/figures/band_curve_sample100/band_curve_sample100_index.csv`.
- Surrogate curves are recovered from `selected_surrogate_predictions` in the no-train sample `candidates.json`, matched by `best_candidate_selected_index`; recovered count is 625 / 625.

## Worst/Best Cases

- Worst 20 table: `runs/20260706/inverse_mpb_replay_mixture_v2_10000_test625_summary_20260706224424/summary_tables/worst_20_targets.csv`.
- Best 20 table: `runs/20260706/inverse_mpb_replay_mixture_v2_10000_test625_summary_20260706224424/summary_tables/best_20_targets.csv`.
- Best 24 structure + band gallery: `runs/20260706/inverse_mpb_replay_mixture_v2_10000_test625_summary_20260706224424/figures/best_worst24/structure_band_gallery_best24.png`.
- Worst 24 structure + band gallery: `runs/20260706/inverse_mpb_replay_mixture_v2_10000_test625_summary_20260706224424/figures/best_worst24/structure_band_gallery_worst24.png`.
- Best 24 index: `runs/20260706/inverse_mpb_replay_mixture_v2_10000_test625_summary_20260706224424/figures/best_worst24/best24_index.csv`.
- Worst 24 index: `runs/20260706/inverse_mpb_replay_mixture_v2_10000_test625_summary_20260706224424/figures/best_worst24/worst24_index.csv`.

## Baseline Comparison

- Baseline is the current 40-target formal replay, so it is a scale/reference comparison only, not a same-N statistical replacement.
- 40-target baseline mean replay MRE bands 2-10: 3.0135754422062084.

## Artifacts

- Summary run: `runs/20260706/inverse_mpb_replay_mixture_v2_10000_test625_summary_20260706224424`.
- Artifact index: `runs/20260706/inverse_mpb_replay_mixture_v2_10000_test625_summary_20260706224424/artifact_index.json`.
- Metrics summary: `runs/20260706/inverse_mpb_replay_mixture_v2_10000_test625_summary_20260706224424/metrics_summary.json`.
