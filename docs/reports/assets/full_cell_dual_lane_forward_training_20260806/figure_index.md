---
title: "全胞正向训练精选图件索引"
summary: "报告图件及来源索引；用于核对资产和证据出处。"
page_type: asset-index
status: draft
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 6 minutes
primary_layer: L3
---

<a id="full-cell-forward-figure-index"></a>

<!--
  Curated figure source and hash index for the full-cell forward report.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 全胞正向训练精选图件索引

本目录保存报告正文长期引用的精选 PNG/PDF。完整 history、逐 run 制品和原始画廊仍位于 `runs/20260806/`；本索引不把图件提升为真实 MPB 证据。

## 汇总图

| 图件 | PNG SHA-256 | PDF SHA-256 | 数值来源与边界 |
|---|---|---|---|
| [完整四模型 validation 矩阵](main_model_validation_matrix.png) | `e0262573bdb43b5b04e3f9e0c9743df8208841f559aa81f17f99a4aedcdd7b61` | `8c2c9d0b3d762705d96d16ec931bbacec699708efa79ce819b85fa07d4046579` | `runs/20260806/full_cell_forward_main_campaign_20260806/main_matrix_leaderboard.csv`，SHA-256 `e8715090421207c4e3adf5f23ffc0f9829731cb438388a3a2e42f9d2097b6131`；84-run validation，非 test |
| [双向跨路线泛化](cross_route_generalization.png) | `828ca6db86bdda62e23757b82d7729d0cabb173fa22bf6d0e708cae1efc31eba` | `edc8bfbeacbe937faf6cd4706713d68534b0c90766d6fe13de0cb5b6e535e5a3` | `runs/20260806/full_cell_forward_cross_eval_20260806/cross_evaluation_summary.csv`，SHA-256 `45f58683aeafee65d52c3e34f4c85b2d18795f971b755c6a85cd67478150361f`；source normalization 保留，selection influence=false |
| [Graded 内部 6×6](graded_internal_cross_representation.png) | `b458d48a922392ecf9e50068e600d7f4683639490c589ed3eeff39664d8959fe` | `f66f0e664ed6ff223089d878634fd4686c498d67db790a8ff517145f564157c4` | `runs/20260806/full_cell_forward_cross_eval_20260806/figures/graded_6x6_cross_evaluation.*`；非 real-MPB |
| [分辨率消融](resolution_ablation.png) | `a21fe9f9d205a3d2c04ecc39ed8d1b26f77a5cc01b13bf03662335d63f4cef94` | `1062666501a6a68ee5e5ebb1c51b8455e238b7afc7ae8b44ca90b187b0311354` | `runs/20260806/full_cell_forward_resolution_acceptance_20260806/figures/resolution_ablation.*`；各分辨率独立训练 |
| [冻结后 test](final_test_comparison.png) | `73c5ad6caea2608e010aafcded683a57d98b05be746f9a19aed7cd5e6c39eed0` | `9feba366af1b4d0291ed301412e86ee1117bcaaab49f743d3642dc815400968e` | `runs/20260806/full_cell_forward_final_test_20260806/figures/final_test_comparison.*`；仅冻结胜者 |

## 七个波次的完整训练曲线

| 波次 | PNG | PNG SHA-256 | PDF SHA-256 | 原始来源 |
|---|---|---|---|---|
| binary | [PNG](training_curves_binary.png) | `0e0d1801534c26f28849f24cf764cd05cd6329297466a83678301ece568a3718` | `543fd8f636853f122aea44a429d771775954d3b40807ed304e49bff9105fc99d` | `runs/20260806/full_cell_forward_binary_native_v1_wave_aggregation_20260806/figures/training_curves_by_model.*` |
| continuous | [PNG](training_curves_continuous.png) | `7d8bfb1a717849757def1cab84eb254b88b91acc263fd71bc747f4488a723e45` | `feb2aab51cfffc842139a81deecaacb372f8140945fd34bef2a1bdf7d4accb5f` | `runs/20260806/full_cell_forward_graded_continuous_v1_wave_aggregation_20260806/figures/training_curves_by_model.*` |
| t0.3 | [PNG](training_curves_t03.png) | `8c51ca864986fd5eeb436d7531f414491031c51dddfd3b78e141eaab391d090f` | `7ad1dbdc55af31c0a50ccfe9ccd1747c58ae0f2c5d836605b9dc8894f371b615` | `runs/20260806/full_cell_forward_graded_t03_v1_wave_aggregation_20260806/figures/training_curves_by_model.*` |
| t0.4 | [PNG](training_curves_t04.png) | `ad07e260cf19b49c2645f8546eabbb21ad76dd70eeb81c9cdec30fe063df9ef4` | `d73a0e74e1d74f01266c2b4c69378ef00267f22687f46705681ae457938950f4` | `runs/20260806/full_cell_forward_graded_t04_v1_wave_aggregation_20260806/figures/training_curves_by_model.*` |
| t0.5 | [PNG](training_curves_t05.png) | `fe9cb0ad80de2edede2c26d34252b54c9ff20a515c074e77bab54538d3fb50b3` | `c667d585c0800083795f950e51c9b9285f6e8e6f46b1aa5213ce7b54038c3f93` | `runs/20260806/full_cell_forward_graded_t05_v1_wave_aggregation_20260806/figures/training_curves_by_model.*` |
| t0.6 | [PNG](training_curves_t06.png) | `c17e1e30c035a9d59faf108008d27a1e7025af9c9f0a0b64bb3ac2a0ed093e8f` | `8f8af81902d6d338bc11732a5e00ee94542b2100c329efcbfb905e3880b72254` | `runs/20260806/full_cell_forward_graded_t06_v1_wave_aggregation_20260806/figures/training_curves_by_model.*` |
| t0.7 | [PNG](training_curves_t07.png) | `8e6fd7a90026bbc0a64373bed86cfcb9a603a1954010928c7f2e24784423fce8` | `c8c4676670571b2efef897ff7bd913287e4585c93fd287e616d1527ef527af83` | `runs/20260806/full_cell_forward_graded_t07_v1_wave_aggregation_20260806/figures/training_curves_by_model.*` |

每张训练图均包含 `mlp`、`res_mlp_wide`、`lifted_cnn`、`fno2d_band_v2` 四个面板，以及 seed 0/1/2、均值和 seed 范围。图像是 validation history，不是 test，也不是 MPB replay。
