---
title: "全胞正向与逆向模型报告图件索引"
summary: "报告图件及来源索引；用于核对资产和证据出处。"
page_type: asset-index
status: draft
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 3 minutes
primary_layer: L3
---

<!--
Purpose: Curated figure index for the 2026-08-10 full-cell forward/inverse model report.
Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 全胞正向与逆向模型报告图件索引

本目录保存新报告的可复现精选图。PNG 用于 Markdown 阅读，SVG 用于论文或汇报后续编辑；文件 hash 见 [`artifact_manifest.json`](artifact_manifest.json)，模型、配置和结果机器源见 [`source_index.csv`](source_index.csv)。

| 图件 | PNG | SVG | 证据边界 |
|---|---|---|---|
| 全胞双路线数据生成链路 | [PNG](figure_2_1_data_generation_pipeline.png) | [SVG](figure_2_1_data_generation_pipeline.svg) | 解析结构是 MPB 真值；三种栅格是同步学习/审计视图，不新增数据验收结论 |
| MLP/ResMLP 同容量结构 | [PNG](figure_4_1_baseline_mlp_residual.png) | [SVG](figure_4_1_baseline_mlp_residual.svg) | 说明 residual 受控差异；全胞数值仍以 50k 结果为准 |
| Lifted CNN baseline | [PNG](figure_4_2_baseline_lifted_cnn.png) | [SVG](figure_4_2_baseline_lifted_cnn.svg) | 说明最小局部卷积下界，不代表现代 CNN |
| FNO v2 baseline | [PNG](figure_4_3_baseline_fno2d_band.png) | [SVG](figure_4_3_baseline_fno2d_band.svg) | 项目原生 band-aware FNO-style 身份 |
| 共享 Full-Wh 输出头 | [PNG](figure_5_1_full_wh_heads.png) | [SVG](figure_5_1_full_wh_heads.svg) | `k_conditioned` 是接口能力；本轮选择的是 dense |
| 两种 MLP 升级 | [PNG](figure_5_2_mlp_upgrades.png) | [SVG](figure_5_2_mlp_upgrades.svg) | 展示 Mixer 与 Gated ResMLP 的张量流和 shape 边界 |
| ConvNeXt upgrade | [PNG](figure_5_3_convnext_band.png) | [SVG](figure_5_3_convnext_band.svg) | 周期性来自 block circular padding，不泛化为所有层均周期 |
| TFNO 与 dual-lane | [PNG](figure_5_4_tfno_dual_lane.png) | [SVG](figure_5_4_tfno_dual_lane.svg) | 项目原生 Tucker-factorized operator；不是官方 TFNO 复刻 |
| 50k native 九模型对照 | [PNG](figure_6_1_native_50k_model_comparison.png) | [SVG](figure_6_1_native_50k_model_comparison.svg) | 同一 held-out test 的 surrogate MRE；不是组件消融或 real-MPB replay |
| Inverse Stage-A 网络 | [PNG](figure_7_1_inverse_stage_a_models.png) | [SVG](figure_7_1_inverse_stage_a_models.svg) | 已实现结构合同；不代表正式 45-run、Stage B 或 replay 完成 |

全部十张图由 [`src/phc2d/plot/full_cell_model_architectures.py`](https://github.com/fangtoast/phc2d/blob/main/src/phc2d/plot/full_cell_model_architectures.py) 生成；图中的数据流、模型名称与参数对应 2026-08-10 的配置、registry 和实现快照。
