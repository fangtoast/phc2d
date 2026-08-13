---
title: "phc2d 总报告图件索引"
summary: "报告图件及来源索引；用于核对资产和证据出处。"
page_type: asset-index
status: draft
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 4 minutes
primary_layer: L3
---

<!--
Purpose: Long-term index for curated phc2d report figures and their full run galleries.
Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
-->

# phc2d 总报告图件索引

本目录只保存适合长期引用、进入 Git 的精选图。完整候选画廊、单案例页面和绘图运行记录仍保存在 `runs/`；数值与文件来源见 [`source_index.csv`](source_index.csv) 和 [`artifact_manifest.json`](artifact_manifest.json)。

## 项目模型与结果

| 图件 | PNG | SVG | 证据边界 |
|---|---|---|---|
| 模型族与证据状态 | [PNG](results/model_inventory_overview.png) | [SVG](results/model_inventory_overview.svg) | 区分实现、历史、占位和 blocked，不是性能排行榜 |
| A1 Full-band 四模型 | [PNG](results/a1_forward_model_comparison.png) | [SVG](results/a1_forward_model_comparison.svg) | 10k V2、三 seed、公平协议 |
| A1 Query 三模型三协议 | [PNG](results/a1_query_model_comparison.png) | [SVG](results/a1_query_model_comparison.svg) | 最终指标；未保存逐 epoch history |
| B3 Field U-Net/FNO/PINO | [PNG](results/b3_field_model_comparison.png) | [SVG](results/b3_field_model_comparison.svg) | PINO 是 `fno_field_v2` 上的训练制度 |
| 原 A2 四方法 100-target（已撤销） | [PNG](results/a2_100_model_comparison.png) | [SVG](results/a2_100_model_comparison.svg) | 仅保留历史谱系；不得进入当前结果、比较或结论 |
| Objective-bridge 625-target | [PNG](results/a2_bridge_625_comparison.png) | [SVG](results/a2_bridge_625_comparison.svg) | 当前配对结果与不同谱系历史参考分面展示 |

## 流程与架构图

| 图件 | SVG | 证据边界 |
|---|---|---|
| 候选筛选与可信物理闭环 | [SVG](figure_8_1_candidate_physics_loop.svg) | $K=128$ 是候选预算，Top-5 是 real-MPB 求解预算；Query 代理只作交叉一致性门控 |

## 学习曲线

| 图件 | PNG | SVG | 说明 |
|---|---|---|---|
| A1 Full-band（含独立 `mlp` run） | [PNG](learning_curves/a1_forward_learning_curves.png) | [SVG](learning_curves/a1_forward_learning_curves.svg) | 每个 LR 的 seed 均值与范围 |
| B3 U-Net/FNO | [PNG](learning_curves/b3_field_learning_curves.png) | [SVG](learning_curves/b3_field_learning_curves.svg) | 监督 data loss；PINO 曲线见 P4/P5/P6 |
| 原 A2（已撤销） | [PNG](learning_curves/a2_original_learning_curves.png) | [SVG](learning_curves/a2_original_learning_curves.svg) | 仅用于定位 reconstruction-only 训练问题，不构成有效 inverse 结果 |
| A2 objective bridge | [PNG](learning_curves/a2_objective_bridge_learning_curves.png) | [SVG](learning_curves/a2_objective_bridge_learning_curves.svg) | 主 objective 与候选健康正则 |
| P4/P5 | [PNG](learning_curves/p4_p5_learning_curves.png) | [SVG](learning_curves/p4_p5_learning_curves.svg) | seed 范围，不是置信区间 |
| P6 | [PNG](learning_curves/p6_learning_curves.png) | [SVG](learning_curves/p6_learning_curves.svg) | 只画实际存在 epoch，partial 状态不提升 |

学习曲线的完整覆盖清单、partial 状态和源 history 哈希位于本地运行目录
`runs/20260729/training_learning_curves_all_models_202607292223/`；该原始目录不进入 Git。

## Inverse 结构案例

- 正文配对四目标：[PNG](inverse/paired_main4.png) · [SVG](inverse/paired_main4.svg)
- 完整 8 目标配对、best/worst 24 例、random40、独立单目标页与冻结 selection manifest 位于本地目录
  `runs/20260728/a2_inverse_objective_bridge_seed0_test625_v1_202607280842/figures/inverse_examples_v2_paired/`；该原始目录不进入 Git。

案例由已冻结 625-target real-MPB 结果后验选择，只用于可视化，不参与训练、checkpoint 或候选筛选。
