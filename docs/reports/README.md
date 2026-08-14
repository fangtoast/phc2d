---
title: "综合报告与快照索引"
summary: "区分仍在正式结果范围内的全胞专题报告与已归档的历史综合材料。"
page_type: landing
status: draft
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-14
reading_time: 3 minutes
primary_layer: L4
---

<!--
  Snapshot-report index for the synchronized phc2d workspace.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 综合报告与快照索引

本目录保存带明确证据日期的综合报告、模型清单和精选图件。稳定知识已迁入“项目与物理基础”“数据与方法”和“参考手册”；长报告保留为快照或兼容入口，避免同时维护多份相同公式与结论。

| 分类 | 入口 | 边界 |
|---|---|---|
| 正式专题报告 | [`全胞双路线正向与逆向模型报告_20260810.md`](全胞双路线正向与逆向模型报告_20260810.md) | 只汇总全胞数据与模型专题；明确列出范围外模型，不等于项目全模型报告 |
| 解释页 | [`phc2d_bandgap_field_inverse_design_rationale_20260806.md`](phc2d_bandgap_field_inverse_design_rationale_20260806.md) | 解释研究动机与证据链，不升级任何实验状态 |
| 历史项目快照 | [`phc2d_project_full_introduction_20260724.md`](phc2d_project_full_introduction_20260724.md) | 2026-07 的综合叙述；稳定知识已迁入新章节 |
| 历史公式手册 | [`phc2d_physics_data_formula_guide_20260803.md`](phc2d_physics_data_formula_guide_20260803.md) | 2026-08 的长篇兼容入口；按主题学习应使用新公式页 |
| 模型状态清单 | [`phc2d_model_inventory_v1.yaml`](phc2d_model_inventory_v1.yaml) | 注册、实现、验证、历史和 blocked 分字段记录 |
| 精选图件索引 | [`assets/phc2d_project_full_introduction_20260724/figure_index.md`](assets/phc2d_project_full_introduction_20260724/figure_index.md) | 来源路径、SHA-256 与使用边界 |
| 全胞模型图件索引 | [`assets/full_cell_forward_inverse_report_20260810/figure_index.md`](assets/full_cell_forward_inverse_report_20260810/figure_index.md) | 十张数据、baseline、升级、结果和 inverse 架构图及来源边界 |

旁支研究计划与调研见 [`side_project/README.md`](side_project/README.md)。

单项实验的当前结论见[正式研究结果](../results/README.md)。修改综合报告前必须运行模型覆盖、来源哈希和本地链接审计；surrogate、field residual 与 real-MPB replay 不可互相替代。
