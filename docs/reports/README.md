<!--
  Snapshot-report index for the synchronized phc2d workspace.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 综合报告与公式参考

本目录保存带明确证据日期的综合报告、公式参考、模型清单和精选图件。它们适合系统阅读和汇报，但不自动代表今天的最新状态；当前状态先看 [`../../PROJECT_MAP.md`](../../PROJECT_MAP.md)。

| 角色 | 入口 | 边界 |
|---|---|---|
| 完整项目叙述 | [`phc2d_project_full_introduction_20260724.md`](phc2d_project_full_introduction_20260724.md) | 截至文档证据更新时间的技术快照 |
| 全胞双路线模型专题 | [`全胞双路线正向与逆向模型报告_20260810.md`](全胞双路线正向与逆向模型报告_20260810.md) | 50k 正向 baseline/升级结构与结果；inverse 仍为 10k preflight/未正式训练 |
| 科学动机与证据链 | [`phc2d_bandgap_field_inverse_design_rationale_20260806.md`](phc2d_bandgap_field_inverse_design_rationale_20260806.md) | 解释带隙、近远场与逆向设计为何需要联合研究；不升级实验状态 |
| 物理与数据公式 | [`phc2d_physics_data_formula_guide_20260803.md`](phc2d_physics_data_formula_guide_20260803.md) | 定义、实现回链与历史指标边界 |
| 模型状态清单 | [`phc2d_model_inventory_v1.yaml`](phc2d_model_inventory_v1.yaml) | 注册、实现、验证、历史和 blocked 分字段记录 |
| 精选图件索引 | [`assets/phc2d_project_full_introduction_20260724/figure_index.md`](assets/phc2d_project_full_introduction_20260724/figure_index.md) | 来源路径、SHA-256 与使用边界 |
| 全胞模型图件索引 | [`assets/full_cell_forward_inverse_report_20260810/figure_index.md`](assets/full_cell_forward_inverse_report_20260810/figure_index.md) | 九张 baseline、升级、结果和 inverse 架构图及来源边界 |

单项实验的当前结论见 [`../results/README.md`](../results/README.md)。修改完整报告前必须运行模型覆盖、来源哈希和本地链接审计；surrogate、field residual 与 real-MPB replay 不可互相替代。
