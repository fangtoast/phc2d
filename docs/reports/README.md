---
title: 综合报告与快照（冻结）
summary: 历史综合报告、解释性快照与精选图件的兼容入口。
page_type: landing
status: draft
audience: 研究用户、结果审阅者与项目维护者
owner: 研究生维护
last_reviewed: 2026-08-21
reading_time: 3 minutes
primary_layer: L4
---

<!--
  Frozen snapshot-report index for phc2d.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 综合报告与快照（冻结）

本目录保留历史长报告、解释性快照、模型清单和精选图件。它不再接收普通实验结果；面向人的当前结论进入[当前结果](../结果/README.md)，明细证据进入[实验与证据库](../results/README.md)。旧路径继续有效，本轮不逐篇改写或搬迁。

| 现有内容 | 当前角色 | 长期去向 |
|---|---|---|
| [全胞双路线正向与逆向模型报告](全胞双路线正向与逆向模型报告_20260810.md) | 日期化实验综合快照 | `结果/` 或 `archive/`，本轮冻结 |
| [带隙、场与逆向设计说明](phc2d_bandgap_field_inverse_design_rationale_20260806.md) | 研究概念解释 | `概念/`，本轮冻结 |
| [项目完整介绍](phc2d_project_full_introduction_20260724.md) | 2026-07 项目快照 | `archive/`，本轮冻结 |
| [物理数据公式总手册](phc2d_physics_data_formula_guide_20260803.md) | 2026-08 长篇参考快照 | `参考/` 或 `archive/`，本轮冻结 |
| [模型状态清单](phc2d_model_inventory_v1.yaml) | 机器可读模型身份与证据边界 | 保留，仍由审计命令使用 |
| [旁支研究](side_project/README.md) | 历史旁支计划与调研 | `archive/`，本轮冻结 |

精选图件继续保留在 `reports/assets/`，并维持来源路径、SHA-256 与证据边界。普通 run 不应在这里生成长篇报告，reports 也不得成为“执行日志 → 更多报告”的默认阅读路径。
