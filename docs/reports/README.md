---
title: 综合报告与快照
summary: 当前跨实验专题总结、历史综合报告、解释性快照与精选图件的兼容入口。
page_type: landing
status: draft
audience: 研究用户、结果审阅者与项目维护者
owner: 研究生维护
last_reviewed: 2026-08-31
reading_time: 3 minutes
primary_layer: L4
---

<!--
  Frozen snapshot-report index for phc2d.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 综合报告与快照

本目录保存跨多个 Experiment ID 的专题综合解释，并保留历史长报告、解释性快照、模型清单和精选图件。普通实验结果仍进入[当前结果](../结果/README.md)和[实验与证据库](../results/README.md)；综合报告不能替代源 run 或独立结果页。

## 当前专题综合

| 报告 | 范围与边界 |
|---|---|
| [模型结构图谱](模型结构图谱_20260909.md) | 基于当前 registry 和实现绘制 27 个注册模型，并单列最新 `global_local_band_operator_v1`；只说明 L2 架构，不新增性能或物理证据 |
| [工作频率与带隙探索总结](工作频率与带隙探索总结_20260831.md) | 汇总 EXP-FWD-012/014/015 与 EXP-PHY-005/006；区分理论频率、gap 标签、surrogate loss 和 real-MPB 证据 |

## 历史综合材料

| 现有内容 | 当前角色 | 长期去向 |
|---|---|---|
| [全胞双路线正向与逆向模型报告](全胞双路线正向与逆向模型报告_20260810.md) | 日期化实验综合快照 | `结果/` 或 `archive/`，本轮冻结 |
| [带隙、场与逆向设计说明](phc2d_bandgap_field_inverse_design_rationale_20260806.md) | 研究概念解释 | `概念/`，本轮冻结 |
| [项目完整介绍](phc2d_project_full_introduction_20260724.md) | 2026-07 项目快照 | `archive/`，本轮冻结 |
| [物理数据公式总手册](phc2d_physics_data_formula_guide_20260803.md) | 2026-08 长篇参考快照 | `参考/` 或 `archive/`，本轮冻结 |
| [模型状态清单](phc2d_model_inventory_v1.yaml) | 机器可读模型身份与证据边界 | 保留，仍由审计命令使用 |
| [旁支研究](side_project/README.md) | 历史旁支计划与调研 | `archive/`，本轮冻结 |

精选图件继续保留在 `reports/assets/`，并维持来源路径、SHA-256 与证据边界。普通 run 不应在这里生成长篇报告，reports 也不得成为“执行日志 → 更多报告”的默认阅读路径。
