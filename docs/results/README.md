---
title: 正式研究结果
summary: 只收录 V2 10k [10000,31,10] 与新全胞数据 [N,81,10] 的正式证据。
page_type: landing
status: draft
audience: 研究使用者、复现执行者与项目维护者
owner: 研究生维护
last_reviewed: 2026-08-14
reading_time: 5 minutes
primary_layer: L3
---

<!--
  Evidence and report navigation for the synchronized phc2d workspace.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="results-home"></a>

# 正式研究结果

这里是正式结果白名单入口。侧栏按数据身份展开，而不是按磁盘文件夹或实验阶段展开；点击目录名称进入总览，点击右侧箭头展开具体页面。

> **证据边界：** 页面整理不改变实验状态。完整 `run.log`、`metrics.json`、checkpoint 和原始图件仍位于 Git 之外的 `runs/`；报告存在也不表示其结论已通过更高层级的 gate。

## 按数据身份进入

<a id="field-physics-results"></a>
<a id="inverse-results"></a>

| 数据范围 | shape 与任务 | 入口 |
|---|---|---|
| V2 10k | `Wh[10000,31,10]`；A1 Forward/Query、inverse 与 real-MPB replay | [V2 10k 正式结果](V2-10k正式结果.md) |
| 新全胞数据 | `Wh[N,81,10]`；数据验收、10k/50k 正向、升级模型与 Meep | [新全胞数据正式结果](全胞数据正式结果.md) |

## 收录规则

只有同时满足下列条件的页面才保持 `page_type: evidence`：

1. 数据身份属于上述两个范围；
2. 有可追溯的 run、manifest、metrics 或 solver 产物；
3. 页面明确声明 evidence level、status、backend 和结论边界；
4. smoke、preflight、dry-run 不冒充模型或物理结果。

1k Query、64/1k/9216 Field/PINO、环境部署、full-cell inverse preflight 和 representation smoke 已移至[历史档案](../历史档案.md)。它们仍可用于追溯，但不再与正式结果并列。

## 使用规则

1. 先确认 `dataset_scope` 和 target shape，再比较指标。
2. 区分 `surrogate_held_out`、`solver_replay` 与 `physical_validation`。
3. 核对关联 run、配置、hash、seed、候选数和失败案例。
4. 新结果必须更新本索引、相关数据集合页和仓库根目录的 `PROJECT_MAP.md`。

旧站内链接仍可通过 [Forward 兼容索引](forward/README.md)和 [inverse 兼容索引](inverse/README.md)进入；它们不在侧栏单独占位，也不改变本页白名单。
