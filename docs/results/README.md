---
title: 实验与证据库
summary: 保存可追溯的结果证据、数据身份集合与历史兼容入口。
page_type: landing
status: draft
audience: 复现执行者、结果审阅者与项目维护者
owner: 研究生维护
last_reviewed: 2026-08-21
reading_time: 4 minutes
primary_layer: L3
---

<!--
  Evidence-store navigation for the synchronized phc2d workspace.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="results-home"></a>

# 实验与证据库

这里回答“结论由哪些运行、指标和求解器制品支持”。如果只想知道当前结论，请先读[当前结果](../结果/README.md)。本目录保留旧路径，不再承担五分钟项目总览，也不把 preflight、smoke 或 dry-run 包装成科研结果。

> **证据边界：** 完整 `run.log`、`metrics.json`、checkpoint 和原始图件位于 Git 之外的 `runs/`。报告存在不等于实验完成；surrogate、solver replay 与 physical validation 不可互相替代。

## 当前证据集合

<a id="field-physics-results"></a>
<a id="inverse-results"></a>

| 数据范围 | shape 与任务 | 入口 |
|---|---|---|
| V2 10k | `Wh[10000,31,10]`；A1 Forward/Query、inverse 与 real-MPB replay | [V2 10k 证据集合](V2-10k正式结果.md) |
| 新全胞数据 | `Wh[N,81,10]`；数据验收、10k/50k/K20 正向、跨域验证与 Meep | [全胞数据证据集合](全胞数据正式结果.md) |

### 日期化历史快照

下列页面只保留当时的局部运行记录，**不进入正式结果白名单**：

| 证据 | 入口 |
|---|---|
| Representation Benchmark | [2026-08-03](representation_benchmark_20260803.md) · [2026-08-13](representation_benchmark_20260813.md) · [2026-08-17](representation_benchmark_20260817.md) · [2026-08-19](representation_benchmark_20260819.md) · [2026-08-20](representation_benchmark_20260820.md) · [2026-08-21](representation_benchmark_20260821.md) · [2026-08-25](representation_benchmark_20260825.md) |

## 证据使用规则

1. 先确认 `dataset_scope`、target shape、seed 和模型身份，再比较指标。
2. 明确区分 `surrogate_held_out`、`solver_replay` 与 `physical_validation`。
3. 结果页必须能够回链到 run、配置、manifest、metrics 或 solver 产物。
4. 环境检查、导入记录、preflight、smoke、dry-run 和普通训练流水账只作为原始或历史证据。

## 冻结与兼容入口

| 内容 | 入口 | 处理方式 |
|---|---|---|
| Forward 旧分类 | [Forward 索引](forward/README.md) | 保留旧链接；当前结论由上级结果总览路由 |
| Inverse 旧分类 | [Inverse 索引](inverse/README.md) | 保留 replay 与 preflight 的层级差异 |
| Field / PINO | [Field/PINO 索引](field-physics/README.md) | 历史 admission 证据，不进入当前主线 |
| Query | [Query 索引](query/README.md) | 旧数据缩放记录，冻结 |
| 环境、导入与数据生成记录 | [数据来源索引](data-provenance/README.md) | 作为机器运行与来源证据保留 |
| 其他 smoke 与历史材料 | [历史档案](../历史档案.md) | 不再扩张为长篇结果报告 |

新结果应先更新[当前结果](../结果/README.md)中的一句话结论和 Experiment ID，再链接到本目录的证据页；不要复制 raw metadata 或生成第二份综合报告。
