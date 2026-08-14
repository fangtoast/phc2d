---
title: 研究结果与报告
summary: 按研究任务进入单项结果、证据记录和综合研究报告。
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

# 研究结果与报告

这里是所有已编入文档站的研究结果入口。侧栏中的“研究结果与报告”可直接打开本页；点击右侧箭头可以继续展开到研究大类和具体结果。

> **证据边界：** 页面整理不改变实验状态。完整 `run.log`、`metrics.json`、checkpoint 和原始图件仍位于 Git 之外的 `runs/`；报告存在也不表示其结论已通过更高层级的 gate。

## 按研究任务进入

| 研究大类 | 内容 | 入口 |
|---|---|---|
| 数据与环境 | 求解器、k 路径、算力环境、数据导入和全胞 pilot | [打开分类总览](data-provenance/README.md) |
| 正向代理 | Forward/Query 公平对照、10k/50k 双路线训练和升级模型评估 | [打开分类总览](forward/README.md) |
| Query | Query 表示的数据规模实验与历史证据 | [打开分类总览](query/README.md) |
| 本征场与物理 | Field/PINO、dense-k、residual、电场度规和 Meep 近远场 | [打开分类总览](field-physics/README.md) |
| 逆向设计 | surrogate rerank、Objective-Bridge、real-MPB replay 和 Stage-A 预检 | [打开分类总览](inverse/README.md) |
| 表征诊断 | 小样本表示消融；不作为正式模型性能主结论 | [打开诊断结果](representation_benchmark_20260803.md) |
| 综合研究报告 | 项目全景、物理公式、科学动机、全胞双路线和旁支研究 | [打开报告总览](../reports/README.md) |

<a id="data-provenance-results"></a>

## 数据与环境结果

[数据与环境分类总览](data-provenance/README.md)负责区分环境预检、数据 provenance 与真实物理验证，避免把“能够运行”误写成“已经得到模型结果”。

## 正向与 Query 结果

- [正向代理分类总览](forward/README.md)：从公平协议进入全胞双路线 10k、50k 和升级模型评估。
- [Query 分类总览](query/README.md)：保留 Query 数据规模实验，同时把当前比较入口指向公平对照报告。

<a id="query-results"></a>

## 本征场与物理结果

<a id="field-physics-results"></a>

[本征场与物理分类总览](field-physics/README.md)按 B1/B2/B3/B4、P6、度规勘误与 Meep 复核组织结果。较低 Maxwell residual 不能单独证明 frequency、inverse reranking 或 field inverse 收益。

<a id="inverse-results"></a>

## 逆向设计结果

[逆向设计分类总览](inverse/README.md)明确区分 surrogate、候选筛选、smoke/preflight 与 real-MPB replay。阅读时必须保留 seed、目标数、候选池和 Top-K 边界。

## 使用规则

1. 先在分类总览确认页面角色，再进入具体结果。
2. 先读结果页顶部的数据、环境、状态和证据边界，再核对关联 run、配置、hash、seed 和失败案例。
3. 历史报告用于追溯，不覆盖更新后的 metric、data contract 或 solver 结果。
4. 新结果应同时更新本索引和仓库根目录的 `PROJECT_MAP.md`，不能只留在聊天或本地草稿中。
