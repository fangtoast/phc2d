---
title: 文档总览
summary: phc2d 文档站的章节地图、读者路线和权威边界。
page_type: landing
status: draft
audience: 研究使用者、导师、开发者与 AI 协作者
owner: 研究生维护，导师审阅研究叙事
last_reviewed: 2026-08-14
reading_time: 6 minutes
primary_layer: L4
---

<!--
  Documentation map and reader routes for phc2d.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="docs-book-home"></a>

# 文档总览

本页是从 GitHub 浏览 `docs/` 时的目录，也是文档站各章节的职责说明。第一次运行请从[开始使用](开始/README.md)进入；核查实验结论请直接进入[正式研究结果](results/README.md)。

## 按任务选择入口

| 你现在要做什么 | 入口 | 读完后应知道什么 |
|---|---|---|
| 安装并检查仓库 | [开始使用](开始/README.md) | 环境层级、第一次正向预测与 smoke 边界 |
| 理解整个项目 | [项目与物理基础](概念/README.md) | Maxwell、MPB、数据、代理、候选与 replay 的关系 |
| 理解公式、shape 和实现 | [数据与方法](方法/README.md) | 方法输入输出、核心代码、配置和局限 |
| 执行具体任务 | [操作指南](指南/README.md) | 前置条件、步骤、产物和停止条件 |
| 查询精确定义 | [参考手册](参考/README.md) | CLI、schema、模型与公式合同 |
| 判断准入数据上做到了哪里 | [正式研究结果](results/README.md) | V2 10k 与新全胞数据的结果、失败和证据边界 |
| 了解计划与规则 | [计划索引](planning/README.md) · [时间线](planning/时间线.md) | 专题、状态、取代关系和未完成 gate |
| 追溯旧结果与工程记录 | [历史档案](历史档案.md) | 哪些材料只保留历史价值、不能覆盖当前证据 |

## 信息架构

```mermaid
flowchart TD
    H["首页"] --> S["开始使用"]
    H --> C["项目与物理基础"]
    C --> M["数据与方法"]
    S --> G["操作指南"]
    M --> R["参考手册"]
    G --> E["正式研究结果"]
    R --> E
    E --> V["计划与治理"]
    V --> A["历史档案"]
```

主导航按读者任务组织。正式结果只展示 `v2_10k_wh31x10` 和 `full_cell_bz9_wh81x10` 两个数据范围；其他日期记录继续可检索，但集中在历史档案并明确标记。

## 权威顺序

冲突时按以下顺序判断项目事实：

```text
实际 run / solver / metrics / checkpoint
  > 当前代码、配置、测试和 CLI
  > 经筛选的结果报告
  > TODO 与研究计划
  > 综合报告和聊天摘要
```

计划不能证明实验完成，smoke 不能证明模型有效，surrogate 不能替代 replay，较低 residual 不能单独证明 inverse 收益。

## 维护入口

- 页面职责与文风：[工程约定](conventions.md)
- 证据、人机协作与人工接受：[协作治理](collaboration.md)
- 正式结果与计划字段：[文档元数据](参考/文档元数据.md)
- 构建与审计：[CLI 与配置](参考/CLI与配置.md#报告维护命令)
- 当前研究状态：仓库根目录的 `PROJECT_MAP.md`
