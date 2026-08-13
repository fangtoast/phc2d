---
title: "文档与人机协作治理"
summary: "规定知识层级、页面职责、证据升级和人机协作的接受边界。"
page_type: reference
status: draft
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 7 minutes
primary_layer: L4
---

<!--
  Shared documentation and human-AI collaboration governance for phc2d.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="collaboration-home"></a>

# 文档与人机协作治理

| Metadata | Value |
|---|---|
| Status | Draft collaboration contract pending human review |
| Audience | 导师、研究生、项目协作者与 AI 代理 |
| Owner | 研究生维护，导师确认研究目标与主张边界 |
| Last reviewed | 2026-08-13 |
| Reading time | 6 minutes |

本文规定项目知识如何分层、证据如何定级，以及 AI 产出怎样经过人工接受。代码、运行、测试和 Git 的固定约束见 [`AGENTS.md`](https://github.com/fangtoast/phc2d/blob/master/AGENTS.md)；目录与文件约定见 [`conventions.md`](conventions.md)。

## 责任与完成定义

| 角色 | 主要责任 | 不应外包的判断 |
|---|---|---|
| 导师 | 科学问题、研究边界、创新性与论证标准 | 证据是否足以支撑论文主张 |
| 研究生 | 系统集成、实验链路、证据关系和当前状态 | 项目实际做到哪里、下一 gate 是什么 |
| AI | 受约束的生成、整理、比较、检查和局部实现 | 发明研究动机、升级证据或自行宣布 Canonical |

“完成”要求产物已生成、已验证、已接入且可解释。至少一名人类负责人应能说明目的、方法、证据和局限。

## 五个知识层级

| 层级 | 回答的问题 | 典型内容 |
|---|---|---|
| L0 研究目标 | 要解决什么问题？ | 科学问题、范围、最终贡献 |
| L1 科学论证 | 为什么选择这条路线？ | 假设、方法依据、论证结构 |
| L2 方法架构 | 系统由哪些模块组成？ | 数据、代理、生成、筛选、replay |
| L3 实验与证据 | 如何证明或否定？ | 协议、基线、指标、结果、失败 |
| L4 实现 | 具体怎样运行？ | 代码、配置、schema、命令、产物 |

任务可以跨层读取，但每次只指定一个主要修改层。不得根据 L4 临时实现反向编造 L0/L1，也不得把局部 L3 结果提升为项目级结论。

## 文档职责

| 文档 | 主要职责 |
|---|---|
| [`README.md`](https://github.com/fangtoast/phc2d/blob/master/README.md) | 稳定项目入口 |
| [`PROJECT_MAP.md`](https://github.com/fangtoast/phc2d/blob/master/PROJECT_MAP.md) | 当前状态、阻塞和最近 gate |
| [`docs/index.md`](index.md) | 发布站首页与受众入口 |
| [`docs/README.md`](index.md) | 完整站点地图与维护入口 |
| [`docs/概念/`](概念/README.md) | 跨实验稳定的解释与证据等级 |
| [`docs/方法/`](方法/README.md) | 方法架构及各模块关系 |
| [`docs/指南/`](指南/README.md) | 面向任务的可执行步骤 |
| [`docs/参考/`](参考/README.md) | CLI、schema、模型和指标契约 |
| [`planning/README.md`](planning/README.md) | 规划与协议路由 |
| [`experiments/`](https://github.com/fangtoast/phc2d/blob/master/experiments/README.md) | 可执行实验合同 |
| [`results/README.md`](results/README.md) | 实际结果和证据边界 |
| [`reports/README.md`](reports/README.md) | 带日期的综合叙述 |

一个文档只承担一种主要职责，细节通过链接下沉，不在多个入口复制。

## 页面契约

`docs/` 中每个页面使用 YAML front matter 声明 `title`、`summary`、`page_type`、`status`、`audience`、`owner`、`last_reviewed`、`reading_time` 和 `primary_layer`。证据与归档页还要声明 `evidence_date` 和 `evidence_scope`。

| `page_type` | 回答的问题 | 典型位置 |
|---|---|---|
| `tutorial` | 第一次怎样走通 | `开始/` |
| `how-to` | 怎样完成一个明确任务 | `指南/` |
| `explanation` | 为什么采用这种设计 | `概念/`、`方法/` |
| `reference` | 精确定义和契约是什么 | `参考/` |
| `evidence` | 某日、某协议实际得到什么 | `results/`、日期报告 |
| `plan` | 准备做什么、门槛是什么 | `planning/` |
| `archive` | 哪段历史仅供追溯 | `archive/` 或兼容入口 |

稳定主题页可以从多个证据记录归纳规则，但必须链接原始证据并保留范围；日期化证据和历史计划的正文不因站点编排而改写。旧路径若仍被仓库或外部书签引用，应保留短兼容页。

## 证据与状态

冲突时按以下顺序判断事实：

```text
实际 run / solver / metrics / checkpoint
  > 当前代码、配置、测试和 CLI 行为
  > 经筛选的结果报告
  > TODO
  > 研究计划
  > 综合报告和聊天摘要
```

计划、实现、配置就绪、smoke、pilot、正式 surrogate 结果、proxy 和 real-MPB 物理验证必须分开记录。文档整理不能提升实验状态。

AI 新建或实质重写的正式文档默认为 Draft。只有人类负责人明确审阅并接管后，才能标记为 Reviewed 或 Canonical。

## 任务与收尾契约

修改项目的任务应明确：主层级、单一目标、受众、证据输入、允许修改、非目标、禁止升级的状态、自动验证和人工接受项。新增或重命名正式文档后，必须从 [`docs/README.md`](index.md) 或相邻索引建立可达链接。

实质任务结束时记录：

```text
解决了什么问题：
为什么采用这个方案：
改变了项目哪一层/哪一模块：
使用什么证据验证：
最大风险、未知或下一 gate：
```

研究 gate 变化时更新结果报告、结果索引和 `PROJECT_MAP.md`；接口或阅读路线变化时更新对应入口。聊天内容不单独充当项目证据。

## 自动检查

```bash
python -m phc2d.cli report docs-audit --root README.md --strict
```

该审计检查已跟踪 Markdown 的链接、可达性、front matter、页面类型和 MkDocs 导航覆盖。站点还应通过 `mkdocs build --clean --strict`；两者都不替代科学语义、数值来源和证据强度的人工审阅。
