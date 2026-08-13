<!--
  Shared documentation and human-AI collaboration governance for phc2d.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="collaboration-home"></a>

# 文档与导师—研究生—AI 协作治理

| Metadata | Value |
|---|---|
| Status | Canonical collaboration contract |
| Audience | 导师、研究生、项目协作者与 AI 代理 |
| Owner | 研究生维护，导师确认研究目标与主张边界 |
| Last reviewed | 2026-08-06 |
| Reading time | 12 minutes |

本文回答“怎样让快速产生的代码、实验和文档被人类理解、验证并纳入统一研究结构”。具体代码与证据约束仍以 [`AGENTS.md`](../AGENTS.md) 为准。

## 1. 三方责任

| 角色 | 主要责任 | 不应外包的判断 |
|---|---|---|
| 导师 | 科学问题、研究边界、创新性、论证标准 | 某项证据是否足以支撑论文主张 |
| 研究生 | 系统集成、实验链路、证据关系、文档入口 | 当前项目实际做到哪里、下一 gate 是什么 |
| AI | 受约束的生成、整理、比较、检查与局部实现 | 自行发明研究动机、升级证据或宣布 Canonical |

“完成”同时要求 artifact produced、verified、integrated 和 explainable。至少一名人类负责人应能说明目的、方法、证据和局限。

## 2. 五个知识层级

| 层级 | 回答的问题 | 典型内容 |
|---|---|---|
| L0 研究目标 | 我们到底解决什么问题？ | 科学问题、范围、最终贡献 |
| L1 科学论证 | 为什么选择这条路线？ | 假设、现有不足、方法选择依据 |
| L2 方法架构 | 系统由哪些模块组成？ | 数据、代理、生成、筛选、replay |
| L3 实验与证据 | 如何证明或否定？ | 协议、基线、指标、结果、失败 |
| L4 实现 | 具体怎样运行？ | 代码、配置、schema、命令、产物 |

任务可以跨层读取上下文，但每次只能指定一个主要修改层。AI 不得根据 L4 临时实现反向编造 L0/L1，也不得把 L3 局部结果升级成全项目结论。

## 3. 文档角色

| 角色 | 唯一主要问题 | 当前入口 |
|---|---|---|
| 稳定入口 | 项目是什么、该去哪里？ | [`README.md`](../README.md) |
| 当前状态 | 现在做到哪里、卡在哪里？ | [`PROJECT_MAP.md`](../PROJECT_MAP.md) |
| 书籍目录 | 理论、公式、代码、实验怎样互相引用？ | [`docs/README.md`](README.md) |
| 权威计划 | 计划做什么、gate 是什么？ | [`planning/README.md`](planning/README.md) |
| 实验协议 | 某实验怎样公平执行？ | [`experiments/`](../experiments/README.md) |
| 结果证据 | 实际得到什么、边界是什么？ | [`results/README.md`](results/README.md) |
| 综合报告 | 某一日期的完整项目叙述 | [`reports/README.md`](reports/README.md) |
| 公式参考 | 物理、数据、损失和指标如何定义？ | [公式手册](reports/phc2d_physics_data_formula_guide_20260803.md) |

一个文档只承担一种主要角色。完整性通过索引和链接保证，不通过把所有细节复制进一个文件保证。

## 4. 权威与冲突处理

当状态或结论冲突时，使用以下顺序：

```text
实际 run / solver / metrics / checkpoint
  > 当前代码、配置、测试与 CLI 行为
  > 经筛选的结果报告
  > Research TODO
  > 研究计划
  > 综合报告与聊天摘要
```

综合报告是带日期快照，不自动代表今天的最新状态。`PROJECT_MAP.md` 可以概括当前状态，但其结论仍必须向下回链。

## 5. 文档状态与元数据

除面向外部读者的根 README 外，当前入口文档必须显示：`Status`、`Audience`、`Owner`、`Last reviewed`、`Reading time`。根 README 只保留稳定的项目定位、最小上手路径和文档入口，不展示内部维护元数据。其他文档按职责记录：

- 计划：authority、状态、被什么 supersede；
- 实验 profile：输入、非目标、执行 gate；
- 结果：环境、数据、run、evidence status、能/不能得出的结论；
- 报告：证据更新时间和快照边界。

AI 新生成的正式文档默认是 Draft。只有人类负责人实际阅读并明确接管后，才能标记为 Reviewed 或 Canonical。

## 6. Codex 任务契约

每个修改项目的任务应包含：

```text
Primary layer: L0 | L1 | L2 | L3 | L4
Goal: 本轮只解决的一个问题
Audience: 谁需要读懂或使用
Evidence inputs: 允许依赖的代码、配置、run、报告
Allowed changes: 允许修改的文件或模块
Non-goals: 不修改什么、不新增什么
Forbidden promotions: 哪些状态或主张不得升级
Required validation: 测试、审计和人工检查
Human acceptance: 人类需要确认的内容
```

文档任务还必须说明该文档的角色、预计阅读时间和它不回答什么。新增或重命名正式文档后，必须从 `docs/README.md` 或相邻子索引建立可达链接。

## 7. 人工理解记录

每个实质任务完成后，负责人保留一份短记录：

```text
解决了什么问题：
为什么采用这个方案：
改变了项目哪一层/哪一模块：
使用什么证据验证：
最大风险、未知或下一 gate：
```

记录可以进入对应结果报告、决策说明或任务交接，不另建无索引的聊天转录仓库。

## 8. 更新节奏

- 研究 gate 改变：更新对应结果报告、结果索引和 `PROJECT_MAP.md`。
- 接口或阅读路线改变：更新根 README 和 `docs/README.md`。
- 公式/数据契约改变：更新公式手册并标出旧指标是否需要重算。
- 完成一轮实验：压缩当前状态，移除重复叙述，避免把历史过程继续追加到入口。
- 超过约 15 分钟才能读完且同时回答多个问题：拆成单一职责文档，原文件保留导航或历史快照。

## 9. 自动检查

```bash
python -m phc2d.cli report docs-audit --root README.md --strict
```

审计检查 Git 已跟踪 Markdown 的链接、跨文件稳定锚点、入口可达性、非根入口元数据和本机绝对路径。它不替代对科学语义、数值来源和证据强度的人工审阅。
