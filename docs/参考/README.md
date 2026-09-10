---
title: 参考手册
summary: 汇总系统边界、命令入口、数据模式、运行产物、模型身份与公式指标。
page_type: landing
status: draft
audience: 复现执行者、模型开发者与证据审阅者
owner: 研究生维护
last_reviewed: 2026-08-13
reading_time: 4 minutes
primary_layer: L4
---

<!--
  Reference landing page for the phc2d documentation site.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 参考手册

本区给出可查询的稳定契约。它回答“名称、位置、输入、输出和判据是什么”，不承担实验结论叙事；日期化结论先进入[当前结果](../结果/README.md)，再回链到[实验与证据库](../results/README.md)。

| 需要确认的问题 | 入口 |
| --- | --- |
| 系统由哪些模块组成，依赖如何隔离 | [系统架构](系统架构.md) |
| 从哪个命令或配置开始 | [CLI 与配置](CLI与配置.md) |
| 数据、shape 和运行目录遵循什么约定 | [数据模式与运行产物](数据模式与运行产物.md) |
| 模型名称代表实现、注册还是证据 | [模型目录](模型目录.md) |
| 方法章节对应哪些源码入口 | [核心代码导航](核心代码导航.md) |
| 公式、符号、单位与指标如何定义 | [公式与指标](公式与指标/README.md) |
| 正式结果与计划需要哪些元数据 | [文档元数据](文档元数据.md) |

## 使用原则

- 先从 metadata、schema 或 config 读取 shape，不把 `16x16`、`31`、`10` 当作通用常量。
- 把实现存在、registry 可见、checkpoint 可用和正式验证分别记录。
- 把 surrogate 指标、物理回放指标与运行成本分栏报告。
- 以命令帮助和当前代码为接口真相，以日期报告为证据快照。
