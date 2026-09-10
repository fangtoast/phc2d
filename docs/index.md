---
title: phc2d 文档
summary: 用五分钟建立项目全景、当前状态、主要结果与证据边界。
page_type: landing
status: draft
audience: 研究使用者、研究生、开发者与评审者
owner: 研究生维护，导师审阅研究主张
last_reviewed: 2026-08-21
reading_time: 5 minutes
primary_layer: L4
---

<!--
  Five-minute reader entrypoint for phc2d.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# phc2d 文档

`phc2d` 是二维光子晶体的统一研究工具链：从周期结构与 MPB 数据生成，经过正向代理、查询与逆向候选生成，最终回到真实求解器复核。这里先回答“项目是什么、做到哪里、结论是什么、证据在哪里”，再进入方法或执行细节。

> **证据边界：** 计划、代码实现、smoke、surrogate 指标、solver replay 与物理验证是不同层级。文档存在不等于实验完成，当前任务优先级以[项目地图](https://github.com/fangtoast/phc2d/blob/master/PROJECT_MAP.md)为准。

## 五分钟阅读路径

| 顺序 | 入口 | 读完后应知道什么 |
|---|---|---|
| 1. Overview | 本页 | 项目对象、研究闭环与证据层级 |
| 2. Current State | [项目地图](https://github.com/fangtoast/phc2d/blob/master/PROJECT_MAP.md) | 当前主线、阻塞项和下一道研究 gate |
| 3. Results | [当前结果](结果/README.md) | 主要实验、当前结论、状态与 Experiment ID |
| 4. Evidence | [实验与证据库](results/README.md) | 数据身份、指标、run、solver 与原始制品入口 |

## 当前研究主线

| 主线 | 当前认知 | 边界 |
|---|---|---|
| 全胞数据与物理标签 | B/G 双路线数据链已建立，MPB 分辨率收敛支持正式数据使用 `resolution=64` | 物理验收不能由 surrogate 指标替代 |
| 全胞正向代理 | 50k 模型升级、K=20 单 seed 与跨数据域验证已有结果 | 本域最优不代表跨域通用；跨域验证仍是 surrogate 层 |
| 逆向设计 | V2 数据已有 seed-0 real-MPB replay 证据 | full-cell inverse 的正式训练与 replay 尚未完成 |
| Field / adapted PINO | 历史 admission gate 未通过 | 不进入当前正式结果主线 |

## 去哪里继续阅读

| 需要 | 入口 | 唯一职责 |
|---|---|---|
| 第一次运行 | [开始](开始/README.md) | 最短可执行路径与环境边界 |
| 理解研究概念 | [概念](概念/README.md) | 物理对象、数据语义与研究闭环 |
| 理解方法 | [方法](方法/README.md) | 输入输出、公式、shape、模型与局限 |
| 查看实验协议 | [实验协议与决策记录](planning/README.md) | 长期计划、正式协议和历史决策 |
| 执行具体任务 | [指南](指南/README.md) | 可重复操作步骤与停止条件 |
| 查询精确定义 | [参考](参考/README.md) | CLI、schema、模型身份、公式与运行合同 |
| 追溯旧材料 | [历史档案](历史档案.md) | 被取代计划、旧报告、preflight 与 smoke |

AI 新建或实质改写的页面默认是 Draft。只有人类负责人明确审阅后，才可标为 Reviewed 或 Canonical。
