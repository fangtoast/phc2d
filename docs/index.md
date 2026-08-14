---
title: phc2d 文档
summary: 面向研究使用者的二维光子晶体数据、代理建模、逆向设计与物理复核手册。
page_type: landing
status: draft
audience: 研究使用者、研究生、开发者与评审者
owner: 研究生维护，导师审阅研究主张
last_reviewed: 2026-08-14
reading_time: 3 minutes
primary_layer: L4
---

<!--
  Reader-oriented homepage for the phc2d documentation site.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# phc2d 文档

`phc2d` 将二维光子晶体的结构表示、MPB 数据生成、神经代理、候选式逆向设计和真实求解器复核组织在同一研究工具链中。本手册按“先运行、再理解、后追溯证据”的顺序组织，不要求读者先熟悉仓库内部目录。

> **证据边界：** 文档站整理不改变实验状态。计划、实现、smoke、surrogate、proxy 与 real-MPB replay 仍是不同证据层级。

<div class="doc-grid">
  <div class="doc-card">
    <h3>第一次使用</h3>
    <p>从安装、CLI 检查和非物理最小训练开始。</p>
    <p><a href="开始/快速上手.html">打开快速上手</a></p>
  </div>
  <div class="doc-card">
    <h3>理解研究方法</h3>
    <p>理解数据、正向代理、本征场、逆向候选与 replay 的关系。</p>
    <p><a href="概念/研究主线与证据等级.html">阅读研究主线</a></p>
  </div>
  <div class="doc-card">
    <h3>复现实验</h3>
    <p>按 R0–R3 选择输入、环境、命令和允许表述。</p>
    <p><a href="指南/分级复现.html">打开分级复现</a></p>
  </div>
  <div class="doc-card">
    <h3>核查结果</h3>
    <p>从日期报告追溯配置、运行制品、失败案例与结论边界。</p>
    <p><a href="results/index.html">查看结果与报告</a></p>
  </div>
</div>

## 推荐阅读顺序

1. [快速上手](开始/快速上手.md)：完成 R0 接口检查；有训练依赖时完成 R1 smoke。
2. [研究主线与证据等级](概念/研究主线与证据等级.md)：理解项目为什么把 surrogate 与 real-MPB replay 分开。
3. [方法总览](方法/README.md)：按 Full-Wh、Query、Field/PINO、Inverse/Replay 选择主题。
4. [分级复现](指南/分级复现.md)：选择 R0、R1、R2 或 R3。
5. [研究结果与报告](results/README.md)：按大类展开并核对具体日期报告；当前项目状态仍以仓库中的 `PROJECT_MAP.md` 为准。

## 文档怎么分工

| 内容 | 回答的问题 | 更新节奏 |
|---|---|---|
| 概念与方法 | 术语、关系和当前稳定合同是什么？ | 接口或方法发生实质变化时 |
| 操作指南与参考 | 怎样执行，输入输出是什么？ | CLI、schema 或工作流变化时 |
| 研究结果与报告 | 某次实际运行得到什么，综合叙述截至哪一天？ | 新 run、复核或综合快照完成时 |
| 计划与治理 | 下一步准备做什么，如何判定？ | 研究 gate 或协作合同变化时 |

AI 新建或实质改写的页面默认是 Draft。只有人类负责人明确审阅和接管后，才可标为 Reviewed 或 Canonical。
