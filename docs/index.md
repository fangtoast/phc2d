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

`phc2d` 是二维光子晶体数据、神经代理、逆向设计和真实求解器复核的统一研究工具链。文档按照“先获得最小成功，再理解物理与方法，最后追溯结果和计划”的顺序组织；读者不需要先浏览仓库文件夹。

> **证据边界：** 文档重排不改变实验状态。计划、实现、smoke、surrogate、proxy、MPB replay 和 Meep physical validation 始终分开记录。

<div class="doc-grid">
  <div class="doc-card">
    <h3>第一次使用</h3>
    <p>安装基础环境，检查 CLI，并完成一次非物理最小训练。</p>
    <p><a href="开始/index.html">打开开始使用</a></p>
  </div>
  <div class="doc-card">
    <h3>理解整个项目</h3>
    <p>从 Maxwell、MPB 和两代数据进入完整研究闭环。</p>
    <p><a href="概念/项目全景.html">阅读项目全景</a></p>
  </div>
  <div class="doc-card">
    <h3>学习公式与代码</h3>
    <p>在同一条阅读路径中查看公式、shape、关键代码和配置。</p>
    <p><a href="方法/index.html">打开数据与方法</a></p>
  </div>
  <div class="doc-card">
    <h3>核查正式结果</h3>
    <p>只查看 V2 10k 与新全胞数据的正式结果集合。</p>
    <p><a href="results/index.html">查看正式结果</a></p>
  </div>
</div>

## 推荐阅读顺序

1. [安装与环境](开始/安装与环境.md)：明确基础 Python、训练依赖和 WSL 求解器环境的边界。
2. [第一次正向预测](开始/第一次正向预测.md)：完成 R1 合成 smoke，认识配置和运行产物。
3. [项目全景与研究闭环](概念/项目全景.md)：理解结构、物理数据、代理、候选和 replay。
4. [数据与方法](方法/README.md)：按公式、shape、代码和配置学习核心模块。
5. [分级复现](指南/分级复现.md)：选择 R0、R1、R2 或 R3。
6. [正式研究结果](results/README.md)：只核查 V2 10k 与新全胞数据；当前状态仍以仓库根目录的 `PROJECT_MAP.md` 为准。

## 文档怎么分工

| 内容 | 回答的问题 | 更新节奏 |
|---|---|---|
| 项目与物理基础 | 研究对象、公式和研究闭环是什么？ | 物理或研究合同变化时 |
| 数据与方法 | 输入输出、shape、模型和核心代码如何连接？ | 接口或方法发生实质变化时 |
| 操作指南与参考 | 怎样执行，输入输出是什么？ | CLI、schema 或工作流变化时 |
| 正式研究结果 | 两个准入数据家族实际得到什么？ | 新 run、复核或白名单变化时 |
| 计划与治理 | 下一步准备做什么，如何判定？ | 研究 gate 或协作合同变化时 |
| 历史档案 | 哪些旧结果、工程预检和计划只供追溯？ | 记录被替代或移出正式结果时 |

AI 新建或实质改写的页面默认是 Draft。只有人类负责人明确审阅和接管后，才可标为 Reviewed 或 Canonical。
