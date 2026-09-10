---
title: "旁支研究"
summary: "研究报告索引；连接稳定主题与完整报告。"
page_type: landing
status: draft
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 1 minutes
primary_layer: L1
---

<!--
  Side-project research index for the synchronized phc2d workspace.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 旁支研究

本目录集中保存依托 phc2d 数据、求解器或研究基础设施提出的旁支研究计划。每个 side project 使用一个独立 Markdown 文件，便于分别审查、扩展和决定是否进入主线。

这些文档中的理论判断、研究假设和实验设计不属于当前主线项目的完成声明。若某个方向正式进入主线，应在研究计划和 TODO 中另行建立任务、验证门槛与证据边界。

## 项目索引

| Side project | 状态 | 内容 |
|---|---|---|
| [光子能带对介电函数傅里叶分量的敏感度](fourier_sensitivity_of_photonic_bands.md) | 调研与计划 | 研究 $\omega_n(\vec k)$ 对 $\varepsilon_{\vec G}$ 的渐近灵敏度、定量截止、现有数据可行性和蒙特卡洛统计关系。 |

## 文件约定

- 每个研究方向对应一个语义明确的 Markdown 文件，不把多个方向混写在同一文档中。
- 文档至少说明研究问题、已有理论、项目数据能否支持、验证方案、风险边界和参考文献。
- 尚未由 real MPB/Meep 或其他正式实验验证的内容，应明确标记为理论推导、研究假设或待验证计划。
- 公式使用 $...$ 或 $$...$$，矢量使用 \vec 记号。
- 新增、重命名或归档 side project 时，同步维护本索引。
