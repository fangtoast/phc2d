---
title: 实验协议与决策记录
summary: 路由仍有效的长期计划，并冻结已完成、被取代或失去 planning 属性的历史记录。
page_type: landing
status: draft
audience: 研究用户、复现执行者与项目维护者
owner: 研究生维护
last_reviewed: 2026-08-27
reading_time: 4 minutes
primary_layer: L4
---

<!--
  Transitional planning and protocol index for phc2d.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 实验协议与决策记录

`planning/` 是过渡中的协议与历史决策库，不是当前状态页。计划不能证明实验完成；先看[项目地图](https://github.com/fangtoast/phc2d/blob/master/PROJECT_MAP.md)和[当前结果](../结果/README.md)，再回到这里核查实验原本如何设计。

## 截稿前补充实验

- [三核心模型与分支消融固定预算确认](../../experiments/research/EXP-FWD-024-三核心模型固定预算确认.md)：2026-10-20授权；状态以run为准。
- [单孔形状迁移测试](../../experiments/research/EXP-DAT-007-单孔形状迁移测试.md)：六种代表孔形，先完成MPB数值pilot。

## 仍可能约束后续工作的入口

| 范围 | 入口 | 使用边界 |
|---|---|---|
| 研究阶段主线 | [FNO 与生成式网络研究计划](fno_generative_network_adjustment_plan_20260617.md) | 只约束其覆盖的任务族、gate 与 replay 边界；不自动覆盖后续 full-cell 任务 |
| Full-cell forward loss | [工作频率与带隙损失探索计划](工作频率与带隙损失探索计划.md) | 先通过加密 band-edge 标签门；固定 backbone，只比较一个新 loss 候选 |
| Full-cell forward 组件 | [FNO、TFNO 与 ConvNeXt 组件归因计划](FNO-TFNO-ConvNeXt组件归因计划.md) | 固定原 loss；先做冻结诊断，再做同家族最小消融和跨家族桥接 |
| Full-cell inverse | [全胞逆向索引](全胞逆向索引.md) | codec、Stage-A、候选筛选与 replay 的现行协议入口 |
| 架构治理 | [项目代码结构重构](项目代码结构重构/README.md) | 作为待复核的长期架构记录；不能据此断言重构已完成 |
| 历史演进 | [时间线](时间线.md) | 用于查日期、取代关系和结果入口，不作为执行日志继续扩张 |

## 冻结区

以下内容保留原路径，但不再作为默认执行入口，也不逐篇重写：

| 冻结类别 | 入口 | 原因 |
|---|---|---|
| 已完成的 V2 / Field / A2 协议 | [早期研究索引](早期研究索引.md) | 已产生结果或准入结论，planning 属性已经结束 |
| 已完成的全胞数据与正向计划 | [全胞数据与正向计划](full_cell_forward_inverse_training/README.md) | 10k、50k、升级模型与 K=20 工作应从结果页核查 |
| 已完成的数据生成与数值验证计划 | [数据验证](数据验证/README.md) · [数据生成](数据生成/20260817%201w条数据高频截断k=20的情况下生成/B与G各1万条K20全胞Fourier数据生成计划.md) | 现有计划文字可能落后于运行证据 |
| 审计、调研、TODO 与验证构想 | [规划参考](reference/README.md) | 属于分析、参考或临时工作记忆，不是长期 planning |
| 被取代草案 | [历史计划索引](历史计划索引.md) | 仅用于追溯 |

## 后续写入规则

- 短任务 planning 默认不落盘。
- 只有研究阶段计划、架构调整、实验 campaign 和多任务 roadmap 可以长期保存。
- 已完成计划只补充结果链接和冻结状态，不在原计划中持续追加运行流水账。
- 新研究决策未来进入 `docs/决策/`；本轮不创建空目录，也不搬迁历史文件。
