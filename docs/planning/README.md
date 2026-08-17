---
title: "计划与治理"
summary: "按专题、状态与时间路由研究规划文档，区分现行计划、已完成协议、被取代草案与参考材料。"
page_type: landing
status: draft
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-17
reading_time: 5 minutes
primary_layer: L4
---

<!--
  Authority-aware planning index for the synchronized phc2d workspace.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 计划与治理

本目录回答“曾经如何计划、当前由哪份协议约束、后来产生了什么结果”。规划记录不证明实验已经完成；完成状态必须回到[正式研究结果](../results/README.md)、运行制品和项目地图核对。

!!! note "规划状态合同"
    `active` 表示仍约束后续工作，`completed` 表示计划规定的工作已有结果入口，`superseded` 表示已被更具体的计划取代，`reference` 只提供背景。页面仍统一保持 `status: draft`，等待人工审阅。

## 从哪里开始

| 需求 | 入口 | 你会看到什么 |
|---|---|---|
| 按日期查演进 | [专题索引与时间线](时间线.md) | 日期、任务、状态、取代关系、对应结果 |
| 查早期 V2 / Field / PINO / A2 | [早期研究索引](早期研究索引.md) | 早期总计划、TODO、物理模型协议与 A2 协议 |
| 查全胞数据与正向训练 | [全胞数据与正向计划](full_cell_forward_inverse_training/README.md) | 10k、50k、模型升级实施与验证 |
| 查新数据生成与数值验证 | [数据验证计划](数据验证/README.md) | MPB 分辨率收敛、统一 Fourier 数据实现与 64+64 试点 |
| 生成 B/G 各 1 万条 K=20 新数据 | [B 与 G 各 1 万条 K=20 全胞 Fourier 数据生成计划](数据生成/20260817%201w条数据高频截断k=20的情况下生成/B与G各1万条K20全胞Fourier数据生成计划.md) | Xingyi 32×64 CPU 统计、B/G 各 8 节点并行、30 分钟只读监控和 HDF5 下载 |
| 查代码复用与模块治理 | [项目代码结构重构](项目代码结构重构/README.md) | 公共接口、依赖方向、薄兼容层和分阶段迁移 |
| 查全胞逆向 | [全胞逆向索引](全胞逆向索引.md) | codec、Stage-A、筛选和 replay 的现行计划 |
| 查审计、调研和验证构想 | [规划参考](reference/README.md) | 数据审计、文献调研与实验验证矩阵 |
| 查已被替代的草案 | [历史计划索引](历史计划索引.md) | 非 p4m 旧草案与单 seed 全胞草案 |

## 专题状态矩阵

| 专题 | 主要计划状态 | 结果入口 | 当前解释 |
|---|---|---|---|
| 早期 FNO / 生成式网络 | active / reference | [V2 10k](../results/V2-10k正式结果.md) | 总体研究方向仍可参考，具体完成情况以结果制品为准 |
| Field / adapted PINO | completed | [历史 Field/PINO 索引](../results/field-physics/README.md) | 现有记录属于历史试验，不进入正式结果白名单 |
| V2 inverse A2 | completed | [A2 单 seed](../results/inverse/a2_inverse_single_seed_formal_20260727.md) | 只支持既定单 seed 与 replay 边界 |
| 全胞数据与正向 | completed | [新全胞数据正式结果](../results/全胞数据正式结果.md) | 10k、50k、升级模型与 Meep 结果分页面核对 |
| 全胞 inverse | active | [全胞逆向索引](全胞逆向索引.md) | 当前计划有效；预检和 dry-run 不等于正式训练结果 |

## 收录完整性

本站保留并索引改造前已有的 **19 份 planning 文档**：17 份位于“计划与治理”，2 份被取代草案位于“历史档案”。新增的时间线与专题索引只负责组织，不替换原始记录。

每个 `page_type: plan` 页面都声明 `plan_state`、`effective_date`、`superseded_by` 和 `related_results`。字段含义见[文档元数据](../参考/文档元数据.md)，协作边界见[文档与协作治理](../collaboration.md)。
