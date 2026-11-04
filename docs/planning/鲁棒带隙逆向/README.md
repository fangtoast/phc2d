---
title: Global–Local 逆向研究入口
summary: 当前纯代理设计路线、实验复核与旧计划封存的唯一目录入口。
page_type: landing
status: draft
audience: 研究负责人、开发者与实验执行者
owner: fangtoast
last_reviewed: 2026-11-04
reading_time: 3 minutes
primary_layer: L1
---

<!--
  Surrogate-only inverse design research protocol.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# Global–Local 逆向研究入口

**当前采用全程 Global–Local 优化与选型，候选锁定后统一 MPB 复核。** 先完成纯代理设计、制造鲁棒与独立验证，再评价在线物理反馈是否值得额外计算。前版“反馈优先”方案已取代，不作为执行依据。方向由作者确认，详细协议为 Draft；尚无本版算法或物理结果。

## 当前规划

| 顺序 | 页面 | 职责 |
|---|---|---|
| 1 | [研究总方案](研究总方案.md) | 科学问题、方法边界与阶段顺序 |
| 2 | [纯代理方法](纯代理方法.md) | Fourier、梯度、硬选型、鲁棒目标和选择锁 |
| 3 | [实验与复核](实验与复核.md) | 对照矩阵、真实谱误差、成功率、成本与数值验收 |
| 4 | [实施与资源](实施与资源.md) | 模块职责、实施顺序、资源和原始产物 |
| 5 | [现有基础与复用](现有基础与复用.md) | 哪些结构、数据、模型和物理能力继续使用，哪些尚需补齐 |

首个开发协议为 [EXP-INV-012 / MVP-INV-001](https://github.com/fangtoast/phc2d/blob/master/experiments/research/EXP-INV-012-梯度链条开发验证.md)。下一 gate 是冻结固定融合 3×3 checkpoint、来源 8、参数化/平滑合同、选择锁与末尾复核预算；不运行在线 MPB 微调。

## 已封存规划

下列文件的完整正文集中归档，原路径已更新至归档目录。**封存的是旧方案的研究次序和执行合同，不是废弃其中的代码、数据或有效方法。** 当前路线需要的几何、正向模型、求解、验证与成本能力已整理到[现有基础与复用](现有基础与复用.md)和相应规划；具体实验仍需核对当前合同。配置、历史 run 和科学失败结论均不改写。

| 版本 | 历史页面 |
|---|---|
| 前版物理反馈优先 | [路线](历史方案/可微物理反馈路线.md) · [方法](历史方案/可微方法协议.md) · [实验](历史方案/实验矩阵与验收.md) · [实施](历史方案/模块复用与实施.md) |
| 更早无梯度优先 | [总计划](历史方案/旧版总计划.md) · [方法](历史方案/研究方法.md) · [实验](历史方案/实验与验证.md) · [架构](历史方案/代码架构.md) · [资源](历史方案/资源与迭代.md) |

旧 EXP-INV-011 保留 DE 名义预检身份；008–010 及[可用性审计](../../results/inverse/鲁棒带隙逆向可用性审计.md)保留各自证据。生成式 Stage-A/Stage-B 的原合同独立保留，不作为当前路线的前置训练。
