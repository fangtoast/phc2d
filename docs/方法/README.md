---
title: 方法总览
summary: 按数据、代理、本征场和逆向闭环组织 phc2d 的稳定方法说明。
page_type: landing
status: draft
audience: 研究使用者、模型开发者与评审者
owner: 研究生维护，导师审阅研究主张
last_reviewed: 2026-08-13
reading_time: 4 minutes
primary_layer: L2
---

<!--
  Method entrypoint for phc2d.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 方法总览

方法页沿“问题与输入输出 → 公式 → 实际 shape → 关键源码 → 配置和命令 → 产物 → 证据边界”解释稳定实现，不维护易过时的排行榜数字。

| 主题 | 读完后应能回答 |
|---|---|
| [数据与表征](../概念/数据与表征.md) | V2 10k 和 full-cell 数据的结构、target 与 shape 有何不同 |
| [全胞双路线](全胞双路线.md) | 解析结构怎样进入 MPB、HDF5 和多分辨率栅格 |
| [正向与查询代理](正向与查询代理.md) | Full-Wh、Query、DeepONet、FNO、ResMLP/CNN/ConvNeXt/TFNO 如何分工 |
| [本征场与 PINO](本征场与PINO.md) | 相位不变场损失、TE residual 和 Rayleigh 项如何计算 |
| [逆向设计与物理回放](逆向设计与物理回放.md) | 候选怎样生成、筛选、排序并接受 real-MPB 裁决 |

源码入口统一列在[核心代码导航](../参考/核心代码导航.md)。数值结果只进入[正式研究结果](../results/README.md)，旧 pilot、smoke 和工程预检进入[历史档案](../历史档案.md)。
