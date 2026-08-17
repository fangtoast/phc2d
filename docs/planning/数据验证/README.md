---
title: "数据验证计划"
summary: "路由全胞数据生成、数值求解收敛与小样本质量验证计划。"
page_type: landing
status: draft
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护，导师确认研究合同"
last_reviewed: 2026-08-17
reading_time: 3 minutes
primary_layer: L3
---

<!--
  Planning index for full-cell data validation.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 数据验证计划

本目录负责回答“新数据合同怎样经过小样本、数值收敛、超算运行和图件审查后才能进入正式生成”。所有页面都是 Draft 计划，不表示代码已经实现、星仪作业已经提交或 real-MPB 验证已经通过。

## 当前计划

| 顺序 | 计划 | 主要问题 | 下一人工门 |
|---:|---|---|---|
| 1 | [MPB 分辨率收敛与耗时验证](MPB分辨率收敛与耗时验证计划.md) | 哪个 MPB resolution 在精度与耗时之间可接受 | 批准星仪计费 pilot 与正式 100 样本作业 |
| 2 | [全胞新数据生成实现与 64+64 试点](全胞新数据生成实现与64加64试点计划.md) | 如何实现老师提出的统一 Fourier 数据合同并验证样本形态 | 接受实现合同，并在计划 1 冻结求解分辨率后批准星仪试点 |

## 共同边界

- 主层级均为 **L3 实验与证据**；代码改造只服务于已冻结的实验合同。
- 星仪上的提交、恢复和计费运行必须取得单独授权；本目录不构成运行授权。
- `32/64/128/256` 是同一解析结构的不同模型输入表示；MPB resolution 是独立的求解器离散精度。
- 集群结果在本机 WSL `Ubuntu` + conda `meep_env` 交叉验收前，最高只能标记为 `cluster_generated_pending_wsl_acceptance`。
- `runs/`、HDF5、worker shard、图件全集和 checkpoint 不进入 Git；只把经筛选且带来源哈希的证据写入 `docs/results/`。
