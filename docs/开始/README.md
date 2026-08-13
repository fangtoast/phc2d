---
title: 开始使用
summary: 为第一次接触 phc2d 的研究使用者提供最短阅读路径。
page_type: landing
status: draft
audience: 第一次使用仓库的研究者、学生和开发者
owner: 研究生维护
last_reviewed: 2026-08-13
reading_time: 3 minutes
primary_layer: L4
---

<!--
  First-use route for the phc2d documentation site.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 开始使用

这一节帮助新用户从干净克隆走到第一个可解释的成功状态。它不会下载正式数据、运行 MPB，也不会重现论文指标。

## 最短路线

1. 按[快速上手](快速上手.md)安装基础包并检查 CLI。
2. 需要模型训练时安装 `train` 可选依赖，并运行非物理 R1 smoke。
3. 再阅读[研究主线与证据等级](../概念/研究主线与证据等级.md)，避免把接口成功误认为实验成功。
4. 需要正式数据、checkpoint 或真实求解器时，转到[分级复现](../指南/分级复现.md)。

## 成功标准

| 阶段 | 成功意味着 | 不意味着 |
|---|---|---|
| R0 | 包、CLI 与轻量 registry 可加载 | 训练、论文复现或物理验证 |
| R1 | 合成数据到训练产物的链路可执行 | 模型对真实物理任务有效 |
| R2 | 指定 surrogate 协议可复跑 | real-MPB 已通过 |
| R3 | 对应求解器 gate 通过 | 结论可无限推广到其他任务 |

不确定自己应运行哪一级时，先停在 R0，并核对所需数据、环境与授权。
