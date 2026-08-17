---
title: "项目代码结构重构"
summary: "路由面向可复用接口、稳定兼容层和模块边界的 L4 重构计划。"
page_type: landing
status: draft
audience: "项目维护者、实现者与代码审查者"
owner: "研究生维护，人工确认公共接口与迁移边界"
last_reviewed: 2026-08-17
reading_time: 3 minutes
primary_layer: L4
---

<!--
  Planning index for reusable code interfaces and structural refactoring.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 项目代码结构重构

本目录规划如何把重复实现收敛为可复用公共接口，并在不改变科学合同、数值结果和历史导入路径的前提下拆分职责过载模块。这里记录的是 Draft 实施计划，不表示重构已经开始或完成。

## 当前计划

| 计划 | 主要问题 | 完成后的下一步 |
|---|---|---|
| [全胞代码可复用接口与模块重构](全胞代码可复用接口与模块重构计划.md) | Full-cell 同名文件、重复函数、私有接口跨模块依赖和职责过载 | 在稳定新接口上实现统一 Fourier、256 母版、哈希和数据试点 |

## 共同原则

- 一个领域概念只有一个权威实现；生成、求解、训练和绘图复用同一接口。
- 旧公共路径先保留为薄兼容层，迁移通过后再讨论废弃，不进行批量破坏性改名。
- 重构提交只改变组织，不同时改变算法、随机分布、schema、数据内容或科学结论。
- `torch`、`h5py`、SciPy、MPB 和 Meep 继续按职责延迟导入，基础 CLI 不依赖重环境。
- 每一阶段都必须在 WSL `Ubuntu` + conda `meep_env` 通过定向测试和兼容性验证。
