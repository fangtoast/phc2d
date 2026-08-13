---
title: 本征场与 adapted PINO
summary: 解释 TE Bloch 本征场表示、监督模型、物理残差和 inverse 准入边界。
page_type: explanation
status: draft
audience: 物理建模研究者、模型开发者与评审者
owner: 研究生维护，导师审阅物理主张
last_reviewed: 2026-08-13
reading_time: 10 minutes
primary_layer: L2
---

<!--
  Stable explanation of eigenfield models and adapted PINO.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 本征场与 adapted PINO

本征场任务从结构、Bloch 波矢和 band 条件预测 TE 本征对。它既需要复数场监督，也需要频率、相位、边界和物理 residual 的一致合同。

## 数据表示

复数场序列化为六个实通道，并保留结构来源、k 点、band、频率和 solver provenance。比较预测场与真实场前，使用一个由 E 场确定的全局复相位，同时作用于 E/H，以保留二者的 Maxwell 关系。

电场比较采用介电加权内积：

$$
\langle E_1,E_2\rangle_\varepsilon
=\int \overline{E_1(\mathbf r)}\,\varepsilon(\mathbf r)\,E_2(\mathbf r)\,d\mathbf r.
$$

## 监督模型

- `unet_field_v2` 使用条件 U-Net 预测空间场与频率；
- `fno_field_v2` 使用频谱算子表达周期场的全局关系；
- 二者都必须在同一 field contract 下比较，而不是与 Full-Wh FNO 混用。

## adapted PINO

项目的 adapted PINO 复用 `fno_field_v2` 主干，并加入 TE Bloch curl–curl residual、归一化和 Rayleigh 相关项。它是项目内的物理约束训练方案，不是独立 faithful PINO 架构复现。

## 为什么 residual 不是最终答案

较低 residual 可能与频率误差、场误差或跨分辨率行为不一致。进入 inverse reranking 前至少需要：

1. 修正后的介电加权 E/H 指标；
2. frequency fidelity；
3. dense-k 检查；
4. 跨分辨率诊断；
5. residual–frequency/error 校准；
6. 冻结的准入决定。

任一关键 gate 失败时，Field/PINO 分支继续保持 `blocked`。历史 B3/B4 结果和当前度规勘误见[Field 证据索引](../results/README.md#field-physics-results)。
