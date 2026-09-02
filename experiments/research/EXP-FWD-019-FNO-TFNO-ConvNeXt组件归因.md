---
title: "EXP-FWD-019：FNO、TFNO 与 ConvNeXt 组件归因"
summary: "以 validation-only 冻结诊断、同家族训练消融和六域冻结评估检验频谱、局部卷积、周期边界、层级下采样与输出头贡献。"
page_type: plan
status: draft
audience: 正向代理研究者、模型实现者与结果审阅者
owner: 研究生维护，导师审阅组件主张
last_reviewed: 2026-09-02
reading_time: 7 minutes
primary_layer: L3
---

<!--
  EXP-FWD-019 experiment contract.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-FWD-019：FNO、TFNO 与 ConvNeXt 组件归因

## 结论与边界

本实验只判断组件删除是否在公平训练合同下造成稳定退化。它不把 Fourier、周期卷积或 Tucker 参数化直接解释为 Maxwell 因果，也不使用 real-MPB/Meep。训练统一使用原 `normalized_full_matrix_mse`；Joint 使用等 lane 的同一全矩阵损失。

用户于 2026-09-02 授权本实验 C0--C3 的 N40 提交、依赖续投和过门候选 seeds 1/2 扩展。授权不包括改数据、改资源规格、失败后调参重跑、EXP-FWD-018 loss 路线或物理 replay。

## 冻结合同

| 项目 | 合同 |
|---|---|
| 数据 | F50 Binary/Graded，各自固定 40k/5k/5k split |
| 输入/目标 | `material_weight_grid_64 -> Wh[81,10]` |
| 选择边界 | C0--C3 选择只读 validation；held-out test 仅在组件与 baseline 冻结后执行一次 |
| 优化 | AdamW、FP32、AMP/TF32 off、effective batch 64、原模型学习率 |
| seed0 筛选 | 20 epochs，同 family、同 lane、同 seed、从零训练 |
| 三 seed 确认 | 200 epochs，seeds 0/1/2；必须同时训练 matched baseline |
| 平台 | N40 `gpu_4090`，每任务 1 GPU、6 CPU、60 GB；最多 3 个并发任务 |

数据哈希固定为 Binary `e2d99e0c...13f00`、Graded `172786bc...38a46`；完整值由配置和 `submission.env` 保存。

## C0：冻结诊断

复用 F50 seed0 validation-best checkpoints。每条 lane 统计 Bands 2--10 MRE，并按输入高频能量、周期边界差分与灰度比例四分位分层。整数周期平移使用 `(1,0)`、`(0,1)`、`(4,4)`；Fourier 输入遮罩使用归一化半径 `0.25`。FNO 屏蔽 primary spectral、额外 band spectral、band gate、low-rank summary；TFNO/Joint 屏蔽 spectral、local、channel。冻结屏蔽只排序训练候选。

FNO、TFNO、Joint 分别按 Binary/Graded 的 mean prediction-delta 排名前二进入 C2；若两条 lane 排名不一致，优先选择每条 lane 第一名的并集并截断为两个。ConvNeXt 固定先检验 `kernel 7->1` 和 `circular->zeros`；层级下采样 bridge 仅在前两项无法解释两 lane 退化时替补。

## C1/C2：seed0 公平筛选

每个 family 先训练一个 matched baseline；FNO baseline 使用新身份 `fno2d_band_component_v1` 与共同 dense head。其余身份为 `tfno2d_band_component_v1`、`convnext_band_component_v1`、`dual_lane_band_component_v1`，参数/状态 schema 与具体开关写入 run identity。

组件删除通过 seed0 门的条件：

- 两条独立 lane 均使 validation Bands 2--10 MRE 相对 matched baseline 退化至少 2%；或
- 一条 lane 退化至少 5%，另一条 lane 不得改善超过 1%；
- 无 NaN/Inf，run 制品、模型身份、参数量、峰值显存和 wall time 完整；
- Joint 使用 Binary/Graded 两条 validation 曲线分别应用同一门。

每个 family 最多保留一个删除候选。未过门即停止该候选，不补 seed、不读 test。

## C3：三 seed 与六域冻结评估

对保留候选及其 matched baseline 从零训练 200 epochs、seeds 0/1/2。三 seed 的 paired validation 差值必须同号，且任一 lane 不得出现超过 1% 的反向改善，才能冻结模型与 checkpoint。随后一次性评估 K5/K20/F50 × Binary/Graded 六域，报告 Band 1、Bands 2--10、Bands 1--10 MRE/MAE、gap decision、参数量、训练时间、峰值显存与推理吞吐。

只有冻结诊断、seed0 删除消融、三 seed 配对确认和六域评估方向一致，才写“该组件在此合同下有贡献”。否则结论为不确定或被证伪。

## 预期证据

`runs/{date}/EXP-FWD-019_*/` 必须保存标准 run 制品，并按阶段附加 `c0_manifest.csv`、`stratified_validation.csv`、`task_manifest.csv`、`candidate_gate.json`、`checkpoint_manifest.csv`、六域逐 cell 指标和资源汇总。原 checkpoint 与原 run 不覆盖、不改写。
