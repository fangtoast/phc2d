---
title: "EXP-FWD-018：K5 理论频率加权全谱训练"
summary: "以冻结的有效介质折叠中心比较完整频谱 MSE 与有界逆距离加权，结论仅限 surrogate validation/test。"
page_type: experiment
plan_state: active
status: draft
audience: "正向代理研究者、N40 执行者与结果审阅者"
owner: "研究生维护，负责人授权远端训练"
last_reviewed: 2026-08-31
primary_layer: L3
---

<!--
  EXP-FWD-018 experiment contract.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="exp-fwd-018"></a>

# EXP-FWD-018：K5 理论频率加权全谱训练

**本实验检验理论频率窗口内的误差是否可降低，同时保持完整 Bands 1–10 代理精度。** 证据仅为 surrogate；$f_0$ 只登记为 `heuristic_folded_center_x`，不是精确工作频率、gap center、complete gap 或器件响应结论。

**目录：** [合同](#合同) · [损失](#损失) · [训练与制品](#训练与制品) · [门与测试](#门与测试)

---

## 一、合同

| 项 | 冻结值 |
|---|---|
| 数据 | Binary / Graded 各 8,000 train、1,000 validation、1,000 test；输入各 lane 的 64×64 材料场，输出完整 `Wh[81,10]` |
| 模型 | `dual_lane_band_operator_v1`；两 arm 同 seed 使用相同初始化与数据顺序 |
| arm | `baseline_mse` 与 `bounded_inverse_frequency`，各 seeds 0/1/2，共 6 个 N40 单卡任务 |
| 训练 | 200 epoch、AdamW、lr $5\times10^{-4}$、weight decay $10^{-5}$、FP32、每步 Binary/Graded 各 32 |
| 选择 | validation equal-lane Bands 2–10 MRE；不按理论窗口或 test 选 checkpoint |
| 非目标 | 不验证 complete/max gap、器件响应或 MPB/Meep；不修改 EXP-FWD-017 |

每条 lane 只用其 train split 计算 target normalizer。训练启动时从 train/validation 冻结栅格一次性预计算中心，不在 batch 中读取高分辨率 HDF5；test 在 validation 判定与全部 checkpoint 冻结后才允许单次读取。

## 二、损失

令材料场对应的介电常数为 $\epsilon$，使用公共 `effective_refractive_index(..., "TE")` 计算：

$$
n_{\mathrm{eff}}=\frac{1}{\sqrt{\langle 1/\epsilon\rangle}},\qquad
f_0=\frac{1}{2n_{\mathrm{eff}}}.
$$

加权 arm 使用已冻结的 $\epsilon_f=0.0008457602483520898$、$\rho=4$、$\alpha=0.25$：

$$
r_{qb}=\frac{\epsilon_f}{|\tilde f_{qb}-f_0|+\epsilon_f},\quad
w_{qb}=\frac{1+\rho r_{qb}}{\operatorname{mean}_{q,b}(1+\rho r_{qb})},\quad
L_B=(1-\alpha)L_A+\alpha\operatorname{mean}_{q,b}w_{qb}(\hat z_{qb}-z_{qb})^2.
$$

全部 $81\times10$ 坐标仍有非零监督。原始无界逆距离仅保留为离线诊断，不参与反向传播。

## 三、训练与制品

每个标准 run 保存常规 run 制品、validation-best `model.pt`、terminal checkpoint，以及 `frequency_center_manifest.jsonl`、`weight_statistics.json`、`window_metrics.json`、`per_band_metrics.csv`、`per_k_metrics.csv`、`frequency_pair_diagnostics.csv` 与 `figures/frequency_window_validation.*`。所有输出位于 `runs/{date}/EXP-FWD-018_.../`。

任务在 hash 漂移、非有限 loss/gradient、OOM、样本覆盖不完整或 checkpoint 恢复失败时为 `blocked`。若第 181–200 epoch 的 validation Bands 2–10 MRE 相对变化超过 0.5%，标记 `inconclusive_nonconverged`，不自动延长或重试。

## 四、门与测试

validation 通过要求每条 lane 同时满足：±10% 窗口 NMAE 改善至少 5%、至少 2/3 seeds 改善、hierarchical bootstrap 95% CI 上界小于 0、B1–B2 条件 NMAE 不退化、Bands 2–10 与 Bands 1–10 MRE 相对退化不超过 2%，以及任一 B3–B10 MRE 相对退化不超过 5%。gap 仅是粗 9×9 标签诊断，不参与 promotion。

只有 validation 通过后，才冻结 6 个 checkpoint、指标代码与报告模板并读取 test 一次；test 采用 seed 配对和分层 hierarchical bootstrap 10,000 次。两阶段均通过才可写 `frequency_weighted_loss_pass`；其余状态严格为 `generalization_fail`、`local_gain_global_regression`、`frequency_weighting_no_gain`、`inconclusive` 或 `blocked`。

[↑ 回顶部](#exp-fwd-018)
