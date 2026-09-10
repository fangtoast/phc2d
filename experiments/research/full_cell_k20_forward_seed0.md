---
title: "K=20 B/G 数据集单 Seed 正向模型训练与跨路线评估"
summary: "在 Xingyi 生成的 binary/graded Fourier K=20 全胞数据上执行 17 个 seed=0、dense-head 正向任务，并完成 34 个 native/cross-lane held-out evaluation cells。"
page_type: plan
plan_state: active
status: draft
audience: "研究执行者、导师与结果审阅者"
owner: "研究生维护"
last_reviewed: 2026-08-20
reading_time: 5 minutes
primary_layer: L3
superseded_by: ""
related_results: "docs/results/forward/full_cell_k20_b_g_seed0_cross_eval_20260820.md"
---

<!--
  K=20 binary/graded single-seed forward campaign profile.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 实验范围

本 profile 执行用户给定的 K=20 B/G 计划。主层级为 **L3 实验与证据**；代码、配置和调度脚本只是执行载体，不能单独证明模型结果。

- 数据：`Wh[10000,81,10]`，B 使用 `material_area_fraction_grid_64`，G 使用 `material_weight_grid_64`。
- 任务：B/G 各 8 个 single-lane 模型，另加 1 个 dual-lane 模型；全部 `seed=0`、dense head、FP32、固定 batch/优化器/early stopping 协议。
- 评估：16 个 single-lane checkpoint 各做 native 与 cross-lane，dual-lane 做 B/G native，共 34 个 evaluation cells；跨路线使用源 checkpoint 的 normalizer。
- 统计：对四组 upgrade-vs-baseline 比较保留 paired bootstrap 10,000 次；surrogate held-out 指标与真实 MPB 验收分开记录。

## Gate 与停止条件

1. 先验证本地 HDF5 SHA-256、schema、split 完整性，并在 WSL Ubuntu + conda `meep_env` 对 B/G 各 8 个结构、9 个代表性 k 点执行真实 MPB 验收。
2. 两条路线均通过后，在 N40 冻结 17-task manifest，先提交 admission，再依次提交 W1 baseline、W2 upgrade、W3 dual-lane；每波必须检查结果后才进入下一波。
3. OOM、非有限值、checkpoint restore 失败、任务覆盖不完整或 hash 漂移均视为失败，不临时修改 batch、learning rate、seed 或模型配置。
4. 结果只允许表述为 single-seed held-out surrogate evidence；不能升级为 seed-robust、全局最优或 real-MPB 模型成功结论。

## 入口与产物

- 冻结与评估控制面：`src/phc2d/forward/full_cell_k20_campaign.py`
- N40 训练与状态脚本：`jobs/paracloud_submit_full_cell_k20.sh`、`jobs/paracloud_full_cell_k20_train.sh`、`jobs/paracloud_full_cell_k20_status.sh`
- N40 evaluation 脚本：`jobs/paracloud_submit_full_cell_k20_eval.sh`、`jobs/paracloud_full_cell_k20_eval.sh`
- 配置：`configs/research/full_cell_k20_forward_*_seed0_v1.yaml`
- 结果入口：[`../../docs/results/forward/full_cell_k20_b_g_seed0_cross_eval_20260820.md`](../../docs/results/forward/full_cell_k20_b_g_seed0_cross_eval_20260820.md)

实际状态以 campaign 下的 `freeze.json`、`task_manifest.csv`、训练 run artifacts、`evaluation_cells.csv` 和 `results.json` 为准。本页和结果页均保持 Draft，等待人工审阅。
