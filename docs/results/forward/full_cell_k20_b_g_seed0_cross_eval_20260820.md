---
title: "K=20 B/G 单 Seed 正向训练与跨路线评估"
summary: "记录 K=20 B/G Fourier 全胞数据上的 17-task seed=0 训练、冻结 validation checkpoint 和已提交的 34-cell held-out evaluation。"
page_type: evidence
status: draft
audience: "研究执行者、导师与结果审阅者"
owner: "研究生维护"
last_reviewed: 2026-08-20
evidence_date: 2026-08-20
evidence_scope: "K=20 B/G Fourier full-cell; Wh[10000,81,10]; seed=0; N40 surrogate training and pending held-out evaluation"
reading_time: 7 minutes
primary_layer: L3
backend: "surrogate training on N40; WSL Ubuntu + meep_env real-MPB data gate"
dataset_scope: full_cell_fourier_k20_wh81x10
evidence_level: surrogate_held_out
evidence_status: completed
validation_backend: mixed
source_artifacts:
  - runs/20260820/full_cell_k20_forward_seed0_202608201430
---

<!--
  Draft evidence record for the K=20 B/G seed-0 forward campaign.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="k20-forward-seed0"></a>

# K=20 B/G 单 Seed 正向训练与跨路线评估

本页汇总 K=20 full-cell Fourier B/G 数据上的 N40 单 seed 正向训练和 34-cell held-out evaluation。训练与评估均已完成；结论仍仅限于 **single-seed held-out surrogate evidence**，不能升级为 seed-robust、物理模型成功或 inverse 成功。

**目录：** [数据 gate](#一数据-gate) · [训练协议](#二训练协议) · [训练结果](#三训练结果) · [评估状态](#四评估状态) · [证据边界](#五证据边界)

---

<a id="一数据-gate"></a>

## 一、数据 gate

| 项目 | B binary | G graded |
|---|---|---|
| HDF5 SHA-256 | `2661eb1e9e3b398a7b69c4083daefa761ea9c26a1081c478b968fd6a951d6b56` | `96b127b43f2c31fb7591e9634317a7ed35332ceae1077b2277de15058902fd72` |
| 输入 grid | `material_area_fraction_grid_64` | `material_weight_grid_64` |
| target shape | `Wh[10000,81,10]` | `Wh[10000,81,10]` |
| split | `8000 / 1000 / 1000` | `8000 / 1000 / 1000` |
| WSL MPB 核查 | 8 structures × 9 k-points | 8 structures × 9 k-points |
| WSL 数据 gate | pass；median `0`，P95 `1.08e-7` | pass；median `0`，P95 `1.02e-7` |

该 gate 在 WSL Ubuntu + conda `meep_env` 下运行，使用 Meep `1.33.0-beta`、MPB `1.12.0`、`resolution=64`、`mesh=3`、TE bands 2–10。它只证明指定样本的数据物理核查通过，不验证 surrogate 预测表现。

---

<a id="二训练协议"></a>

## 二、训练协议

- Campaign：`/HOME/scw6g7a/run/phc2d/runs/20260820/full_cell_k20_forward_seed0_202608201430`。
- 共 17 个任务：B/G 各 8 个 single-lane 模型和 1 个 `dual_lane_band_operator_v1`；全部 `seed=0`、dense head、FP32、100 epoch 上限、validation checkpoint selection。
- Admission 17/17 成功；W1 `5422465` 和 W2 `5422496` 均 8/8 `formal_training_success`；W3 `5423180` 为 `dual_lane_training_success`。
- 每任务请求 1 GPU、6 CPU、12 小时。训练无 OOM、nonfinite 或 checkpoint restore 错误；W3 `MaxRSS` 约 954 MiB，GPU 峰值约 6.45 GiB。

下表是 best-validation epoch 的 `val_bands_2_10_mre_pct`，用于 checkpoint 冻结；测试集尚未用于模型选择。

---

<a id="三训练结果"></a>

## 三、训练结果

| task | 模型 | 路线 | 状态 | best epoch | validation MRE (%) |
|---:|---|---|---|---:|---:|
| 0 | `mlp` | B | formal | 3 | 1.1623 |
| 1 | `mlp` | G | formal | 5 | 4.4873 |
| 2 | `res_mlp_wide` | B | formal | 85 | 0.5389 |
| 3 | `res_mlp_wide` | G | formal | 34 | 0.7246 |
| 4 | `lifted_cnn` | B | formal | 76 | 0.5050 |
| 5 | `lifted_cnn` | G | formal | 100 | 0.4344 |
| 6 | `fno2d_band_v2` | B | formal | 19 | 0.4890 |
| 7 | `fno2d_band_v2` | G | formal | 68 | 0.4091 |
| 8 | `mlp_mixer_band_v1` | B | formal | 8 | 0.5002 |
| 9 | `mlp_mixer_band_v1` | G | formal | 12 | 0.6652 |
| 10 | `res_mlp_gated_v2` | B | formal | 9 | 0.6359 |
| 11 | `res_mlp_gated_v2` | G | formal | 23 | 0.7472 |
| 12 | `convnext_band_v1` | B | formal | 27 | 0.5088 |
| 13 | `convnext_band_v1` | G | formal | 58 | 0.4434 |
| 14 | `tfno2d_band_v3` | B | formal | 94 | 0.2846 |
| 15 | `tfno2d_band_v3` | G | formal | 41 | 0.2721 |
| 16 | `dual_lane_band_operator_v1` | joint | dual-lane | 99 | 0.2153 |

远端 `checkpoint_inventory.csv` 已冻结 17 个 checkpoint，其 SHA-256 是 `0f3c46391b10616fa8a7083c3597823f4e2a7272562f6daa7cd6c5cb443cbf55`。这些 validation 数值不等价于最终 native/cross held-out 排名。

---

<a id="四评估状态"></a>

## 四、评估状态

| 项目 | 当前值 |
|---|---|
| evaluation plan | `evaluation_plan_frozen` |
| cells | 34 |
| 构成 | 16 个 single-lane checkpoint × native/cross（32）+ joint checkpoint 的 B/G native（2） |
| 提交作业 | N40 Slurm array `5423464[0-7%8]` |
| 分片资源 | 1 GPU、6 CPU、12 小时，最多 8 并发 |
| 当前状态 | 8/8 shard 完成并已 finalizer 汇总 |

评估只在所有 validation checkpoint 冻结后读取 hold-out test。8 个 shard 均以 `COMPLETED/0:0` 结束，finalizer 输出 `evaluation_complete_pending_report`：17 个训练 checkpoint、34/34 evaluation cells（18 native、16 cross），无失败记录，并完成 10,000-replicate paired bootstrap。

### 4.1 Held-out 汇总

| 路线 | 最低 native MRE | 最低 cross MRE | 结论边界 |
|---|---:|---:|---|
| B | dual-lane native `0.2450%`；single-lane TFNO `0.2881%` | G→B FNO `1.5712%` | joint 模型不构成跨路线诊断；single-lane cross 仍显著变差 |
| G | dual-lane native `0.1860%`；single-lane TFNO `0.2724%` | B→G lifted CNN `3.4137%` | 同上；B→G/G→B 均保留 domain-shift 边界 |

### 4.2 完整 single-lane B/G 交叉验证

所有 cell 使用源 checkpoint 的 normalizer，并在目标路线的 1,000 个 held-out samples 上评估。`MRE` 是 bands 2–10 MRE；`R²` 是 `r2_centered_mean`。joint model 训练时已同时见过 B/G，因此不纳入本表的 cross-lane 诊断。

| 源模型 | B→G MRE (%) | B→G R² | G→B MRE (%) | G→B R² |
|---|---:|---:|---:|---:|
| `mlp` | 5.3029 | -0.0766 | 10.6465 | -0.0853 |
| `res_mlp_wide` | 3.9891 | 0.3890 | 4.1200 | 0.8072 |
| `lifted_cnn` | **3.4137** | 0.5505 | 2.3775 | 0.9441 |
| `fno2d_band_v2` | 3.8473 | 0.4398 | **1.5712** | **0.9773** |
| `mlp_mixer_band_v1` | 3.8803 | 0.4267 | 4.3587 | 0.7870 |
| `res_mlp_gated_v2` | 3.8218 | 0.4346 | 4.3556 | 0.7983 |
| `convnext_band_v1` | 3.6117 | 0.4907 | 2.7355 | 0.8958 |
| `tfno2d_band_v3` | 3.7373 | 0.4619 | 3.1784 | 0.8295 |

交叉误差与 native 误差相比仍有明显 domain shift，且最优模型随方向改变：B→G 是 lifted CNN，G→B 是 FNO。该方向差异是单 seed 实验观察，不能据此推广为模型族的普遍优劣。

在预注册的四组 upgrade–baseline 对比中，TFNO 相对 FNO 的 native MRE 改善为 B `-0.1970` pp（95% CI `[-0.2070,-0.1873]`）和 G `-0.1351` pp（`[-0.1428,-0.1276]`）；但 G→B cross 为 `+1.6072` pp（`[+1.4173,+1.7991]`），不能描述为跨路线替代。ConvNeXt 相对 lifted CNN 的 native 差异在 B/G 的置信区间均跨 0。

---

<a id="五证据边界"></a>

## 五、证据边界

1. 当前训练结果是 **single-seed、validation-only surrogate evidence**，不构成三 seed 稳健性或全局模型优劣结论。
2. 表中的 held-out 结果只来自一个 seed；不构成 seed-robust 或全局模型优劣结论。
3. WSL real-MPB gate 验证的是数据样本，不是模型的 real-MPB replay。
4. 本页不能作为 inverse design 成功证据；real-MPB replay 仍是下游任务最终裁判。

**当前完成状态：** 训练和 34-cell held-out evaluation 已完成；后续如需 promotion、论文证据或 inverse 使用，仍需单独决定多 seed 与 real-MPB replay gate。

[↑ 回顶部](#k20-forward-seed0)
