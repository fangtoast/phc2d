---
title: "第一阶段 A0/B0：接管与协议冻结"
summary: "研究计划或执行协议；实施前需与当前代码和证据重新核对。"
page_type: plan
status: draft
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 11 minutes
primary_layer: L0
plan_state: completed
effective_date: 2026-07-18
superseded_by: null
related_results:
  - docs/results/forward/a1_forward_query_fair_v1_20260726.md
---

<!--
  A0/B0 handover and protocol-freeze record for the 0718-0801 phc2d sprint.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="phase1-a0-b0-protocol"></a>

# 第一阶段 A0/B0：接管与协议冻结

本文将现有真实 run、模型审计和后续实验接口收敛为可执行的 A0/B0 输入。A0/B0 本身未启动训练、批量 field 数据生成、PINO/MaxwellNet 或新的 MPB replay；其后已完成 [64 条 B1 real-MPB field-data pilot](../../results/field-physics/field_eigen_te_b1_64_20260720.md)。A1/A2/B1 必须从本文链接的版本化协议开始。

**目录：** [状态](#状态) · [A0](#a0) · [B0](#b0) · [门槛](#门槛)

---

<a id="状态"></a>

## 一、状态

10k V2 的 dataset SHA256 为 `991cb3810e881938755495ad0bccfa5b091194c2b74451842a9a50ddce974ad4`，split 为 train/val/test=`6875/2500/625`。当前 selected mixture pipeline 已完成 pure-test 625 targets、top-5 的 real-MPB replay：bands 2–10 best-of-K mean 为 `3.0775%`，95% CI 为 `[2.9658%, 3.1828%]`；这是单一 selected checkpoint 的规模验证，不是跨方法或三 seed 的公平 superiority 结论。证据来自本地 `runs/20260706/inverse_mpb_replay_mixture_v2_10000_test625_summary_20260706224424/metrics_summary.json`。

可执行配置为 [`baseline_protocol_v1.yaml`](https://github.com/fangtoast/phc2d/blob/master/configs/research/baseline_protocol_v1.yaml)，B1–B3 物理任务契约为 [`field_eigen_te_contract.yaml`](https://github.com/fangtoast/phc2d/blob/master/configs/research/field_eigen_te_contract.yaml)。后者已完成 B2 residual gate、1k data scale-up 与 9,216 条 B3 operator-data；首轮 data-only field training 的负结果见 [`field_eigen_te_supervised_baseline_20260720.md`](../../results/field-physics/field_eigen_te_supervised_baseline_20260720.md)，通过监督门槛的 B3 v2 与 PINO gate 进展见 [`field_eigen_te_b3_v2_pino_20260720.md`](../../results/field-physics/field_eigen_te_b3_v2_pino_20260720.md)。PINO validation-only lambda sweep 已绑定冻结的 data-only FNO v2；MaxwellNet 与 inverse 训练仍不允许。

---

## 二、A0：模型身份、证据与公平协议

### 2.1 三维模型登记

| 模型组 | 代码与参数量证据 | 当前配置 / run | 三维状态 | 允许表述 |
|---|---|---|---|---|
| `res_mlp_wide` | [`baselines.py`](https://github.com/fangtoast/phc2d/blob/master/src/phc2d/models/baselines.py)；formal leaderboard 记录 `878646` 参数 | [`forward_formal_retrain_v2_10000.yaml`](https://github.com/fangtoast/phc2d/blob/master/configs/research/forward_formal_retrain_v2_10000.yaml) / 本地 `runs/20260702/forward_formal_retrain_v2_10000_202607021842` | `trained` / `project_native` / `multi_seed_surrogate` | 10k vector baseline；异常 seed 必须单列。 |
| `lifted_cnn` | [`baselines.py`](https://github.com/fangtoast/phc2d/blob/master/src/phc2d/models/baselines.py)；参数量待 A1 同协议 run 记录 | A1 使用 baseline protocol | `implemented` / `project_native` / `historical multi-seed` | 可作为 grid baseline，尚无 10k formal claim。 |
| `fno2d_band_v2` | [`baselines.py`](https://github.com/fangtoast/phc2d/blob/master/src/phc2d/models/baselines.py)；formal leaderboard 记录 `579430` 参数 | [`forward config`](https://github.com/fangtoast/phc2d/blob/master/configs/research/forward_formal_retrain_v2_10000.yaml) / 本地 `runs/20260702/forward_formal_retrain_v2_10000_202607021842` | `trained` / `paper_inspired_adaptation` / `multi_seed_surrogate` | FNO-inspired band-aware spectral encoder，非 faithful FNO。 |
| `mlp_query` | [`baselines.py`](https://github.com/fangtoast/phc2d/blob/master/src/phc2d/models/baselines.py)；参数量待 A1 run 记录 | A1 使用 baseline protocol | `implemented` / `project_native` / `pilot` | query 低复杂度 baseline，缺 10k formal evidence。 |
| `deep_onet_query_v2` | [`baselines.py`](https://github.com/fangtoast/phc2d/blob/master/src/phc2d/models/baselines.py)；formal leaderboard 输出参数量 | [`query config`](https://github.com/fangtoast/phc2d/blob/master/configs/research/query_formal_retrain_v2_10000.yaml) / 本地 `runs/20260702/query_formal_retrain_v2_10000_202607022213` | `trained` / `paper_inspired_adaptation` / `multi_seed_surrogate` | DeepONet-style adaptation，非 faithful DeepONet。 |
| `fno2d_query_v2` | [`baselines.py`](https://github.com/fangtoast/phc2d/blob/master/src/phc2d/models/baselines.py)；formal leaderboard 输出参数量 | [`query config`](https://github.com/fangtoast/phc2d/blob/master/configs/research/query_formal_retrain_v2_10000.yaml) / 本地 `runs/20260702/query_formal_retrain_v2_10000_202607022213` | `trained` / `paper_inspired_adaptation` / `multi_seed_surrogate` | query spectral encoder；held-out segment fail 需单独报告。 |
| direct / single-mode inverse | [`inverse`](https://github.com/fangtoast/phc2d/tree/master/src/phc2d/inverse)；新 run 必须写 checkpoint identity 与参数量 | [`direct baseline`](https://github.com/fangtoast/phc2d/blob/master/configs/research/direct_inverse_baseline_v2_10000.yaml) | `trained` / `project_native` / `small real replay` | 历史结果不能加入新 paired 主表。 |
| `mixture_conditional_generator` | [`inverse`](https://github.com/fangtoast/phc2d/tree/master/src/phc2d/inverse)；[模型清单](../../reports/phc2d_model_inventory_v1.yaml)记录实现与证据状态 | [`test625 sample`](https://github.com/fangtoast/phc2d/blob/master/configs/research/inverse_mixture_generator_v2_10000_test625_sample.yaml) / 本地 `runs/20260706/inverse_mixture_generator_v2_10000_test625_sample_202607061921` | `trained` / `project_native` / `real_solver_replay` | selected pipeline 已经 real-MPB 验证；未证明公平优于其他方法。 |
| `unet_field` / `fno_field` | 已实现并完成 1k data-only 三 seed | [`field baseline report`](../../results/field-physics/field_eigen_te_supervised_baseline_20260720.md) | `implemented` / `project-native` or `paper-inspired adaptation` / `partial` | FNO 未超过零预测；不能作 inverse evaluator。 |
| `unet_field_v2` / `fno_field_v2` | active TE codec、conditioner 与 v2 architecture 已实现；9,216 三 seed data-only gate 通过 | [`B3 report`](../../results/field-physics/field_eigen_te_b3_v2_pino_20260720.md) | `trained` / `project-native` or `eigenfield-adapted` / `multi_seed_supervised_field` | 只允许称监督场基线，不是 faithful reproduction 或 inverse evidence。 |
| PINO-adapted / MaxwellNet | PINO 已完成 lambda gate 与三 seed；MaxwellNet 仅更新适配边界 | [`B3 report`](../../results/field-physics/field_eigen_te_b3_v2_pino_20260720.md) | `trained` or `config_only` / `eigenfield-adapted` or `planned` / `multi_seed_physics_gate` or `blocked` | PINO 不是独立架构或 faithful reproduction；MaxwellNet 缺原始 driven/source task。 |

### 2.2 Baseline protocol v1.0

[`baseline_protocol_v1.yaml`](https://github.com/fangtoast/phc2d/blob/master/configs/research/baseline_protocol_v1.yaml) 固定 dataset lineage、seeds=`0/1/2`、CUDA FP32、训练/学习率预算、validation checkpoint selection、test evaluation 与 bands 2–10 headline。每个 forward/query 主表必须同时给出参数量、训练/评估/推理 runtime 和三维状态。

inverse 使用版本化 [`100-target manifest`](https://github.com/fangtoast/phc2d/blob/master/configs/research/target_manifests/inverse_paired_test100_v1.json)：它从 existing 625-target no-train candidate artifact 按仓库 `_select_target_records` 的 low/median/high/fixed-random 四层 surrogate-MRE 策略冻结，每层 25 个 target。后续每种方法、每个适用 seed 均使用同一清单、`K=128`、`top-M=5`、500 real-MPB solves、相同阈值/二值去重/最小 Hamming 规则；只有 ranking 本身是消融变量时才可不同。

### 2.3 Claim–evidence matrix

| Claim | 可用证据 | 当前允许度 | 禁止替代 |
|---|---|---|---|
| selected mixture 可在 pure test target 上得到可 replay 的候选 | 625-target real-MPB summary | 允许 | 不得表述为优于 direct、single-mode、cVAE 或 cINN。 |
| FNO / DeepONet 已在项目任务上运行 | 多 seed surrogate run 与代码审计 | 允许，必须写 `paper_inspired_adaptation` | 不得写 faithful reproduction。 |
| query held-out 能力 | query formal run | 必须报告 fail | 不得用 all-k/interpolation 均值掩盖。 |
| field/PINO/MaxwellNet 可训练 | B2 residual、1k/9,216 数据、data-only v2 与 adapted PINO 三 seed已完成 | partial | 当前 eigenfield-adapted PINO gate 通过；MaxwellNet 仍 blocked。 |
| physics-aware inverse 有收益 | 无 paired real replay | blocked | 不能用 surrogate error 或 residual 代替 solver replay。 |

---

<a id="b0"></a>

## 三、B0：TE Bloch 本征场契约

首个 physics task 固定为 square-lattice、TE、Bloch-periodic eigenfield。每条未来 field record 是 `(structure, k-point, band)`，band 使用 one-based 2–10；k-point 来自现有 `square_gxm` physical-k metadata。完整字段、solver、normalization 和 B1 handoff 见 [`field_eigen_te_contract.yaml`](https://github.com/fangtoast/phc2d/blob/master/configs/research/field_eigen_te_contract.yaml) 与 [research profile](https://github.com/fangtoast/phc2d/blob/master/experiments/research/field_eigen_te.md)。

field shape 由序列化 schema 推导，不在 runtime 写死空间网格、k 点或 band 数。复数 E/H field 需保留显式 schema，采用 cell-volume L2 normalization 和最大幅值分量相位固定；residual 的约定是 $a=1$、$c=1$、$\omega=2\pi f$ 与 Bloch-periodic boundary。source、PML、driven-field、TM 不属于本轮。

---

<a id="门槛"></a>

## 四、门槛与交接

| Gate | 必须满足 | 失败时动作 |
|---|---|---|
| A1/A2 entry | protocol config、target manifest、模型 identity、checkpoint/source/run 链接完整 | 缺一项即从主表排除，不启动 paired claim。 |
| B1 field-data pilot | 64 条分层 field schema 包含频率、复数 E/H、Bloch BC、`dx/dy`、normalization、solver provenance | 已通过。 |
| B2 residual 与 1k | residual unit tests、solver reference 优于随机场、1,000 条可训练 records | 已通过；v1 field baseline 仅作失败诊断。 |
| B3 data-only v2 | 9,216 条 records、active TE 表示、两个模型三 seed | 已通过；两个模型全部 seed 的 E/H relative L2 `<1.0`。 |
| B3 PINO entry | 匹配 FNO v2 冻结、validation-only lambda gate、预测 E/预测频率 residual | 已通过；`lambda_pde=0.01`，三 seed residual median 降低 37.48%。 |
| B4 frequency guard | P1/P2/P3 validation 消融、frequency/E/residual 三重阈值、固定候选三 seed test | 运行中；P0 仅历史 comparator，test 不参与选择。 |
| B4 dense-k admission | 31 path indices、30 unique coordinates、source lineage、64/96/128 artifacts 与 promotion decision | blocked pending frozen candidate；不授权 inverse。 |

A0/B0 已完成的含义仍是协议冻结。B1/B2、B3 数据、data-only v2 与 adapted PINO 三 seed正式 run 已在 WSL `Ubuntu` + conda `meep_env` 完成；MaxwellNet、field inverse 与新的 inverse replay 仍须按各自 gate 独立记录。

[↑ 回顶部](#phase1-a0-b0-protocol)
