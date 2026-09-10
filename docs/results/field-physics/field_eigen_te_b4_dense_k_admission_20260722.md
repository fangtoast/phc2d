---
title: "B4：dense-k 与跨分辨率准入结果"
summary: "日期化实验与验证证据；正文保持原始记录。"
page_type: archive
status: historical
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 3 minutes
primary_layer: L3
evidence_date: 2026-07-22
evidence_scope: "日期化证据或历史记录；正文未因站点重构而改写。"
---

<!--
  B4 dense-k admission result report.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="field-eigen-te-b4-dense-k-admission"></a>

# B4：dense-k 与跨分辨率准入结果

!!! warning "历史证据"
    本页记录 dense-k 与跨分辨率准入过程，不属于正式结果白名单；准入结论只在页面声明的合同内有效。

**结论：** B4 的 frequency-guard、冻结三 seed 和 dense-k 正式评估均已完成，但三项准入状态均为 `blocked`。PINO 不能进入 inverse cross-check、residual rerank、跨分辨率 claim、inverse sampling 或 replay。

**目录：** [实验范围](#一实验范围) · [裁决结果](#二裁决结果) · [可追溯产物](#三可追溯产物) · [边界](#四边界)

---

<a id="一实验范围"></a>

## 一、实验范围

- 环境：WSL `Ubuntu` + conda `meep_env`，CUDA `NVIDIA GeForce RTX 4060 Laptop GPU`。
- 输入：冻结的 FNO/PINO 各三 seed checkpoint、B3 field HDF5、V2 频率真值和 31-index $\Gamma$–X–M–$\Gamma$ sidecar；所有 SHA256 与 source lineage 已在运行前验证。
- 评估：val/test 各 32 structures，31 path indices、30 个唯一坐标、8 seen-k、22 unseen-k、bands 2–10 和 64/96/128 分辨率；共 321,408 条 case-row，36 个原子 partial 全部完成。

---

<a id="二裁决结果"></a>

## 二、裁决结果

| 准入项 | 结果 | 直接证据 |
|---|---|---|
| `physics_frequency_crosscheck_ready` | `blocked` | seen-k E relative L2 三 seed 均满足阈值，但 unseen-k frequency MRE 的 PINO 均值为 3.1121%，FNO 为 2.9960%，比值 1.0387，高于 1.02；seed 2 比值 1.2024，也高于 1.05。 |
| `physics_residual_rerank_ready` | `blocked` | unseen-k residual median 的平均降幅为 62.48%，但 seed 1/2 的 Spearman bootstrap CI 下界分别为 -0.0580/-0.0817，四分位 frequency-MRE 降幅仅 1.62%/-7.14%，未满足校准条件。 |
| `cross_resolution_zero_shot_pass` | `blocked` | 96 的 residual median 相对 64 增加 65.92%–71.07%，128 增加 209.42%–243.76%；且 128 的 frequency MRE 也超过冻结上限。 |

`promotion_decision.json` 同时固定 `inverse_status=blocked` 与 `real_solver_benefit_status=blocked`。本次是冻结 surrogate 的 GPU forward 评估，不是新的 MPB/Meep solver validation，也未运行 inverse sampling、rerank 或 replay。

---

<a id="三可追溯产物"></a>

## 三、可追溯产物

正式 run 位于本地 `runs/20260722/field_eigen_te_b4_dense_k_gate_202607220819`。

- `promotion_decision.json`：唯一准入裁决。
- `dense_k_per_case_metrics.csv`、`k_support_metrics.csv`：case/support 指标。
- `residual_calibration.csv`、`resolution_transfer.csv`：校准与 transfer 证据。
- `lineage.json`、`history.csv`：输入绑定和 36/36 分片完成记录。

---

<a id="四边界"></a>

## 四、边界

下一步只能进入针对性的 B4 诊断：解释 unseen-k frequency 失配、residual-error 校准失效和分辨率残差退化。不得把 residual 下降单独升级为 rerank 或 inverse 收益，也不得基于本结果启动 MaxwellNet、field inverse、inverse replay 或 100k 数据生成。

[返回顶部](#field-eigen-te-b4-dense-k-admission)
