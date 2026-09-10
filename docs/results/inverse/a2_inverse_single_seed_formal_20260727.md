---
title: "A2 单 Seed 逆向设计公平对照正式结果"
summary: "日期化实验与验证证据；正文保持原始记录。"
page_type: evidence
status: draft
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 4 minutes
primary_layer: L3
evidence_date: 2026-07-27
evidence_scope: "日期化证据或历史记录；正文未因站点重构而改写。"
dataset_scope: v2_10k_wh31x10
evidence_level: solver_replay
evidence_status: completed
validation_backend: wsl_mpb
source_artifacts:
  - runs/20260726/a2_inverse_single_seed_v1_202607261911
---

<!--
  Formal A2 single-seed fair inverse benchmark result.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# A2 单 Seed 逆向设计公平对照正式结果

## 结论

A2 已按派生协议完整执行，状态为 `formal_completed`，证据标签为
`single_seed_formal_replay`。本结果固定 `seed=0` 和同一组 100 个 test
targets，不提供跨 seed 稳定性证据。

train-only retrieval 的 real-MPB replay mean 最低，为 `1.521%`。在三个可训练
inverse 方法中，mixture generator 优于 direct inverse，但相对 matched
single-mode generator 的差异经三项 Holm 校正后不显著，因此不能声称 mode
架构具有显著优势。

## 协议与完整性

| 项目 | 正式证据 |
|---|---|
| dataset | 10k V2，SHA-256 `991cb381…974ad4` |
| split | `6875/2500/625` |
| target manifest | 100 个固定 test targets |
| seed | `0` |
| validation cases | `3 methods × 3 LR = 9` |
| checkpoint evaluations | `54` |
| candidate records | `4 methods × 100 targets = 400` |
| replay references | `1546` |
| unique real-MPB solves | `1335/1335` |
| cache reuse | `211` |
| solver failure | `0` |
| real-MPB wall time | `39018.75 s`，约 `10.84 h` |
| environment | WSL `Ubuntu` + conda `meep_env` |

候选冻结后重新核验 SHA-256。retrieval 的 100 条记录均声明且实际只访问 train
split；direct inverse 的重复输出二值去重后为每 target 一个有效候选，没有为
凑预算重复调用 MPB。

## Real-MPB 主表

MRE 均以百分比报告，bootstrap 为 paired-target 协议下固定 seed 0 的 95% CI。

| method | selected LR / epoch | replay mean | median | p10 | p90 | bootstrap 95% CI | effective K/M |
|---|---|---:|---:|---:|---:|---|---:|
| retrieval | — | 1.521% | 1.428% | 1.123% | 2.054% | [1.451%, 1.597%] | 128.00 / 5.00 |
| direct inverse | `3e-4 / 200` | 18.119% | 17.825% | 4.981% | 33.945% | [15.792%, 20.583%] | 1.00 / 1.00 |
| single-mode generator | `1e-4 / 150` | 13.183% | 9.896% | 3.522% | 31.746% | [11.007%, 15.440%] | 18.32 / 4.79 |
| mixture generator | `3e-4 / 200` | 11.558% | 7.121% | 3.118% | 25.729% | [9.612%, 13.647%] | 19.68 / 4.67 |

## Surrogate 诊断与规模

| method | ranking FNO MRE | query FNO diagnostic MRE | surrogate–replay gap | params |
|---|---:|---:|---:|---:|
| retrieval | 1.517% | 1.527% | 0.004 pp | 0 |
| direct inverse | 16.901% | 17.213% | 1.218 pp | 154,660 |
| single-mode generator | 12.561% | 12.656% | 0.623 pp | 491,812 |
| mixture generator | 10.943% | 11.133% | 0.615 pp | 492,580 |

query FNO 仅用于 cross-surrogate 诊断，不参与候选排序或 checkpoint 选择。

## Paired 比较

差值定义为 `mixture − baseline`；负值表示 mixture 的 MRE 更低。

| comparison | mean delta | paired bootstrap 95% CI | sign-test p | Holm p |
|---|---:|---|---:|---:|
| mixture vs retrieval | +10.037 pp | [8.079, 12.135] pp | `1.58e-30` | `4.73e-30` |
| mixture vs direct | −6.560 pp | [−8.165, −5.013] pp | `4.83e-13` | `9.65e-13` |
| mixture vs single-mode | −1.625 pp | [−3.229, −0.056] pp | `0.0886` | `0.0886` |

mixture 的 effective-K 略高于 single-mode，且 Holm 校正后差异不显著。该结果
最多支持“当前候选预算下 mixture 的均值较低”，不能分离 candidate-budget
advantage 与 mode 架构贡献；K/top-M/mode 消融属于 A3。

## 边界

- 不报告 seed mean ± std。
- 不使用“跨 seed 稳定”或“三 seed 显著”等措辞。
- PINO、field inverse、B4 checkpoint 和 physics-aware rerank 均未进入 A2。
- `formal_completed` 只表示该 single-seed 协议完整执行。
- 原始 checkpoint、candidate manifest、MPB shards 和 run outputs 保留在
  `runs/20260726/a2_inverse_single_seed_v1_202607261911/`，不进入 Git。
