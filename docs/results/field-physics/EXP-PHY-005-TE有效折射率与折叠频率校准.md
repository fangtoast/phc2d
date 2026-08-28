---
title: "EXP-PHY-005：TE 有效折射率与折叠频率校准结果"
summary: "192 结构 near-Gamma real-MPB 表明单一标量 TE 调和候选不成立，且方向各向异性超门；停止 resonance-prior 训练准入。"
page_type: evidence
status: draft
audience: "物理建模研究者、正向代理研究者与结果审阅者"
owner: "研究生维护，导师审阅科学结论"
last_reviewed: 2026-08-28
reading_time: 5 minutes
evidence_date: 2026-08-28
primary_layer: L3
evidence_scope: "EXP-PHY-005 K5 Binary/Graded train 64 + validation 32 per lane near-Gamma TE real-MPB slope calibration and fair 4k+1k per-lane all-adjacent-gap audit; no test or training"
dataset_scope: "K5 Binary/Graded Wh[10000,81,10] with frozen 256-resolution material fields"
evidence_level: physical_validation
evidence_status: completed
validation_backend: wsl_mpb
source_artifacts:
  - runs/20260828/EXP-PHY-005_202608281751
---

<!--
  EXP-PHY-005 curated result.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-PHY-005：TE 有效折射率与折叠频率校准结果

**结论：当前单一标量 TE 调和候选不成立，不能作为 resonance-prior loss 的中心。** 正式终态为 `scalar_anisotropy_fail`：Binary 的 Γ–X/Γ–M 有效折射率差异 P90 为 `5.36%`，超过 `5%` 门。即使暂时忽略该门，TE 候选相对 MPB slope 的误差也在两条 lane 显著劣于旧公式，因此 EXP-FWD-017 不解锁。

## 实验与证据完整性

| 项目 | 正式证据 |
|---|---|
| Run | `runs/20260828/EXP-PHY-005_202608281751/` |
| 代码身份 | `51937a59aad3acdce5755aaa9b8bf19a2a5e96f8` |
| near-Γ 样本 | Binary/Graded 各 train 64 + validation 32，共 192 条 |
| gap 审计 | Binary/Graded 各 train 4,000 + validation 1,000；B1–B2 至 B9–B10 |
| 求解合同 | TE，resolution 64，mesh 3，tolerance `1e-7`，2 bands + 4 guard bands，9 个唯一 k 点 |
| 完整性 | 192 个唯一 key，`bands[192,9,2]`，零 solver failure，零 test overlap |
| 禁止项 | `test_targets_loaded=false`，`training_started=false`，`n40_used=false` |

三个均匀介质 sentinel 全部通过；最大有效折射率相对误差为 `1.43e-8`。每条 lane、每个方向的 `R²>=0.999` 通过率和内外窗口漂移通过率均为 `100%`，因此该负结果不是 small-k 数值拟合失效造成的。MPB 的二维 TE 定义为电场在平面内、磁场仅有 z 分量，与本实验 `run_te()` 合同一致，参见 [MPB Python User Interface](https://mpb.readthedocs.io/en/latest/Python_User_Interface/#run-functions)。

## 有效介质与方向门

| Lane | 旧公式 median / P90 | TE 调和 median / P90 | TM 算术候选 median | 各向异性 median / P90 | TE 优于旧公式比例 |
|---|---:|---:|---:|---:|---:|
| Binary | 7.77% / 15.08% | 24.81% / 27.03% | 14.94% | 1.77% / 5.36% | 1.04% |
| Graded | 5.83% / 9.24% | 20.71% / 23.08% | 11.74% | 1.39% / 3.64% | 0.00% |

pooled paired bootstrap 对“旧误差减 TE 误差”的 95% CI 为 `[-15.83,-14.60]` 个百分点，方向与预期改善相反。旧的 `mean(sqrt(epsilon))` 在本数据上更接近 slope，但这只是经验对照，不把旧公式升级为严格均匀化公式。

该结果说明：对当前非 p4m、方向不受约束的全胞结构，简单的 `1/sqrt(mean(1/epsilon))` 不能替代真实二维 TE 均匀化。需要时应转向方向相关的有效张量或胞元 corrector，而不是继续调整一个标量中心。

## 三个折叠候选与真实 gap

| 候选 | Validation pair 一致率 Binary / Graded | Oracle 归一化中心距离中位数 Binary / Graded | Oracle 落入 complete gap | 判定 |
|---|---:|---:|---:|---|
| `gamma_g1` | 0% / 0% | 0.845 / 0.980 | 0/32、0/32 | 解析式与 oracle 选择不同 band pair |
| `x` | 96.88% / 100% | 1.126 / 1.182 | 0/32、0/32 | 只稳定选择 B1–B2，不等于找到 gap 中心 |
| `m` | 0% / 0% | 0.620 / 0.657 | 0/32、0/32 | 解析式与 oracle 选择不同 band pair |

`x` 通过了预注册的相对 gap 子门：两条 lane 相对旧公式的中心距离均改善、pair 分布稳定且不越出 Bands 1–10 频率范围。但该子门没有绝对距离条件；本次 oracle 中位距离大于一个 spacing scale，且没有 validation 样本落入 complete gap。因此不能把 `x` 的子门通过写成“已找到工作频率和对应 gap”。

`gamma_g1` 和 `m` 的 oracle 对某些 pseudo-gap 中心较近，说明能带折叠图像本身尚未被完全否定；被正式否定的是“用一个 TE 标量平均式同时代表两个方向，并据此生成训练中心”的当前合同。

## 决策

- 固定正式状态为 `scalar_anisotropy_fail`，并同时记录 `estimator_pass=false`。
- 不登记、不提交 EXP-FWD-017，不在 N40 上训练 resonance-prior loss。
- 不使用 `x` 的 pair 一致率单独宣称工作频率命中。
- 若继续该研究问题，下一实验应先定义方向相关有效张量/胞元均匀化，并重新预注册绝对 gap 距离或 complete-gap 命中门；这属于新方法与新授权，不是本 run 的补丁。

原始指标见 `slope_metrics.json`、`gap_alignment_metrics.json` 与 `effective_index_samples.csv`；near-Γ 原始频率见 `near_gamma_bands.npz`。本结果不改写 EXP-FWD-012，也不产生任何 surrogate 训练或性能结论。
