<!--
  Preregistered TE effective-index and folded-gap calibration experiment.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-PHY-005：TE 有效折射率与折叠频率校准

结论先行：本实验已完成，正式终态为 `scalar_anisotropy_fail`，且 `estimator_pass=false`。单一标量 TE 调和候选不能作为 resonance-prior loss 中心，EXP-FWD-017 不解锁。结果见[正式结果页](../../docs/results/field-physics/EXP-PHY-005-TE有效折射率与折叠频率校准.md)。

| 元数据 | 冻结值 |
|---|---|
| 范围 | 当前 K5 TE 全胞 Binary / Graded 数据 |
| Owner | 研究生执行，导师接受物理结论 |
| 状态 | `completed` |
| 生效日期 | 2026-08-28 |
| 主层级 | L3 实验/证据；本轮实现为 L4 支撑 |
| 关联计划 | [工作频率加权与自适应联合算子计划](../../docs/planning/工作频率加权与自适应联合算子计划.md) |

## 问题与对照

实验检验偏振相关的 TE 候选

\[
n_{\mathrm{eff}}=\frac{1}{\sqrt{\langle1/\epsilon\rangle}}
\]

能否逼近 Band 1 在 Γ 点附近的 real-MPB 斜率，并使 `gamma_g1=1/n_eff`、`x=1/(2n_eff)` 或 `m=sqrt(2)/(2n_eff)` 中至少一个预注册候选稳定对应 Bands 1–10 的真实相邻 gap。旧的 `mean(sqrt(epsilon))` 是冻结对照，不在本实验中改写。

## 冻结合同

| 项目 | 合同 |
|---|---|
| 数据 | K5 Binary 10k 与 K5 Graded 10k；验证 dataset、solver、geometry、material hash |
| slope 样本 | 每 lane train 64 + validation 32；按 `family_id × material_kind` 分层并由 `geometry_hash` 稳定抽样 |
| 材料积分 | 冻结 256 raster；同时计算旧公式、TM 算术候选、TE 调和候选 |
| near-Γ | `kappa={0,1/256,1/128,1/64,1/32}`；Γ–X 与 Γ–M 共 9 个唯一 k 点 |
| real-MPB | TE，resolution 64，mesh 3，tolerance `1e-7`，保留 2 bands 加 4 guard bands |
| gap 审计 | 每 lane train 4,000 + validation 1,000；枚举 B1–B2 至输入最后一对相邻 band |
| 禁止项 | 不访问 test，不训练，不使用 N40，不修改 EXP-FWD-012 历史语义 |

正式求解前必须先通过 `epsilon=1,4,8.9` 三个均匀介质 sentinel。数值门、有效介质门、gap 对齐门及固定终态分类以冻结配置和关联计划为准；smoke 只能证明求解链与制品链可运行，不能触发 promotion。

## 配置、命令与制品

正式配置为 [`exp_phy_005_te_effective_index_gap_calibration_v1.yaml`](../../configs/research/exp_phy_005_te_effective_index_gap_calibration_v1.yaml)。一次性正式授权已由 `EXP-PHY-005_202608281751` 消耗，规范配置恢复为 `preflight` 且 `formal_solver_authorized=false`；该 run 内的 `config.resolved.yaml` 保留实际正式输入。工程 smoke 配置为 [`exp_phy_005_te_effective_index_smoke_v1.yaml`](../../configs/smoke/exp_phy_005_te_effective_index_smoke_v1.yaml)。

```bash
PYTHONPATH=src python -m phc2d.cli replay effective-index \
  --config configs/research/exp_phy_005_te_effective_index_gap_calibration_v1.yaml
```

标准 run 除通用制品外保存 `sample_manifest.jsonl`、`near_gamma_bands.npz`、`effective_index_samples.csv`、`slope_metrics.json`、`gap_alignment_metrics.json` 和分层图件。原始数值留在 `runs/`，本页不复制 raw evidence。

## Gate 与停止条件

- `preflight_pass` 已用于取得单独的 192 结构 WSL real-MPB 执行授权；授权不等于物理 gate 通过。
- 只有正式 run 的全部数值门、有效介质门与至少一个预注册 gap 候选门同时通过，才可形成 `prior_physical_gate_pass`。
- `effective_medium_fail`、`folding_hypothesis_fail`、`scalar_anisotropy_fail` 均停止 resonance-prior 训练路线；`blocked` 要求补齐环境、身份或 solver 制品。
- 人工负责人接受正式结论后，才允许登记 EXP-FWD-017；届时仅比较原 normalized full-matrix MSE 与一个冻结 resonance-prior loss，不混入 ranked gap 或 adaptive gate。

## 当前证据边界

192 结构正式 real-MPB 已完成；结果证明当前标量合同未通过，但不否定所有方向相关均匀化或能带折叠理论，也不产生 surrogate 训练结论。
