<!--
  Full-Wh surrogate spectral and response-sensitivity experiment contract.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-PHY-004：Full-Wh 代理选频与真实响应差异

本实验检验一个限定问题：K5 Full-Wh held-out MRE 较低时，代理模型是否仍能保持最大带隙的 band、k 点和上边缘频率选择，以及残余选频误差是否会改变同一结构在冻结 real-Meep 合同下的近远场响应。

## 实验矩阵

| 维度 | 冻结值 |
|---|---|
| 数据 | EXP-DAT-005 K5-B / K5-G，`Wh[10000,81,10]` |
| split | 各路线固定 1000 条 test；不得重采样 |
| 模型 | `convnext_band_v1` native lane 与 `dual_lane_band_operator_v1` joint lane |
| 频谱统计 | 全 1000 条 test，bands 2–10 sample MRE、逐 band、逐 k、paired bootstrap |
| 物理选样 | 各路线按两模型平均 sample MRE 的 p10/p50/p90 选 3 条，共 6 条 |
| MPB 场 | 每条结构仅回放 stored real-MPB 最大带隙上边缘 TE 本征场 |
| Meep 响应 | 同一结构分别在 stored-MPB、ConvNeXt 与 joint 所选频率运行真实 Meep，共 18 条 |

配置入口为 [`exp_phy_004_full_wh_response_sensitivity_v1.yaml`](../../configs/research/exp_phy_004_full_wh_response_sensitivity_v1.yaml)。四个 surrogate 评估单元必须从 EXP-FWD-007 的 `evaluation_cells.csv` 按模型、lane、family、seed 与 target domain 唯一解析，不允许手写替代文件。

## 指标与展示顺序

频谱图先展示介质、stored real-MPB 能带、两种 surrogate 能带和叠图，再展示 full-BZ signed relative error。全量统计使用 1000 条 test；代表图只用于观察分位点案例，不能替代全量分布。

最大带隙在 lower bands 2–9 中选择；记录 band、k 点、上边缘频率和 gap。对同一结构的三次 real-Meep 响应计算：选频相对误差、公共尺度近场 relative-L2、近场 cosine、能量质心位移、单位面积远场 angular-L1、远场 cosine 与主瓣角偏移。

展示顺序必须是：

1. stored real-MPB 与 surrogate 的原始频谱；
2. 频谱差异；
3. truth-only real-MPB 本征场；
4. 三个频率下的真实 Meep 原始响应；
5. surrogate 选频响应减去 stored-MPB 选频响应的差异。

不存在 Full-Wh 场预测：所有近场、远场均来自真实求解器，surrogate 只提供频率。

## 执行 gate

- 频谱阶段必须验证两份 HDF5 SHA-256、四个 NPZ SHA-256、数组 shape、sample MRE 重算和 HDF5 test target 完全一致。
- 正式物理阶段只能在 WSL `Ubuntu` 的 conda `meep_env` 执行。
- 先运行 p50 预检：2 个 MPB 场和 6 个 Meep 响应；通过后才扩展 p10/p90。
- MPB 回放频率 MRE 必须不高于 1%。
- Meep 必须在最大时间前达到 DFT 衰减阈值；任一 case 失败即停止，不自动更改资源或参数。
- 最终必须有 6 个 MPB 分片、18 个 Meep 分片、12 条 response-difference 记录与 PNG/SVG/PDF/TIFF 图件。

## 命令

频谱阶段可在本地环境执行：

```bash
PYTHONPATH=src python -m phc2d.cli plot full-cell-surrogate-physics \
  --config configs/research/exp_phy_004_full_wh_response_sensitivity_v1.yaml \
  --run-dir runs/20260825/EXP-PHY-004_202608251101 \
  --stage spectral
```

p50 与全量物理阶段在 WSL `meep_env` 中依次运行：

```bash
PYTHONPATH=src python -m phc2d.cli plot full-cell-surrogate-physics \
  --config configs/research/exp_phy_004_full_wh_response_sensitivity_v1.yaml \
  --run-dir runs/20260825/EXP-PHY-004_202608251101 \
  --stage physics --physics-quantiles 0.5

PYTHONPATH=src python -m phc2d.cli plot full-cell-surrogate-physics \
  --config configs/research/exp_phy_004_full_wh_response_sensitivity_v1.yaml \
  --run-dir runs/20260825/EXP-PHY-004_202608251101 \
  --stage physics
```

## 证据边界

该实验把 `surrogate_held_out`、stored real-MPB label、real-MPB eigenfield replay 与 real-Meep response sensitivity 分开记录。它不训练场代理、不证明绝对散射功率、不升级 inverse 状态，也不把 6 条代表样本推广为完整 test split 的物理响应分布。
