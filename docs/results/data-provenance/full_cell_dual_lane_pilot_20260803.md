---
title: "全胞双路线数据构建与验收"
summary: "记录 binary/graded 各 64 样本 real-MPB 试点、解析几何 provenance 与 64/96/128 分辨率收敛门。"
page_type: evidence
status: draft
audience: "研究用户、数据生产者、复现执行者与证据审阅者"
owner: "研究生维护"
last_reviewed: 2026-08-14
reading_time: 10 minutes
primary_layer: L3
evidence_date: 2026-08-03
evidence_scope: "全胞 binary/graded 各 64 样本数据接受；不等同于后续 10k/50k 训练数据或模型结果。"
dataset_scope: full_cell_bz9_wh81x10
evidence_level: data_acceptance
evidence_status: completed
validation_backend: wsl_mpb
source_artifacts:
  - runs/20260803/full_cell_solver_preflight_v1_202608031352
  - runs/20260803/full_cell_graded_final64_202608031822
  - runs/20260803/full_cell_binary_final64_202608031822
  - runs/20260803/full_cell_resolution_convergence_v1_202608032339
  - data/full_cell_dual_lane_pilot64_manifest.yaml
---

<!--
  Full-cell dual-lane dataset implementation and acceptance report.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="full-cell-dual-lane-data-acceptance"></a>

# 全胞双路线数据构建与验收

## 结论

Binary 与 graded 两条全胞路线的最终 64 样本 real-MPB 试点均完成 `64/64`、失败为 0；随后进行的 64/96/128 分辨率收敛检查状态为 `resolution_convergence_pass`。联合清单 `data/full_cell_dual_lane_pilot64_manifest.yaml` 状态为 `pilot_pass`，其 SHA-256 为 `27c0ed5d7c287c315d33a81a23733a0bd27eb92a529f56eab4973ea97a3e5635`。

这项验收证明“解析周期结构 → real-MPB → 可追溯 HDF5 → 分辨率 gate”的数据链路可用。它不把早期 4-sample preflight、被取代的生成规则或后续 10k/50k 模型训练混入本页结论。

## 数据身份

| 字段 | 冻结值 |
|---|---|
| 结构范围 | 无 p4m 约束的 full-cell binary / graded |
| 解析几何 | `FullCellStructureSpec` + periodic material callback |
| 晶格 / 极化 | square，TE，`a=1` |
| 材料 | $\varepsilon_{air}=1.0$，$\varepsilon_{material}=8.9$ |
| k 网格 | centered unique $9\times9$ full BZ，`k_points[81,2]` |
| bands | MPB 求解 14（含 4 guard bands），按频率升序保留 10 |
| target | `Wh[64,81,10]`（每条路线） |
| 学习视图 | `material_weight_grid_32/64/128` |
| solver | `meep.mpb.ModeSolver`，resolution 64，mesh 3，tolerance `1e-7` |
| 正式环境 | WSL Ubuntu + conda `meep_env` |

两条路线只是共享 schema 和物理求解合同，并非 64 组逐样本配对结构。联合清单设置 `merge_arrays: false`，保留各自材料语义与 hash。

## 结构分布

| 路线 | 结构族 | 64 样本组成 | 材料语义 |
|---|---|---:|---|
| graded | `graded_rbf_fourier` | 32 个 single-core + 32 个 double-core | 连续 $u(\mathbf r)\in[0,1]$ |
| binary | `binary_polar_holes`、`binary_spectral_threshold` | 32 + 32 | 二值 $u(\mathbf r)\in\{0,1\}$ |

介电常数由

$$
\varepsilon(\mathbf r)=1+7.9u(\mathbf r)
$$

得到。MPB 在查询位置直接调用解析材料函数；32/64/128 栅格只是从同一 spec 重采样的学习和审计视图，不是 MPB 的离散几何输入。

## 求解器 oracle

在大规模试点前，`runs/20260803/full_cell_solver_preflight_v1_202608031352/` 对均匀介质、重复圆孔和 material-function/native geometry 进行核对。

| 指标 | 结果 | 门限 | 状态 |
|---|---:|---:|---|
| uniform air median absolute difference | `0` | `≤1e-6` | 通过 |
| uniform material median absolute difference | `0` | `≤1e-6` | 通过 |
| circle repeat max absolute difference | `1.19e-7` | `≤1e-6` | 通过 |
| circle material-function/native MRE，bands 2+ | `0.02083%` | `≤0.5%` | 通过 |

均匀介质的少数精确简并带可能以不同顺序返回，所以最大逐带差只作为诊断；稳健谱匹配、非均匀圆孔对照和重复性共同构成 oracle。

## 两条最终 64 样本结果

| 路线 | run | 完成/失败 | wall time | HDF5 SHA-256 |
|---|---|---:|---:|---|
| graded | `full_cell_graded_final64_202608031822` | `64/0` | `17135.75 s` | `1f711d084e327a0407d3e8d072ce6e20dbad1d61a440af1326a1bbe58c3ad6d3` |
| binary | `full_cell_binary_final64_202608031822` | `64/0` | `17613.47 s` | `f1d607ae69ccfc88e7ffcc533efe5aed72bf07803ed0784365c5e588cc531a6d` |

两条 run 的 `solver_provenance.json` 均声明：

- `execution_boundary = WSL Ubuntu + conda meep_env`；
- `acceptance_eligible = true`；
- `geometry_convention = analytic_periodic_full_cell_material_function`；
- `use_proxy = false`；
- `completed_count = 64`、`failed_count = 0`。

## 64/96/128 分辨率收敛门

收敛 run 为 `runs/20260803/full_cell_resolution_convergence_v1_202608032339/`，固定门限为 median MRE `64→96 ≤0.5%`、`96→128 ≤0.25%`，p95 MRE `≤1%`。

| 路线 | median 64→96 | median 96→128 | p95 64→96 | p95 96→128 | 状态 |
|---|---:|---:|---:|---:|---|
| binary | `0.00971%` | `0.01135%` | `0.04722%` | `0.02975%` | 通过 |
| graded | `0.00189%` | `0.00067%` | `0.00482%` | `0.00176%` | 通过 |
| combined | `0.00333%` | `0.00174%` | `0.03353%` | `0.02357%` | 通过 |

gate 使用 bands 2–10，并保留逐 transition 统计。此结果支持当前试点采用 resolution 64；它不等于任意新结构族或任意 solver 参数都已收敛。

## 执行入口与产物

真实数据生成与收敛命令为：

```bash
PYTHONPATH=src python -m phc2d.cli data generate \
  --config configs/research/full_cell_graded_bz9_pilot64.yaml
PYTHONPATH=src python -m phc2d.cli data generate \
  --config configs/research/full_cell_binary_holes_bz9_pilot64.yaml
PYTHONPATH=src python -m phc2d.cli data converge \
  --config configs/research/full_cell_resolution_convergence_v1.yaml
```

每条 lane 保存 HDF5、dataset manifest、sample plan、worker shards、failed-case ledger、solver provenance、resolved config、metrics 和 results；收敛 run 保存 selection、逐 lane metrics 与 finalization。联合 manifest 只链接已通过 gate 的最终文件。

## 被排除的尝试

`full_cell_graded_bz9_pilot64_202608031427`、`full_cell_binary_holes_bz9_pilot64_202608031427` 以及随后各完成 6 个 shard 的中间规则目录都写有 `superseded.json`。这些数据不能与最终 64 样本混合，也不进入联合 manifest。早期 4-sample run 只承担 preflight，不替代 64+64 与 convergence gate。

## 能证明与不能证明什么

本页能够证明：两条全胞结构路线可以在声明的 WSL/MPB 协议下生成无失败、可哈希、可恢复且通过分辨率门的 64 样本试点。

本页不能证明：后续 10k/50k 数据天然无误、正向模型达到某个精度、full-cell inverse 已训练，或有限阵列散射性能已验证。对应结论分别见[10k 正向](../forward/full_cell_dual_lane_forward_training_20260806.md)、[50k 正向](../forward/full_cell_dual_lane_50000_forward_training_20260808.md)、[全胞逆向计划](../../planning/全胞逆向索引.md)与[Meep 近远场](../field-physics/full_cell_dual_lane_meep_scattering_20260806.md)。
