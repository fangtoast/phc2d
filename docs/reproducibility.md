<!--
  Evidence-graded reproducibility contract for phc2d experiments.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="reproducibility-home"></a>

# 分级复现指南

| Metadata | Value |
|---|---|
| Status | Canonical reproducibility contract |
| Audience | 需要复现接口、代理实验或物理验证的研究者 |
| Owner | 研究生维护，导师审阅证据边界 |
| Last reviewed | 2026-08-06 |
| Reading time | 12 minutes |

本文回答“复现某项工作需要哪些输入、执行什么入口、得到什么产物、最多能声称什么”。安装步骤见 [`getting-started.md`](getting-started.md)，当前项目状态见 [`../PROJECT_MAP.md`](../PROJECT_MAP.md)。

## 1. 四级复现契约

| 等级 | 最小输入 | 目标 | 允许表述 |
|---|---|---|---|
| R0 | 干净 Git 克隆、Python 3.10+ | 安装、help、版本和 registry | 接口检查通过 |
| R1 | `train` extra、合成配置 | 数据到训练产物的最小闭环 | 非物理 smoke 通过 |
| R2 | 指定数据、manifest、split、配置和可选 checkpoint | 重跑代理训练、评估或统计 | 对应 surrogate 协议结果 |
| R3 | WSL `Ubuntu`、`meep_env`、真实 MPB 输入和 gate | 数据生成、replay、跨环境物理复核 | 对应 real-MPB 物理证据 |

R0/R1 不能升级为 R2，surrogate 或 field residual 也不能替代 R3。

## 2. 公共仓库包含什么

公共 Git 仓库包含源码、配置、实验 profiles、数据 manifest、精选报告和图件，但不包含完整 `.npz`/`.h5` 数据、checkpoint 或原始 runs。正式复现开始前必须从证据拥有者获取数据/权重，或按对应 profile 重新生成，并核对：

1. 数据文件 SHA-256 与 manifest；
2. train/val/test split 及 sample lineage；
3. 配置、模型 identity、seed、precision 与设备；
4. 源 run 的 `run.meta.json`、`config.resolved.yaml` 和结果文件；
5. MPB/Meep 任务的 solver、geometry、k 点和环境 provenance。

不依据文件名或“已有报告”推断本地数据已经可用。

## 3. R0 / R1

R0 和 R1 的逐条命令见 [快速上手](getting-started.md)。R1 生成的 manifest 固定包含：

```text
evidence_status: synthetic_nonphysical_smoke
physical_verification_status: not_applicable
scientific_claims_allowed: false
```

任何报告或任务总结都必须保留这三个边界。

## 4. R2：当前代理实验

| 研究问题 | 数据与配置 | 执行入口 | 结果入口 | 证据边界 |
|---|---|---|---|---|
| A1 Full-Wh 公平对照 | 10k V2 数据；`a1_forward_fair_v1.yaml` | `phc2d forward benchmark` | [A1 报告](results/forward/a1_forward_query_fair_v1_20260726.md) | 三 seed neural surrogate |
| A1 Continuous Query | 同一结构 split 与 physical-k metadata；`a1_query_fair_v1.yaml` | `phc2d query benchmark` | [A1 报告](results/forward/a1_forward_query_fair_v1_20260726.md) | Query surrogate，不是 replay |
| A2 inverse surrogate 流程 | 10k V2、冻结 checkpoint、候选谱系；A2 配置 | `phc2d inverse benchmark` 的显式 stage | [Inverse 结果索引](results/README.md#inverse-results) | 按 seed、候选池和 replay 状态分别声明 |
| Field / adapted PINO | field HDF5、field contract 与相应配置 | `phc2d forward train` 或 profile 指定入口 | [Field 结果索引](results/README.md#field-physics-results) | 修正后指标未重训前保留历史边界 |

A1 的冻结命令由 [`a1_fair_forward_query.md`](../experiments/research/a1_fair_forward_query.md) 维护。A2 必须先运行 preflight，并在后续 stage 中复用同一 run 目录；Field/PINO 必须遵守 [`field_eigen_te.md`](../experiments/research/field_eigen_te.md) 的数据与准入门。

R2 验收至少检查标准 run 产物、数据/切分 hash、checkpoint identity、实际 seed 数、运行时间和失败案例。单 seed 结果不得改写为跨 seed 稳定性。

<a id="repro-r3"></a>

## 5. R3：real-MPB 数据与 replay

R3 只在 WSL `Ubuntu` + conda `meep_env` 或由协议明确允许的原生集群环境中执行。原生集群生成仍需完成固定 WSL 跨环境 replay 才能获得项目正式验收。

| 物理任务 | 权威 profile / 配置 | 必须检查 |
|---|---|---|
| 全胞双路线 MPB 数据 | [实验 profile](../experiments/research/full_cell_dual_lane_data.md) | solver preflight、完整 shard、HDF5/manifest、分辨率收敛 |
| 集群渐变 10k 验收 | [M9 profile](../experiments/research/full_cell_graded_paracloud_m9.md) | 固定结构、固定 k 点、hash、WSL replay 与误差门 |
| inverse real-MPB replay | [A2 profile](../experiments/research/a2_inverse_single_seed.md) | candidate manifest、去重、缓存谱系、逐例 solve 与 aggregate |
| Field 物理准入 | [Field profile](../experiments/research/field_eigen_te.md) | 修正度规、frequency、dense-k、跨分辨率与 residual-error 校准 |

R3 完成也只支持对应任务和对应数据范围；不能自动推广为模型普遍优越、跨 seed 稳定或 field inverse 成功。

## 6. Wang 2026 / POD-DeepONet 状态

[`experiments/reproduction/wang2026`](../experiments/reproduction/wang2026/README.md) 当前是复现 profile 的目录接口，不是 faithful reproduction 完成证明。模型清单中的 `pod_res_mlp` 仍为 `placeholder`，当前 A1 Forward/Query 直接预测 `Wh` 或 query band vector，并未使用已实现的 POD basis 管线。

因此当前允许的说法是“仓库保留 Wang 2026 对齐和后续研究接口”；在 POD basis、训练、评估和同协议结果全部实现并通过前，不得写成“已完整复现 POD-DeepONet”。

## 7. 复现完成清单

- [ ] 明确复现等级 R0/R1/R2/R3。
- [ ] 记录 commit、环境、配置和命令。
- [ ] 核对数据、split、checkpoint 和 solver lineage。
- [ ] 只报告实际生成的产物和指标。
- [ ] 区分 surrogate、field residual 与 real-MPB replay。
- [ ] 写出能得出的结论、不能得出的结论和剩余 gate。
- [ ] 更新对应结果索引，而不是只在聊天中留下结论。

任务如何限制在一个知识层级内，见 [协作治理](collaboration.md)。
