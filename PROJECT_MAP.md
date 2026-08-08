<!--
  Concise project map for the unified phc2d workspace.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

> 2026-08-07 工程状态：通用超算物理 campaign 控制面与 `full_cell_band` 参考适配器已实现；query/k-path、field-eigen、resolution-convergence 与 inverse replay 仅完成适配器登记，尚未完成多节点迁移。详见 [`docs/hpc_physics_campaigns.md`](docs/hpc_physics_campaigns.md)。这不是新的物理验证结论。

> 2026-08-08 架构实现状态：五个 Full-Wh 升级模型已进入 registry、模型 inventory 和 Full-Wh runner，并形成 [实现记录](docs/planning/full_cell_forward_inverse_training/full_cell_forward_model_upgrade_implementation_plan.md) 与 [10k/50k 训练验证方案](docs/planning/full_cell_forward_inverse_training/full_cell_forward_model_upgrade_10k_50k_training_validation_plan.md)。当前严格保持 `implemented / no_result`；未启动 N40 正式训练。

# PROJECT MAP

| Metadata | Value |
|---|---|
| Status | Current state snapshot |
| Audience | 导师、研究生与项目负责人 |
| Owner | 研究生维护，导师审阅科学主张 |
| Last reviewed | 2026-08-08 |
| Reading time | 8 minutes |

[项目封面](README.md) · [书籍式文档目录](docs/README.md) · [当前实验进展](docs/results/README.md) · [协作治理](docs/collaboration.md)

> 本文件是“当前状态”的唯一短入口。“已实现”不等于“已完成研究验证”；surrogate、field residual 与 real-MPB replay 证据不得混用。具体数值、配置和运行命令继续留在下层文档。

## 项目要解决什么问题

`phc2d` 面向二维光子晶体，研究如何用神经网络/神经算子可靠加速 Maxwell/MPB 求解，并从目标能带反向生成可由 real-MPB 验证的结构。

## 当前研究主线

主线是 **Replay-Calibrated Hierarchical Inverse Design**：

`结构 → real-MPB 数据 → Full-Wh / Query / Field 代理 → 多候选生成 → 合法性/多样性/交叉代理门 → real-MPB replay`

近期重点是全胞、多形态、全 BZ 数据底座。现有 V2 p4m 数据继续支撑基线，但不能替代全胞数据上的重新验证。

## 模块与当前进度

| 模块 | 负责什么 | 当前做到哪里 | 详细入口 |
|---|---|---|---|
| 数据与 MPB 物理底座 | 结构、k 路径、标签、provenance | V2 p4m 已物理复核；全胞 64+64 试点与分辨率门通过；二值与渐变两套 10k 均已通过 WSL 验收 | [数据目录](data/README.md) · [全胞试点](docs/results/data-provenance/full_cell_dual_lane_pilot_20260803.md) |
| 表征与任务协议 | `rho`/`epsilon`、Full-Wh、Query、Field | spec、target 与 NPZ/HDF5 schema inspection 已实现；小样本表征实验仅属诊断 | [研究计划](docs/planning/fno_generative_network_adjustment_plan_20260617.md) · [公式手册](docs/reports/phc2d_physics_data_formula_guide_20260803.md) |
| Forward / Query | 完整能带与任意 `(band,k)` 查询 | 10k 的 102-run 参考矩阵已完成；50k 的 17 波、204/204 runs、906-cell 冻结评估、300 组 paired bootstrap 和训练曲线均已完成，等待人工接受 Draft 报告；五个升级模型已实现并完成代码级验证，但仍无训练结果 | [升级实现记录](docs/planning/full_cell_forward_inverse_training/full_cell_forward_model_upgrade_implementation_plan.md) · [升级训练方案](docs/planning/full_cell_forward_inverse_training/full_cell_forward_model_upgrade_10k_50k_training_validation_plan.md) · [50k 结果](docs/results/forward/full_cell_dual_lane_50000_forward_training_20260808.md) · [10k 结果](docs/results/forward/full_cell_dual_lane_forward_training_20260806.md) |
| Field / adapted PINO | TE Bloch 本征场与物理残差 | U-Net、Field FNO、adapted PINO 已实现；介电加权指标需重算，residual–frequency 门未过，inverse 准入仍 `blocked` | [指标修正](docs/results/field-physics/field_electric_metric_correction_20260803.md) · [准入结果](docs/results/field-physics/field_eigen_te_b4_dense_k_admission_20260722.md) |
| 生成式逆向与 replay | 多候选、筛选、真实 MPB 复核 | objective-bridge 已完成 seed0 的 100-target 与 test625 replay；full-cell codec 与真实 10k 的 15-case Stage-A CPU 合同已通过，N40/正式训练/replay 尚未准入 | [逆向 seed0](docs/results/inverse/a2_inverse_objective_bridge_seed0_v1_20260727.md) · [Stage-A CPU 预检](docs/results/inverse/full_cell_inverse_stage_a_cpu_preflight_20260806.md) |
| 工程与证据治理 | CLI、配置、run 产物、审计 | 统一源码/配置/实验/运行结构已建立；N40 已证明可同时分配 3×RTX4090，单卡训练预检已通过；N50 登录权限仍受平台阻塞 | [GPU 准备报告](docs/results/data-provenance/paracloud_gpu_preflight_20260806.md) · [仓库约定](docs/conventions.md) |

## 当前最重要的结论

1. 核心成果是“**代理加速 + 多候选搜索 + real-MPB 最终裁判**”的闭环；只有 real-MPB replay 能支撑物理成功结论。
2. Forward / Query 已形成正式对照基础；inverse 闭环可运行，但当前最强证据仍是单 seed，不能宣称架构普遍优越。
3. residual 下降不等于频率或 inverse 改善；adapted PINO 在校准通过前不得进入候选排序。
4. 二值与渐变全胞 10k 均已通过固定 WSL real-MPB 跨环境验收；这只解除数据门，不等于模型训练已完成。
5. M9 已补齐 binary/graded 两份 10k 数据并与 N40 使用同一静态部署 commit `b93cc2d`；这只完成文件/代码交接，不等于新的 M9 environment preflight 或 inverse replay。
6. 50k 全胞 Forward 的 40k held-out test 中，FNO 在全部 12 个方向/epoch 组均为最低三 seed 均值；binary 100/200 epoch 均为约 `0.630%`，continuous 200 epoch 为约 `0.160%`。50k final-fit 和 threshold 结果只属于训练重叠 monitor。
7. CNN/FNO 能接受 32/128 输入不等于分辨率不变；相对 64 的 156 个 paired comparisons 中有 151 个 MRE CI 显著变差。该结果不是 MPB 分辨率收敛或物理验证。

## 当前最重要的未解决问题

- N50R5 登录仍被平台 `loginCluster` 权限拒绝；N40 已部署当前 inverse Stage-A runner/orchestrator（`b93cc2d`），但当前正式训练只能以既有三卡分配证据为下限，9/18 卡仍需真实 admission 作业证明。
- Full-cell 50k forward 计算和自动审计已完成，但 AI 整理的 Draft 报告仍需人工接受；inverse 仍未在全胞、多形态、全 BZ 数据上完成正式 45-run Stage-A 或 soft-surrogate Stage B。
- Full-Wh 升级模型目前只有实现规格与 10k/50k 验证方案；五个模型、两种输出头、WSL 回归测试、N40 admission 和 54-job 10k campaign 均未执行。
- inverse 缺 seeds 1/2 和满足 candidate-health/Top-5 coverage 的中立候选池。
- adapted PINO 尚未通过修正指标、residual–frequency、dense-k 与跨分辨率门。
- OOD、cross-dataset、cross-surrogate 统计和失败分析仍不完整。

## 下一步优先任务

1. **P0：人工审阅 50k 全胞 Forward 冻结结果**；确认 40k best-validation 与 50k final-epoch final-fit 的角色，接受或要求修复 [Draft 报告](docs/results/forward/full_cell_dual_lane_50000_forward_training_20260808.md)。
2. **P0：推进 full-cell inverse GPU admission**；另行授权 N40 Stage-A 正式训练；Stage B 使用已冻结 Forward surrogate 前仍需人工准入。
3. **P0：保持 M9-only 物理计算边界**；threshold MPB/Meep preflight、pilot 与 replay 均只能在 M9 执行且需要单独授权，本轮未提交。
4. **P1：重做 Field/PINO 校准**；任一 frequency、dense-k、跨分辨率或 residual–error 门失败即保持 blocked。
5. **P1：汇总 OOD / cross-dataset / cross-surrogate**；统一报告统计量、运行时间、失败案例和 surrogate–replay gap。
6. **P1：人工审阅 Full-Wh 升级规格**；先冻结五个模型与 dense/k-conditioned 接口，再另开 L4 代码任务；未通过 WSL tests 前不得授权 10k N40 campaign。

## 详细内容去哪里查看

- 研究范围、Gate、失败判据：[研究计划](docs/planning/fno_generative_network_adjustment_plan_20260617.md)
- 逐项任务：[Research TODO](docs/planning/research_todo_20260617.md)（早期勾选可能滞后）
- 模型实现与证据状态：[模型清单](docs/reports/phc2d_model_inventory_v1.yaml)
- Full-Wh 升级代码合同：[实现规格](docs/planning/full_cell_forward_inverse_training/full_cell_forward_model_upgrade_implementation_plan.md)
- Full-Wh 升级实验合同：[10k/50k 训练验证方案](docs/planning/full_cell_forward_inverse_training/full_cell_forward_model_upgrade_10k_50k_training_validation_plan.md)
- 方法、结果与边界：[完整项目报告](docs/reports/phc2d_project_full_introduction_20260724.md)
- 物理、数据与损失公式：[公式手册](docs/reports/phc2d_physics_data_formula_guide_20260803.md)
- 最新实验：[结果索引](docs/results/README.md)
- 渐变 10k 验收：[M9 验收合同](experiments/research/full_cell_graded_paracloud_m9.md)
- GPU 环境与三卡准备：[N40/N50/M9 全链路预检](docs/results/data-provenance/paracloud_gpu_preflight_20260806.md)
