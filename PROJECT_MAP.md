<!--
  Concise project map for the unified phc2d workspace.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

> 2026-08-07 工程状态：通用超算物理 campaign 控制面与 `full_cell_band` 参考适配器已实现；query/k-path、field-eigen、resolution-convergence 与 inverse replay 仅完成适配器登记，尚未完成多节点迁移。详见 [`docs/hpc_physics_campaigns.md`](docs/hpc_physics_campaigns.md)。这不是新的物理验证结论。

> 2026-08-10 升级评估状态：五个 Full-Wh 升级模型已完成 50k 正式主评估。ConvNeXt 是 B/G native held-out test 共同最优，Mixer 与 TFNO 也通过直接基线 promotion；Gated ResMLP 退化，dual-lane 未达到联合替代门。threshold proxy 因 `AssocMaxSubmitJobLimit` 未提交成功；没有 real-MPB inverse replay。详见[升级结果](docs/results/forward/full_cell_forward_model_upgrade_50k_evaluation_20260810.md)与[全胞模型报告](docs/reports/全胞双路线正向与逆向模型报告_20260810.md)。

> 2026-08-13 维护状态：CLI、模型与 inverse runtime 模块化已分别提交为 `082ac2d`、`5c01bf9`，架构边界记录于 `75bab09`。WSL `Ubuntu` + conda `meep_env` 下 compileall、基础 CLI、`models list`、40 项定向测试与完整 `482 passed` 回归均通过；这只证明 L4 工程兼容性，不新增模型性能或物理验证结论。同期生成的多份 `paper_like_64` representation benchmark 仅为重复小样本 smoke，不构成新的正式模型或物理结论。

> 2026-08-25 EXP-PHY-004 状态：K5-B/G 各 1000 条 frozen test 的 Full-Wh 频谱审计与 6 条代表结构的 real-MPB/18 条 real-Meep response sensitivity 已完成。整体 MRE 较低不保证最大带隙 band/k 选择稳定；该结论不等于 field surrogate 已建立，也不升级 inverse 状态。详见[代理选频与真实响应差异](docs/results/field-physics/代理选频与真实响应差异_20260825.md)。

> 2026-08-25 EXP-FWD-009 状态：五个 epoch-200 权重热启动身份均通过主收敛门，最终 17 个 K5 身份完成六域 510-cell、408/272 数值回归和 102-cell paired bootstrap。最终 inventory 仅 9/17 满足主收敛门，另 8 个依据人工豁免纳入；不得写成原 EXP-FWD-008 的 17/17 收敛通过。详见[EXP-FWD-009 结果](docs/results/forward/EXP-FWD-009-K5权重热启动与六域验证.md)。

> 2026-08-26 EXP-FWD-010/011 状态：joint 改为 K5 Binary/Graded 各 4,000 条后完成 seed0 200 轮从零训练、200→300 权重续训和两组六域评估。300 轮时 Binary 收敛通过，Graded 与等权未通过；相对 200 轮为五域显著改善、K20 Graded 退化，只在 K5 Binary 超过最佳单路线。该结果仍是 surrogate seed0 负对照，不支持 joint 全域 promotion。详见[EXP-FWD-009/010/011 汇总](docs/results/forward/EXP-FWD-009-K5权重热启动与六域验证.md)。

> 2026-08-27 EXP-FWD-012 状态：公平 4k+4k K5 train/validation 的工作频率先验审计完成。$X/M$ 一阶 Bragg 尺度对两条 lane 的最大 gap-center 命中均为 0，未通过相对 train-only 固定中心改善门；EXP-FWD-013 的四个 loss-only 训练任务未启动。后续 L4 已实现不依赖 Bragg 中心的可选 `equal_lane_gap_aware_v1`，并通过真实 K5 CPU admission smoke；该状态仅为 `implemented + smoke_pass`，尚未登记或执行正式性能实验。详见[工作频率先验负结果](docs/results/forward/EXP-FWD-012-工作频率先验覆盖审计.md)。

> 2026-08-28 EXP-FWD-014 状态：两个 N40 seed0、20 epoch 配对任务正常完成。新损失把 Binary/Graded joint band-k accuracy 分别提高 `10.9/15.1` 个百分点，并改善多数 gap 指标；但 Bands 2–10 MRE 分别恶化 `13.44%/12.65%`，且 complete-gap 子集状态命中仍为 `0`，因此 `gap_screen_fail`，停止自动三 seed 扩展。详见[EXP-FWD-014 结果](docs/results/forward/EXP-FWD-014-gap决策损失短预算筛选.md)。

> 2026-08-28 EXP-FWD-015 状态：对 EXP-FWD-014 两个冻结检查点完成 K5 validation-only 近简并审计。train-only 容差最多只提高 joint accuracy `3.0` 个百分点，排除近简并后 v1 仍只有 Binary/Graded `27.7%/24.8%`；结论为模型决策受限，而非指标近简并主导。`equal_lane_gap_ranked_v2` 已实现并通过 WSL CPU admission smoke，但 EXP-FWD-016 的一个 N40 短筛任务尚未授权或提交。详见[EXP-FWD-015 结果](docs/results/forward/EXP-FWD-015-gap决策近简并审计.md)。

# PROJECT MAP

| Metadata | Value |
|---|---|
| Status | Current state snapshot |
| Audience | 导师、研究生与项目负责人 |
| Owner | 研究生维护，导师审阅科学主张 |
| Last reviewed | 2026-08-27 |
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
| 数据与 MPB 物理底座 | 结构、k 路径、标签、provenance | V2 p4m 已物理复核；全胞 B/G 100 结构五档 MPB resolution 技术门通过并推荐 r64，待负责人冻结；二值与渐变两套 10k 均已通过 WSL 验收 | [数据目录](data/README.md) · [全胞试点](docs/results/data-provenance/full_cell_dual_lane_pilot_20260803.md) · [MPB 分辨率验证](docs/results/data-provenance/全胞MPB分辨率收敛验证_20260817.md) |
| 表征与任务协议 | `rho`/`epsilon`、Full-Wh、Query、Field | spec、target 与 NPZ/HDF5 schema inspection 已实现；小样本表征实验仅属诊断 | [研究计划](docs/planning/fno_generative_network_adjustment_plan_20260617.md) · [公式手册](docs/reports/phc2d_physics_data_formula_guide_20260803.md) |
| Forward / Query | 完整能带与任意 `(band,k)` 查询 | 10k 参考矩阵和 50k baseline 冻结评估均完成；五个升级模型的 50k native/cross-lane/resolution 主评估完成，ConvNeXt native 最优，threshold proxy 仍受 QoS 阻塞 | [全胞模型报告](docs/reports/全胞双路线正向与逆向模型报告_20260810.md) · [升级结果](docs/results/forward/full_cell_forward_model_upgrade_50k_evaluation_20260810.md) · [50k baseline](docs/results/forward/full_cell_dual_lane_50000_forward_training_20260808.md) · [10k 结果](docs/results/forward/full_cell_dual_lane_forward_training_20260806.md) |
| Field / adapted PINO | TE Bloch 本征场与物理残差 | U-Net、Field FNO、adapted PINO 已实现；介电加权指标需重算，residual–frequency 门未过，inverse 准入仍 `blocked` | [指标修正](docs/results/field-physics/field_electric_metric_correction_20260803.md) · [准入结果](docs/results/field-physics/field_eigen_te_b4_dense_k_admission_20260722.md) |
| 生成式逆向与 replay | 多候选、筛选、真实 MPB 复核 | objective-bridge 已完成 seed0 的 100-target 与 test625 replay；full-cell codec 与真实 10k 的 15-case Stage-A CPU 合同已通过，N40/正式训练/replay 尚未准入 | [逆向 seed0](docs/results/inverse/a2_inverse_objective_bridge_seed0_v1_20260727.md) · [Stage-A CPU 预检](docs/results/inverse/full_cell_inverse_stage_a_cpu_preflight_20260806.md) |
| 工程与证据治理 | CLI、配置、run 产物、审计 | 统一源码/配置/实验/运行结构已建立；CLI、模型与 inverse runtime 模块化已通过 WSL 完整工程回归；N40 已证明可同时分配 3×RTX4090，N50 登录仍受平台阻塞 | [架构说明](docs/architecture.md) · [协作治理](docs/collaboration.md) · [GPU 准备报告](docs/results/data-provenance/paracloud_gpu_preflight_20260806.md) · [仓库约定](docs/conventions.md) |

## 当前最重要的结论

1. 核心成果是“**代理加速 + 多候选搜索 + real-MPB 最终裁判**”的闭环；只有 real-MPB replay 能支撑物理成功结论。
2. Forward / Query 已形成正式对照基础；inverse 闭环可运行，但当前最强证据仍是单 seed，不能宣称架构普遍优越。
3. residual 下降不等于频率或 inverse 改善；adapted PINO 在校准通过前不得进入候选排序。
4. 二值与渐变全胞 10k 均已通过固定 WSL real-MPB 跨环境验收；这只解除数据门，不等于模型训练已完成。
5. M9 已补齐 binary/graded 两份 10k 数据并与 N40 使用同一静态部署 commit `b93cc2d`；这只完成文件/代码交接，不等于新的 M9 environment preflight 或 inverse replay。
6. 50k 全胞 Forward 的 40k held-out test 中，FNO 在全部 12 个方向/epoch 组均为最低三 seed 均值；binary 100/200 epoch 均为约 `0.630%`，continuous 200 epoch 为约 `0.160%`。50k final-fit 和 threshold 结果只属于训练重叠 monitor。
7. CNN/FNO 能接受 32/128 输入不等于分辨率不变；相对 64 的 156 个 paired comparisons 中有 151 个 MRE CI 显著变差。该结果不是 MPB 分辨率收敛或物理验证。
8. Full-Wh 升级 50k held-out test 中，ConvNeXt 的 B/G MRE 为约 `0.321%/0.050%`；Mixer 的 32/128 直接输入诊断总体最稳。Gated ResMLP 负结果和 dual-lane 未通过替代门必须保留。
9. K5 Full-Wh 的整体 test MRE 不能替代最大带隙选择审计；EXP-PHY-004 的 6 条代表结构中，ConvNeXt 仅 1/6、joint 仅 4/6 同时保持 real-MPB band 与 k 选择，选频差异可显著改变冻结 real-Meep 响应。

## 当前最重要的未解决问题

- EXP-FWD-009 已完成五个身份的权重热启动续训与六域 510-cell held-out surrogate 评估；54 身份 preflight、10 shards、408/272 数值回归和 102-cell paired bootstrap 均闭合。最终 inventory 仍只有 9/17 满足主收敛门，8 个身份经人工豁免纳入，因此当前需要人工决定是否接受其为探索性 K5 对照；这不构成 real-MPB 或物理验证。详见[EXP-FWD-009 结果](docs/results/forward/EXP-FWD-009-K5权重热启动与六域验证.md)。
- EXP-FWD-010/011 已完成 joint 各 4k 的 seed0 200 轮训练、续训至 300 轮与两组六域增量评估；300 轮时 Graded 与等权收敛曲线未过门，六域只在 K5 Binary 击败最佳单路线。当前需人工决定是否接受其为负对照；若要 promotion 或数据量因果结论，仍缺多 seed、同训练与优化器状态合同的成对实验。
- EXP-FWD-012 已否定当前 $X/M$ 一阶 Bragg 工作频率先验对最大 gap-center 的准入价值；不得扩大窗口或读取 test 追调。该负结果不否定 EXP-FWD-014 直接监督真实 gap 决策的独立假设。
- EXP-FWD-014 已完成两个 N40 seed0 短预算任务并触发负 gate：k-profile 对应 k 点方向有效，但当前辅助权重和 complete-gap 状态约束不合格；不得自动调参重跑或进入三 seed/test。
- EXP-FWD-015 已确认低命中主要来自模型候选排序与 complete-gap 召回，而非近简并评价口径；EXP-FWD-016 只登记了一个 ranked-v2 N40 短筛任务，等待单独远端训练授权。
- N50R5 登录仍被平台 `loginCluster` 权限拒绝；N40 已部署当前 inverse Stage-A runner/orchestrator（`b93cc2d`），但当前正式训练只能以既有三卡分配证据为下限，9/18 卡仍需真实 admission 作业证明。
- Full-cell 50k forward 计算和自动审计已完成，但 AI 整理的 Draft 报告仍需人工接受；inverse 仍未在全胞、多形态、全 BZ 数据上完成正式 45-run Stage-A 或 soft-surrogate Stage B。
- Full-Wh 升级 native/cross-lane/resolution 主评估已完成，但 `k_conditioned` 头未被选择，threshold proxy 尚无作业或结果；这些 surrogate 结果尚未进入 full-cell inverse Stage B 或 real-MPB replay。
- inverse 缺 seeds 1/2 和满足 candidate-health/Top-5 coverage 的中立候选池。
- adapted PINO 尚未通过修正指标、residual–frequency、dense-k 与跨分辨率门。
- OOD、cross-dataset、cross-surrogate 统计和失败分析仍不完整。

## 下一步优先任务

1. **P0：人工审阅 50k 全胞 Forward 与升级冻结结果**；确认 held-out、final-fit、native、cross-lane 和 resolution 角色，接受或要求修复 [baseline 报告](docs/results/forward/full_cell_dual_lane_50000_forward_training_20260808.md)、[升级报告](docs/results/forward/full_cell_forward_model_upgrade_50k_evaluation_20260810.md)与[模型综合报告](docs/reports/全胞双路线正向与逆向模型报告_20260810.md)。
2. **P0：推进 full-cell inverse GPU admission**；另行授权 N40 Stage-A 正式训练；Stage B 使用已冻结 Forward surrogate 前仍需人工准入。
3. **P0：保持 M9-only 物理计算边界**；threshold MPB/Meep preflight、pilot 与 replay 均只能在 M9 执行且需要单独授权，本轮未提交。
4. **P1：重做 Field/PINO 校准**；任一 frequency、dense-k、跨分辨率或 residual–error 门失败即保持 blocked。
5. **P1：汇总 OOD / cross-dataset / cross-surrogate**；统一报告统计量、运行时间、失败案例和 surrogate–replay gap。
6. **P1：冻结 full-cell inverse 的 forward ranker 与数据版本**；决定继续 10k 还是迁移 50k，更新 Stage-B admission、normalization 和 checkpoint manifests 后再单独授权训练/replay。
7. **P1：把 gap-selection 纳入 forward→inverse gate**；先对完整 test split 做低成本 band/k/gap 一致率审计，再决定是否按失配类型扩大 real-MPB/Meep replay。

## 详细内容去哪里查看

- 研究范围、Gate、失败判据：[研究计划](docs/planning/fno_generative_network_adjustment_plan_20260617.md)
- 逐项任务：[Research TODO](docs/planning/research_todo_20260617.md)（早期勾选可能滞后）
- 模型实现与证据状态：[模型清单](docs/reports/phc2d_model_inventory_v1.yaml)
- Full-Wh 升级代码合同：[实现规格](docs/planning/full_cell_forward_inverse_training/full_cell_forward_model_upgrade_implementation_plan.md)
- Full-Wh 升级实验合同：[10k/50k 训练验证方案](docs/planning/full_cell_forward_inverse_training/full_cell_forward_model_upgrade_10k_50k_training_validation_plan.md)
- Full-Wh baseline、升级与 inverse 架构：[全胞双路线模型报告](docs/reports/全胞双路线正向与逆向模型报告_20260810.md)
- 方法、结果与边界：[完整项目报告](docs/reports/phc2d_project_full_introduction_20260724.md)
- 物理、数据与损失公式：[公式手册](docs/reports/phc2d_physics_data_formula_guide_20260803.md)
- 最新实验：[结果索引](docs/results/README.md)
- 渐变 10k 验收：[M9 验收合同](experiments/research/full_cell_graded_paracloud_m9.md)
- GPU 环境与三卡准备：[N40/N50/M9 全链路预检](docs/results/data-provenance/paracloud_gpu_preflight_20260806.md)
