<!--
  Evidence-oriented result index for the synchronized phc2d workspace.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="results-home"></a>

# 当前实验进展与证据索引

本目录回答“实际运行得到什么、证据边界是什么”。完整 `run.log`、`metrics.json`、checkpoint 和原始图件仍在 Git 之外的 `runs/`；报告存在不等于其结论仍是最新或已通过更高 gate。

## 项目级入口

- [项目地图](../../PROJECT_MAP.md)
- [完整项目报告](../reports/phc2d_project_full_introduction_20260724.md)
- [模型清单](../reports/phc2d_model_inventory_v1.yaml)
- [精选图表与哈希](../reports/assets/phc2d_project_full_introduction_20260724/figure_index.md)

<a id="data-provenance-results"></a>

## Data provenance

| 文档 | 证据角色 |
|---|---|
| [`data-provenance/gpu_environment_audit_20260618.md`](data-provenance/gpu_environment_audit_20260618.md) | GPU/环境审计，不是模型结果 |
| [`data-provenance/paracloud_gpu_preflight_20260806.md`](data-provenance/paracloud_gpu_preflight_20260806.md) | N40 三卡分配、单卡训练预检、数据传输与 N50 阻塞边界 |
| [`data-provenance/paracloud_m9_binary_import_and_code_sync_20260806.md`](data-provenance/paracloud_m9_binary_import_and_code_sync_20260806.md) | M9 binary 10k 补传、双数据 hash 和 N40/M9 同 commit 静态部署；未运行 MPB |
| [`data-provenance/k_path_physical_validation_v2_1000_20260621.md`](data-provenance/k_path_physical_validation_v2_1000_20260621.md) | physical-k 路径验证 |
| [`data-provenance/full_cell_dual_lane_pilot_20260803.md`](data-provenance/full_cell_dual_lane_pilot_20260803.md) | 全胞双路线 pilot 与收敛边界 |

## Representation diagnostics

- [`representation_benchmark_20260803.md`](representation_benchmark_20260803.md)：小样本表征诊断，不进入正式模型性能主结论。

## Forward results

- [`forward/full_cell_forward_model_upgrade_50k_evaluation_20260810.md`](forward/full_cell_forward_model_upgrade_50k_evaluation_20260810.md)：五个 Full-Wh 升级模型的 50K 三 seed held-out test、直接基线/FNO 对比、逐 band、cross-lane、32/128 与资源结果；主评估完成，threshold proxy 待补充。
- [`forward/full_cell_dual_lane_50000_forward_training_20260808.md`](forward/full_cell_dual_lane_50000_forward_training_20260808.md)：50k 双路线 17 波、204/204 runs、旧 10k→50k、40k held-out test、双向交叉与 CNN/FNO 32/64/128 零样本输入分辨率的完整 Draft 证据；50k final-fit 仅为训练重叠 monitor。
- [`forward/full_cell_dual_lane_forward_training_20260806.md`](forward/full_cell_dual_lane_forward_training_20260806.md)：全胞双路线完整实验报告；展示 4 models × 3 seeds × 7 表征、七组训练曲线、576-cell 双向跨数据评估、独立分辨率重训与冻结后 test；threshold real-MPB 未执行。
- [`forward/a1_forward_query_fair_v1_20260726.md`](forward/a1_forward_query_fair_v1_20260726.md)：A1 10k V2 Forward/Query 三 seed 公平协议结果。

<a id="query-results"></a>

## Query results

- [`query/query_data_scaling_v2_1000_full_20260621.md`](query/query_data_scaling_v2_1000_full_20260621.md)：早期 query 数据缩放证据；A1 当前主结论仍见 Forward/Query 公平报告。

<a id="field-physics-results"></a>

## Field physics results

| 文档 | 状态与边界 |
|---|---|
| [`field-physics/field_eigen_te_b1_64_20260720.md`](field-physics/field_eigen_te_b1_64_20260720.md) | 64 条 real-MPB field-data pilot |
| [`field-physics/field_eigen_te_b2_b1_1000_20260720.md`](field-physics/field_eigen_te_b2_b1_1000_20260720.md) | B2 / 1k 数据与 residual gate |
| [`field-physics/field_eigen_te_supervised_baseline_20260720.md`](field-physics/field_eigen_te_supervised_baseline_20260720.md) | 首轮 data-only 负结果 |
| [`field-physics/field_eigen_te_b3_v2_pino_20260720.md`](field-physics/field_eigen_te_b3_v2_pino_20260720.md) | 修正度规前的 B3/PINO 历史证据 |
| [`field-physics/field_eigen_te_b4_dense_k_admission_20260722.md`](field-physics/field_eigen_te_b4_dense_k_admission_20260722.md) | frequency、dense-k 与 inverse 准入 |
| [`field-physics/field_eigen_te_p6_seed0_target_residual_pilot_20260729.md`](field-physics/field_eigen_te_p6_seed0_target_residual_pilot_20260729.md) | seed0 target-residual pilot，准入未通过 |
| [`field-physics/field_electric_metric_correction_20260803.md`](field-physics/field_electric_metric_correction_20260803.md) | 当前介电加权指标契约；旧指标需重算 |

较低 Maxwell residual 不能单独证明 frequency、inverse reranking 或 field inverse 收益。

<a id="inverse-results"></a>

## Inverse results

| 文档 | 状态与边界 |
|---|---|
| [`inverse/inverse_retrain_v2_10000_20260706.md`](inverse/inverse_retrain_v2_10000_20260706.md) | 历史 inverse retrain 证据 |
| [`inverse/inverse_surrogate_rerank_v2_10000_20260630.md`](inverse/inverse_surrogate_rerank_v2_10000_20260630.md) | 历史 surrogate rerank 证据 |
| [`inverse/inverse_mpb_replay_mixture_v2_10000_test625_20260706.md`](inverse/inverse_mpb_replay_mixture_v2_10000_test625_20260706.md) | 625-target real-MPB replay，保留协议边界 |
| [`inverse/a2_inverse_single_seed_formal_20260727.md`](inverse/a2_inverse_single_seed_formal_20260727.md) | A2 单 seed 正式协议结果 |
| [`inverse/a2_inverse_objective_bridge_seed0_v1_20260727.md`](inverse/a2_inverse_objective_bridge_seed0_v1_20260727.md) | Objective-Bridge seed0 证据 |
| [`inverse/a2_inverse_surrogate_postrepair_validation_v2_20260802.md`](inverse/a2_inverse_surrogate_postrepair_validation_v2_20260802.md) | surrogate 修复后候选健康与 pilot 验证 |
| [`inverse/full_cell_inverse_codec_preflight_20260806.md`](inverse/full_cell_inverse_codec_preflight_20260806.md) | 五变体 analytic codec、hard/soft decoder 与真实 10k train-split 合同预检；尚非训练/replay 证据 |
| [`inverse/full_cell_inverse_stage_a_cpu_preflight_20260806.md`](inverse/full_cell_inverse_stage_a_cpu_preflight_20260806.md) | 真实双路线 10k 上的 15-case Stage-A CPU 合同预检；N40 正式单卡执行尚待授权 |
| [`inverse/full_cell_inverse_stage_a_runner_smoke_20260806.md`](inverse/full_cell_inverse_stage_a_runner_smoke_20260806.md) | 可断点续训 Stage-A epoch runner 的 Windows/WSL 小数据测试；正向 surrogate 与 N40 正式训练均未开始 |
| [`inverse/full_cell_inverse_stage_a_wave_dryrun_20260806.md`](inverse/full_cell_inverse_stage_a_wave_dryrun_20260806.md) | 18-task/9-concurrency N40 波次零提交演练；验证身份、节流、manifest/status，但不证明实际 GPU 容量 |

阅读 inverse 文档时必须保留 seed、目标数、候选池、Top-K、surrogate 与 replay 层级；单 seed pilot 不能升级为跨 seed 或普遍 superiority。

## 使用规则

1. 先读报告顶部的数据、环境、状态和证据边界。
2. 再核对关联 run、配置、hash、seed 和失败案例。
3. 历史报告用于追溯，不覆盖更新后的 metric/data contract。
4. 新结果应更新本索引和 [`../../PROJECT_MAP.md`](../../PROJECT_MAP.md)，而不是只写在聊天中。
