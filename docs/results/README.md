<!--
  Current-results index for the synchronized phc2d workspace.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 当前实验进展

本目录只同步当前项目报告直接依赖的证据和各主线最新进展。旧的逐日快照、被替代报告和 legacy 结果不进入远端；完整 `run.log`、`metrics.json`、checkpoint 与原始图件仍保存在本地 `runs/`。

## 项目级入口

- [完整项目报告](../reports/phc2d_project_full_introduction_20260724.md)
- [模型清单](../reports/phc2d_model_inventory_v1.yaml)
- [精选图表与哈希](../reports/assets/phc2d_project_full_introduction_20260724/figure_index.md)

## 最新进展

| 主线 | 当前文档 | 边界 |
|---|---|---|
| Field physics | [`field-physics/field_electric_metric_correction_20260803.md`](field-physics/field_electric_metric_correction_20260803.md) | 修正电场指标定义，不自动解除 inverse 准入阻塞 |
| Field P6 | [`field-physics/field_eigen_te_p6_seed0_target_residual_pilot_20260729.md`](field-physics/field_eigen_te_p6_seed0_target_residual_pilot_20260729.md) | 单 seed pilot，不替代正式多 seed 结论 |
| Inverse | [`inverse/a2_inverse_surrogate_postrepair_validation_v2_20260802.md`](inverse/a2_inverse_surrogate_postrepair_validation_v2_20260802.md) | surrogate 修复后验证；real-MPB 边界以报告原文为准 |
| Forward / Query | [`forward/a1_forward_query_fair_v1_20260726.md`](forward/a1_forward_query_fair_v1_20260726.md) | A1 公平协议正式结果 |
| Representation | [`representation_benchmark_20260803.md`](representation_benchmark_20260803.md) | 小样本诊断，不进入正式性能主结论 |

## 主报告保留的必要证据

- [`field-physics/field_eigen_te_b3_v2_pino_20260720.md`](field-physics/field_eigen_te_b3_v2_pino_20260720.md)
- [`field-physics/field_eigen_te_b4_dense_k_admission_20260722.md`](field-physics/field_eigen_te_b4_dense_k_admission_20260722.md)
- [`inverse/a2_inverse_objective_bridge_seed0_v1_20260727.md`](inverse/a2_inverse_objective_bridge_seed0_v1_20260727.md)
- [`inverse/a2_inverse_single_seed_formal_20260727.md`](inverse/a2_inverse_single_seed_formal_20260727.md)
- [`inverse/inverse_mpb_replay_mixture_v2_10000_test625_20260706.md`](inverse/inverse_mpb_replay_mixture_v2_10000_test625_20260706.md)
- [`inverse/inverse_retrain_v2_10000_20260706.md`](inverse/inverse_retrain_v2_10000_20260706.md)
- [`inverse/inverse_surrogate_rerank_v2_10000_20260630.md`](inverse/inverse_surrogate_rerank_v2_10000_20260630.md)

阅读时必须区分 local surrogate、field-physics residual 与 real-MPB replay；较低 residual 不能单独证明 inverse-design 收益。
