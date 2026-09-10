---
title: "Representation Benchmark（2026-08-17）"
summary: "表征消融的日期化历史快照，不构成正式精度或物理验证结论。"
page_type: archive
status: historical
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-17
reading_time: 1 minutes
primary_layer: L3
evidence_date: 2026-08-17
evidence_scope: "日期化证据或历史记录；正文保持原始记录。"
---

<!--
  Archived representation benchmark snapshot.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# Representation Benchmark（2026-08-17）

> **历史证据：** 本页只保留局部表征链路的日期化记录；不属于正式结果白名单，也不构成模型精度或物理验证。

- experiment_id: representation_ablation_benchmark
- completed_at: 2026-08-17T20:22:54.834220
- dataset: data/paper_like_64_mpb.npz

## Case leaderboard

| case_id | input_repr | model | mse | mae | rmse | avg_MRE_bands_2_10_pct |
|---|---|---|---|---|---|---|
| rho_wedge_res_mlp_1 | rho_wedge | res_mlp | 0.379383 | 0.556390 | 0.615941 | 90.117218 |
| epsilon_grid_fno_lite_4 | epsilon_grid | fno_lite | 0.411000 | 0.549252 | 0.641093 | 89.952919 |
| rho_grid_lifted_cnn_2 | rho_grid | lifted_cnn | 0.438740 | 0.589977 | 0.662374 | 95.680885 |
| epsilon_grid_lifted_cnn_3 | epsilon_grid | lifted_cnn | 0.443393 | 0.594581 | 0.665877 | 96.573517 |

Best by MSE: **rho_wedge_res_mlp_1**

Per-band rows: 40
