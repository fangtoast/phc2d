# Representation Benchmark

- experiment_id: representation_ablation_benchmark
- completed_at: 2026-08-03T11:32:06.480610
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
