<!--
  Representation ablation research profile.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# Representation Ablation

目标：把 forward surrogate 从固定 `rho -> Wh` 扩展为多输入表征实验，比较 `rho_wedge`、`rho_grid`、`epsilon_grid` 与 `image_epsilon` 对 band matrix surrogate 的影响。

配置入口：[`../../configs/research/representation_ablation.yaml`](../../configs/research/representation_ablation.yaml)

## Gate

1. `phc2d data inspect --path data/tiny.npz` 能识别 `rho`、`Wh`、split 与 shape。
2. 同一 raw sample id 派生多种 input representation。
3. MLP/CNN/FNO 类模型声明自己支持的 representation，不强行共享一种输入。
4. 输出 `leaderboard.csv`、`metrics.json`、`per_band_metrics.csv` 与 figures。

## Decision

如果 FNO 在 `epsilon_grid -> Wh` 上没有优势，优先转向 `epsilon_grid + k/query -> band_query` 或 near-field/operator surrogate，而不是继续堆叠 band-matrix FNO。
