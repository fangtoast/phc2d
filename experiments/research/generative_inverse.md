<!--
  Generative inverse design research profile.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# Generative Inverse Design

目标：用多候选生成表达 2D photonic crystal inverse design 的一对多解空间，并通过 surrogate rank、morphology filter、diversity selection 与 MPB replay 验证闭环筛选候选。

配置入口：

- [`../../configs/research/generative_inverse_cvae.yaml`](../../configs/research/generative_inverse_cvae.yaml)
- [`../../configs/research/generative_inverse_diffusion_pilot.yaml`](../../configs/research/generative_inverse_diffusion_pilot.yaml)

## Protocol

```text
condition / target Wh or descriptors
-> sample K rho_candidates or grid_candidates
-> validate CandidateBatch and compute validity/unique/diversity/fill metrics
-> compute simplified morphology metrics: connected components, isolated pixels, total variation
-> forward surrogate prediction
-> surrogate ranking by best_of_k_surrogate_mre
-> diversity-aware top-M selection
-> WSL meep_env MPB replay in a later replay stage
-> replay report with replay metrics separated from surrogate metrics
```

当前实现范围：

- `CandidateBatch` 保存 `target_id`、`condition`、`rho_candidates` 或 `grid_candidates`、`input_repr` 与候选级 `metadata`。
- candidate protocol metrics 包括 `validity_rate`、`unique_rate`、`diversity_hamming` 与 `fill_fraction`。
- morphology metrics 包括 `num_connected_components`、`isolated_pixel_count` 与 `total_variation` 的简化版本。
- surrogate ranking 接口可以使用 mock surrogate 或后续 frozen forward surrogate adapter，输出 `best_of_k_surrogate_mre` 与 `surrogate_ranking_time_s`。
- 本阶段不实现 diffusion，不做大模型训练，不调用 MPB，也不声称完成生成式 inverse。

## Gate

1. 先定义 `CandidateBatch` 与 morphology/diversity metrics。
2. cVAE `rho_wedge` 先做 smoke；diffusion 等 candidate/replay 协议稳定后再做。
3. 每次报告必须区分 `best_of_k_surrogate_mre` 与 `best_of_k_replay_mre`。
4. MPB replay 前不能宣称 inverse design 物理成功。
