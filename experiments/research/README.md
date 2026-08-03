<!--
  Research experiment profile index.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# Research Profiles

本目录用于后续研究 profile。旧 `research/ai4phc` 顶层目录已从活跃树移除；后续研究代码应直接进入 [`../../src/phc2d`](../../src/phc2d)，统一入口通过 `phc2d forward`、`phc2d inverse`、`phc2d replay` 和 `phc2d plot` 调度。

优先收束的研究主线：

| profile | 说明 |
|---|---|
| `forward_benchmark` | 多 forward surrogate 对比 |
| [`representation_ablation`](representation_ablation.md) | `rho_wedge`、grid、epsilon、image 输入表征消融 |
| `dispersion_inverse` | frozen surrogate 驱动的色散逆向设计 |
| [`generative_inverse`](generative_inverse.md) | cVAE / diffusion 多候选逆向设计与 replay 闭环 |
| [`a2_inverse_surrogate_ablation_v1`](a2_inverse_surrogate_ablation_v1.md) | A2 七个频率 surrogate 固定候选池筛选、primary-only objective 与两模型三种子门禁 |
| `mpb_replay` | 生成结构的 MPB physics replay |
| [`near_field_surrogate`](near_field_surrogate.md) | TE/TM near-field surrogate、NeurOLight 与 Maxwell residual feasibility |
