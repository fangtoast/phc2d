<!--
  Generative inverse-design protocol and historical model-routing boundary.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# Generative Inverse Design

目标是用多候选生成表达二维光子晶体 inverse design 的一对多解空间，并通过合法性、形态、多样性、surrogate 排序与 real-MPB replay 筛选候选。

## 当前模型身份

当前已实现并进入 A2 证据链的是 `single_mode_generator` 与 `mixture_conditional_generator`。它们是带 latent noise / mode 条件的 MLP 生成器，**不是 cVAE，也不是 diffusion**。当前入口见：

- [`a2_inverse_single_seed.md`](a2_inverse_single_seed.md)
- [`a2_inverse_objective_bridge_seed0_v1.md`](a2_inverse_objective_bridge_seed0_v1.md)
- [`a2_inverse_objective_bridge_seed0_test625_v1.md`](a2_inverse_objective_bridge_seed0_test625_v1.md)

历史 `generative_inverse_cvae.yaml` 与 `generative_inverse_diffusion_pilot.yaml` 只保留为 configured/planned lineage；它们没有完成可引用的训练与 real-MPB 验证，不能用来命名当前 mixture generator。

## Candidate protocol

```text
condition / target Wh or descriptors
-> sample K rho_candidates or grid_candidates
-> validate CandidateBatch and compute validity / unique / diversity / fill metrics
-> morphology and collision checks
-> primary surrogate objective or ranking
-> cross-surrogate consistency gate / diagnostic
-> diversity-aware top-M selection
-> real-MPB replay
-> replay report separated from surrogate metrics
```

`CandidateBatch` 保留 target、condition、candidate、input representation 和候选级 lineage。报告必须分别记录 `best_of_k_surrogate_mre`、`best_of_k_replay_mre`、`surrogate_replay_gap`、候选健康度与运行时间。

## Gate

1. 固定数据、split、checkpoint 和候选谱系。
2. 先通过 candidate validity、unique、diversity、morphology 与 Top-K coverage。
3. surrogate objective/ranking 与 cross-surrogate gate 分角色记录，不写成无差别“双代理排序”。
4. MPB replay 前不能声明 inverse design 物理成功。
5. 单 seed、pilot 或 configured-only 模型不得升级为跨 seed 或架构 superiority。

当前状态和可引用结果以 [`../../docs/results/README.md#inverse-results`](../../docs/results/README.md#inverse-results) 为准。
