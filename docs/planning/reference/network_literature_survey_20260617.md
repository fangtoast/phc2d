<!--
  Literature and code survey for multi-representation neural operators and generative inverse design.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 网络文献与开源代码调研记录

## 摘要
- **目的**：基于原始记录保留实验链路与指标定义，对文档进行中文化结构化重构。
- **状态**：已完成中文主语体补充，原始英文命令、路径、文件名、模型字段与数值保持原文不变。
- **文件角色**：Project Docs | **原始语言比例**：英中混合（7%）
- **证据入口**：文内无直接 runs/artifact 命令字段引用；请按实验组入口补充。

## 任务边界
- 目标边界：文档结构重写与中文化表达，不修改代码、实验脚本与模型参数。
- 不覆盖运行产物，仅整理文档链路和表述。

## 进展状态
- 已完成：中文化主框架与摘要落位。
- 未完成：全文逐段逐句人工校对（保留英文原文仅作证据对照）。

## 当前风险
- 部分文件存在大量英文原文，术语一致性依赖后续逐条复核。
- 命令/路径/文件名中英文杂糅时需保持原写。

## 下一步
- 统一 `runs/`、`run_dir`、`surrogate_eval`、`mpb_replay_eval` 术语解释。
- 对新增章节与原文索引做一次 `rg` 链接可读性扫描。


调研日期：2026-06-17
目标项目：`phc2d` 二维光子晶体多表征 surrogate、field/operator surrogate 与生成式逆向设计

## 调研结论

1. FNO / DeepONet 适合作为 operator-learning 方法库来源，但本项目不能把它们都硬套成 `rho -> Wh` 回归；更合理的落点是 `epsilon_grid + k/query -> band_query` 与 `epsilon/source/frequency -> field`。
2. NeurOLight 与 modified FNO electromagnetic surrogate 的核心任务都是 field/operator surrogate，不应被简化成普通 band matrix predictor。
3. MaxwellNet 的关键是 Maxwell residual loss，需要 source、frequency、boundary condition、grid spacing、epsilon 和 field/residual 定义；当前只适合 feasibility 和 toy residual smoke。
4. Christensen 等 PhC CNN/GAN 工作最贴近本项目的 `unit cell -> band` 与 generative inverse，但也要经由 `CandidateBatch -> surrogate rank -> MPB replay` 闭环。
5. diffusion/AdjointDiffusion 应作为 P2，不应先于 candidate protocol、validity/diversity metrics 和 replay selection。

## 条目

| 名称 | 年份 / venue | 论文链接 | 开源代码链接 | license | 原始任务 | 原始输入 | 原始输出 | 与本项目关系 | 难度 | 数据需求 | 需要修改模块 | 决策 | 理由 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fourier Neural Operator | 2020/2021, arXiv / ICLR-era operator learning | https://arxiv.org/abs/2010.08895 | https://github.com/neuraloperator/neuraloperator | MIT | PDE operator surrogate | coefficient/function grid | solution field/function | `fno2d_band`、`fno2d_query`、`fno_field` 的方法来源 | medium | grid/query/field dataset | `phc2d.models`、`phc2d.training` | prototype | band-matrix pilot 可做，但更有价值的是 query/field |
| DeepONet | 2021, Nature Machine Intelligence | https://doi.org/10.1038/s42256-021-00302-5 | https://github.com/lululxvi/deeponet | CC BY-NC-SA 4.0 | operator learning | branch input function + trunk coordinates | queried operator value | `deep_onet_query` 候选 | medium | k/query 坐标与目标频率 | `phc2d.models`、`phc2d.data.transforms` | prototype | 适合 `epsilon_grid + k -> frequency`，不是首选 full Wh baseline |
| NeuralOperator library | 2025, arXiv library paper | https://arxiv.org/abs/2412.10354 | https://github.com/neuraloperator/neuraloperator | MIT | neural operator library | general function grids | solution fields | 可作为参考实现，不直接混入源码 | low | optional torch dataset | `external_references` 记录即可 | literature_only | 先写最小本地 FNO，不引入完整外部库 |
| MaxwellNet | 2021/2022, APL Photonics | https://arxiv.org/abs/2107.06164 | https://github.com/limjoowon/maxwellnet | 未在 GitHub 页面明确显示 | Maxwell residual field solver | material map, coordinates, source/frequency context | electric field | physics-informed branch feasibility | high | source/frequency/BC/grid spacing/residual collocation | `phc2d.physics`、`phc2d.models.physics` | prototype | 当前数据不足以完整复现，先做 residual feasibility |
| NeurOLight | 2022, NeurIPS | https://arxiv.org/abs/2209.10098 | https://github.com/JeremieMelo/NeurOLight | MIT | photonic field/operator surrogate | unified domain, compact wave prior, source/frequency condition | light field | `neurolight_adapted_field` P2 | high | epsilon、source、frequency、TE/TM field、domain metadata | `phc2d.models`、`phc2d.data.field_dataset` | prototype | 当前有 tiny field data，但 source/domain metadata 还不够 |
| Neural Operator-Based Surrogate Solver for Free-Form EM Inverse Design | 2023, ACS Photonics | https://doi.org/10.1021/acsphotonics.3c00156 | https://github.com/tfp-photonics/neurop_invdes | MIT | FNO/UNet field surrogate + VAE inverse | permittivity / field dataset | EM field, candidate design | FNO field + VAE inverse 的强参考 | medium | HDF5 field datasets, pretrained surrogate | `phc2d.models`、`phc2d.inverse` | prototype | 与本项目 field+inverse 闭环高度相关 |
| Predictive and Generative ML Models for Photonic Crystals | 2020, Nanophotonics | https://doi.org/10.1515/nanoph-2020-0197 | 未找到官方主代码 | unknown | PhC CNN predictor + GAN inverse | 2D unit cell image | band structure / generated unit cell | `cnn_grid_band`、cGAN/cVAE 逆向对照 | medium | unit cell + band dataset | `phc2d.models`、`phc2d.inverse` | implement_now | 任务最接近本项目，但要使用本项目 replay protocol |
| Physics-Informed Neural Operator | 2021, arXiv | https://arxiv.org/abs/2111.03794 | https://github.com/neuraloperator/physics_informed | Apache-2.0 | operator + physics residual | PDE parameters, residual points | PDE solution | `pino_field_pilot` 可行性 | high | residual equations and field data | `phc2d.physics`、`phc2d.training` | literature_only | 先做 Maxwell residual feasibility，再考虑 PINO |
| AdjointDiffusion | 2025, photonic inverse design | https://arxiv.org/abs/2504.17077 | https://github.com/dongjin-seo2020/AdjointDiffusion | MIT | diffusion + adjoint-guided structural optimization | fabrication-aware binary masks, adjoint gradients | optimized photonic structures | `surrogate_guided_diffusion` / replay-aware diffusion 参考 | high | adjoint/FoM gradients or surrogate guidance | `phc2d.inverse.diffusion`、`phc2d.replay` | literature_only | diffusion 放在 candidate/replay 稳定之后 |
| Tandem / frozen surrogate inverse | 2018, ACS Photonics and related inverse-design literature | https://doi.org/10.1021/acsphotonics.7b01377 | no primary unified code | unknown | inverse design with frozen forward model | target spectrum/response | structure parameters | inverse MLP baseline 与一对多问题对照 | low | forward surrogate checkpoint | `phc2d.inverse`、`phc2d.surrogates` | implement_now | 必须作为生成式 best-of-K 的对照 |

## 外部代码策略

如果后续下载外部代码，只能放入：

```text
external_references/
  README.md
  <project_name>/
    SOURCE.md
    LICENSE_OR_LINK.txt
    commit.txt
```

`src/phc2d` 只接收适配后的最小实现。不得复制外部项目的大段源码；MIT/Apache 代码也必须记录来源、commit/hash、下载时间和参考模块。