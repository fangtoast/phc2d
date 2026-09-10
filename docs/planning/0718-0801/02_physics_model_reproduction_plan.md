---
title: "Physics Forward、PINO、MaxwellNet 与 Inverse Integration 计划"
summary: "研究计划或执行协议；实施前需与当前代码和证据重新核对。"
page_type: plan
status: draft
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 10 minutes
primary_layer: L0
plan_state: active
effective_date: 2026-07-18
superseded_by: null
related_results:
  - docs/results/field-physics/README.md
---

<!--
  Physics-model reproduction plan for forward surrogates and inverse integration.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# Physics Forward、PINO、MaxwellNet 与 Inverse Integration 计划

## 一、修订后的核心认识

physics model 不是只在 MPB replay 后面补一层分析。正确顺序是：

```text
physics data/constraints
-> independently train and validate a forward physics surrogate
-> freeze or expose its differentiable gradients
-> integrate it into inverse training/ranking/refinement
-> real MPB/Meep final verification
```

当前仓库已具备 1k 与 9,216 条 real-MPB datasets、active TE codec、schema-driven `unet_field_v2`/`fno_field_v2`、NumPy/Torch TE Bloch Maxwell residual 与 adapted-PINO training regime。首轮 v1 data-only 三 seed sasdwa仅构成失败诊断；v2 六个正式 run 已通过监督场门槛。PINO 已在 validation split 冻结 `lambda_pde=0.01`，并以三 seed residual median 降低 `37.48%`、mean E 不退化通过 physics gate；MaxwellNet 因缺少 driven/source/PML 原任务继续 blocked。

## 二、两类 physics forward model

### 2.1 Band physics-regularized surrogate

该路线使用当前 `rho/epsilon -> Wh` 数据，可以较快建立。建议损失为：

```text
L_band = L_data
       + lambda_relative * L_relative_bands_2_10
       + lambda_order * L_band_ordering
       + lambda_continuity * L_k_continuity
       + lambda_symmetry * L_k_path_symmetry
       + lambda_gap * L_gap_descriptor
```

当前 `band_aware` loss 已包含 MSE、bands 2-10 relative error、ordering 和 k-index smoothness，但没有 Maxwell PDE residual，也没有完整 symmetry/gap loss。它应被描述为结构化或 physics-regularized data loss，不能写成 PINO。

该模型可作为独立 forward baseline，并在 inverse 中提供更符合 band 结构的梯度。

### 2.2 True physics-informed field/eigen operator

PINO 和 MaxwellNet 的自然任务是学习 PDE solution operator 或 field solver，而不是普通 `rho -> Wh` regression。模型需要输出 complex field/eigen solution，并计算：

```text
L_physics = lambda_data * L_field_data
          + lambda_pde * L_Maxwell_residual
          + lambda_bc * L_boundary
          + lambda_norm * L_normalization
          + lambda_div * L_divergence
          + lambda_orth * L_mode_orthogonality
```

band eigenfiel d 与 driven field 必须拆开：

| 任务 | 输入 | 输出 | 关键约束 |
|---|---|---|---|
| eigen operator | epsilon + Bloch k + polarization/band context | eigenfrequency + eigenfield | eigen residual、Bloch BC、normalization、orthogonality |
| driven field | epsilon + source + frequency + polarization | complex E/H field | source Maxwell residual、absorbing/periodic BC、field normalization |

## 三、模型矩阵

| 模型 | 自然任务 | 损失 | 当前状态 | 本阶段目标 |
|---|---|---|---|---|
| existing MLP/spectral band model | structure -> `Wh` | data/band-aware | trained | physics-regularized loss control |
| `unet_field_v2` | physics context -> active TE complex field | supervised field | implemented / formal run in progress | field baseline v2 |
| adapted `fno_field_v2` | coefficient/condition -> active TE field | supervised operator | trained / three-seed gate passed | matched data-only architecture |
| `fno_field_v2` + `pino_adapted` regime | parametric eigenproblem -> field | data + Maxwell residual | trained / three-seed physics gate passed | eigenfield-adapted PINO pilot |
| `maxwellnet_residual_pilot` | driven material/source context -> field | Maxwell residual + BC | blocked feasibility | requires separate driven/source/PML task |
| `neurolight_adapted_field` | source/frequency-conditioned field | supervised/masked field | feasibility only | 后续候选，不在首轮主线 |

## 四、Phase P0：复现与公式审计

每个外部模型先完成：

- paper、official repository、commit、license 和下载时间；
- 原论文输入、输出、网络、loss、BC、数据和 metric；
- 本项目适配点与不能复现的部分；
- `project_native / paper_inspired_adaptation / faithful_reproduction` 定级；
- 不把外部代码直接复制到 `src/phc2d`。

FNO faithful comparator 优先放到 grid-to-grid field operator；当前 `fno2d_band_v2` 保持 adapted spectral baseline。

## 五、Phase P1：FieldSample 与 residual contract

`FieldSample` 至少包含：

```text
sample_id
task_type: eigen | driven
epsilon
k_points and/or source
frequency / eigenfrequency
polarization and band_index
boundary_condition
domain_extent and dx/dy
complex E/H field or real/imag channels
field_normalization
material_units and source_convention
solver_provenance
```

先为 residual 写不依赖训练的验证：

1. 解析/构造解 residual 接近离散误差下限；
2. solver reference field residual 显著低于随机场；
3. 错误 BC、单位、dx/dy 会被测试捕获；
4. complex derivative、TE/TM 分量和 source term convention 有独立 tests；
5. residual 在 grid refinement 下趋势合理。

无法满足时保持 `blocked`，不制造 physics loss 数值。

## 六、Phase P2：真实 field pilot 数据

已按 64-sample smoke、1k 开发集和 9,216 条 operator-style records 逐级扩容。当前不生成 100k full-field samples。

要求：

- WSL `meep_env` MPB/Meep 生成；
- eigen 与 driven dataset 分开；
- 覆盖多个 structure、k/frequency、TE/TM 条件；
- 保存 source/BC/domain/grid/normalization metadata；
- shard、resume、failed-case ledger 和 duplicate hash 完整；
- 抽样重复求解验证 solver consistency。

## 七、Phase P3：独立 supervised baseline

先训练 `unet_field` 与 `fno_field`。只有二者在同一 field pilot 上形成稳定 baseline，才评估 physics loss 带来的额外收益。

正式指标：

- complex field relative L2；
- amplitude 与 phase error；
- TE/TM 分开结果；
- Maxwell residual 与 boundary residual；
- inference/runtime/memory；
- 3-seed convergence；
- in-resolution 与 higher-resolution evaluation。

## 八、Phase P4：PINO 与 MaxwellNet pilot

### PINO

参考 [Physics-Informed Neural Operator](https://arxiv.org/abs/2111.03794)，完成：

- data-only FNO；
- residual-only；
- data + residual；
- coarse data + higher-resolution residual；
- in-resolution 与 zero-shot higher-resolution。

Promotion：data+residual 相比 data-only 同时改善 field error、physics residual 或 data efficiency，且收益跨 seed 稳定。

### MaxwellNet

参考 [MaxwellNet](https://arxiv.org/abs/2107.06164)，复现范围限定为一个明确的 2D TE/TM 问题，不宣称覆盖原论文全部微透镜场景。

必须证明：

1. residual 对正确 reference field 接近离散下限；
2. 随机场和错误 BC residual 更高；
3. 训练后 residual 与 field error 同时下降；
4. 与 MPB/Meep reference field 对齐；
5. source、units、domain 和 normalization 可追溯。

## 九、Phase P5：接入 inverse

按风险由低到高推进：

```text
target band / field
-> inverse generator candidates
-> existing query/full surrogate ranking
-> physics surrogate cross-check
-> physics-aware reranking
-> optional differentiable physics loss/refinement
-> real MPB/Meep final verification
```

必须进行四组对照：

| 组别 | Physics 的作用 | 回答的问题 |
|---|---|---|
| no-physics | 不使用 | 当前基线 |
| physics cross-check/rerank | 只在选择阶段使用 | 是否减少 surrogate exploit |
| physics train loss | 参与 generator objective | 是否改善生成分布 |
| physics local refinement | 对 shortlist 做梯度细化 | 额外计算是否换来 replay 收益 |

每组固定 candidate 与 solver budget。最终报告不能把 physics forward 指标代替 inverse replay，也不能把增加 solver calls 带来的收益归因于模型本身。

## B4：频率保真与 dense-k 准入

B3 的 adapted PINO 只证明当前 eigenfield task 上的 residual/E 指标，不证明其 frequency head 可安全进入 inverse。B4 的实现与运行设计见 [`10_b4_frequency_fidelity_dense_k_admission.md`](10_b4_frequency_fidelity_dense_k_admission.md)：PDE residual 的 `full|detach` 频率梯度路径、固定 frequency supervision、40-epoch 审计、31-index dense-k 和 64→96/128 边界都必须在独立 artifacts 中完成。

在 B4 的 `promotion_decision.json` 出现前，PINO 不可作为 inverse cross-check 或 rerank score；更不得因 residual 数值改善启动 physics-train-loss、local refinement 或任何 replay。

## 十、半月完成定义

- 最低完成：公式审计、FieldSample、residual tests、64 real samples、one-batch train smoke。
- 目标完成：256/1k pilot、U-Net/FNO supervised baseline、PINO 或 MaxwellNet 至少一个真实 data+physics pilot。
- 进阶完成：physics surrogate 进入 inverse rerank 或 objective，并完成固定预算 real-solver 对照。
- 不允许声称：PINO/MaxwellNet 已成为正式论文主模型，除非有 multi-seed、solver-backed forward 和 inverse 证据。
