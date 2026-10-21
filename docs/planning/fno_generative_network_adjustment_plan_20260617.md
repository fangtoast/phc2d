---
title: "多表征神经算子与生成式逆向设计项目规划书"
summary: "研究计划或执行协议；实施前需与当前代码和证据重新核对。"
page_type: plan
status: draft
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 29 minutes
primary_layer: L0
plan_state: active
effective_date: 2026-06-17
superseded_by: null
related_results:
  - docs/results/forward/a1_forward_query_fair_v1_20260726.md
  - docs/results/inverse/inverse_mpb_replay_mixture_v2_10000_test625_20260706.md
---

<!--
  Multi-representation neural-operator and generative inverse-design research plan.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 多表征神经算子与生成式逆向设计项目规划书

## 摘要
- **目的**：基于原始记录保留实验链路与指标定义，对文档进行中文化结构化重构。
- **状态**：已完成中文主语体补充，原始英文命令、路径、文件名、模型字段与数值保持原文不变。
- **文件角色**：Project Docs | **原始语言比例**：英中混合（9%）
- **证据入口**：runs/`、`checkpoints/`；runs/20260608/plot_dataset_near_fields_202606081551`；runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/`，至少包括；run.meta.json；config.resolved.yaml

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


报告日期：2026-06-17
代码仓库：`D:\project\article_reproduction\POD-DeepONET-2d(codex)`
当前架构：统一 `src/phc2d` 公共包 + `configs` / `experiments` profile + Git history 历史来源
研究对象：2D photonic crystal 的 band surrogate、query surrogate、near-field/operator surrogate、生成式逆向设计与 MPB replay 验证

## 1. 摘要

本计划把项目目标从“替换一个固定 `rho/p4m -> Wh` 网络”升级为“统一 `phc2d` 架构下的多输入表征、多输出物理目标、多模型家族、多候选逆向设计和 MPB replay 验证平台”。

当前项目不再以旧 `legacy/reproduction` 或 `research/ai4phc` 目录作为活跃后端。公共能力进入 [`../../src/phc2d`](https://github.com/fangtoast/phc2d/tree/main/src/phc2d)，实验组合由 [`../../configs`](https://github.com/fangtoast/phc2d/tree/main/configs) 和 [`../../experiments/research`](https://github.com/fangtoast/phc2d/tree/main/experiments/research) 描述，结果分析进入 [`../results`](../results/README.md)。历史代码只作为 Git history 中的算法来源，需要时恢复并迁入 `src/phc2d`。

执行进度使用 [`research_todo_20260617.md`](research_todo_20260617.md) 跟踪。后续完成任务时，优先更新该 checklist，再同步结果报告。

核心决策：

1. 不把 `16x16`、`31`、`10` 当作平台常量，只把它们视为当前数据实例的 metadata。
2. FNO 不再是唯一主线，而是 neural operator 家族中的一个候选；它在 `epsilon_grid -> Wh` 上若无优势，应转向 `band_query` 或 field surrogate。
3. NeurOLight 与 MaxwellNet 不应被简化成普通 band predictor。NeurOLight 属于 source/frequency-conditioned field operator；MaxwellNet 属于 Maxwell residual field solver。
4. 生成式逆向必须先定义 `CandidateBatch`、validity、diversity、morphology、surrogate rank 和 replay selection，再实现 cVAE/diffusion。
5. 任何 inverse design 结论都必须区分 WSL `meep_env` smoke、surrogate eval 与 WSL `meep_env` MPB replay，不能把 surrogate 指标包装成物理验证成功。

## 2. 当前仓库与数据约束

权威边界：

| 区域 | 路径 | 当前职责 |
|---|---|---|
| 公共包 | [`../../src/phc2d`](https://github.com/fangtoast/phc2d/tree/main/src/phc2d) | data、generators、models、training、inverse、surrogates、replay、plot、runs、cli |
| 配置入口 | [`../../configs`](https://github.com/fangtoast/phc2d/tree/main/configs) | `reproduction`、`research`、`smoke`、`formal` |
| 研究 profile | [`../../experiments/research`](https://github.com/fangtoast/phc2d/tree/main/experiments/research) | forward、query、near-field、generative inverse、MPB replay |
| 规划与审计 | [`../planning`](README.md) | 本计划、文献调研、repo/data audit、里程碑 |
| 结果报告 | [`../results`](../results/README.md) | 人工整理后的结果分析，不放原始 run 产物 |
| 本地产物 | `data/`、`runs/`、`checkpoints/` | 大数据、运行输出、权重，Git ignored |

当前可用数据概况见 [`repo_data_audit_20260617.md`](reference/repo_data_audit_20260617.md)：

| 数据 | 关键字段 |
|---|---|
| `data/tiny.npz` | `rho=(12,36)`、`Wh=(12,31,10)`、split、`config_yaml` |
| `data/paper_like_64_mpb.npz` | `rho=(64,36)`、`Wh=(64,31,10)`、split、gap test fields |
| `data/paper_like_50k_mpb.npz` | `rho=(50000,36)`、`Wh=(50000,31,10)`、split、gap test fields |
| `data/field_tiny_mpb.h5` | `epsilon`、`rho`、`k_points`、`freqs`、`efield`、`hfield` |
| `runs/20260608/plot_dataset_near_fields_202606081551` | 100 samples，TE/TM，bands 2-4，resolution 64，WSL MPB probe ok |

硬约束：

1. PyTorch、h5py、MPB/Meep 等重依赖必须 optional / lazy import。
2. 后续测试统一通过 WSL `Ubuntu` + conda `meep_env` 执行。
3. Windows `.venv` 只作为编辑期临时调试便利，不再作为验收测试结果。
4. 正式物理验证必须继续明确标注为 WSL `Ubuntu` + conda `meep_env` MPB/Meep validation。
5. 运行产物统一写入 `runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/`，至少包括 `run.log`、`run.meta.json`、`config.resolved.yaml`、`results.json`、`metrics.json`、`history.csv`、`figures/`。

## 3. 为什么不能固定为 16x16 -> 31x10 Wh

固定 `rho_wedge(36) -> Wh(31,10)` 只对应当前 Wang 2026 复现数据的一种表达。它有价值，但不应成为平台边界。

原因：

1. `rho_wedge` 是 p4m symmetry 下的 baseline design vector，不代表所有结构输入。
2. `16x16` 是当前离散 resolution，不代表 near-field replay 的 `64` resolution，也不代表未来 MPB/FEM 生成参数。
3. `Wh=(Nk,Nb)` 中的 `Nk=31`、`Nb=10` 来自当前 k path 和 band count，query-form task、IBZ dataset、field dataset 的 shape 不同。
4. FNO、DeepONet、NeurOLight、MaxwellNet 对输入输出的自然形式不同。强行统一成 full matrix regression 会掩盖模型真正适合的任务。
5. inverse design 是一对多问题。直接监督 `target -> rho` 容易学到平均结构，需要多候选生成与 solver replay。

因此，所有模型、loss、metric、plot 和 config 都应从 dataset schema 或 `DataRepresentationSpec` 读取 shape。

### 3.1 Metrics 必须包含运行时间

所有 smoke / pilot / full / replay 实验都必须记录运行时间，避免只按精度选择模型。时间指标与精度指标同等进入 `metrics.json`、leaderboard 和结果报告。

统一建议键：

| 指标 | 含义 | 适用阶段 |
|---|---|---|
| `wall_time_total_s` | 端到端 wall-clock 时间 | all |
| `setup_time_s` | 配置解析、数据加载、模型构建等准备时间 | train/eval/inverse/replay |
| `train_time_s` | 训练循环时间，不含数据生成 | forward/inverse train |
| `eval_time_s` | 评估循环时间 | forward/inverse eval |
| `inference_time_s` | 总推理时间 | forward eval / surrogate rank |
| `inference_time_per_sample_ms` | 单样本平均推理耗时 | forward eval / surrogate rank |
| `candidate_sampling_time_s` | 生成 K candidates 的总时间 | generative inverse |
| `candidate_sampling_time_per_target_ms` | 单 target 生成耗时 | generative inverse |
| `surrogate_ranking_time_s` | surrogate rank + diversity selection 时间 | generative inverse |
| `mpb_replay_time_s` | MPB replay 总时间 | replay |
| `mpb_replay_time_per_candidate_s` | 单候选 replay 平均时间 | replay |
| `throughput_samples_per_s` | forward/eval throughput | train/eval |
| `peak_gpu_memory_mb` | CUDA 可用时记录 GPU peak allocated memory | neural train/eval/inverse |
| `device_name` | CPU/GPU/WSL 环境标识 | all |
| `requested_device` / `resolved_device` | 配置请求与实际解析设备 | neural train/eval/inverse |
| `precision` / `amp_enabled` / `tf32_enabled` | 数值精度与加速策略 | neural train/eval/inverse |

报告口径：

1. smoke 报告至少记录 `wall_time_total_s`。
2. forward benchmark leaderboard 必须同时包含误差指标、`train_time_s`、`inference_time_per_sample_ms`、parameter count。
3. generative inverse 必须同时报告 `candidate_sampling_time_s`、`surrogate_ranking_time_s`、`mpb_replay_time_s`。
4. MPB replay 时间不能与 local surrogate 时间混写。
5. Forward 与 inverse 神经网络训练在 CUDA GPU 可用时必须优先使用 GPU。pilot/full/formal 配置必须 `require_accelerator=true`；CPU 只用于 smoke/debug/reference。所有运行必须记录 requested/resolved device、precision 和 GPU memory。CPU/GPU runtime 不得混合比较。

### 3.2 GPU-first Training And Runtime Validation

统一设备策略由 `src/phc2d/training/device.py` 提供：`DeviceConfig` 解析 `runtime.device`，`device=auto` 在 CUDA 可用时解析为 `cuda:<device_index>`；若 `require_accelerator=true` 且 CUDA 不可用，训练、评估、采样或 ranking 必须立即失败，不能静默回退 CPU。Forward train/eval/benchmark 与 inverse sample/eval 必须使用同一个 resolved device；只有写文件、NumPy metric、plot 或 morphology CPU 分析时才把张量转回 CPU。

当前 WSL `meep_env` CUDA 审计见 [`../results/gpu_environment_audit_20260618.md`](../results/data-provenance/gpu_environment_audit_20260618.md)。`runs/20260618/forward_band_benchmark_202606180848` 保留为 CPU reference benchmark；修正 `fno_lite` / `fno2d_band` identity 问题后，Full-1 必须在同一 GPU、同一 precision、同一 split、seeds `0/1/2` 下重跑。GPU Full-1 完成前不得启动 Full-2。

## 4. 输入表征与输出任务协议

本计划新增或规划以下协议模块：

```text
src/phc2d/data/representations.py
src/phc2d/data/target_specs.py
src/phc2d/data/transforms.py
src/phc2d/data/field_dataset.py
src/phc2d/data/lift_torch.py
```

当前最小实现已提供：

```python
DataRepresentationSpec
SampleMetadata
TargetSpec
infer_dataset_schema(path)
make_input(sample, spec)
make_target(sample, spec)
validate_representation_compatibility(schema, spec)
inspect_field_dataset(path)
lift_rho_to_grid_torch(...)
epsilon_from_rho_torch(...)
```

输入表征：

| input_repr | 用途 |
|---|---|
| `rho_wedge` | p4m wedge / design vector baseline |
| `rho_grid` | binary or soft occupancy grid |
| `epsilon_grid` | material permittivity grid |
| `image_binary` | image-like binary input |
| `image_epsilon` | image-like dielectric map |
| `param_vector` | 几何/材料参数 |
| `multimodal` | structure + metadata + k/query/source |
| `field_conditioned` | structure + source/frequency/polarization |

输出目标：

| output_target | 用途 |
|---|---|
| `Wh` | full band matrix，shape 由 metadata 决定 |
| `band_query` | selected k/band/frequency query |
| `gap_descriptors` | gap location、width、gap-midgap ratio |
| `near_field_te` | TE field map |
| `near_field_tm` | TM field map |
| `near_field_te_tm` | multi-polarization fields |
| `scalar_fom` | gap score、transmission、refraction metric 等 |
| `multi_task` | Wh + gap + field + morphology |

同一个 raw sample 必须保留 `sample_id`、split 和 metadata，允许派生出不同 representation。

## 5. 文献调研与开源代码复现策略

详细调研见 [`network_literature_survey_20260617.md`](reference/network_literature_survey_20260617.md)。

结论摘要：

| 方向 | 决策 |
|---|---|
| FNO / U-FNO | P1 prototype。先做 band matrix pilot，再做 query/field |
| DeepONet | P1 query prototype，更适合 `k/query -> frequency` |
| NeurOLight | P2 field/operator prototype，不硬套 `rho -> Wh` |
| MaxwellNet | feasibility first，缺字段时只做 toy residual |
| modified FNO EM surrogate | P2 field surrogate 重要参考 |
| Christensen PhC CNN/GAN | P1/P2 候选，适合作为 PhC CNN/GAN 对照 |
| Tandem / frozen surrogate inverse | P1 baseline |
| diffusion / AdjointDiffusion | P2，必须排在 candidate/replay 协议之后 |

外部代码策略：

```text
external_references/
  README.md
  <project_name>/
    SOURCE.md
    LICENSE_OR_LINK.txt
    commit.txt
```

不得把外部仓库代码直接混入 `src/phc2d`。`src/phc2d` 只放适配后的最小实现，并记录来源和 license。

## 6. 模型家族与适配决策

`phc2d.models.registry` 应保持 `register_model` / `build_model` / `list_models` 兼容，同时新增 `ModelSpec`：

```python
ModelSpec(
    name="fno2d_band",
    family="neural_operator",
    input_repr="epsilon_grid",
    output_target="Wh",
    supports_variable_grid=False,
    supports_query=False,
    supports_field=False,
    requires=("torch",),
)
```

模型分层：

| 阶段 | 模型 | 输入 | 输出 | 决策 |
|---|---|---|---|---|
| P0 baseline | `res_mlp` | `rho_wedge` | `Wh` | 从历史逻辑恢复或重写 |
| P0 baseline | `pod_res_mlp` | `rho_wedge` / POD coefficients | `Wh` | 恢复为对照 |
| P0 baseline | `lifted_cnn` | `rho_grid` / `epsilon_grid` | `Wh` | 图像输入 baseline |
| P0 baseline | `fno_lite` | `epsilon_grid` | `Wh` | 最小 FNO baseline |
| P1 forward | `cnn_grid_band` | `image_epsilon` | `Wh` | representation benchmark |
| P1 forward | `fno2d_band` | `epsilon_grid` | `Wh` | pilot，不承诺 full |
| P1 query | `fno2d_query` | `epsilon_grid + k` | `band_query` | FNO 更自然的落点 |
| P1 query | `deep_onet_query` | branch + trunk | `band_query` | query-form 对照 |
| P1 inverse | `conditional_vae_rho` | target condition | `rho_wedge` candidates | 先做 best-of-K |
| P1 inverse | `tandem_inverse` | target condition | single candidate | 对照 baseline |
| P2 field | `fno_field` | field-conditioned | TE/TM field | 依赖 field dataset |
| P2 field | `neurolight_adapted_field` | source/frequency-conditioned | field | feasibility first |
| P2 physics | `maxwellnet_residual_pilot` | epsilon + physics context | field/residual | toy first |
| P2 generative | `conditional_diffusion_grid` | target condition | `rho_grid` candidates | replay protocol 后 |
| P2 generative | `surrogate_guided_diffusion` | target + surrogate gradient | candidates | replay-aware 后期 |

## 7. MaxwellNet / NeurOLight 可行性专题

### MaxwellNet

MaxwellNet 的核心不是 supervised `rho -> Wh`，而是用 Maxwell 方程残差训练 field solver。当前判断：

| 条件 | 当前状态 |
|---|---|
| near-field TE/TM 场 | 有 tiny HDF5 与 20260608 replay 可视化 |
| source / frequency | `freqs` 有，source 表示不足 |
| boundary condition | 缺 |
| grid spacing / domain metadata | 缺明确字段 |
| epsilon grid | 有 |
| residual 定义 | 需新增 `phc2d.physics.maxwell_residual` |

决策：新增 `configs/research/maxwellnet_feasibility.yaml`，先写 feasibility report 与 toy TE/TM residual smoke。字段不足时不做完整 MaxwellNet benchmark。

### NeurOLight

NeurOLight 的核心是 photonic field/operator surrogate：统一 domain、compact wave prior、source/frequency 条件和 masked source modeling。当前判断：

| 条件 | 当前状态 |
|---|---|
| `epsilon_grid` | 有 |
| TE/TM field maps | 有 tiny data 与 replay 可视化 |
| source / frequency / wavelength | frequency 有，source 表示不足 |
| boundary/domain metadata | 不足 |
| 大规模 field dataset | 不足 |

决策：新增 `configs/research/neurolight_field_pilot.yaml`，先做 `fno_field` / `unet_field` schema 和 tiny smoke，再决定是否做 `neurolight_adapted_field`。

## 8. Forward surrogate 实验矩阵

Task A: band matrix surrogate

| 项 | 设计 |
|---|---|
| input | `rho_wedge` / `rho_grid` / `epsilon_grid` / `image_epsilon` |
| output | `Wh`，shape from metadata |
| models | MLP、POD-MLP、CNN、FNO、U-FNO |
| metrics | MSE、MAE、RMSE、avg_MRE_pct、avg_MRE_bands_2_10_pct、per-band、ordering violation、gap_f1、train/eval/inference runtime |
| config | [`../../configs/research/representation_ablation.yaml`](https://github.com/fangtoast/phc2d/blob/main/configs/research/representation_ablation.yaml)、[`../../configs/research/fno_benchmark.yaml`](https://github.com/fangtoast/phc2d/blob/main/configs/research/fno_benchmark.yaml) |

Task B: query-form band surrogate

| 项 | 设计 |
|---|---|
| input | `epsilon_grid + k/query coordinate + optional band index` |
| output | selected frequency or band vector |
| models | DeepONet、FNO query、MLP query、NeurOLight-inspired query encoder |
| metrics | query MAE/RMSE、k interpolation/extrapolation、band continuity、ordering violation、query inference latency |
| config | [`../../configs/research/query_band_benchmark.yaml`](https://github.com/fangtoast/phc2d/blob/main/configs/research/query_band_benchmark.yaml) |

## 9. Near-field / operator surrogate 实验矩阵

Task C: near-field surrogate

| 项 | 设计 |
|---|---|
| input | `epsilon_grid + source/frequency/k/polarization` |
| output | TE/TM field maps |
| models | FNO field、U-Net field、NeurOLight adapted、MaxwellNet/PINO pilot |
| metrics | relative L2 field error、amplitude/phase error、physics residual if feasible、boundary/source consistency、field inference runtime |
| config | [`../../configs/research/near_field_surrogate.yaml`](https://github.com/fangtoast/phc2d/blob/main/configs/research/near_field_surrogate.yaml) |

已知 near-field run 支持 feasibility：`runs/20260608/plot_dataset_near_fields_202606081551`。该 run 显示 TE/TM 可视化成功，`n_ok=100`，TE frequency check mean relative error 约 `2.40e-08`。下一步不是直接训练大模型，而是把 field HDF5 schema、field target spec、field metrics 和 optional loader 打通。

## 10. Generative inverse 实验矩阵

Task D: inverse candidate generation

| 项 | 设计 |
|---|---|
| input condition | target `Wh` / gap descriptors / band summary / optional field objective |
| output | K candidate `rho_wedge` or `rho_grid` |
| models | conditional VAE、conditional GAN、diffusion、surrogate-guided diffusion |
| metrics | validity_rate、unique_rate、diversity_hamming、morphology_score、best_of_k_surrogate_mre、best_of_k_replay_mre、surrogate_replay_gap、candidate/replay runtime |
| config | [`../../configs/research/generative_inverse_cvae.yaml`](https://github.com/fangtoast/phc2d/blob/main/configs/research/generative_inverse_cvae.yaml)、[`../../configs/research/generative_inverse_diffusion_pilot.yaml`](https://github.com/fangtoast/phc2d/blob/main/configs/research/generative_inverse_diffusion_pilot.yaml) |

当前最小协议：

```python
CandidateBatch(
    target_id=...,
    condition=...,
    rho_candidates=...,
    input_repr="rho_wedge" or "rho_grid",
    metadata=[...],
)
```

固定评估流程：

```text
condition / target
-> sample K candidates
-> validity + morphology filters
-> forward surrogate prediction
-> surrogate ranking
-> diversity-aware top-M selection
-> MPB replay
-> replay report
```

## 11. MPB replay 验证闭环

Task E: solver-validated replay

| 项 | 设计 |
|---|---|
| input | selected candidates |
| solver | WSL `Ubuntu` + conda `meep_env` MPB replay |
| output | replay `Wh` / gap / pass-fail / comparison plots |
| metrics | avg_MRE_bands_2_10_pct、gap match、replay success rate、invalid geometry rate、mpb_replay_time_s、mpb_replay_time_per_candidate_s |

报告必须分开：

1. WSL `meep_env` tests
2. surrogate_eval
3. WSL `meep_env` MPB validation

只有第 3 项可以支撑物理成功结论。

## 12. 完整实验验证与结果分析路线

完整 validation 在本计划中指 **研究实验验证 benchmark protocol**，不是工程 acceptance test、`pytest`、CLI smoke 或 compileall。工程检查只能证明入口可用；研究结论必须来自数据集准备、模型对比、任务对比、cross-dataset 泛化、cross-surrogate inverse validation、WSL `meep_env` MPB replay 和失败分析。

详细方案写入 [`experimental_validation_plan_20260617.md`](reference/experimental_validation_plan_20260617.md)。主计划只保留总览：

| 模块 | 完整验证内容 | 当前证据边界 |
|---|---|---|
| dataset preparation | `tiny`、`paper_like_64`、`paper_like_50k`、derived small-data splits、derived OOD splits、field pilot/main dataset | `data/tiny.npz` 只用于 dev；`paper_like_64` 只用于 pilot sanity；`paper_like_50k` 是当前 forward 主 benchmark |
| forward band surrogate benchmark | `rho_wedge`、`rho_grid`、`epsilon_grid`、`image_epsilon` 表征消融；`res_mlp`、`pod_res_mlp`、CNN、FNO 对比 | tiny / smoke leaderboard 不能作为研究结论 |
| query-form band surrogate benchmark | 对比 full `Wh` matrix 与 `band_query`；覆盖 k interpolation、k extrapolation、held-out path segment | 若 FNO full matrix 无优势，应转向 query-form 或 field/operator |
| representation and data scaling | `N=64/256/1k/5k/10k/50k`，至少 multi-seed，报告误差与 runtime scaling curves | small-data 胜负不能直接外推到 50k |
| cross-dataset / OOD generalization | generator OOD、fill-ratio OOD、morphology OOD、gap difficulty、未来 p4m -> non-p4m | in-domain 成功不能支持“全设计空间替代 MPB” |
| near-field / operator surrogate benchmark | `unet_field`、`fno_field`、`neurolight_adapted_field`、`maxwellnet_residual_pilot` | `field_tiny_mpb.h5` 只能支持 feasibility；正式 NeurOLight / MaxwellNet benchmark 需要新增大规模 field dataset 与 source/domain metadata |
| generative inverse benchmark | tandem、cVAE、grid cVAE、optional GAN、later diffusion；K sweep 与 top_M sweep | inverse design 不能只看 surrogate rank |
| cross-surrogate inverse validation | surrogate A ranking、surrogate B cross-check、MPB replay 验证 | 目标是发现 surrogate exploit |
| cross-dataset inverse validation | generator train set A，target set A/B/OOD，surrogate train set A/B/mixed | 必须进入 stratified MPB replay subset |
| MPB replay validation | sanity、pilot、formal、stress 四级 replay；target 按 easy/hard/high gap/low gap/OOD/disagreement/high-confidence 分层 | 物理成功结论必须来自 WSL `Ubuntu` + conda `meep_env` MPB replay |
| failure analysis and reports | forward、query、scaling、OOD、near-field、inverse replay 报告进入 `docs/results` | surrogate_eval 与 replay metrics 必须分开 |

结果报告必须至少包含 experiment_id、dataset/split/seeds、model/input/output、metrics table、runtime table、leaderboard、per-band/per-k plots、failure cases、conclusion 和 promote / stop / pivot decision。raw run artifacts 继续留在 `runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/`，不放进 `docs/results`。

## 13. 工程改造路径

P0 已开始补齐的基础设施：

| 模块 | 状态 |
|---|---|
| `phc2d.data.representations` | 已新增 `DataRepresentationSpec` / `SampleMetadata` |
| `phc2d.data.target_specs` | 已新增 `TargetSpec` 与输入/输出 target 名称 |
| `phc2d.data.transforms` | 已新增 schema inspection 与 representation transforms |
| `phc2d.data.field_dataset` | 已新增 field schema inspection helper |
| `phc2d.data.lift_torch` | 已新增 optional torch lifting |
| `phc2d.models.registry` | 已扩展 `ModelSpec`，保留原 factory API |
| `phc2d.inverse.candidates` | 已新增 `CandidateBatch`、diversity、unique_rate |
| `phc2d.inverse.morphology` | 已新增 binary morphology metrics |
| `phc2d data inspect` | 已新增 NPZ/HDF5 schema inspection CLI |
| `configs/research/*` | 已新增 8 个 profile 草案 |
| `experiments/research/*` | 已新增 representation、near-field、generative profile 说明 |

下一步实现顺序：

```text
Gate 1 representation protocol
-> phc2d data inspect / data convert
-> baseline model specs + CPU shape tests
-> native forward train/eval smoke
-> representation benchmark
-> field/operator feasibility
-> cVAE best-of-K inverse
-> MPB replay pilot
```

## 14. 里程碑与验收标准

### Gate 0: Repo and data audit

产物：

- [`repo_data_audit_20260617.md`](reference/repo_data_audit_20260617.md)
- [`network_literature_survey_20260617.md`](reference/network_literature_survey_20260617.md)

验收：当前仓库边界、forward 数据、near-field 数据、CLI stub 状态已记录。

### Gate 1: Representation protocol

产物：

- `DataRepresentationSpec`
- `TargetSpec`
- schema inspection
- representation smoke tests

验收：

1. `rho -> Wh` 旧任务仍可表达。
2. `epsilon_grid -> Wh` 可表达。
3. near-field task 可表达，即使暂时没有大规模真实数据。
4. 不在模型中硬编码 data shape。

### Gate 2: Baseline restoration

模型：

```text
res_mlp
pod_res_mlp
lifted_cnn
fno_lite
```

验收：registry 可列出，CPU shape test 通过，smoke train 输出标准 run artifact。

### Gate 3: Representation benchmark

比较：

```text
rho_wedge vs rho_grid vs epsilon_grid vs image_epsilon
MLP vs CNN vs FNO
Wh vs band_query
```

输出：`leaderboard.csv`、`metrics.json`、`per_band_metrics.csv`、figures、`docs/results/representation_benchmark_YYYYMMDD.md`。

### Gate 4: Field/operator pilot

模型：

```text
fno_field
unet_field
neurolight_adapted_field
maxwellnet_residual_pilot, only if data supports it
```

验收：先完成 near-field schema 和 loader；有数据则 tiny train/eval；无数据则生成数据需求与 MPB/Meep 生成计划。

### Gate 5: Generative inverse pilot

模型：

```text
conditional_vae_rho
conditional_vae_grid
optional diffusion pilot
```

验收：K candidates 可复现采样，validity/diversity/morphology 可统计，surrogate rank 可运行，少量 target 进入 MPB replay。

### Gate 6: MPB replay and final report

验收：WSL `meep_env` replay 命令和日志可追溯，replay 指标与 surrogate 指标分开报告，对 surrogate 好但 replay 差的案例做 failure analysis。

## 15. 风险、失败判据与转向策略

| 风险 | 失败判据 | 转向 |
|---|---|---|
| FNO band matrix 无优势 | `epsilon_grid -> Wh` 不优于 MLP/CNN | 转向 `band_query` 或 `fno_field` |
| shape 被隐式写死 | 新数据 shape 导致模型/metric 失败 | 所有 shape 改由 schema/spec 注入 |
| near-field 数据不足 | 缺 source/BC/grid spacing/normalization | 先做 HDF5 schema 和 MPB/Meep 生成计划 |
| MaxwellNet 名不副实 | 只做普通 supervised field predictor | 改名为 field surrogate，MaxwellNet 留在 residual feasibility |
| NeurOLight 被误用 | 仅用于 `rho -> Wh` | 保留为 source/frequency-conditioned field/operator 分支 |
| diffusion 过早主线化 | 无 validity/diversity/replay 仍训练 diffusion | 回到 cVAE + candidate protocol |
| surrogate exploit | surrogate rank 好但 MPB replay 差 | 报告 `surrogate_replay_gap`，以 replay 为准 |
| 旧目录复活 | 新逻辑写回 `research/ai4phc` | 只从 Git history 恢复算法，入口与输出按 `phc2d` 重写 |

## 16. 参考文献与代码链接

| 编号 | 来源 |
|---|---|
| R1 | Li, Z. et al. Fourier Neural Operator for Parametric Partial Differential Equations. https://arxiv.org/abs/2010.08895 |
| R2 | NeuralOperator official library. https://github.com/neuraloperator/neuraloperator |
| R3 | Lu, L. et al. DeepONet. https://doi.org/10.1038/s42256-021-00302-5 and https://github.com/lululxvi/deeponet |
| R4 | Lim, J. and Psaltis, D. MaxwellNet. https://arxiv.org/abs/2107.06164 and https://github.com/limjoowon/maxwellnet |
| R5 | Gu, J. et al. NeurOLight. https://arxiv.org/abs/2209.10098 and https://github.com/JeremieMelo/NeurOLight |
| R6 | Augenstein, Y. et al. Neural Operator-Based Surrogate Solver for Free-Form Electromagnetic Inverse Design. https://doi.org/10.1021/acsphotonics.3c00156 and https://github.com/tfp-photonics/neurop_invdes |
| R7 | Christensen, T. et al. Predictive and generative machine learning models for photonic crystals. https://doi.org/10.1515/nanoph-2020-0197 |
| R8 | Li, Z. et al. Physics-Informed Neural Operator. https://arxiv.org/abs/2111.03794 and https://github.com/neuraloperator/physics_informed |
| R9 | Seo, D. et al. Physics-guided and fabrication-aware inverse design of photonic devices using diffusion models. https://arxiv.org/abs/2504.17077 and https://github.com/dongjin-seo2020/AdjointDiffusion |
| R10 | Liu, D. et al. Training Deep Neural Networks for the Inverse Design of Nanophotonic Structures. https://doi.org/10.1021/acsphotonics.7b01377 |
