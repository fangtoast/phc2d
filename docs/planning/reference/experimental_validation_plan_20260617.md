---
title: "完整实验验证与结果分析计划"
summary: "历史研究记录；正文保持原始时间语境，不代表当前状态。"
page_type: archive
status: historical
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 34 minutes
primary_layer: L3
evidence_date: 2026-06-17
evidence_scope: "日期化证据或历史记录；正文未因站点重构而改写。"
---

<!--
  Complete experimental validation and result-analysis protocol for phc2d research.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="experimental-validation-plan-20260617"></a>

# 完整实验验证与结果分析计划

## 摘要
- **目的**：基于原始记录保留实验链路与指标定义，对文档进行中文化结构化重构。
- **状态**：已完成中文主语体补充，原始英文命令、路径、文件名、模型字段与数值保持原文不变。
- **文件角色**：Project Docs | **原始语言比例**：英中混合（9%）
- **证据入口**：runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/`；runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/metrics.json`；runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/leaderboard.csv`；runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/figures/`；runs/{yyyymmdd}/dataset_split_audit_{YYYYMMDDHHmm}/metrics.json`

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


日期：2026-06-17
适用计划：[`fno_generative_network_adjustment_plan_20260617.md`](../fno_generative_network_adjustment_plan_20260617.md)
适用 TODO：[`research_todo_20260617.md`](../research_todo_20260617.md)

本文定义后续完整研究实验验证的 benchmark protocol。这里的 **validation** 指科研实验验证和结果分析，不是 CI、`pytest`、CLI smoke 或工程 acceptance suite。工程检查只证明代码入口、shape、依赖边界和输出格式可用；研究验证必须覆盖数据集准备、模型对比、输入输出任务对比、cross-dataset 泛化、cross-surrogate inverse validation、WSL `meep_env` MPB replay 和失败分析。

**目录：** [一、目标与范围](#scope) · [二、数据集准备矩阵](#dataset-matrix) · [三、数据划分与随机种子策略](#splits-and-seeds) · [四、Forward band surrogate benchmark](#forward-band) · [五、Query-form band surrogate benchmark](#query-band) · [六、Data scaling experiment](#data-scaling) · [七、Cross-dataset / OOD generalization](#ood-generalization) · [八、Near-field / operator surrogate benchmark](#near-field-operator) · [九、Generative inverse benchmark](#generative-inverse) · [十、Cross-surrogate inverse validation](#cross-surrogate-inverse) · [十一、Cross-dataset inverse validation](#cross-dataset-inverse) · [十二、MPB replay validation](#mpb-replay) · [十三、结果分析模板](#result-template) · [十四、Promotion / stop criteria](#promotion-stop) · [十五、First executable experiment batch / MVP benchmark](#mvp-benchmark)

---

<a id="scope"></a>

## 一、目标与范围

本计划的目标是把后续研究实验从“能跑一个 smoke”升级为可复现、可横向比较、可写入论文级报告的 benchmark protocol。

范围内：

| 范围 | 说明 |
|---|---|
| 数据集准备 | 明确 dev、pilot、main、OOD、field/operator 数据层级 |
| forward band surrogate | 比较输入表征、模型家族、误差、gap 与运行时间 |
| query-form band surrogate | 比较 full `Wh` matrix 与 `band_query` 任务 |
| near-field / operator surrogate | 只在数据字段满足 source/domain/residual schema 后进入正式 benchmark |
| generative inverse | 比较单候选、best-of-K、多候选生成、validity、diversity 和 replay |
| MPB replay | 使用 WSL `Ubuntu` + conda `meep_env` 作为物理验证边界 |
| result analysis | 把人工整理后的结论写入 `docs/results`，不把 raw run artifacts 放进去 |

范围外：

| 不属于本文 validation 的内容 | 处理方式 |
|---|---|
| `compileall`、`pytest`、CLI help | 只属于工程检查或 acceptance suite |
| tiny smoke 的 leaderboard | 只能证明管线可运行，不能写成研究结论 |
| Windows `.venv` 本地结果 | 不能替代 WSL `meep_env` validation |
| 单次 surrogate rank | 不能替代 inverse design 的物理验证 |
| raw run artifacts | 保留在 `runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/` |

统一产物约定：

| 产物 | 路径模板 |
|---|---|
| config | `configs/research/<experiment_id>.yaml` |
| run | `runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/` |
| metrics | `runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/metrics.json` |
| leaderboard | `runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/leaderboard.csv` |
| figures | `runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/figures/` |
| curated report | `docs/results/<report_name>_YYYYMMDD.md` |

统一设备原则：

1. Forward 与 inverse 神经网络训练在 CUDA GPU 可用时必须优先使用 GPU。
2. pilot/full/formal neural configs 必须使用 `runtime.require_accelerator=true`；CPU 只用于 smoke/debug/reference 或无 GPU 环境下的显式 CPU reference。
3. 所有 neural train/eval/sample/ranking run 必须记录 `requested_device`、`resolved_device`、`device_name`、`precision`、`amp_enabled`、`tf32_enabled`、`peak_gpu_memory_mb`、`batch_size`、`effective_batch_size`、`num_workers`、`pin_memory`、`non_blocking_transfer`。
4. CPU 与 GPU runtime 不得汇总为同一个 `mean ± std`；CPU Full-1 只能作为 CPU reference，GPU runtime leaderboard 必须来自 GPU 重跑。
5. MPB replay 是 solver validation，不因为神经网络 GPU-first 策略而伪装成 CUDA solver。

[↑ 回顶部](#experimental-validation-plan-20260617)

---

<a id="dataset-matrix"></a>

## 二、数据集准备矩阵

| 数据集 / 派生集 | 证据等级 | 用途 | 结论边界 | 必要 metadata |
|---|---|---|---|---|
| `data/tiny.npz` | dev only | schema、shape、runner、plot、metric smoke | 不能作为研究结论 | `sample_id`、split、raw config |
| `data/paper_like_64_mpb.npz` | pilot sanity | small pilot、模型/metric 接口 sanity | 只能说明 pilot 可行性 | `sample_id`、split、generator metadata |
| `data/paper_like_50k_mpb.npz` | main in-domain benchmark | 当前 forward 主 benchmark | 支撑 in-domain forward 结论 | `sample_id`、fixed split、raw metadata、generator metadata |
| derived small-data splits | scaling benchmark | `N=64/256/1k/5k/10k/50k` 数据规模曲线 | 只比较同一生成分布下的数据规模效应 | parent sample id、split seed、subset seed |
| derived OOD splits | OOD benchmark | generator family、fill fraction、morphology complexity、gap target difficulty | 支撑 domain shift 分析 | OOD 标签、生成器参数、难度分层 |
| `data/field_tiny_mpb.h5` | field feasibility | field schema、loader、visualization、metric feasibility | 只能支持 near-field feasibility | frequency、polarization、field component metadata |
| future `field_pilot_1k` | field pilot | `unet_field`、`fno_field`、NeurOLight adapted pilot | 支撑 field pilot，不支撑 full conclusion | source、domain、grid spacing、BC、normalization |
| future `field_main_10k` | field main benchmark | near-field/operator 正式 benchmark | 支撑 NeurOLight / MaxwellNet 类正式对比 | source/domain/residual schema、TE/TM labels |

关键判定：

1. `paper_like_50k_mpb.npz` 是当前 forward 主 benchmark。
2. `paper_like_64_mpb.npz` 只作为 pilot sanity，不写成模型优劣结论。
3. `data/tiny.npz` 和所有 smoke run 只证明开发管线可用。
4. `field_tiny_mpb.h5` 只能支持 field feasibility；正式 NeurOLight / MaxwellNet benchmark 必须新增大规模 field dataset，并补齐 source、domain、boundary condition、grid spacing、normalization 或 residual metadata。

[↑ 回顶部](#experimental-validation-plan-20260617)

---

<a id="splits-and-seeds"></a>

## 三、数据划分与随机种子策略

固定原则：

1. 每个研究 benchmark 必须使用固定 train / val / test split，并把 split 文件或 split id 写入 run metadata。
2. 每个正式实验至少运行 3 个 random seeds，报告 `mean ± std`。
3. small data 可以考虑 k-fold；`paper_like_50k` 主实验优先 fixed split + multi-seed，不默认 full k-fold。
4. 所有 split 必须保留 `sample_id`、raw metadata 和 generator metadata。
5. derived split 不能重排后丢失 parent sample id，否则 cross-dataset、failure analysis 和 replay 追踪会断链。

建议 split 产物：

| 产物 | 路径模板 | 内容 |
|---|---|---|
| split spec | `configs/research/splits/<dataset_id>_<split_id>.yaml` | train / val / test index 来源、seed、strata |
| small-data split | `configs/research/splits/paper_like_50k_N<size>_seed<seed>.yaml` | subset parent ids |
| OOD split | `configs/research/splits/<ood_axis>_<level>_seed<seed>.yaml` | OOD 轴、难度、source distribution |
| split audit | `runs/{yyyymmdd}/dataset_split_audit_{YYYYMMDDHHmm}/metrics.json` | split size、分布、gap stats、metadata 完整性 |
| split report | `docs/results/dataset_split_audit_YYYYMMDD.md` | 人工整理后的数据划分说明 |

[↑ 回顶部](#experimental-validation-plan-20260617)

---

<a id="forward-band"></a>

## 四、Forward band surrogate benchmark

目标：比较不同输入表征和模型家族对 full band matrix `Wh` 的预测能力，明确谁适合作为 in-domain forward 主 surrogate，谁只适合 smoke 或被转向 query/field。

实验矩阵：

| 维度 | 取值 |
|---|---|
| representation ablation | `rho_wedge`、`rho_grid`、`epsilon_grid`、`image_epsilon` |
| model benchmark | `res_mlp`、`pod_res_mlp`、`lifted_cnn`、`cnn_grid_band`、`fno_lite`、`fno2d_band` |
| dataset | main: `data/paper_like_50k_mpb.npz`; pilot: `data/paper_like_64_mpb.npz`; dev: `data/tiny.npz` |
| metrics | MSE、MAE、RMSE、`avg_MRE_pct`、`avg_MRE_bands_2_10_pct`、per-band metrics、ordering violation、`gap_f1`、runtime、parameter count |
| runtime keys | `wall_time_total_s`、`train_time_s`、`eval_time_s`、`inference_time_per_sample_ms`、`throughput_samples_per_s` |

输出路径：

| 产物 | 路径 |
|---|---|
| config | `configs/research/representation_ablation.yaml`、`configs/research/forward_band_benchmark.yaml` |
| run | `runs/{yyyymmdd}/forward_band_benchmark_{YYYYMMDDHHmm}/` |
| leaderboard | `runs/{yyyymmdd}/forward_band_benchmark_{YYYYMMDDHHmm}/leaderboard.csv` |
| metrics | `runs/{yyyymmdd}/forward_band_benchmark_{YYYYMMDDHHmm}/metrics.json` |
| per-band metrics | `runs/{yyyymmdd}/forward_band_benchmark_{YYYYMMDDHHmm}/per_band_metrics.csv` |
| figures | `runs/{yyyymmdd}/forward_band_benchmark_{YYYYMMDDHHmm}/figures/` |
| curated report | `docs/results/forward_benchmark_YYYYMMDD.md` |

报告要求：

1. leaderboard 必须同时列出精度、runtime、throughput 和 parameter count。
2. band 1 的 relative error 必须与 bands 2-10 分开解释，避免 near-zero 频率放大相对误差。
3. 任何 shape 都来自 dataset schema、`DataRepresentationSpec` 或 `TargetSpec`，不写死当前数据实例的 shape。

[↑ 回顶部](#experimental-validation-plan-20260617)

---

<a id="query-band"></a>

## 五、Query-form band surrogate benchmark

目标：比较 full `Wh` matrix regression 与 `band_query` 任务，判断 FNO / DeepONet 是否更适合 query-form band surrogate，而不是硬套成 full matrix predictor。

实验矩阵：

| 维度 | 取值 |
|---|---|
| task | full `Wh` matrix vs `band_query` |
| query axes | k interpolation、k extrapolation、held-out path segment |
| models | `mlp_query`、`deep_onet_query`、`fno2d_query` |
| metrics | query MAE / RMSE、band continuity、ordering violation、query latency |
| OOD split | held-out k path segment；后续可加入 non-p4m query if dataset supports it |

输出路径：

| 产物 | 路径 |
|---|---|
| config | `configs/research/query_band_benchmark.yaml` |
| run | `runs/{yyyymmdd}/query_band_benchmark_{YYYYMMDDHHmm}/` |
| leaderboard | `runs/{yyyymmdd}/query_band_benchmark_{YYYYMMDDHHmm}/leaderboard.csv` |
| metrics | `runs/{yyyymmdd}/query_band_benchmark_{YYYYMMDDHHmm}/metrics.json` |
| per-k metrics | `runs/{yyyymmdd}/query_band_benchmark_{YYYYMMDDHHmm}/per_k_metrics.csv` |
| figures | `runs/{yyyymmdd}/query_band_benchmark_{YYYYMMDDHHmm}/figures/` |
| curated report | `docs/results/query_benchmark_YYYYMMDD.md` |

[↑ 回顶部](#experimental-validation-plan-20260617)

---

<a id="data-scaling"></a>

## 六、Data scaling experiment

目标：回答不同模型在数据规模变化下的误差、训练时间、推理时间和稳定性如何变化，避免只用 50k 单点比较模型。

实验矩阵：

| 维度 | 取值 |
|---|---|
| N | `64`、`256`、`1k`、`5k`、`10k`、`50k` |
| models | MLP、CNN、FNO、query models |
| seeds | 至少 3 个 random seeds |
| metrics | forward band metrics、query metrics、runtime、parameter count |
| figures | scaling curves：error vs N、train time vs N、inference latency vs N |

输出路径：

| 产物 | 路径 |
|---|---|
| config | `configs/research/data_scaling_benchmark.yaml` |
| run | `runs/{yyyymmdd}/data_scaling_benchmark_{YYYYMMDDHHmm}/` |
| leaderboard | `runs/{yyyymmdd}/data_scaling_benchmark_{YYYYMMDDHHmm}/leaderboard.csv` |
| metrics | `runs/{yyyymmdd}/data_scaling_benchmark_{YYYYMMDDHHmm}/metrics.json` |
| scaling curves | `runs/{yyyymmdd}/data_scaling_benchmark_{YYYYMMDDHHmm}/figures/` |
| curated report | `docs/results/data_scaling_benchmark_YYYYMMDD.md` |

判读方式：

1. small data 下的胜负不能直接外推到 50k。
2. 50k 下的胜负不能自动证明 OOD 泛化。
3. 若 FNO 在 small data 与 50k 都没有优势，应考虑转向 query-form 或 field/operator task。

[↑ 回顶部](#experimental-validation-plan-20260617)

---

<a id="ood-generalization"></a>

## 七、Cross-dataset / OOD generalization

目标：区分 in-domain 精度、generator OOD、fill-ratio OOD、morphology OOD、query OOD 和未来 symmetry OOD，不把单一 test split 误写成“全设计空间替代 MPB”。

实验矩阵：

| 任务 | 含义 | 目的 |
|---|---|---|
| train A test A | in-domain | 主 benchmark 基线 |
| train A test B | generator OOD | 检查生成器族迁移 |
| low fill train high fill test | fill-ratio OOD | 检查填充率分布偏移 |
| morphology simple train complex test | morphology OOD | 检查结构复杂度偏移 |
| k-path held-out segment | query OOD | 检查 query interpolation / extrapolation |
| p4m -> non-p4m symmetry OOD | future | 仅当后续生成非 p4m 数据后加入 |

输出路径：

| 产物 | 路径 |
|---|---|
| config | `configs/research/ood_generalization_benchmark.yaml` |
| run | `runs/{yyyymmdd}/ood_generalization_{YYYYMMDDHHmm}/` |
| leaderboard | `runs/{yyyymmdd}/ood_generalization_{YYYYMMDDHHmm}/leaderboard.csv` |
| metrics | `runs/{yyyymmdd}/ood_generalization_{YYYYMMDDHHmm}/metrics.json` |
| OOD slices | `runs/{yyyymmdd}/ood_generalization_{YYYYMMDDHHmm}/ood_slice_metrics.csv` |
| curated report | `docs/results/ood_generalization_YYYYMMDD.md` |

结论边界：

1. 只有 in-domain 成功不能支持全设计空间替代 MPB。
2. OOD 结论必须写清训练分布、测试分布和 OOD 轴。
3. 若 surrogate 用于 inverse ranking，OOD 高风险 target 必须进入 MPB replay subset。

[↑ 回顶部](#experimental-validation-plan-20260617)

---

<a id="near-field-operator"></a>

## 八、Near-field / operator surrogate benchmark

当前定位：只做 `field feasibility` / tiny pilot。正式 NeurOLight / MaxwellNet benchmark 需要新增 `field_pilot_1k` / `field_main_10k`，并补齐 source、domain、boundary condition、grid spacing、field normalization、TE/TM metadata 和 residual schema。

模型矩阵：

| 模型 | 启动条件 | 输出 |
|---|---|---|
| `unet_field` | field loader + relative L2 metric ready | TE/TM field maps |
| `fno_field` | field loader + grid/domain metadata ready | TE/TM field maps |
| `neurolight_adapted_field` | source/frequency/domain metadata ready | source/frequency-conditioned field |
| `maxwellnet_residual_pilot` | residual schema supports it | field / residual pilot only |

metrics：

| 指标 | 说明 |
|---|---|
| relative L2 | field map 相对误差 |
| TE/TM separated metrics | 分开报告 TE 与 TM |
| amplitude / phase error | 若 real/imag schema 清晰则报告 |
| field inference time | `field_inference_time_s`、`field_inference_time_per_sample_ms` |
| physics residual | 仅在 source/domain/residual schema 有效时报告 |

输出路径：

| 产物 | 路径 |
|---|---|
| config | `configs/research/near_field_operator_benchmark.yaml`、`configs/research/neurolight_field_pilot.yaml`、`configs/research/maxwellnet_feasibility.yaml` |
| run | `runs/{yyyymmdd}/near_field_operator_pilot_{YYYYMMDDHHmm}/` |
| leaderboard | `runs/{yyyymmdd}/near_field_operator_pilot_{YYYYMMDDHHmm}/leaderboard.csv` |
| metrics | `runs/{yyyymmdd}/near_field_operator_pilot_{YYYYMMDDHHmm}/metrics.json` |
| figures | `runs/{yyyymmdd}/near_field_operator_pilot_{YYYYMMDDHHmm}/figures/` |
| curated report | `docs/results/near_field_operator_pilot_YYYYMMDD.md` |

硬边界：

1. MaxwellNet 不能被简化成 `rho -> Wh`。如果没有 Maxwell residual schema，只能称为 field surrogate 或 MaxwellNet feasibility。
2. `field_tiny_mpb.h5` 不能支撑正式 NeurOLight / MaxwellNet benchmark。
3. field benchmark 的正式结论必须写清数据规模、source/domain metadata 和 residual 是否有效。

[↑ 回顶部](#experimental-validation-plan-20260617)

---

<a id="generative-inverse"></a>

## 九、Generative inverse benchmark

目标：比较不同 inverse design 方法生成 K 个候选结构的能力，并用 surrogate ranking + MPB replay 验证候选是否真实复现 target band / gap。

模型矩阵：

| 模型 | 阶段 |
|---|---|
| `tandem_inverse` | baseline |
| `conditional_vae_rho` | first generative baseline |
| `conditional_vae_grid` | grid candidate baseline |
| `conditional_gan` | optional |
| `conditional_diffusion_grid` | later |
| `surrogate_guided_diffusion` | later |

sweep：

| sweep | 取值 |
|---|---|
| K sweep | `K=1/4/16/64/256` |
| top_M sweep | `top_M=1/3/5/10` |

metrics：

| 指标 | 说明 |
|---|---|
| `validity_rate` | candidate 几何/表征合法率 |
| `unique_rate` | 去重后候选比例 |
| `diversity_hamming` | candidate 之间的 Hamming diversity |
| `morphology_score` | 结构形态质量 |
| `best_of_k_surrogate_mre` | surrogate 上 best-of-K 误差 |
| `best_of_k_replay_mre` | MPB replay 后 best-of-K 误差 |
| `surrogate_replay_gap` | surrogate 预测与 MPB replay 的差距 |
| `candidate_sampling_time_s` | 候选采样时间 |
| `surrogate_ranking_time_s` | surrogate ranking 时间 |
| `mpb_replay_time_s` | MPB replay 时间 |

输出路径：

| 产物 | 路径 |
|---|---|
| config | `configs/research/generative_inverse_benchmark.yaml`、`configs/research/generative_inverse_cvae.yaml` |
| run | `runs/{yyyymmdd}/generative_inverse_benchmark_{YYYYMMDDHHmm}/` |
| leaderboard | `runs/{yyyymmdd}/generative_inverse_benchmark_{YYYYMMDDHHmm}/leaderboard.csv` |
| metrics | `runs/{yyyymmdd}/generative_inverse_benchmark_{YYYYMMDDHHmm}/metrics.json` |
| candidate table | `runs/{yyyymmdd}/generative_inverse_benchmark_{YYYYMMDDHHmm}/candidates.csv` |
| replay subset | `runs/{yyyymmdd}/generative_inverse_benchmark_{YYYYMMDDHHmm}/replay_selection.csv` |
| curated report | `docs/results/generative_inverse_replay_YYYYMMDD.md` |

硬边界：inverse design 不能只看 surrogate rank。任何物理成功结论都必须进入 WSL `Ubuntu` + conda `meep_env` MPB replay。

[↑ 回顶部](#experimental-validation-plan-20260617)

---

<a id="cross-surrogate-inverse"></a>

## 十、Cross-surrogate inverse validation

目标：发现 surrogate exploit。若 generator 学会欺骗某个 surrogate，候选可能在 surrogate A 上排名很高，但被 surrogate B 或 MPB replay 判为失败。

验证矩阵：

| candidates | ranking surrogate | cross-check surrogate | replay |
|---|---|---|---|
| cVAE candidates | MLP | FNO | MPB replay |
| cVAE candidates | FNO | MLP | MPB replay |
| grid candidates | CNN | FNO | MPB replay |
| diffusion candidates | stable surrogate after protocol | second surrogate | only after replay protocol stable |

记录指标：

| 指标 | 说明 |
|---|---|
| rank correlation | surrogate A 与 surrogate B 的候选排名相关性 |
| surrogate disagreement | 两个 surrogate 的误差或排序差异 |
| MPB replay MRE | 物理 replay 后的 MRE |
| `surrogate_replay_gap` | surrogate score 与 replay score 的差距 |

输出路径：

| 产物 | 路径 |
|---|---|
| config | `configs/research/cross_surrogate_inverse_validation.yaml` |
| run | `runs/{yyyymmdd}/cross_surrogate_inverse_validation_{YYYYMMDDHHmm}/` |
| metrics | `runs/{yyyymmdd}/cross_surrogate_inverse_validation_{YYYYMMDDHHmm}/metrics.json` |
| disagreement table | `runs/{yyyymmdd}/cross_surrogate_inverse_validation_{YYYYMMDDHHmm}/surrogate_disagreement.csv` |
| curated report | `docs/results/cross_surrogate_inverse_validation_YYYYMMDD.md` |

[↑ 回顶部](#experimental-validation-plan-20260617)

---

<a id="cross-dataset-inverse"></a>

## 十一、Cross-dataset inverse validation

目标：检查 inverse generator、ranking surrogate 和 target set 发生分布偏移时，候选是否仍能通过 stratified MPB replay。

实验矩阵：

| 维度 | 取值 |
|---|---|
| generator train set | A |
| target set | A / B / OOD |
| surrogate train set | A / B / mixed |
| target strata | in-domain target、OOD target、hard target、high-gap target |
| replay subset | 必须进入 stratified MPB replay subset |

输出路径：

| 产物 | 路径 |
|---|---|
| config | `configs/research/cross_dataset_inverse_validation.yaml` |
| run | `runs/{yyyymmdd}/cross_dataset_inverse_validation_{YYYYMMDDHHmm}/` |
| metrics | `runs/{yyyymmdd}/cross_dataset_inverse_validation_{YYYYMMDDHHmm}/metrics.json` |
| replay subset | `runs/{yyyymmdd}/cross_dataset_inverse_validation_{YYYYMMDDHHmm}/replay_selection.csv` |
| curated report | `docs/results/cross_dataset_inverse_validation_YYYYMMDD.md` |

报告要求：

1. 不能把 in-domain target 成功外推到 OOD target。
2. surrogate train set 与 generator train set 必须分开记录。
3. hard / high-gap target 的失败案例必须进入 failure analysis。

[↑ 回顶部](#experimental-validation-plan-20260617)

---

<a id="mpb-replay"></a>

## 十二、MPB replay validation

MPB replay 是 inverse design 的物理验证环节，必须运行在 WSL `Ubuntu` + conda `meep_env`。报告必须分开 **surrogate_eval** 与 **WSL `meep_env` MPB validation**。

replay 等级：

| 等级 | 规模 | 用途 |
|---|---|---|
| replay sanity | 5 targets × top 1 | 检查 replay 接口和 target/candidate 对齐 |
| replay pilot | 20 targets × top 3-5 | 检查 replay 稳定性与初步误差 |
| replay formal | 100-500 targets × top M | 支撑正式 inverse 结论 |
| replay stress | 1000+ targets | 只在 pilot 稳定后启动 |

target selection 必须 stratified：

| strata | 目的 |
|---|---|
| easy | 验证基本可达 target |
| hard | 检查难目标失败模式 |
| high gap | 检查高 gap target |
| low gap | 检查低 gap 或无 gap target |
| OOD | 检查分布外 target |
| surrogate disagreement | 发现 surrogate exploit |
| high-confidence | 检查 surrogate 高置信候选是否真的通过 MPB |

输出路径：

| 产物 | 路径 |
|---|---|
| config | `configs/research/mpb_replay_validation.yaml` |
| run | `runs/{yyyymmdd}/mpb_replay_validation_{YYYYMMDDHHmm}/` |
| metrics | `runs/{yyyymmdd}/mpb_replay_validation_{YYYYMMDDHHmm}/metrics.json` |
| replay bands | `runs/{yyyymmdd}/mpb_replay_validation_{YYYYMMDDHHmm}/replay_bands.csv` |
| figures | `runs/{yyyymmdd}/mpb_replay_validation_{YYYYMMDDHHmm}/figures/` |
| curated report | `docs/results/generative_inverse_replay_YYYYMMDD.md` |

报告指标：

1. `best_of_k_replay_mre`
2. `surrogate_replay_gap`
3. replay success rate
4. invalid geometry rate
5. gap match / gap miss
6. `mpb_replay_time_s`
7. `mpb_replay_time_per_candidate_s`

[↑ 回顶部](#experimental-validation-plan-20260617)

---

<a id="result-template"></a>

## 十三、结果分析模板

每个实验报告必须写入 `docs/results`，只保存人工整理后的分析和表格，不保存 raw run artifacts。原始 `metrics.json`、`leaderboard.csv`、figures、logs 留在对应 `runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/`。

每个报告必须包含：

| 模块 | 内容 |
|---|---|
| experiment_id | 与 run dir、config 对齐 |
| dataset / split / seeds | 数据路径、split id、random seeds、是否 fixed split |
| model / input / output | 模型、input representation、output target |
| metrics table | `mean ± std` over seeds；forward 与 replay 分开 |
| runtime table | training、eval、inference、candidate sampling、ranking、replay runtime |
| leaderboard | 按主指标排序，同时列出 runtime 与 parameter count |
| per-band / per-k plots | forward band 与 query-form 必须包含 |
| failure cases | surrogate 失败、OOD 失败、replay 失败、无效结构 |
| conclusion | 当前证据支持什么、不支持什么 |
| promotion decision | 是否 promote to full experiment、stop 或 pivot |

结果判读不得按单个 seed 的 best result 做最终 leaderboard 排序。正式报告必须以 `mean ± std` over seeds 为主；若模型差距小于跨 seed std，不应声称该模型显著优于 baseline，只能写作“当前证据未显示稳定优势”或“需要更多 seed / 更大 split 复核”。

报告路径清单：

| 报告 | 路径 |
|---|---|
| forward benchmark | `docs/results/forward_benchmark_YYYYMMDD.md` |
| query benchmark | `docs/results/query_benchmark_YYYYMMDD.md` |
| data scaling | `docs/results/data_scaling_benchmark_YYYYMMDD.md` |
| OOD generalization | `docs/results/ood_generalization_YYYYMMDD.md` |
| near-field / operator pilot | `docs/results/near_field_operator_pilot_YYYYMMDD.md` |
| generative inverse replay | `docs/results/generative_inverse_replay_YYYYMMDD.md` |

[↑ 回顶部](#experimental-validation-plan-20260617)

---

<a id="promotion-stop"></a>

## 十四、Promotion / stop criteria

promotion / stop 规则：

| 决策 | 允许条件 | 停止或转向条件 |
|---|---|---|
| smoke -> pilot | tiny / smoke 管线可复现，标准 artifacts 齐全，metrics schema 稳定 | smoke 只证明可运行，不产生研究结论 |
| pilot -> full 50k | paper_like_64 pilot 无 shape/metric/runner 问题，至少 3 seeds 方案明确 | pilot 结果不稳定或数据 metadata 断链 |
| inverse -> MPB replay | candidate validity/diversity/ranking 可追踪，top-M selection 固定 | 只看 surrogate rank；无 replay selection strata |
| stop a model | 多 seed 下误差、runtime 或 OOD 表现持续劣于 baseline | 仅一次 tiny smoke 失败不能直接淘汰 |
| pivot to query / field | full `Wh` matrix task 上 FNO/CNN 无优势，query 或 field 任务更符合模型归纳偏置 | field 数据缺 source/domain metadata 时不能直接 full field benchmark |
| allow diffusion | cVAE / tandem baseline、CandidateBatch、cross-surrogate、MPB replay protocol 稳定 | 无 validity/diversity/replay 的 diffusion 先不启动 |
| allow NeurOLight full benchmark | `field_pilot_1k` 稳定，`field_main_10k` 准备完成，source/domain metadata 齐全 | 只有 `field_tiny_mpb.h5` 或 source/residual schema 不足 |
| allow MaxwellNet full benchmark | Maxwell residual schema、source、BC、grid spacing、field target 均有效 | 只能做 supervised `rho -> Wh` 时不得称 MaxwellNet |

最小下一步：

1. 在 `configs/research` 下补正式实验矩阵 YAML。
2. 为 forward、query、scaling、OOD、field、inverse、replay 补 native runner。
3. 固定 `docs/results/*_YYYYMMDD.md` 报告模板。
4. 先从 `paper_like_50k` forward 主 benchmark 和 cVAE inverse pilot 开始，等 replay protocol 稳定后再启动 diffusion 与 full field/operator benchmark。

[↑ 回顶部](#experimental-validation-plan-20260617)

---

<a id="mvp-benchmark"></a>

## 十五、First executable experiment batch / MVP benchmark

第一批 MVP benchmark 的目标不是覆盖完整矩阵，而是把数据划分、forward、query、inverse 和 replay 的最小闭环跑通，并产出可复用的 config、run artifact 和报告骨架。MVP 完成前，Gate 7 只能表示“实验矩阵设计完成”，不能表示实验已经执行完成。

| MVP | 目的 | 前置条件 | 数据集 | 模型 | 最小 sweep | 输出路径 | promotion / stop 判据 |
|---|---|---|---|---|---|---|---|
| MVP-0 dataset split audit | 固定第一批 benchmark 的 split、seed、metadata 追踪口径 | `infer_dataset_schema` / `phc2d data inspect` 可读 NPZ/HDF5；split spec 草案存在 | `data/tiny.npz`、`data/paper_like_64_mpb.npz`、`data/paper_like_50k_mpb.npz`、`data/field_tiny_mpb.h5` | none | audit fixed train/val/test；检查至少 3 seeds；抽查 `sample_id`、raw metadata、generator metadata | `configs/research/splits/*.yaml`；`runs/{yyyymmdd}/dataset_split_audit_{YYYYMMDDHHmm}/metrics.json`；`docs/results/dataset_split_audit_YYYYMMDD.md` | promote：split 可复现且 metadata 不断链；stop：缺少 parent sample id、split 不可复现或 OOD strata 无法追踪 |
| MVP-1 forward_band_mvp | 建立 first runnable forward band leaderboard | MVP-0 通过；forward train/eval runner 可写标准 artifacts；候选模型已在 registry 可构建 | pilot: `data/paper_like_64_mpb.npz`；main smoke: `data/paper_like_50k_mpb.npz` 的固定 small split | `res_mlp` baseline；`pod_res_mlp` / `lifted_cnn` / `fno_lite` 仅在可用时加入 | representations: `rho_wedge` + one grid/effective medium repr；seeds 3；epochs 使用 pilot budget；不做 full 50k 训练 | `configs/research/forward_band_benchmark.yaml`；`runs/{yyyymmdd}/forward_band_mvp_{YYYYMMDDHHmm}/leaderboard.csv`；`docs/results/forward_benchmark_YYYYMMDD.md` | promote：baseline + 至少一个对照模型 multi-seed 完成且 runtime/metrics/per-band 输出齐全；stop：runner 或 metrics schema 不稳定、单 seed 结果无法复现 |
| MVP-2 query_band_mvp | 验证 `band_query` 任务是否能形成独立 benchmark | MVP-0 通过；query target builder 可从 `Wh` 派生 query samples；query metrics 可输出 per-k / continuity | pilot: `data/paper_like_64_mpb.npz`；main smoke: `data/paper_like_50k_mpb.npz` 的固定 small split | `mlp_query` baseline；`deep_onet_query` / `fno2d_query` 仅在可用时加入 | k interpolation + one held-out path segment；seeds 3；query latency 记录 | `configs/research/query_band_benchmark.yaml`；`runs/{yyyymmdd}/query_band_mvp_{YYYYMMDDHHmm}/leaderboard.csv`；`docs/results/query_benchmark_YYYYMMDD.md` | promote：query MAE/RMSE、ordering violation、latency 和 held-out segment 指标齐全；stop：query 构造丢失 band/k metadata 或与 full `Wh` 对齐失败 |
| MVP-3 inverse_mvp | 跑通 candidate generation、surrogate rank、diversity-aware top-M 的最小 inverse pipeline | MVP-1 至少有一个 frozen forward surrogate；CandidateBatch / selection metrics 可追踪 | target 来自 `data/paper_like_64_mpb.npz` pilot split；后续接 `paper_like_50k` target strata | `tandem_inverse` baseline；`conditional_vae_rho` first generative baseline | `K=1/16`；`top_M=1/3`；seeds 3；先只做 surrogate eval，不把 surrogate rank 当物理结论 | `configs/research/generative_inverse_benchmark.yaml`；`runs/{yyyymmdd}/inverse_mvp_{YYYYMMDDHHmm}/candidates.csv`；`docs/results/generative_inverse_replay_YYYYMMDD.md` 的 surrogate-only 草稿段 | promote：validity/diversity/best-of-K/surrogate ranking 可复现，且 replay selection file 生成；stop：candidate 无效率过高、top-M 不可追踪或 frozen surrogate 不稳定 |
| MVP-4 replay_sanity | 对 MVP-3 的少量候选做 WSL `meep_env` MPB replay sanity | MVP-3 生成 replay selection；MPB wrapper/config 可追溯；只在 WSL `Ubuntu` + conda `meep_env` 执行 | 5 个 stratified targets × top 1 candidate | no training model；使用 MVP-3 selected candidates 与 MPB solver | replay sanity: 5 targets × top 1；记录 replay success、bands 2-10 MRE、gap match、`mpb_replay_time_s` | `configs/research/mpb_replay_validation.yaml`；`runs/{yyyymmdd}/replay_sanity_{YYYYMMDDHHmm}/metrics.json`；`docs/results/generative_inverse_replay_YYYYMMDD.md` | promote：5 个 target replay 命令、日志、metrics 和失败原因可追溯；stop：WSL `meep_env` 不可用、candidate/target 对齐失败或 replay 输出无法和 surrogate_eval 分开 |

[↑ 回顶部](#experimental-validation-plan-20260617)