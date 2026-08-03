<!--
  Progress checklist for the multi-representation neural-operator and generative inverse-design plan.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# Research TODO Checklist

## 摘要
- **目的**：基于原始记录保留实验链路与指标定义，对文档进行中文化结构化重构。
- **状态**：已完成中文主语体补充，原始英文命令、路径、文件名、模型字段与数值保持原文不变。
- **文件角色**：Project Docs | **原始语言比例**：英文主导（4%）
- **证据入口**：runs/20260608/plot_dataset_near_fields_202606081551`。；run.meta.json；config.resolved.yaml；results.json；metrics.json

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


更新时间：2026-06-17

用途：跟踪“多表征神经算子与生成式逆向设计项目规划书”的执行进度。后续每完成一项，就把 `[ ]` 改成 `[x]`，并尽量在备注里写明对应代码、配置、run 或报告路径。

## Gate 0: Repo And Data Audit

- [x] 重读当前 `src/phc2d` + `configs` + `experiments` 架构。
- [x] 审计 forward NPZ 数据：`rho`、`Wh`、split、metadata/config 字段。
- [x] 审计 near-field 数据：`data/field_tiny_mpb.h5`。
- [x] 审计 near-field run：`runs/20260608/plot_dataset_near_fields_202606081551`。
- [x] 记录 CLI 状态：`plot` 已实现，`data/forward/inverse/replay` 多数仍需 native backend。
- [x] 输出 repo/data audit 文档：`docs/planning/repo_data_audit_20260617.md`。
- [x] 输出文献与开源代码调研：`docs/planning/network_literature_survey_20260617.md`。

## Gate 1: Representation Protocol

- [x] 新增 `DataRepresentationSpec` / `SampleMetadata`。
- [x] 新增 `TargetSpec` 与输入/输出 target 名称。
- [x] 新增 NPZ/HDF5 `infer_dataset_schema`。
- [x] 新增 `make_input` / `make_target` / compatibility validation。
- [x] 新增 `inspect_field_dataset`。
- [x] 新增 optional `lift_torch` 可微 lifting。
- [x] 接入 `phc2d data inspect --path ...`。
- [x] 增加 representation/schema/CLI inspect tests。
- [ ] 实现 `phc2d data convert --config ...`。
- [ ] 为 derived representation 写入 raw sample id / split / metadata 保真测试。

## Gate 2: Baseline Restoration

- [x] 扩展 `ModelSpec`，保持 `register_model/build_model/list_models` 兼容。
- [ ] 从 Git history 恢复或重写 `res_mlp`。
- [ ] 从 Git history 恢复或重写 `pod_res_mlp`。
- [ ] 从 Git history 恢复或重写 `lifted_cnn`。
- [ ] 从 Git history 恢复或重写 `fno_lite`。
- [ ] 每个 baseline 增加 CPU smoke shape test。
- [x] 实现 `phc2d forward train --config ...` native smoke backend：`src/phc2d/forward/runner.py`。
- [x] 实现 `phc2d forward eval --config ...` native smoke backend：`src/phc2d/forward/runner.py`。
- [x] 输出标准 run artifacts：`run.log`、`run.meta.json`、`config.resolved.yaml`、`results.json`、`metrics.json`、`history.csv`、`figures/`；smoke run 见 `runs/20260617/forward_tiny_res_mlp_smoke_202606171537`。
- [x] `metrics.json` 记录 runtime keys：`wall_time_total_s`、`train_time_s`、`eval_time_s`、`inference_time_per_sample_ms`、`throughput_samples_per_s`。

## Gate 3: Representation Benchmark

- [x] 新增 `configs/research/representation_ablation.yaml`。
- [x] 新增 `configs/research/fno_benchmark.yaml`。
- [x] 新增 `configs/research/query_band_benchmark.yaml`。
- [x] 新增 `experiments/research/representation_ablation.md`。
- [ ] 跑 `rho_wedge` vs `rho_grid` vs `epsilon_grid` vs `image_epsilon` smoke。
- [ ] 跑 MLP / CNN / FNO pilot 对比。
- [ ] 跑 `Wh` vs `band_query` pilot 对比。
- [ ] 生成 `leaderboard.csv`、`metrics.json`、`per_band_metrics.csv`、figures。
- [ ] leaderboard 同时包含精度、运行时间、吞吐和 parameter count。
- [ ] 写 `docs/results/representation_benchmark_YYYYMMDD.md`。

## Gate 4: Field / Operator Pilot

> 2026-07-20 update: the valid 64-record real-MPB pilot passed the B2 residual reference-vs-random gate, and the B1 scale-up produced a 1,000-record TE Bloch eigenfield dataset. Evidence: `docs/results/field_eigen_te_b2_b1_1000_20260720.md`. No field/PINO/MaxwellNet training has started.

- [x] 新增 `configs/research/near_field_surrogate.yaml`。
- [x] 新增 `configs/research/neurolight_field_pilot.yaml`。
- [x] 新增 `configs/research/maxwellnet_feasibility.yaml`。
- [x] 新增 `experiments/research/near_field_surrogate.md`。
- [x] 明确 MaxwellNet 完整实现缺 source、BC、grid spacing、residual schema。
- [ ] 实现 field dataset loader smoke。
- [ ] 实现 field metrics：relative L2、amplitude/phase 可选、TE/TM 分开统计。
- [ ] field metrics 记录 `field_inference_time_s` / `field_inference_time_per_sample_ms`。
- [ ] 实现 `unet_field` CPU shape smoke。
- [ ] 实现 `fno_field` CPU shape smoke。
- [x] 写 Maxwell residual feasibility report：`docs/planning/near_field_feasibility_20260617.md`。
- [ ] 若字段足够，做 toy TE/TM residual smoke loss。
- [x] 若字段不足，输出 MPB/Meep field dataset 生成计划：`docs/planning/near_field_feasibility_20260617.md`。

## Gate 5: Generative Inverse Pilot

- [x] 新增 `CandidateBatch` 与 `GenerativeInverseAdapter` 协议。
- [x] 新增 `unique_rate` / `diversity_hamming`。
- [x] 新增 binary morphology metrics 与 `morphology_score`。
- [x] 新增 `configs/research/generative_inverse_cvae.yaml`。
- [x] 新增 `configs/research/generative_inverse_diffusion_pilot.yaml`。
- [x] 新增 `experiments/research/generative_inverse.md`。
- [ ] 实现 `conditional_vae_rho` CPU smoke。
- [ ] 实现 `phc2d inverse sample --config ...`。
- [x] 实现 surrogate rank 与 diversity-aware top-M selection：`src/phc2d/inverse/selection.py`。
- [x] 记录 `best_of_k_surrogate_mre`：candidate protocol ranking metrics。
- [ ] 记录 `candidate_sampling_time_s` 与 `surrogate_ranking_time_s`：本次 ranking 接口记录 `surrogate_ranking_time_s`，sampling timer 等 `phc2d inverse sample` backend。
- [ ] 选少量 target 进入 MPB replay pilot。
- [ ] 记录 `best_of_k_replay_mre` 与 `surrogate_replay_gap`。
- [ ] diffusion pilot 只在 candidate/replay 协议稳定后启动。
- [x] 实现 direct inverse baseline 训练接口：`model.name=direct_inverse_mlp`，配置 `configs/research/direct_inverse_baseline_v2_10000.yaml`。
- [x] 实现 surrogate-objective generator fine-tuning 接口：`phc2d inverse objective-train --config ...`，loss 默认按 bands 2-10 训练，边界为 surrogate-side no MPB/Meep。
- [x] 新增 GLOnet-style surrogate objective pilot 配置：`configs/research/inverse_glonet_pilot_v2_10000.yaml`。
- [x] 跑 direct inverse baseline V2 10k training/sample/real-MPB replay，并写 `docs/results/direct_inverse_baseline_20260630.md`；结果未改善旧 query-only replay baseline。
- [x] 跑 objective fine-tune V2 10k training/sample/real-MPB replay；surrogate MRE 明显下降但 MPB replay 变差，判定为 surrogate exploit。
- [ ] GLOnet-style V2 10k pilot 暂缓；需先重设 diversity / gap-aware / cross-surrogate consistency objective。
- [x] 固化 10k inverse 双闭环路线：`docs/planning/inverse_dual_loop_route_20260701.md`，明确 candidate health 是 replay 前门槛，real MPB replay 是最终裁判。
- [x] 跑 `inverse_generator_uniqueness_repair_v2_10000` surrogate-side uniqueness repair；`unique_rate` 与 surrogate MRE 过线，但 `nearest_neighbor_hamming_p10=0`，未启动 real MPB replay，报告 `docs/results/inverse_generator_uniqueness_repair_v2_10000_20260701.md`。
- [x] 跑 Stage A 最后一轮 MLP collision 修复 `inverse_generator_uniqueness_repair_v2b_10000`：`collision_weight=0.9`、`target_nearest_neighbor_hamming=0.05`、`n_objective_candidates=12`；采样 preflight 仍 `nearest_neighbor_hamming_p10=0`，未启动 real MPB replay，报告 `docs/results/inverse_generator_uniqueness_repair_v2b_10000_20260701.md`。
- [x] 按用户要求诊断性覆盖 v2b preflight gate，运行 `inverse_mpb_replay_uniqueness_repair_v2b_10000_unique24_diagnostic` real MPB replay；`best_of_k_replay_mre_bands_2_10_pct=15.462788`、`mpb_replay_success_rate=1.0`，但不改变 `nearest_neighbor_hamming_p10=0` 的 Stage A fail 结论；报告 `docs/results/inverse_mpb_replay_uniqueness_repair_v2b_10000_diagnostic_20260701.md`。
- [ ] 实现 Stage B `mixture conditional generator`：`mode_count=4`、mode embedding、mode-round-robin sampling、mode-conditioned collision，并复用相同 preflight gate。

## Gate 6: MPB Replay And Reports

- [x] 实现 `phc2d replay mpb --config ...` CLI 分发（placeholder 解封；当前以 MVP proxy 实现过渡）。
- [x] replay run 写入标准 artifacts（`run.log` / `run.meta.json` / `config.resolved.yaml` / `results.json` / `metrics.json` / `history.csv` / `figures`）。
- [x] replay metrics 记录 `mpb_replay_time_s`、`mpb_replay_time_per_candidate_s`、`best_of_k_surrogate_mre`、`best_of_k_replay_mre`、`surrogate_replay_gap`。
- [ ] 明确报告 WSL `meep_env` tests 与 WSL `meep_env` MPB validation（proxy 与真实 MPB replay 仍需分离）。
- [ ] 输出 replay band/gap 对比图。
- [ ] 对 surrogate 好但 replay 差的案例做 failure analysis。
- [ ] 写 `docs/results/generative_inverse_replay_YYYYMMDD.md`（当前临时报告文件为 `mpb_replay_sanity_20260617.md`）。

## Gate 7: Experimental Validation Matrix Design

- [x] 新增 `docs/planning/experimental_validation_plan_20260617.md`。
- [x] 定义 dataset matrix：dev、pilot、main、small-data、OOD、field feasibility、future field pilot/main。
- [x] 定义 split/seeds：fixed train/val/test split、至少 3 seeds、small data 可 k-fold、50k fixed split + multi-seed。
- [x] 定义 representation ablation：`rho_wedge`、`rho_grid`、`epsilon_grid`、`image_epsilon`。
- [x] 定义 model benchmark：`res_mlp`、`pod_res_mlp`、`lifted_cnn`、`cnn_grid_band`、`fno_lite`、`fno2d_band`。
- [x] 定义 data scaling experiment：`N=64/256/1k/5k/10k/50k`。
- [x] 定义 cross-dataset / OOD benchmark：generator、fill-ratio、morphology、gap target、query held-out segment。
- [x] 定义 query-form benchmark：full `Wh` matrix vs `band_query`，含 k interpolation / extrapolation / held-out path segment。
- [x] 定义 near-field/operator benchmark 数据需求：`field_pilot_1k`、`field_main_10k`、source/domain/residual metadata。
- [x] 定义 generative inverse K/top_M sweep：`K=1/4/16/64/256`，`top_M=1/3/5/10`。
- [x] 定义 cross-surrogate inverse validation：surrogate A ranking、surrogate B cross-check、MPB replay。
- [x] 定义 cross-dataset inverse validation：generator train set A、target set A/B/OOD、surrogate train set A/B/mixed。
- [x] 定义 MPB replay stratified target selection：easy、hard、high gap、low gap、OOD、surrogate disagreement、high-confidence。
- [x] 为每个实验指定 config、run、metrics、leaderboard、report 路径；正式可执行 YAML / runner 仍需后续实现。

## Gate 8: Result Analysis And Paper-Level Report

- [ ] `docs/results/forward_benchmark_YYYYMMDD.md`。
- [ ] `docs/results/query_benchmark_YYYYMMDD.md`。
- [ ] `docs/results/data_scaling_benchmark_YYYYMMDD.md`。
- [ ] `docs/results/ood_generalization_YYYYMMDD.md`。
- [ ] `docs/results/near_field_operator_pilot_YYYYMMDD.md`。
- [ ] `docs/results/generative_inverse_replay_YYYYMMDD.md`。
- [ ] `docs/results/cross_surrogate_inverse_validation_YYYYMMDD.md`。
- [ ] `docs/results/cross_dataset_inverse_validation_YYYYMMDD.md`。
- [ ] 每个报告包含 mean/std over seeds。
- [ ] 每个报告包含 runtime metrics。
- [ ] 每个 forward 报告包含 per-band/per-k/gap analysis。
- [ ] 每个 inverse 报告包含 validity/diversity/best-of-K/surrogate_replay_gap。
- [ ] 每个 replay 报告包含 failure analysis。
- [ ] 总结 promote / stop / pivot decisions。

## Gate 9: Executable Benchmark Configs And Pilot Runs

- [x] `configs/research/forward_band_benchmark.yaml` 已创建（Full-1 主配置）
- [x] `configs/research/query_band_benchmark.yaml` 已创建（Full-2 计划）
- [x] `configs/research/data_scaling_benchmark.yaml` 已创建（Full-2 计划）
- [x] `configs/research/ood_generalization_benchmark.yaml` 已创建（Full-2 计划，待 OOD metadata）
- [ ] `configs/research/generative_inverse_benchmark.yaml`（待创建）
- [ ] `configs/research/cross_surrogate_inverse_validation.yaml`（待创建）
- [ ] `configs/research/cross_dataset_inverse_validation.yaml`（待创建）
- [ ] `configs/research/mpb_replay_validation.yaml`（待创建）
- [x] MVP-0 `dataset split audit`（配置与执行文档在位）
- [x] MVP-1 `forward_band_mvp` pilot（完成）
- [x] MVP-2 `query_band_mvp` pilot（完成）
- [x] MVP-3 `inverse_mvp` pilot（完成）
- [x] MVP-4 `replay_sanity`（proxy 已完成）

### Full-1 / Full-2 notes

- [x] Full-1: `forward_band_benchmark` on `data/paper_like_50k_mpb.npz` 已执行（`runs/20260618/forward_band_benchmark_202606180848`）
- [x] Full-1: `docs/results/forward_benchmark_20260617.md` 已按 Full-1 真实结果重写
- [x] Full-2: query 已完成 guarded V2 Full-2A；data scaling 已完成 guarded V2 `N=64` Full-2B 最近可执行门槛；OOD 仍不执行
- [ ] Full-2：如 `pod_res_mlp` 不可用，记录为 blocker 并从 leaderboard 排除

## Gate 10: GPU-first Training And Runtime Validation

- [x] WSL `meep_env` CUDA audit：`docs/results/gpu_environment_audit_20260618.md`。
- [x] shared `DeviceConfig`：`src/phc2d/training/device.py`。
- [x] forward GPU training：`phc2d forward train` 解析统一 GPU device，并记录 metadata。
- [x] forward GPU eval/benchmark：`phc2d forward eval/benchmark` 使用统一 device、checkpoint `map_location` 与 CUDA timing sync。
- [x] inverse GPU sampling/ranking：`phc2d inverse sample/eval` 在统一 device 上执行 tensor sampling 与 surrogate scoring。
- [x] GPU metrics and metadata：`run.meta.json` / `metrics.json` 增加 requested/resolved device、precision、TF32、pinning、peak GPU memory 等字段。
- [x] GPU integration tests：新增 `tests/test_phc2d_device_runtime.py`；CUDA 可用时执行 forward/inverse GPU smoke，否则 skip。
- [x] CPU Full-1 relabeled as reference：`docs/results/forward_benchmark_20260617.md`。
- [x] corrected Full-1 rerun on GPU：`runs/20260618/forward_band_benchmark_gpu_202606181551`。
- [x] GPU forward report generated：`docs/results/forward_benchmark_gpu_20260618.md`。

项目级规则：Forward 与 inverse 神经网络训练在 CUDA GPU 可用时必须优先使用 GPU。pilot/full/formal 配置必须 `require_accelerator=true`；CPU 只用于 smoke/debug/reference。所有运行必须记录 requested/resolved device、precision 和 GPU memory。CPU/GPU runtime 不得混合比较。

## Gate: Formal GPU Full-1 Forward Benchmark

- [x] existing CPU Full-1 integrity audit：`docs/results/forward_benchmark_integrity_audit_20260618.md`
- [x] fno_lite / fno2d_band identity audit：`runs/20260618/forward_band_benchmark_gpu_202606181551/model_identity_summary.json`
- [x] alias documented or registry fixed：`ModelSpec.alias_of` 与 benchmark alias guard 已实现；本轮 `alias_cases=[]`
- [x] independent fno2d_band verified：`fno2d_band` class/factory/params/state schema/fingerprint 均不同于 `fno_lite`
- [x] model identity tests：`tests/test_phc2d_model_registry.py`
- [x] GPU model-matrix preflight：`runs/20260618/forward_model_matrix_gpu_preflight_202606181550`
- [x] formal GPU benchmark config：`configs/research/forward_band_benchmark_gpu.yaml`
- [x] seeds 0/1/2 completed：formal run 15/15 case rows completed
- [x] no duplicate cases：`integrity_summary.json` 中 `duplicate_case_ids=[]`
- [x] GPU leaderboard generated：`runs/20260618/forward_band_benchmark_gpu_202606181551/leaderboard.csv`
- [x] per-band metrics generated：`runs/20260618/forward_band_benchmark_gpu_202606181551/per_band_metrics.csv`
- [x] GPU runtime/memory recorded：`metrics.json` 与 `leaderboard.csv`
- [x] GPU forward report generated：`docs/results/forward_benchmark_gpu_20260618.md`
- [x] forward surrogate shortlist frozen：`docs/results/forward_surrogate_shortlist_20260618.md`

Full-2 remains pending：下一步仅建议 Full-2A Query-form Benchmark，不在本 gate 内执行。

## Gate: Formal GPU Full-2A Query-form Benchmark

- [x] query dataset and k-path audit：`runs/20260618/query_band_benchmark_gpu_202606181908`，报告 `docs/results/query_dataset_audit_20260618.md`
- [ ] physical k coordinate source verified：本轮已恢复 candidate sidecar，但尚未 5-sample MPB physical validation；见 `docs/results/k_path_provenance_audit_20260618.md`
- [x] structure-level leakage prevention verified：`integrity_summary.json` 记录 train/val/test sample_id disjoint
- [x] lazy BandQueryDataset implemented：`src/phc2d/data/query_dataset.py`
- [x] Q1 unseen_structure_all_k split：`configs/research/splits/query_unseen_structure_all_k.yaml`
- [x] Q2 k_interpolation split：`configs/research/splits/query_k_interpolation.yaml`
- [x] Q3 heldout_segment split：`configs/research/splits/query_heldout_segment.yaml`；当前数据缺 segment metadata，因此只可降级为 `contiguous_index_holdout`
- [x] mlp_query implemented and verified：`src/phc2d/models/baselines.py`，`tests/test_phc2d_query_models.py`
- [x] deep_onet_query implemented and verified：`src/phc2d/models/baselines.py`，`tests/test_phc2d_query_models.py`
- [x] fno2d_query implemented and verified：`src/phc2d/models/baselines.py`，`tests/test_phc2d_query_models.py`
- [x] query model identity tests：`tests/test_phc2d_query_models.py`
- [x] GPU query model preflight：`runs/20260618/query_model_matrix_gpu_preflight_202606181909`
- [x] formal query GPU config：`configs/research/query_band_benchmark_gpu.yaml`
- [x] seeds 0/1/2 completed：guarded V2 formal run `runs/20260619/query_band_benchmark_gpu_v2_202606191632`，27/27 cases
- [x] per-k metrics generated：`runs/20260619/query_band_benchmark_gpu_v2_202606191632/per_k_metrics.csv`
- [ ] continuity metrics generated：formal guarded V2 run currently writes empty `continuity_metrics.csv`; explicit continuity analysis still pending
- [ ] full-Wh baseline comparison generated：formal guarded V2 run currently writes empty `full_wh_baseline_comparison.csv`；Q2/Q3 frozen full-Wh comparator 仍需 masked/full-reference 边界实现
- [x] query GPU blocked report generated：`docs/results/query_benchmark_gpu_20260618.md`
- [x] query surrogate shortlist frozen：guarded V2 shortlist `docs/results/query_surrogate_shortlist_20260619.md`，`fno2d_query` ranked first across Q1/Q2/Q3

Full-2B Data Scaling 已完成最近可执行门槛：只运行当前实际存在的 guarded V2 `N=64` baseline；更大 `N` 保持 pending。Full-2C OOD 保持 pending。旧 50k physical-k recovery 仍保持 candidate/blocker 边界，不被 V2 结果回填。

## Gate: Full-2B Guarded V2 Query Data Scaling

- [x] 审计既有 data scaling 配置：旧 `configs/research/data_scaling_benchmark.yaml` 指向 `data/paper_like_50k_mpb.npz` / full-Wh，占位性质；本轮已改为 guarded V2 query path。
- [x] 实现 guarded V2 query scaling wrapper：`src/phc2d/query/scaling.py`，CLI 为 `phc2d query data-scaling --config ...`。
- [x] wrapper 测试：`tests/test_phc2d_query_data_scaling.py` 覆盖 missing exact V2 sizes、child run_dir 不继承 parent run_dir、blocked artifacts。
- [x] Full-2B parent run completed：`runs/20260619/query_data_scaling_v2_202606191735`。
- [x] Full-2B N=64 child run completed：`runs/20260619/query_data_scaling_v2_N64_202606191735`，`27/27` cases，`0` failures，CUDA `NVIDIA GeForce RTX 4060 Laptop GPU`。
- [x] N=64 leaderboard generated：`leaderboard.csv` 主排序仍为 `fno2d_query`，bands 2-10 MRE 为 `8.535197 ± 0.769185`、`9.586228 ± 0.061334`、`9.596021 ± 0.549813`。
- [x] Full-2B report generated：`docs/results/query_data_scaling_v2_20260619.md`。
- [x] N=256 guarded V2 dataset generated and physically validated：canonical generation `runs/20260620/query_dataset_v2_pilot_256_202606201232` 使用 `band_order_policy=frequency_ascending`、`workers_requested=16`、`workers_resolved=16`，`failed_count=0`；physical validation `runs/20260620/k_path_physical_validation_v2_256_202606201454` 返回 `physically_verified_with_documented_numerical_tolerance`，manifest/sidecar 已 promote。
- [x] N=256 guarded V2 target-side mismatch root-cause audit and targeted diagnostic completed：`runs/20260619/v2_smoke_mismatch_audit_202606192322` 与 `runs/20260619/v2_smoke_guard_band_diagnostic_202606192322` 已完成；修复策略为 canonical `band_order_policy=frequency_ascending` 并重生成 N=256，sample 241 / k_index 3 / band 6-7 pair max abs error 降至 `5.960464477539063e-08`，低于 accepted tolerance `9.212543833347908e-06`。
- [x] N=1000 guarded V2 dataset generated and physically validated：generation `runs/20260620/query_dataset_v2_pilot_1000_202606201751` 使用 `band_order_policy=frequency_ascending`、`workers_requested=20`、`workers_resolved=20`，`completed_count=1000`、`failed_count=0`；physical validation `runs/20260621/k_path_physical_validation_v2_1000_202606210915` 返回 `physically_verified_with_documented_numerical_tolerance`，`max_abs_error=1.1324882507324219e-06` <= `accepted_tolerance=1.2345293748383668e-05`，manifest/sidecar 已 promote；用户已明确允许本 gate 使用 `workers=20`。
- [ ] N=1000 guarded V2 query data-scaling benchmark completed：本轮已把 `data/query_dataset_v2_pilot_1000_mpb.npz` 加入 `configs/research/data_scaling_benchmark.yaml` 的 `paths_by_scale`，scale plan 正确识别 N=64/256/1000 为 runnable；但 `runs/20260621/query_data_scaling_v2_202606211009` 仅完成 N=64 child（27/27, failed_count=0），N=256 child 在 4 个 checkpoint 后因异常长 runtime 人工终止，N=1000 未开始；见 `docs/results/query_data_scaling_v2_20260621.md`。
- [x] N=1000 guarded V2 query scale-selective preflight completed：新增 `scaling.only_sizes` 支持与 `configs/research/data_scaling_benchmark_v2_1000_preflight.yaml`；run `runs/20260621/query_data_scaling_v2_1000_preflight_202606211638` 直接跳过 N=64/N=256 并执行 N=1000 child `runs/20260621/query_data_scaling_v2_1000_preflight_N1000_202606211638`，9/9 cases、failed_count=0、CUDA `NVIDIA GeForce RTX 4060 Laptop GPU`；这是 one-seed/limited preflight，不是 full 3-seed benchmark。
- [x] N=1000 guarded V2 forward full-Wh benchmark completed：`configs/research/forward_band_benchmark_v2_1000.yaml`；run `runs/20260621/forward_band_benchmark_v2_1000_202606211417`；15/15 case rows, failed_count=0, seeds 0/1/2, CUDA `NVIDIA GeForce RTX 4060 Laptop GPU`；报告 `docs/results/forward_benchmark_v2_1000_20260621.md`。
- [ ] N=1000 generative inverse pilot preflight completed：当前 `phc2d inverse sample/eval` 使用 `_MockConditionalGenerator` 与 `_MockSurrogate`，尚未接入来自 N=1000 query/forward benchmark 的 frozen surrogate checkpoint；本轮不运行 mock surrogate 结果，也不生成 replay metrics。
- [ ] N=5000 guarded V2 dataset generated and physically validated。
- [ ] N=10000 guarded V2 dataset generated and physically validated。
- [ ] N=50000 guarded V2 dataset generated and physically validated。
- [ ] continuity metrics generated：current child run still writes placeholder `continuity_metrics.csv` only。
- [ ] full-Wh baseline comparison generated：current child run still writes placeholder `full_wh_baseline_comparison.csv` only。

Full-2B current exact gate：`N=1000` guarded V2 canonical generation 与 physical validation 已通过。`N=5000/N=10000/N=50000` 尚未生成或物理验证；不要把旧 50k physical-k recovery 当作 V2 scaling 替代结果，也不要把本次 N=1000 结果扩写成更大规模验证。

## Gate: WSL MPB Parallel Workers Runtime

- [x] 串行 MPB generation baseline 证据记录：`runs/20260619/query_dataset_v2_pilot_256_202606191912`，`N=256` 串行完成，`wall_time_total_s=5354.780390006999`，`mpb_generation_time_per_sample_s=20.899588770281262`。
- [x] MPB worker policy plan written：`docs/planning/mpb_parallel_workers_plan_20260619.md`。
- [x] 公共 sample-level worker layer implemented：`src/phc2d/solvers/mpb_parallel.py`。
- [x] generation runner supports MPB workers：`scripts/run_query_dataset_v2_pilot_generation.py` 写入 `sample_plan.jsonl`、`worker_summary.json`、`worker_shards/`。
- [x] physical validation runner supports MPB workers：`scripts/run_k_path_physical_validation.py` selected samples 并行 recompute，repeat-first tolerance 单独记录。
- [x] worker unit tests added：`tests/test_phc2d_mpb_parallel.py`、`tests/test_query_dataset_v2_pilot_generation.py`、`tests/test_phc2d_k_path_physical_validation.py`。
- [x] WSL `meep_env` real MPB workers smoke passed with `workers_resolved > 1`：`runs/20260619/query_dataset_v2_pilot_64_202606192208`，`workers_resolved=2`，`sample_count=4`，`failed_count=0`，`worker_wall_time_s=241.261775253`。
- [x] N=256 guarded V2 physical validation rerun with workers：canonical validation `runs/20260620/k_path_physical_validation_v2_256_202606201454` 通过（workers requested `16`，resolved `5`，backend `subprocess`，failed_count `0`），decision 为 `physically_verified_with_documented_numerical_tolerance`，sidecar/manifest 已 promote；旧失败 run `runs/20260619/k_path_physical_validation_v2_256_202606192235` 保留为 mismatch 历史证据。

Boundary：MPB workers 是 WSL solver sample-level parallelism；neural GPU training/eval/scaling 仍由 `DeviceConfig` 管理。当前不声明 MPB GPU acceleration。

## Gate: Physical K-Path Recovery And Formal Query Unblocking

- [x] search current configs/code/runs：见 `runs/20260618/k_path_provenance_audit_202606182125/candidates.json`
- [x] search Git history：定位 generation commit `6c1b95e029441d1d8f47265f594cd87077e3663d`
- [x] authoritative source identified：`high_candidate_unverified`
- [x] dataset association verified：50k generation run + embedded NPZ config + historical generator/solver code
- [x] KPathSpec implemented：`src/phc2d/data/k_path.py`
- [x] sidecar metadata generated：`data/metadata/paper_like_square_lattice_gxm_kpath_v1.json`
- [x] k-path hash recorded：`6e91d2e0025c5caf1fa081f30c3628130f576c0b194b7fd53ceddd6278ff5e7c`
- [x] coordinate convention documented：`docs/results/k_path_provenance_audit_20260618.md`
- [x] node/segment metadata documented：`docs/results/k_path_provenance_audit_20260618.md`
- [x] duplicate physical coordinate groups detected：Gamma duplicate group `[0, 30]`
- [x] validation script implemented：`scripts/run_k_path_physical_validation.py`
- [x] validation config implemented：`configs/research/k_path_physical_validation_5sample.yaml`
- [x] 5 samples selected：`runs/20260618/k_path_physical_validation_5sample_202606182357/selected_samples.json`
- [ ] solver provenance complete：blocked by missing `eigensolver_tolerance`
- [ ] real MPB execution started：not started; script stopped before solver invocation
- [ ] real MPB execution completed：not completed
- [ ] direct alignment verified：no recomputed bands
- [ ] tolerance justified：requires explicit solver tolerance or completed repeat-run evidence
- [ ] sidecar promoted to physically_verified：not promoted; remains `candidate_unverified`
- [ ] Formal Full-2A unblocked：blocked
- [ ] 5-sample MPB validation completed：blocked, latest run `runs/20260618/k_path_physical_validation_5sample_202606182357`
- [ ] recovered path verified against stored Wh
- [x] BandQueryDataset consumes physical coordinates：`src/phc2d/data/query_dataset.py`
- [x] physical-k leakage tests pass：`tests/test_phc2d_k_path.py`
- [ ] physical-k GPU preflight passes
- [ ] Formal Full-2A unblocked
- [ ] Formal Full-2A physical query benchmark completed
- [x] physical query report generated：candidate/blocker reports in `docs/results`
- [ ] physical query shortlist frozen

## Gate: Query Dataset V2 Pilot Generation

- [x] v2 generation plan written：`docs/planning/query_dataset_v2_pilot_generation_plan_20260619.md`
- [x] v2 config implemented：`configs/research/query_dataset_v2_pilot_64.yaml`
- [x] v2 generation script implemented：`scripts/run_query_dataset_v2_pilot_generation.py`
- [x] solver provenance complete：smoke run `runs/20260619/query_dataset_v2_pilot_64_202606190032/solver_provenance.json`
- [x] real MPB generation started：smoke run `runs/20260619/query_dataset_v2_pilot_64_202606190032`
- [ ] real MPB generation completed：64-sample config 未完整生成；最新状态为 `pilot_smoke`
- [x] manifest/sidecar written：smoke run `dataset_manifest.yaml`、`k_path_sidecar.json`、`metadata/*.manifest.yaml`、`metadata/*.kpath.v1.json`
- [x] generated dataset inspected：smoke dataset `runs/20260619/query_dataset_v2_pilot_64_202606190032/query_dataset_v2_pilot_64_mpb_smoke_5.npz`
- [x] v2 smoke validation config implemented：`configs/research/k_path_physical_validation_v2_smoke.yaml`
- [x] v2 validation runner supports V2 provenance：`scripts/run_k_path_physical_validation.py`
- [x] v2 smoke validation started：`runs/20260619/k_path_physical_validation_v2_smoke_202606191452`
- [x] v2 smoke validation completed：5 samples replayed by real MPB, decision `mismatch_detected`
- [x] v2 smoke mismatch root-cause audit completed：`runs/20260619/v2_smoke_mismatch_audit_202606191514`，status `mismatch_root_cause_identified`
- [x] sample identity / sidecar / solver provenance mismatch ruled out：audit shows all hashes and solver fields match
- [x] localized sample `query_dataset_v2_pilot_64_000003` band-10 replay divergence root cause fixed：`band_guard_count=4`，diagnostic `runs/20260619/v2_smoke_guard_band_diagnostic_202606191620`
- [x] guarded V2 smoke dataset generated：`runs/20260619/query_dataset_v2_pilot_64_202606191621`
- [x] direct alignment verified：guarded validation `runs/20260619/k_path_physical_validation_v2_smoke_202606191625`，decision `physically_verified_with_documented_numerical_tolerance`
- [x] V2 smoke validation passed after fix：max_abs_error `3.550558750342603e-7` <= accepted tolerance `3.6668601799405565e-6`
- [x] fixed audit completed：`runs/20260619/v2_smoke_mismatch_audit_202606191632`，status `fixed_and_v2_smoke_validation_passed`
- [x] tolerance justified：latest accepted tolerance `3.6668601799405565e-6`
- [x] 64-sample V2 generation completed：`runs/20260619/query_dataset_v2_pilot_64_202606191559`，dataset `data/query_dataset_v2_pilot_64_mpb.npz`
- [x] initial 5-sample validation rerun on v2 dataset：completed with `mismatch_detected` before guard-band fix
- [x] 5-sample guarded validation rerun on v2 dataset：completed and passed
- [x] full V2 5-sample physical validation completed：`runs/20260619/k_path_physical_validation_v2_full_202606191628`，decision `physically_verified_with_documented_numerical_tolerance`
- [x] Formal Full-2A unblocked：V2 full physical validation passed
- [x] Formal Full-2A guarded V2 query benchmark completed：`runs/20260619/query_band_benchmark_gpu_v2_202606191632`

## Gate: B1 TE Bloch Eigenfield Data Pilot

- [x] 冻结 TE Bloch eigenfield contract、normalization、Bloch BC 与 solver provenance。
- [x] 完成单条 real-MPB preflight：`runs/20260720/field_eigen_te_preflight_202607200913`。
- [x] 实现不可变 field selection manifest、source split lineage、native-complex shard 和 resume-aware isolated MPB workers。
- [x] 完成 64 条真实 MPB field-data pilot：`runs/20260720/field_eigen_te_b1_64_202607201021`，`completed_count=64`、`failed_count=0`、train/val/test=`48/8/8`。
- [x] 审计 native-complex E/H、cell-volume L2、固定相位和完整 solver provenance：见 `docs/results/field_eigen_te_b1_64_20260720.md`。
- [x] 实现 NumPy Maxwell residual unit tests，并完成 solver reference-vs-random gate：`runs/20260720/field_eigen_te_b2_residual_202607201057`。
- [x] 完成 1,000 条真实 MPB records：`runs/20260720/field_eigen_te_b1_1000_202607201058`，作为 v1 baseline 与 B3 消融输入。

## Gate: B3 Supervised Field v2 And Adapted PINO

- [x] 生成 9,216 条 operator-style real-MPB records：256 structures、4 k/structure、bands 2–10，run 为 `runs/20260720/field_eigen_te_b3_9216_202607201608`。
- [x] 审计 source split `6912/1152/1152`、每 band 1,024 条、1,024 个多 band jobs 全部成功、normalization/provenance 零失败。
- [x] 实现 TE active-component codec、完整 E/H 重建、周期坐标、独立 conditioner、train-only log-frequency normalization 与共享相位对齐 loss。
- [x] 实现 `unet_field_v2`、`fno_field_v2`，并在 1k seed-0 消融上冻结候选；结果见 `docs/results/field_eigen_te_b3_v2_pino_20260720.md`。
- [x] 实现可微 Torch TE Bloch residual、梯度测试、预测 E/预测频率 physics loss 与 normalization loss。
- [x] 完成 9,216 数据上 data-only U-Net/FNO v2 三 seed 正式训练：run `runs/20260720/field_eigen_te_v2_baseline_9216_202607201723`，两个模型全部 seed 的 test E/H relative L2 均 `<1.0`。
- [x] 在 validation split 完成 PINO lambda sweep，并按 `1.02×` E relative L2 约束冻结 `lambda_pde=0.01`：`runs/20260721/field_eigen_te_pino_lambda_sweep_9216_202607210044`。
- [x] 完成 PINO-adapted 三 seed：`runs/20260721/field_eigen_te_pino_9216_202607210812`；residual median 降低 `37.48%`，mean E 改善 `0.99%`，通过 physics gate。
- [x] MaxwellNet 仅更新适配边界；因缺少 driven/source/PML 原任务，不创建伪 faithful 模型、不进入主表。
- [x] PINO/MaxwellNet 未通过独立 forward gate 前保持 field inverse blocked。

## Gate: B4 PINO Frequency Fidelity, Dense-k, And Inverse Admission

- [x] Implement `physics_frequency_gradient: full|detach`; residual always uses model-predicted frequency.
- [x] Implement configurable frequency/E/H weights, fixed 200 epochs, epoch-40 checkpoints, and frequency/field PDE-gradient audit.
- [x] Add B4 frequency-guard/dense-k configs and tests for 31/30 k-path support plus gradient isolation.
- [x] Start the CUDA frequency-guard run after WSL `meep_env` full-test success.
- [x] Validation P1/P2/P3 ablation, frozen-candidate selection, and three-seed test: P1 selected and manifest frozen.
- [x] 31-point dense-k and 64/96/128 evaluation: completed in `runs/20260722/field_eigen_te_b4_dense_k_gate_202607220819`.
- [x] `physics_frequency_crosscheck_ready`: evaluated and blocked by unseen-k frequency criteria.
- [x] `physics_residual_rerank_ready`: evaluated and blocked by residual-error calibration criteria.
- [x] `cross_resolution_zero_shot_pass`: evaluated and blocked by 96/128 transfer degradation.
- [x] Run the validation-only fixed-64 target-frequency residual diagnostic for frozen P4/P5: all six seed calibrations are blocked; evidence is `runs/20260727/field_eigen_te_b4_target_residual_64_202607271848` and `docs/results/field_eigen_te_b4_target_residual_64_20260727.md`.
- [ ] Train strict P6A/P6B target-frequency PDE ablations before any fixed-64 inverse adapter preflight.
- [ ] Field inverse / inverse sampling / replay: remains blocked; this gate does not authorize it.

## Hygiene

## Gate: A1 10k V2 Forward/Query Fair Baseline

- [x] Forward 原三模型完成 validation-selected `27 / 9 / 3`。
- [x] Query 三协议、三模型完成 validation-selected `81 / 27 / 9`。
- [x] Query physical interpolation 与 held-out segment 使用真实 k 坐标并保持 train/eval k 隔离。
- [x] 完成 matched plain `mlp` 残差消融，独立产物为 `9 / 3 / 1`。
- [x] 合并 Forward 验收为 `36 validation / 12 selected-LR test / 4 summary`。
- [x] 审计数据/split hash、selected-LR-only `n=3`、checkpoint lineage、CUDA FP32、有限指标和累计 runtime。
- [x] 正式报告：`docs/results/a1_forward_query_fair_v1_20260726.md`。
- [x] A1：`formal_completed`；A2 保持未升级。

- [x] WSL `meep_env`: `python -m compileall src tests` 通过。
- [x] WSL `meep_env`: `python -m pytest -q tests` 通过，当前为 70 tests。
- [x] WSL `meep_env`: `python -m phc2d.cli --help` 通过。
- [x] WSL `meep_env`: `python -m phc2d.cli plot --help` 通过。
- [x] WSL `meep_env`: `python -m phc2d.cli data --help` 通过。
- [x] WSL `meep_env`: `python -m phc2d.cli forward benchmark --help` 通过。
- [x] WSL `meep_env`: `python -m phc2d.cli data inspect --path data/tiny.npz` 通过。
- [x] WSL `meep_env`: `python -m phc2d.cli data inspect --path data/field_tiny_mpb.h5` 通过。
- [ ] 全局 `git diff --check` 通过；当前仍被既有 `runs/README.md:40 new blank line at EOF` 拦住。
