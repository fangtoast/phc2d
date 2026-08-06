<!--
  Full-cell dual-lane forward training evidence report.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 全胞双路线正向训练、交叉评估与冻结结果

| Metadata | Value |
|---|---|
| Status | Draft，待人工负责人审阅 |
| Primary layer | L3 实验与证据 |
| Report date | 2026-08-06 |
| Training environment | N40 `gpu_4090`，NVIDIA GeForce RTX 4090，PyTorch 2.1.2+cu118，FP32 |
| Training source snapshot | `c746078`；后处理补丁见 `6e15ac0` |
| Evidence status | 102/102 正式训练完成；冻结后 test 完成；threshold real-MPB 未执行 |
| Reading time | 10 minutes |

本文记录双路线全胞 Full-Wh 正向训练的正式证据，不承担逆向训练结论，也不把 threshold proxy 指标解释为真实阈值结构的 MPB 物理响应。实验协议见[全胞双路线训练与混合算力计划](../../planning/full_cell_forward_inverse_training/full_cell_dual_lane_hybrid_compute_plan.md)。

## 执行结论

- N40 完成 `84 main + 18 resolution = 102` 个正式单卡训练 run，全部为 `COMPLETED/0:0` 与 `formal_training_success`；失败、缺失和非有限值计数均为 0。
- 独立终局审计逐项检查 102 个唯一 run 目录的 9 类必需制品、checkpoint SHA-256、history 和 metrics 有限性，得到 `audited_runs=102`、`error_count=0`。
- 七个 64 分辨率主波次均完成 4 models × 3 seeds；每个波次的 validation 胜者都是 `fno2d_band_v2`。
- 仅基于三 seed validation 均值冻结三类配置：binary 选择 128 分辨率，graded-continuous 与 graded-t0.6 均保留 64 分辨率。
- 冻结后在 N40 执行 test：binary 为 `0.80394 ± 0.00705%`，graded-continuous 为 `0.21579 ± 0.00494%`，graded-t0.6 为 `1.24295 ± 0.01729%`，均为 bands 2–10 MRE 的三 seed mean ± std。
- 576/576 个交叉推理单元完成，失败数为 0；交叉结果不参与模型、阈值或分辨率选择。
- 本轮没有提交任何 M9/MPB/Meep 作业。人工负责人已明确：后续所有 MPB/Meep 物理计算只能在 M9 执行，并仍需单独授权。

## 数据与固定划分

| 路线 | HDF5 | Dataset SHA-256 | Train / val / test hash |
|---|---|---|---|
| binary | `data/full_cell_binary_bz9_10000_xingyi_v1.h5` | `1f131f2b...530c2f` | `b87806ed...a96878e` / `8b40be08...20eba1` / `f9287656...0eba1` |
| graded | `data/full_cell_graded_bz9_10000_paracloud_m9_v1.h5` | `d16d838e...b8b728` | `34f595c4...d61fd` / `2b750661...400305` / `c54196a2...b75b61` |

两套数据均使用数据内冻结的 `idx_train`、`idx_val` 和 `idx_test`。训练目标 normalization 只由源路线 train split 估计；交叉路线评估继续使用源模型 normalization，不从目标路线重新校准。

## 主训练矩阵

主矩阵覆盖 binary、graded-continuous 和 graded threshold `0.3/0.4/0.5/0.6/0.7`，每个表征训练 `mlp`、`res_mlp_wide`、`lifted_cnn`、`fno2d_band_v2` 三个 seed，共 84 runs。

| 波次 | 表征 | Validation 胜者 | MRE mean ± std (%) |
|---|---|---|---:|
| binary | binary-native | `fno2d_band_v2` | 0.81305 ± 0.00214 |
| continuous | graded continuous | `fno2d_band_v2` | 0.22039 ± 0.00571 |
| t0.3 | deterministic threshold | `fno2d_band_v2` | 1.64805 ± 0.00953 |
| t0.4 | deterministic threshold | `fno2d_band_v2` | 1.51491 ± 0.00407 |
| t0.5 | deterministic threshold | `fno2d_band_v2` | 1.37851 ± 0.00776 |
| t0.6 | deterministic threshold | `fno2d_band_v2` | 1.23493 ± 0.00772 |
| t0.7 | deterministic threshold | `fno2d_band_v2` | 1.26180 ± 0.00704 |

主矩阵机器结论位于 `runs/20260806/full_cell_forward_main_campaign_20260806/campaign.meta.json`，其状态为 `main_84_training_complete_validation_winners_frozen`。选择阶段没有执行 test。

## 交叉评估

交叉矩阵由 graded 内部 6×6 表征迁移的 432 cells 与 binary ↔ graded 的 144 cells 组成：

| 项目 | 结果 |
|---|---:|
| Planned / completed cells | 576 / 576 |
| Failed cells | 0 |
| Total inference time | 109.23 s |
| Total wall time | 119.34 s |
| Selection influence | `false` |

原始 cell、三 seed 汇总和 6×6 热图位于 `runs/20260806/full_cell_forward_cross_eval_20260806/`。这些数据只属于表征迁移和 OOD/generalization 诊断，不能替代同路线 validation，也不能升级为 real-MPB replay。

## 分辨率消融与最终冻结

18 个消融 run 只针对 validation 已冻结的三类 FNO 配置，比较 32 与 128，并与主矩阵 64 共同排序。

| 路线 / 表征 | r32 MRE (%) | r64 MRE (%) | r128 MRE (%) | 冻结分辨率 |
|---|---:|---:|---:|---:|
| binary / binary | 0.84461 ± 0.00503 | 0.81305 ± 0.00214 | **0.81154 ± 0.00313** | 128 |
| graded / continuous | 0.22289 ± 0.00737 | **0.22039 ± 0.00571** | 0.22116 ± 0.01068 | 64 |
| graded / t0.6 | 1.25241 ± 0.00880 | **1.23493 ± 0.00772** | 1.23497 ± 0.00997 | 64 |

`runs/20260806/full_cell_forward_resolution_acceptance_20260806/acceptance.meta.json` 记录 `all_102_forward_training_runs_complete_final_resolutions_frozen`、`failed_runs=0`、`missing_runs=0` 和 `test_executed_during_selection=false`。

## 冻结后 test

| 路线 / 表征 | Model | Resolution | Test MRE mean ± std (%) | MAE mean | RMSE mean |
|---|---|---:|---:|---:|---:|
| binary / binary | `fno2d_band_v2` | 128 | 0.80394 ± 0.00705 | 0.004236 | 0.006929 |
| graded / continuous | `fno2d_band_v2` | 64 | 0.21579 ± 0.00494 | 0.001084 | 0.001778 |
| graded / t0.6 | `fno2d_band_v2` | 64 | 1.24295 ± 0.01729 | 0.006043 | 0.008274 |

最终测试保存 9 行 seed 指标、90 行 per-band、729 行 per-k、3 份紧凑 prediction NPZ 和 4 行 bootstrap 结果。graded-continuous 减 graded-t0.6 的 paired bootstrap MRE 差为 `-1.02716` 个百分点，95% CI 为 `[-1.07225, -0.98238]`。这说明在当前 graded 原始 `Wh` 代理目标上 continuous 明显优于 t0.6 proxy；它不说明阈值后二值结构具有相同的真实 MPB 响应。

云端最终测试状态为 `final_test_evaluation_success`，位于 `runs/20260806/full_cell_forward_final_test_20260806/`。最终测试脚本与作业脚本的远端 SHA-256 分别为 `b95c2306...e3d380` 和 `bd42e426...68af0`，与本地 `6e15ac0` 内容一致。由于静态 archive 部署不更新远端 `.git`，`run.meta.json` 中自动探测到的 `git_commit=c746078` 不能单独代表后处理文件版本；此差异保留为 provenance 注记。

## 异常、恢复与排除项

1. 消融 admission probe 一次性提交 18 个作业时触发 `AssocMaxSubmitJobLimit`，报错前已有 15 个被接受。审计后只补交缺失的 3 个，最终 18/18 均为 `full_cell_forward_admission_probe_pass`；没有重复训练。
2. 正式消融按计划允许的 `12 + 6` 节流运行，最终合并清单严格包含 18 行、每行 10 个字段。
3. 曾启动本机 WSL checkpoint CPU 推理复核。人工负责人要求停止所有本地 replay 后，相关进程已终止；该目录只有两个不完整 prediction 文件，没有 `results.json`，不进入任何数值结论。此任务不是 MPB，但仍按人工边界排除。
4. 没有失败训练 run；没有自动重启、取消或恢复正式训练作业。

## 物理证据边界与下一 gate

- t0.3–t0.7 结果是固定 raster 表征输入下对原 graded `Wh` 标签的 proxy 训练，不是阈值结构重新求解后的 `Wh_threshold_MPB`。
- 本轮没有执行 threshold analytic codec、M9 preflight、M9 pilot 或 real-MPB replay。
- 所有后续 MPB/Meep 计算只能提交到 M9；是否执行 40-structure preflight 或 320-structure pilot，必须由人工负责人另行授权。
- Forward 冻结结果可作为 full-cell inverse 的候选 surrogate 输入，但能否准入 Stage B 仍需人工审阅本 Draft、确认 checkpoint 角色和接受 provenance 注记。

## 人工理解记录

解决了什么问题：完成双路线全胞正向代理的 102-run 公平矩阵、交叉评估、分辨率冻结和冻结后 test。

为什么采用这个方案：模型 × seed 独占单卡，validation-only 冻结，test 延后执行；这样并发只影响墙钟时间，不改变训练协议或选择口径。

改变了项目哪一层：L3 实验与证据；L0/L1 研究目标和 M9 物理验证规则未改变。

使用什么证据验证：102-run manifests、每 run 核心制品、7 个 wave aggregation、84-run campaign、576-cell cross evaluation、18-run resolution acceptance 和冻结后 9-checkpoint test。

最大风险、未知或下一 gate：threshold proxy 尚无 M9 real-MPB 物理验证；静态部署的 `.git` 元数据不能完全表达后处理补丁；人工负责人需决定是否接受 Forward 冻结结果并授权 inverse 或 M9 的下一阶段。
