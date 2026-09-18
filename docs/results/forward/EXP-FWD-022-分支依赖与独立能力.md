---
title: EXP-FWD-022 分支依赖与独立能力
summary: 冻结分支关断与30项独立训练已完成；固定200 epoch预算下Full优于单支及参数量控制，但仅11项通过收敛门，待人工审阅。
page_type: evidence
status: draft
audience: 导师、正向模型研究者与结果审阅者
owner: 研究生维护，导师审阅科学主张
last_reviewed: 2026-09-18
date: 2026-09-18
scope: frozen_validation_branch_dependency_and_fixed_budget_independent_capability
reading_time: 12 minutes
primary_layer: L3
evidence_date: 2026-09-18
evidence_scope: frozen_validation_dependency_and_k5_k20_lane_separated_200_epoch_capability
dataset_scope: full_cell_k5_k20_f50_binary_graded_validation_wh81x10
evidence_level: surrogate_held_out
evidence_status: completed
validation_backend: surrogate_only
completion_qualification: validation_only_checkpoint_selection_split_capability_11_of_30_converged_pending_human_review_not_test_confirmed
source_artifacts:
  - runs/20260916/EXP-FWD-022_g1_validation_202609161602
  - runs/20260917/EXP-FWD-022_capability_202609171226
---

<!--
  EXP-FWD-022 frozen branch dependency and independent capability results.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="exp-fwd-022-results"></a>

# EXP-FWD-022：分支依赖与独立能力

**结论：同域 Full 模型关闭任一分支均显著退化；Binary 的 Local 关断损失更大，K20 Graded 的 Global 关断损失更大。该证据只回答共同训练模型的当前依赖，不回答单分支独立能力。** 部分跨域关断反而改善；内部一阶 attribution 与关断排序并不完全一致。

**独立能力补充：固定 200 epoch 预算下，Binary/Graded 两条 K5+K20 mixed lane 的 Full 均优于普通单支及参数量控制，三个 seed 方向一致。** 但仅 11/30 通过末段收敛门，不能写成充分优化后的能力上限。本页保持 Draft，等待人工审阅；第一至五节仍为 2026-09-16 固化的 G1 证据，第六节单独记录 2026-09-18 完成的独立训练，不改写 G1 原始指标。

**目录：** [证据范围](#scope) · [同域结果](#native) · [尺度控制](#scale) · [跨域与内部归因](#attribution) · [结论边界](#limits) · [独立训练](#capability)

<a id="scope"></a>

## 一、证据范围与统计单位

| 项目 | 已核实状态 |
|---|---|
| 上游模型 | EXP-FWD-020 已训练的 36 个 checkpoint；6 source domain × 2 kernel × 3 seed |
| 干预 | Full、G-off raw、L-off raw、G-off renorm、L-off renorm；参数冻结，不再训练 |
| 评估 | 6 target validation，各 1,000 条；216 checkpoint-target、1,080 intervention cells |
| 配对统计 | 864 条比较，10,000 次 sample-level paired bootstrap |
| 内部 attribution | 每个 checkpoint-target 随机 512 主集与结构分层 512 探索集；共 5,184 分支层记录 |
| 工程终态 | 8/8 分片及 finalizer 均 COMPLETED/0:0；1,080 唯一身份、主指标有限 |
| 未执行 | test、独立训练能力评估、real-MPB/Meep；不作 promotion |

以下为三个 seed 的算术均值。MRE 为 Bands 2–10，单位 `%`；误差差值为百分点（pp）。样本 bootstrap CI 不衡量跨训练 seed 不确定性，seed 方向另行检查。kernel 指 Local kernel；K5/K20 是 Fourier cutoff，F50 是历史 50k 数据合同，不是 K50。

同域 validation 曾用于上游 checkpoint 选择，因此这不是独立 held-out test。元数据 evidence_level 使用站点既有 surrogate 层枚举，实际证据由 evidence_scope 与 completion_qualification 限定为 validation-only；不得按枚举名升级 test-confirmed 结论。

<a id="native"></a>

## 二、同域干预结果

| domain | kernel | Full MRE | G-off renorm | L-off renorm | I_G ctrl（pp） | I_L ctrl（pp） |
|---|---|---:|---:|---:|---:|---:|
| K5 Binary | 3×3 | 0.558 | 3.291 | 9.758 | 2.734 | 9.201 |
| K5 Binary | 7×7 | 0.569 | 3.553 | 10.940 | 2.984 | 10.371 |
| K5 Graded | 3×3 | 0.407 | 5.506 | 4.904 | 5.098 | 4.496 |
| K5 Graded | 7×7 | 0.402 | 4.036 | 5.688 | 3.634 | 5.286 |
| K20 Binary | 3×3 | 0.198 | 2.774 | 10.740 | 2.576 | 10.542 |
| K20 Binary | 7×7 | 0.196 | 3.699 | 10.721 | 3.503 | 10.525 |
| K20 Graded | 3×3 | 0.130 | 9.209 | 4.919 | 9.079 | 4.789 |
| K20 Graded | 7×7 | 0.148 | 9.002 | 4.890 | 8.854 | 4.741 |
| F50 Binary | 3×3 | 0.392 | 3.527 | 13.704 | 3.136 | 13.312 |
| F50 Binary | 7×7 | 0.391 | 3.099 | 16.055 | 2.708 | 15.664 |
| F50 Graded | 3×3 | 0.097 | 3.037 | 3.480 | 2.940 | 3.383 |
| F50 Graded | 7×7 | 0.095 | 2.888 | 3.920 | 2.793 | 3.825 |

每行两种 renorm 关断在三个 seed 均退化，各 seed 的 95% CI 均不跨 0。G/L 大小排序是描述性比较，尚未直接检验 I_G−I_L；K5 Graded 排序随 kernel 改变，不使用统一主导标签。Graded 从 K5 到 K20 的 Global 关断损失增大，两 kernel 一致；Binary 不呈现相同趋势。只有两档 K 且 checkpoint 分别训练，不支持单调复杂度机制。

依据：第五节所列 G1 原始目录中的 `intervention_cells.csv`、`branch_dependency_metrics.csv`、`paired_bootstrap.csv`。

<a id="scale"></a>

## 三、尺度控制与输出扰动

Full update=(G+L)/2；G-off raw=L/2，G-off renorm=L；Local 对称。I ctrl=MRE off,renorm−MRE Full，S=I raw−I ctrl。3×3 结果如下：

| domain | G-off raw MRE | L-off raw MRE | S_G（pp） | S_L（pp） | D_G ×100% | D_L ×100% |
|---|---:|---:|---:|---:|---:|---:|
| K5 Binary | 2.901 | 10.176 | −0.390 | 0.418 | 3.65 | 11.60 |
| K5 Graded | 2.766 | 4.954 | −2.740 | 0.050 | 5.93 | 5.94 |
| K20 Binary | 0.498 | 10.720 | −2.276 | −0.020 | 2.99 | 11.72 |
| K20 Graded | 0.447 | 4.986 | −8.762 | 0.067 | 9.11 | 5.30 |
| F50 Binary | 3.243 | 12.054 | −0.284 | −1.649 | 4.26 | 16.08 |
| F50 Graded | 3.163 | 2.910 | 0.126 | −0.569 | 3.76 | 4.60 |

D 是各样本 renorm 预测变化的相对 Frobenius 范数均值，不是 MRE。K20 Graded 的 G-off raw→renorm 为 0.447%→9.209%；7×7 同样为 0.344%→9.002%。**renorm 控制的是融合系数，并未匹配实际 residual RMS**；可能改变剩余分支与后续非线性的相互作用。因此 I ctrl 不称“纯信息贡献”，S 不称严格因果尺度分解。

<a id="attribution"></a>

## 四、跨域及内部 attribution

下列改善在两种 kernel、三个 seed 中均成立，各 seed CI 均不跨 0：

| source→target | 关断 | 3×3 ΔMRE（pp） | 7×7 ΔMRE（pp） |
|---|---|---:|---:|
| K5 Graded→K20 Binary | G-off renorm | −1.505 | −0.774 |
| K20 Binary→K5 Graded | G-off renorm | −1.300 | −1.548 |
| F50 Graded→K20 Graded | L-off renorm | −0.844 | −0.858 |

这是分支更新在特定迁移目标可能不利的线索，不证明该分支整体无用。完整 source→target 结果保存在 dependency CSV，不与同域平均混合。

随机 512 主集的 Global saliency share R_C，各层及各 seed 的算术均值：

| 同域 domain | 3×3 | 7×7 |
|---|---:|---:|
| K5 Binary | 53.2% | 48.2% |
| K5 Graded | 51.7% | 50.0% |
| K20 Binary | 35.9% | 34.3% |
| K20 Graded | 33.1% | 36.8% |
| F50 Binary | 58.5% | 61.2% |
| F50 Graded | 58.3% | 57.6% |

R_C 基于归一化全矩阵 MSE 的 absolute Gradient×Activation，不是 MRE 的归因比例，也不是信息占比。K20 Graded 与 F50 Binary 的内部比例和关断排序不一致；局部一阶敏感度不能替代有限幅度干预。activation、各层 C、结构分层探索集见 G1 原始目录的 `branch_internal_metrics.csv`。

<a id="limits"></a>

## 五、尚未形成的结论

per-band 与 per-k 产物已生成，但空间模式的跨 seed 稳定性及 BH-FDR 尚未完成，本页不作 band/k 显著机制主张。G1 不升级 EXP-FWD-020 的收敛状态；没有物理因果或独立能力结论。

原始证据在 N40 `/HOME/scw6g7a/run/phc2d/runs/20260916/EXP-FWD-022_g1_validation_202609161602/`；本地镜像为仓库 `runs/20260916/EXP-FWD-022_g1_validation_202609161602/`，终态文件 `results.json`、`metrics.json`。原始产物不发布到文档站，路径是证据定位信息而非公开下载链接。失败 G0 作业 5460192 保留；修正执行使用 efc8309，G0=5460397、G1=5460685、finalizer=5460686。

<a id="capability"></a>

## 六、独立训练能力：固定预算下 Full 领先，收敛门未闭合

**工程终态已完成，科学状态为待人工审阅的固定预算 validation 能力证据。** N40 的 30/30 训练任务与 finalizer 均 `COMPLETED/0:0`；每项完成 200 epoch，配置、身份、参数量、checkpoint SHA 与终态审计通过。汇总表有 30 个唯一 case、24 条配对比较、12 条组合收益，主指标均有限；仅 11/30 通过预先固定的收敛门，未自动延长训练。

### 6.1 统一训练合同与参数量控制

2026-09-17 用户授权从零训练与 head fallback。Binary/Graded 分开，各用 K5+K20 等权 mixed：每域 8,000 train、1,000 validation，每 lane 共 16,000 train、2,000 validation；两种材料 lane 不混合，不加载 test 标签。固定 kernel 3×3、三 seed、200 epoch、AdamW、learning rate 5×10⁻⁴、weight decay 10⁻⁵、FP32、micro batch 8、effective batch 64、无 early stopping。按 validation Bands 2–10 MRE 选择 best checkpoint，另保留 epoch200。

参数量目标仍为 Full ±5%。G-only width-only 的相邻候选无法满足该门，因此启用获批的 dense head；旧 EXP-FWD-020 checkpoint 不作本合同的公平训练基线。

| 训练变体 | Trunk width | Dense head | 参数量 | 相对 Full |
|---|---:|---|---:|---:|
| Full | 64 | 原单层线性 | 2,571,978 | 基线 |
| G-only | 64 | 原单层线性 | 2,369,610 | −7.87% |
| L-only | 64 | 原单层线性 | 256,170 | −90.04% |
| G-only 参数量控制 | 64 | 隐藏层 291、GELU | 2,572,395 | +0.02% |
| L-only 参数量控制 | 224 | 原单层线性 | 2,614,890 | +1.67% |

参数量经同一公开 factory 与训练终态核验。按现有训练器 `sum(p.numel())` 计数，spectral 复数 core 每个元素计一，不是实数自由度匹配。`dense_head_hidden_dim` 仅在 G-only 参数量控制中启用。G-only 对照改变 head，L-only 对照改变特征维度；二者是 **parameter-count control，不是 architecture-equivalent control**，不能消除优化难度混杂。

### 6.2 独立训练性能

以下来自 finalizer 对各 best-validation checkpoint 的统一重评。MRE 为 Bands 2–10、81 个 k 点的样本平均相对误差，单位 `%`；mixed 为 K5/K20 等权平均。报告三 seed **均值 ± 样本标准差（ddof=1）**，不是均值 CI。

| lane | 训练变体 | mixed MRE（%） | K5 MRE（%） | K20 MRE（%） | 收敛通过 |
|---|---|---:|---:|---:|---:|
| Binary | Full | 0.3804 ± 0.0037 | 0.5512 ± 0.0051 | 0.2095 ± 0.0035 | 1/3 |
| Binary | G-only | 0.5375 ± 0.0016 | 0.6523 ± 0.0039 | 0.4226 ± 0.0010 | 2/3 |
| Binary | L-only | 0.6601 ± 0.0032 | 1.0278 ± 0.0068 | 0.2923 ± 0.0018 | 3/3 |
| Binary | G-only 参数量控制 | 0.4856 ± 0.0082 | 0.5719 ± 0.0124 | 0.3994 ± 0.0044 | 0/3 |
| Binary | L-only 参数量控制 | 0.6568 ± 0.0023 | 1.0190 ± 0.0025 | 0.2945 ± 0.0069 | 1/3 |
| Graded | Full | 0.2737 ± 0.0049 | 0.3969 ± 0.0065 | 0.1505 ± 0.0060 | 0/3 |
| Graded | G-only | 0.4057 ± 0.0080 | 0.4723 ± 0.0147 | 0.3391 ± 0.0022 | 1/3 |
| Graded | L-only | 0.5505 ± 0.0011 | 0.8704 ± 0.0013 | 0.2306 ± 0.0010 | 1/3 |
| Graded | G-only 参数量控制 | 0.3625 ± 0.0070 | 0.4039 ± 0.0052 | 0.3211 ± 0.0089 | 1/3 |
| Graded | L-only 参数量控制 | 0.5468 ± 0.0004 | 0.8601 ± 0.0036 | 0.2335 ± 0.0030 | 1/3 |

证据：本节末所列 capability 目录的 `capability_metrics.csv`；每项 `runs/{case_id}/results.json` 与 `history.csv` 保存选择 epoch、完整训练历史和收敛状态。

**不同 K 下的单支能力排序不同。** 两 lane 中 G-only 在 K5 的描述性均值优于 L-only，而 L-only 在 K20 优于 G-only，普通和参数量控制均呈现这一模式。它是统一 mixed 训练下的目标域性能差异，不是 G1 的关断依赖，也不能据此推断随 K 单调变化的因果机制。Graded K5 的 G-only 参数量控制与 Full 均值接近，但 seed 0 方向相反；尚无该域配对检验或等价门，不能称“独立充分”。

### 6.3 配对比较与组合收益

下表为同 lane、同 seed、同 validation 样本的 $\Delta E=E_{single}-E_{Full}$，单位 pp；方括号为 10,000 次 sample-level paired bootstrap 的 95% percentile CI。正值表示单支更差。重采样在 2,000 条 mixed validation 样本池内进行，未按 K 分层；CI 条件于已选 checkpoint，不包含重新训练、checkpoint 选择或跨 seed 不确定性。

| lane | 对照 Full 的变体 | seed 0：Δ [95% CI] | seed 1：Δ [95% CI] | seed 2：Δ [95% CI] |
|---|---|---:|---:|---:|
| Binary | G-only | 0.1579 [0.1522, 0.1638] | 0.1543 [0.1484, 0.1603] | 0.1590 [0.1533, 0.1646] |
| Binary | L-only | 0.2834 [0.2730, 0.2943] | 0.2718 [0.2615, 0.2825] | 0.2838 [0.2730, 0.2943] |
| Binary | G-only 参数量控制 | 0.0995 [0.0931, 0.1061] | 0.1012 [0.0949, 0.1077] | 0.1150 [0.1085, 0.1216] |
| Binary | L-only 参数量控制 | 0.2766 [0.2658, 0.2871] | 0.2742 [0.2638, 0.2848] | 0.2785 [0.2683, 0.2887] |
| Graded | G-only | 0.1298 [0.1252, 0.1345] | 0.1364 [0.1320, 0.1407] | 0.1297 [0.1252, 0.1342] |
| Graded | L-only | 0.2754 [0.2659, 0.2848] | 0.2740 [0.2640, 0.2838] | 0.2811 [0.2714, 0.2909] |
| Graded | G-only 参数量控制 | 0.0791 [0.0738, 0.0842] | 0.0894 [0.0841, 0.0950] | 0.0979 [0.0924, 0.1034] |
| Graded | L-only 参数量控制 | 0.2718 [0.2626, 0.2811] | 0.2693 [0.2600, 0.2788] | 0.2782 [0.2688, 0.2876] |

24 条比较的 CI 均高于 0，每种 lane×变体的三个 seed 方向一致。相对 Full 的误差增加均超过 1% practical threshold，最小为 26.3%；该阈值与统计方向分开报告，不用它替代 CI。依据 `paired_bootstrap.csv`，仅针对 mixed 主指标，不将其 CI 套用于 K5/K20 子域。

定义 $B_{GL}=\min(E_G,E_L)-E_{Full}$，参数量控制对应 $B_{GL}^{cap}$。每个 seed 内先计算 min，再汇总；以下单位 pp。

| lane | 指标 | seed 0 | seed 1 | seed 2 | 三 seed 均值 ± SD |
|---|---|---:|---:|---:|---:|
| Binary | B_GL | 0.1579 | 0.1543 | 0.1590 | 0.1571 ± 0.0025 |
| Binary | B_GL cap | 0.0995 | 0.1012 | 0.1150 | 0.1052 ± 0.0085 |
| Graded | B_GL | 0.1298 | 0.1364 | 0.1297 | 0.1320 ± 0.0039 |
| Graded | B_GL cap | 0.0791 | 0.0894 | 0.0979 | 0.0888 ± 0.0094 |

所有 seed 的两类组合收益均为正；本次 mixed 指标中较好的单支均为 G-only。`combination_benefits.csv` 未生成 min 统计量的 bootstrap CI，上表 SD 不是该 CI；统计支持来自前表各单支对 Full 的配对比较，各 CI 是逐项区间，不是联合置信区域。普通与参数量控制均落后，表明**本预算下组合优势不能简单归因于 Full 参数总量更多**，不证明充分优化后的架构优越性。

### 6.4 收敛与解释边界

| 证据层 | 支持的结论 | 仍不支持的结论 |
|---|---|---|
| G1 冻结关断 | 已共同训练的 Full 对分支的域相关依赖与系数控制敏感度 | 单支从零训练能力、纯信息因果分解 |
| 本次独立训练 | 固定合同和 200 epoch 下，Full 的 mixed validation 性能一致优于四种单支；K5/K20 单支排序不同 | 单支理论能力不足、无限预算上限、等价或独立充分 |
| 参数量控制 | Full 领先不只发生在参数量较少的普通单支中 | 架构等价、优化难度完全可比 |
| 收敛门 | Binary 7/15、Graded 4/15 通过；合计 11/30 | 全组收敛、能力确认 gate 全部通过 |

收敛定义保持不变：epochs 181–190 与 191–200 的 validation MRE 窗口均值相对变化不超过 0.5%。未通过是科学资格限制，不是 Slurm 失败；通过也只是该局部窗口标准，不代表全局最优。尤其 Full 仅 Binary 1/3、Graded 0/3 通过，因此本页只称“固定预算优势”，不称充分优化后的稳定能力结论。validation 同时用于 checkpoint 选择和重评；没有独立 test、物理验证或 model promotion。

### 6.5 原始证据与下一决策

有效 campaign 在 N40 `/HOME/scw6g7a/run/phc2d/runs/20260917/EXP-FWD-022_capability_202609171226/`，部署 commit `f4f1625aad3d70654ccd1167c89e7fc40a5eb139`。GPU 准入 `5461528`、Binary 数组 `5461531`、Graded 数组 `5462590`、finalizer `5462591` 均已完成且 exit 0。两批训练使用默认每卡 60GB、单 GPU/6 CPU、最多并发 3；没有改变科学合同。

本地镜像为仓库相同 `runs/20260917/EXP-FWD-022_capability_202609171226/`：224 个小型 JSON/CSV/YAML 文件已与 N40 SHA256 核对，不含数据集或 checkpoint。定位文件为 `freeze.json`、`task_manifest.csv`、`config.resolved.yaml`、`mixed_data_manifest.json`、`results.json`，三张结果表及每项训练终态文件。原始产物不发布到文档站。旧 `EXP-FWD-022_capability_202609171224` 的提交失败记录与部署 `56c1e0f` 保留为修正 lineage；原显式内存参数被平台拒绝，不计为训练结果。

实验协议为仓库 `experiments/research/EXP-FWD-022-全局局部分支归因.md`，训练配置为 `configs/research/full_cell_branch_capability_exp_fwd_022_v1.yaml`。**下一 gate 是人工审阅此固定预算结果与未收敛限制**：若需要更强的能力确认，应另行批准统一的后续训练合同；不得据此自动续训、修改门槛、评估 test 或升级物理主张。
