---
title: EXP-FWD-022 分支依赖与独立能力
summary: 已完成冻结 validation 分支关断；记录域相关依赖及尺度控制差异，独立训练能力证据待补充。
page_type: evidence
status: draft
audience: 导师、正向模型研究者与结果审阅者
owner: 研究生维护，导师审阅科学主张
last_reviewed: 2026-09-17
date: 2026-09-17
scope: frozen_validation_branch_dependency_and_pending_independent_capability
reading_time: 8 minutes
primary_layer: L3
evidence_date: 2026-09-16
evidence_scope: frozen_validation_branch_dependency_only
dataset_scope: full_cell_k5_k20_f50_binary_graded_validation_wh81x10
evidence_level: surrogate_held_out
evidence_status: completed
validation_backend: surrogate_only
completion_qualification: validation_only_reuses_checkpoint_selection_split_not_independent_held_out_test
source_artifacts:
  - runs/20260916/EXP-FWD-022_g1_validation_202609161602
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

本页保持 Draft。2026-09-17 用户授权进入独立训练；训练完成后在第六节补充能力证据，不改写本次冻结干预结果。

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

## 六、独立训练能力：等待新证据

2026-09-17 用户明确授权训练，并批准参数量控制在 width-only 不可行时额外调整预测 head。固定 Binary/Graded 分开 K5+K20 等权 mixed train/validation，kernel 3×3；Full、G-only、L-only、G-only 参数量控制、L-only 参数量控制 × 三 seed，共 30 个从零训练任务。参数量目标仍为 Full ±5%，不是 architecture-equivalent control。旧 EXP-FWD-020 checkpoint 不作公平训练基线。

| 后续需补充 | 准入/证据要求 | 当前科学状态 |
|---|---|---|
| 训练性能 | 统一合同、相同 epoch/batch/optimizer；validation 选 checkpoint，保留收敛状态 | 未有结果 |
| 单分支能力 | 分别报告 K5/K20 validation 与等权 mixed MRE | 未有结果 |
| 组合收益 | B_GL=min(E_G,E_L)−E_Full；参数量控制同理；配对 bootstrap 和 seed 方向 | 未有结果 |
| 依赖与能力解释 | 两层结论分列，不用 G1 直接推断单支能力 | 未有结果 |

**参数量与 GPU 准入已通过，第一批正式训练已提交；尚无终态能力结果。** 原 G-only width=64 为 −7.87%，width=72 为 +16.28%，不满足 ±5%。保持固定搜索和 tolerance，优先选择最近 width，仅在失败时启用用户批准的可调隐藏层 dense head。

| 训练变体 | Trunk width | Dense head | 参数量 | 相对 Full |
|---|---:|---|---:|---:|
| Full | 64 | 原单层线性 | 2,571,978 | 基线 |
| G-only | 64 | 原单层线性 | 2,369,610 | −7.87% |
| L-only | 64 | 原单层线性 | 256,170 | −90.04% |
| G-only 参数量控制 | 64 | 隐藏层 291、GELU | 2,572,395 | +0.02% |
| L-only 参数量控制 | 224 | 原单层线性 | 2,614,890 | +1.67% |

上述参数量由同一公开 factory 的工程检查给出，不是性能证据。按现有训练器 `sum(p.numel())` 计数，spectral 的复数 core 每个元素计一，因此不是实数自由度匹配。旧 `head_hidden_dim` 仅作用于 k-conditioned head；新显式 `dense_head_hidden_dim` 只在 G-only 参数量控制中启用，普通三组与默认 checkpoint schema 保持不变。G-only 对照因此同时改变 head 结构，L-only 对照改变特征维度；二者只控制此口径的参数总量，不能证明架构等价或无优化混杂。

每条 lane 使用 16,000 train、2,000 validation；不加载 test 标签。统一从零训练 200 epoch、AdamW、learning rate 5×10⁻⁴、weight decay 10⁻⁵、FP32、micro batch 8、effective batch 64、无 early stopping。按 validation bands 2–10 MRE 选择 checkpoint，并保留 epoch200 与末 20 epoch 收敛检查。先完成十种 lane×结构的单 batch GPU 准入，再推进 30-run；N40 在途任务上限为 16，故两批各 15 个、最多并发 3 个，前批 Slurm 完成且终态审计通过后才提交后批；失败不自动重试或改资源。

首次冻结产物在 N40 `/HOME/scw6g7a/run/phc2d/runs/20260917/EXP-FWD-022_capability_202609171224/`：30-run manifest、两条 lane mixed 数据及来源 SHA 已生成，执行起始 commit 为 `56c1e0f`。`submission_preflight.json` 保留失败意图，未获得 job ID。平台诊断明确拒绝显式 `--mem=60000`，要求单卡默认 60GB；不带 `--mem` 的 `sbatch --test-only` 检查通过，但这不是作业或 GPU 准入通过。用户随后明确授权修正并重提，使用新的授权标识与运行目录，保留原失败记录；移除显式内存参数，使用分区默认每卡 60GB。科学合同不变，不把提交检查当成训练结果。

修正后的 campaign 为 N40 `/HOME/scw6g7a/run/phc2d/runs/20260917/EXP-FWD-022_capability_202609171226/`，部署 commit `f4f1625`，GPU 准入 job `5461528` 已十种结构全部通过且 Slurm `COMPLETED/0:0`。第一批 Binary 15-run 已提交数组 `5461531`；第二批 Graded 15-run 尚未提交，需等待前批全部完成和终态审计。30 分钟监控已启用，按 gate 推进两批训练并在全部终态审计后补充本节。原始定位文件为 `preflight_manifest.csv`、`submission_training_batch_0.json`、`campaign_status.json`，本地小型证据镜像位于仓库相同 `runs/20260917/` 路径。

训练结束后仅用新训练证据更新本节与摘要；不回填预测值、不改写 G1 原始指标。实验协议位于仓库 `experiments/research/EXP-FWD-022-全局局部分支归因.md`，独立训练配置草案位于 `configs/research/full_cell_branch_capability_exp_fwd_022_v1.yaml`。
