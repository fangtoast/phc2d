---
title: EXP-FWD-019：FNO、TFNO与ConvNeXt组件归因三种子确认
summary: 汇总C0、seed0筛选和42项三种子validation-only确认，区分稳定删除敏感性、容量混杂、收敛缺口与尚未执行的六域评估。
page_type: evidence
status: draft
audience: 正向代理研究者、模型实现者与结果审阅者
owner: 研究生维护，导师审阅组件主张
last_reviewed: 2026-09-08
reading_time: 9 minutes
primary_layer: L3
evidence_date: 2026-09-08
evidence_level: surrogate_held_out
evidence_scope: "EXP-FWD-019 C0 frozen diagnostics, 21-case seed0 screen lineage, and 42-case F50 validation-only three-seed confirmation"
dataset_scope: full_cell_f50_binary_graded_wh81x10_validation_only
evidence_status: completed
validation_backend: surrogate_only
completion_qualification: three_seed_confirmation_complete_but_23_of_42_terminal_convergence_not_met_and_six_domain_evaluation_pending
source_artifacts:
  - runs/20260902/EXP-FWD-019_C0_accepted_20260902173322
  - runs/20260905/EXP-FWD-019_seed0_accepted_20260905201533
  - runs/20260905/EXP-FWD-019_confirmation_20260905201622
---

<!--
  Curated EXP-FWD-019 component-attribution confirmation result.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="exp-fwd-019-result"></a>

# EXP-FWD-019：FNO、TFNO与ConvNeXt组件归因三种子确认

**结论：**42项200轮validation-only训练全部完成并保留完整标准制品。删除FNO primary spectral、Joint spectral或把ConvNeXt卷积核从$7\times7$改为$1\times1$，都在Binary/Graded三个种子上稳定增大Bands 2--10 validation MRE；其中ConvNeXt结果最接近“局部大核有贡献”的干净证据，因为参数量只减少3.21%。FNO与Joint虽然退化幅度很大，但参数量分别减少89.73%和85.65%，当前只能证明**该频谱分支及其容量组合不可删**，不能把全部差异单独归因给频谱计算。

TFNO `no_channel`只在Binary三个种子稳定退化；Graded的配对相对变化为$-1.07\%/-0.49\%/+2.24\%$，方向不一致且seed0反向改善超过1%，因此未通过三种子通用贡献门。整个确认矩阵有23/42项终态为`training_complete_convergence_not_met`，且尚未执行K5/K20/F50六域冻结评估、held-out test或real-MPB/Meep。故本实验状态是**三种子确认完成、组件归因部分成立、整体计划仍未完成**。

**目录：** [证据状态](#evidence-status) · [实验合同](#experiment-contract) · [漏斗来源](#funnel-lineage) · [主结果](#confirmation-results) · [辅助指标](#secondary-metrics) · [资源](#resource-summary) · [审计](#evidence-audit) · [解释](#attribution-interpretation) · [决策](#next-gate)

---

<a id="evidence-status"></a>

## 一、证据状态

| 核查项 | 观察 | 判定 |
|---|---:|---|
| C0冻结诊断 | 7/7身份；无训练、无test、无MPB/Meep | 完整 |
| seed0筛选派生 | 21/21身份；11项以source-to-replacement清单替换受影响切片 | 谱系完整，原始证据未覆盖 |
| 三种子确认 | 42/42 Slurm任务`COMPLETED`，退出码均为`0:0` | 执行完整 |
| 标准制品 | 42/42 `results.json`、`metrics.json`、`run.meta.json`、`config.resolved.yaml`、`history.csv`、`run.log`、`model.pt`、`epoch200.pt` | 完整 |
| 终态收敛 | 19/42成功终态；23/42未满足末窗收敛门 | **未整体通过** |
| validation边界 | 0项validation overlap；42/42只使用validation选择 | 通过 |
| held-out test | 0/42执行 | 保持冻结 |
| 六域冻结评估 | 未发现checkpoint manifest、六域逐cell指标或资源汇总聚合 | **待执行** |
| 物理证据 | 无real-MPB、无Meep | 不在本证据范围 |

本页报告的是F50 validation上的组件删除敏感性，不是held-out泛化结论、跨域稳健性结论或物理因果结论。

---

<a id="experiment-contract"></a>

## 二、实验合同

| 项目 | 冻结值 |
|---|---|
| 输入与目标 | `material_weight_grid_64 -> Wh[81,10]` |
| 数据 | F50 Binary/Graded，各40,000 train、5,000 validation、5,000 test |
| 主指标 | validation Bands 2--10 MRE，越低越好 |
| 训练 | AdamW、200 epochs、seeds 0/1/2、effective batch 64 |
| 数值模式 | FP32；AMP关闭；TF32关闭 |
| 平台 | N40 `gpu_4090`，每项1×RTX 4090、6 CPU、60 GB |
| 代码身份 | `655a31945f563af4c33cfe2ee79f96f930031484` |
| 数据SHA-256 | Binary `e2d99e0c...13f00`；Graded `172786bc...38a46` |
| 确认manifest | `feace68d6b72b8c7bc628aa05eb07b8130f80f15ccd2d575cfd51718fa8a18e4` |
| 调度批次 | `5443722`（0--15）、`5444973`（16--31）、`5448526`（32--41） |

第三批只改变数组并发上限为8；每项GPU、CPU、内存、训练轮数、代码、数据和配置未改变。由于不同批次并发度不同，训练时间、吞吐和显存只作资源描述，不能用于跨家族严格因果比较。

---

<a id="funnel-lineage"></a>

## 三、C0与seed0漏斗

C0复用7个冻结checkpoint做validation诊断，接受目录记录`expected_cases=7`、`observed_cases=7`，并明确`training_started=false`、`test_targets_loaded=false`。其后21项seed0筛选从每个家族最多两个候选中选出一个进入三种子确认：

| 家族 | seed0候选 | 进入确认的删除项 | seed0判定 |
|---|---|---|---|
| FNO | `no_band_gate`、`no_primary_spectral` | `no_primary_spectral` | 两路线明显退化 |
| TFNO | `no_local`、`no_channel` | `no_channel` | 两路线约3%退化 |
| Joint | `no_local`、`no_spectral` | `no_spectral` | 两路线明显退化 |
| ConvNeXt | `kernel_1`、`zero_padding` | `kernel_1` | 退化幅度最大 |

seed0派生目录不复制全部原始run，而用`source_to_replacement_manifest.csv`连接原始、修复与续投制品；21个接受身份中11个来自最小替换切片。这种处理保留了失败与替换谱系，没有通过覆盖原始run制造“全成功”历史。

---

<a id="confirmation-results"></a>

## 四、三种子主结果

下表为Bands 2--10 validation MRE的三种子`mean ± sample std`。`相对变化`按“三种子均值之比减1”计算，正值表示删除组件后变差；括号内为seed0/1/2逐种子配对相对变化。

| 家族与删除项 | 路线 | Baseline MRE (%) | 删除后MRE (%) | 相对变化 | 三种子同向 | 方向门 |
|---|---|---:|---:|---:|---|---|
| FNO `no_primary_spectral` | Binary | 0.6437 ± 0.0014 | 1.5059 ± 0.0015 | +133.93%（+133.54/+134.55/+133.71%） | 是 | 通过 |
| FNO `no_primary_spectral` | Graded | 0.1653 ± 0.0009 | 0.8527 ± 0.0065 | +415.72%（+420.37/+412.70/+414.11%） | 是 | 通过 |
| TFNO `no_channel` | Binary | 0.3966 ± 0.0017 | 0.4130 ± 0.0038 | +4.14%（+3.61/+5.61/+3.21%） | 是 | 通过 |
| TFNO `no_channel` | Graded | 0.1027 ± 0.0020 | 0.1029 ± 0.0005 | +0.20%（-1.07/-0.49/+2.24%） | **否** | **失败** |
| Joint `no_spectral` | Binary | 0.4595 ± 0.0022 | 0.8333 ± 0.0072 | +81.35%（+81.76/+82.89/+79.40%） | 是 | 通过 |
| Joint `no_spectral` | Graded | 0.1160 ± 0.0007 | 0.1619 ± 0.0007 | +39.61%（+40.03/+39.43/+39.39%） | 是 | 通过 |
| Joint `no_spectral` | 两路线等权 | 0.2877 ± 0.0008 | 0.4976 ± 0.0037 | +72.94%（+73.42/+74.05/+71.35%） | 是 | 通过 |
| ConvNeXt `kernel_1` | Binary | 0.3079 ± 0.0024 | 1.0609 ± 0.0045 | +244.58%（+244.19/+248.54/+241.07%） | 是 | 通过 |
| ConvNeXt `kernel_1` | Graded | 0.0415 ± 0.0020 | 0.1606 ± 0.0047 | +287.07%（+291.02/+279.39/+291.47%） | 是 | 通过 |

这张表支持三类**固定合同内删除敏感性**：ConvNeXt大核、FNO primary spectral和Joint spectral均不可无损删除。它不自动证明这些组件在不同数据域、相同参数预算或真实Maxwell求解下仍具有相同作用。

---

<a id="secondary-metrics"></a>

## 五、Band 1与全频带

单路线run同时记录Band 1与Bands 1--10 MRE。表中均为三种子均值；Joint当前训练输出只记录Binary、Graded及等权的Bands 2--10主指标，没有同口径Band 1/全频带聚合。

| 家族与路线 | Band 1 baseline → 删除后 (%) | Bands 1--10 baseline → 删除后 (%) |
|---|---:|---:|
| FNO Binary | 0.5020 → 2.3963 | 0.6295 → 1.5949 |
| FNO Graded | 0.1090 → 1.2946 | 0.1597 → 0.8969 |
| TFNO Binary | 0.2496 → 0.3435 | 0.3819 → 0.4060 |
| TFNO Graded | 0.0646 → 0.0692 | 0.0989 → 0.0995 |
| ConvNeXt Binary | 0.3267 → 1.4502 | 0.3098 → 1.0999 |
| ConvNeXt Graded | 0.0972 → 0.1184 | 0.0471 → 0.1563 |

辅助指标与主指标总体同向；TFNO Graded仍只表现为很小的均值差，不能修复其逐种子方向冲突。

---

<a id="resource-summary"></a>

## 六、资源与效率

| 家族 | 参数量 baseline → 删除后 | 参数变化 | 平均训练时间范围 | 峰值显存 baseline → 删除后 |
|---|---:|---:|---:|---:|
| FNO | 406,282 → 41,738 | -89.73% | 79.9--83.0 min → 70.5--71.1 min | 737.7 → 722.5 MiB |
| TFNO | 2,690,652 → 2,591,190 | -3.70% | 320.4--330.9 min → 223.5--235.0 min | 5,237.4 → 3,316.3 MiB |
| Joint | 2,703,740 → 387,926 | -85.65% | 699.4 min → 568.7 min | 5,621.7 → 4,975.2 MiB |
| ConvNeXt | 2,298,138 → 2,224,410 | -3.21% | 40.0--42.7 min → 34.9--36.8 min | 282.3 → 242.8 MiB |

FNO和Joint删除频谱分支时也删除了绝大多数参数，因此大幅精度退化同时包含组件与容量变化。ConvNeXt核大小消融只减少约3.21%参数，且两路线退化远大于该容量差，当前归因证据相对更强。TFNO删除channel节省约36%显存并提高吞吐，但Graded精度没有稳定收益或损失；它更像路线相关的效率--精度折中，而非已确认的通用必要组件。

---

<a id="evidence-audit"></a>

## 七、完整性与收敛审计

### 7.1 完整性、谱系与碰撞

- 42个manifest身份对应42个唯一case ID和42个唯一run目录；没有缺失、重复或冲突身份。
- 42个best checkpoint SHA-256和42个terminal checkpoint SHA-256均无重复，未发现checkpoint覆盖碰撞。
- 全部run使用同一部署commit、RTX 4090、PyTorch `2.1.2+cu118`、FP32且AMP/TF32关闭。
- Binary/Graded各自的数据、train/validation/test索引哈希在相关run中一致；Joint同时记录两条路线哈希。
- 42个Slurm `.err`只包含同一条PyTorch `TypedStorage`弃用警告，不是训练失败。
- `test_executed=false`为42/42，未发现held-out test被提前读取。

### 7.2 收敛边界

| 家族与路线 | Baseline通过/3 | 删除后通过/3 | 对解释的影响 |
|---|---:|---:|---|
| FNO Binary | 2 | 3 | 方向稳定，但baseline seed2未过末窗门 |
| FNO Graded | 1 | 2 | 方向稳定，仍有3项未过门 |
| TFNO Binary | 0 | 3 | 删除后稳定，baseline三项均未过门 |
| TFNO Graded | 0 | 0 | 两侧均未过门且方向冲突 |
| Joint Binary曲线 | 2 | 2 | 方向稳定，仍各有1项未过门 |
| Joint Graded曲线 | 1 | 1 | 方向稳定，仍各有2项未过门 |
| Joint等权曲线 | 1 | 2 | 方向稳定，终态仍需按双路线联合门解释 |
| ConvNeXt Binary | 2 | 3 | 方向稳定，baseline seed0未过门 |
| ConvNeXt Graded | 0 | 1 | 方向稳定但收敛资格最弱 |

终态计数为17项`formal_training_success`、2项`dual_lane_training_success`和23项`training_complete_convergence_not_met`。最佳validation checkpoint的配对差异可作为方向证据，但未过收敛门的run不能被重新标记为成功，也不能据此直接进入模型promotion。

---

<a id="attribution-interpretation"></a>

## 八、归因解释

### 8.1 ConvNeXt：大核贡献信号最强

$7\times7\to1\times1$在Binary和Graded的三个种子全部显著退化，参数差仅3.21%，Band 1与全频带也同向。当前可写为：**在F50原生路线、固定层级结构和优化合同下，ConvNeXt的局部大核是稳定性能来源。**仍不能把作用细化为“周期边界”“感受野”或“材料边界建模”，因为zero-padding与层级下采样bridge尚未做三种子确认，Graded baseline也未满足收敛门。

### 8.2 FNO：primary spectral不可删，但容量混杂明显

删除primary spectral后两路线均出现超过100%的相对退化，三个种子高度一致。这强烈否定“primary spectral可由剩余pointwise/head无损替代”，但参数量同时减少89.73%。在完成parameter-matched control之前，结论应限定为**primary spectral及其承载容量组合有贡献**，不能把全部差异解释为Fourier物理归纳偏置。

### 8.3 Joint：spectral对两条路线都有稳定贡献

Joint删除spectral后Binary、Graded和等权指标均三种子同向退化，说明同一条件算子内的频谱支路对两条路线都提供有效信息。参数量减少85.65%，因此仍需容量匹配；多项双路线收敛曲线未过门，也阻止“已完成联合架构因果归因”的表述。

### 8.4 TFNO：channel不是通用必要组件

Binary在三个种子上稳定退化4.14%，但Graded均值仅变化0.20%，逐种子一正两负，并触发超过1%的反向改善停止条件。当前结论是**channel贡献依赖lane，通用贡献猜想未通过**。这也与删除channel后的显存和吞吐改善共同构成后续轻量化候选，但任何精度--效率取舍都需新的预注册目标，不能在本实验中事后改门。

---

<a id="next-gate"></a>

## 九、决策与下一门

1. 停止把TFNO `no_channel`描述为两路线通用贡献；不因Binary阳性结果自动扩大候选。
2. ConvNeXt `kernel_1`通过三种子方向门，是当前最干净的组件贡献候选；FNO `no_primary_spectral`与Joint `no_spectral`通过方向门，但保留容量混杂标签。
3. 不自动重跑23项未收敛训练，不调整阈值、轮数、学习率或资源；是否接受validation方向证据进入后续评估，需要人工决定。
4. 若继续原计划，先冻结通过方向门的checkpoint与身份，再执行K5/K20/F50 × Binary/Graded六域评估，并补齐Band 1、Bands 2--10、Bands 1--10、gap decision与资源汇总。
5. held-out test继续保持未读；只有组件、baseline与跨域评估全部冻结后，才能按合同执行一次最终test。
6. 任何“Fourier更符合物理”“周期卷积解释了全部优势”或“组件普遍有效”的表述仍被禁止；本实验没有real-MPB/Meep证据。

原始证据位于N40仓库相对路径：

- C0接受结果：`runs/20260902/EXP-FWD-019_C0_accepted_20260902173322/`
- seed0接受派生：`runs/20260905/EXP-FWD-019_seed0_accepted_20260905201533/`
- 42项确认矩阵：`runs/20260905/EXP-FWD-019_confirmation_20260905201622/`

本页是AI整理的Draft。人工接受应重点确认：是否允许未全部收敛的FNO、Joint和ConvNeXt候选进入六域冻结评估，以及FNO/Joint是否必须先补parameter-matched control。
