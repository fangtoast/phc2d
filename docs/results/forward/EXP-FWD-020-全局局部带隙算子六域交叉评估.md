---
title: EXP-FWD-020 全局局部带隙算子六域交叉评估
summary: 36 个冻结 checkpoint 在 K5/K20/F50、Binary/Graded 六域 held-out test 上的 216-cell 探索性代理评估；大核没有跨域一致优势。
page_type: evidence
status: draft
audience: 模型研究者、结果审阅者与复现执行者
owner: 研究生维护，导师审阅研究主张
last_reviewed: 2026-09-12
reading_time: 8 minutes
primary_layer: L3
evidence_date: 2026-09-12
evidence_level: surrogate_held_out
evidence_scope: exploratory_held_out_surrogate_only
dataset_scope: full_cell_k5_k20_f50_binary_graded_wh81x10_held_out
evidence_status: completed
validation_backend: surrogate_only
completion_qualification: exploratory_held_out_surrogate_only_with_g2_formal_status_not_passed
source_artifacts:
  - runs/EXP-FWD-020/20260909/EXP-FWD-020_g2_all_lanes_override_20260909191759
  - runs/EXP-FWD-020/20260911/EXP-FWD-020_epoch300_continuations_20260911115515
  - runs/EXP-FWD-020/20260912/EXP-FWD-020_g3_cache_reuse_20260912123541
---

<!--
  EXP-FWD-020 held-out six-domain result.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-FWD-020：全局—局部带隙算子六域交叉评估

## 结论

36 个冻结 checkpoint 的 216/216 个 held-out surrogate cell 已完成。模型在自己的训练域上保持低误差，但离开训练域后误差明显上升：native、同数据集跨 lane、跨数据集同 lane、同时跨数据集和 lane 的 Bands 2--10 MRE 均值依次为 `0.297%`、`2.703%`、`3.295%`、`5.409%`。因此，结果支持“存在强域偏移”，不支持把任一 checkpoint 写成六域通用 ranker。

把局部周期核从 $3\times3$ 改为 $7\times7$ 也没有形成全局一致优势：108 个同源域、同目标域、同 seed 的 paired bootstrap 中，`7x7` 改善 43 个、退化 63 个、不确定 2 个；36 个源域到目标域方向中仅 6 个方向三种子一致改善，13 个一致退化，17 个混合。原生域中只有 F50 Graded 稳定改善，K20 Graded 稳定退化。

这是一项完整但**探索性**的 held-out surrogate 结果。`g2_formal_status=not_passed`，不产生 model promotion、real-MPB/Meep replay 或 physical claim。

## 评估对象与证据边界

| 项目 | 冻结事实 |
|---|---|
| checkpoint inventory | 36 个；7 个为原 G2 validation-best，29 个为 correction campaign 的 epoch 1--300 validation-best |
| target | K5/K20/F50 的 Binary/Graded 六个 held-out test target |
| 矩阵 | $6$ source domain $\times2$ kernel $\times3$ seed $\times6$ target domain = 216 cells |
| 关系数 | native 36、跨 lane 36、跨数据集 72、跨数据集且跨 lane 72 |
| 测试读取 | 首次成功 preflight 后各目标 test 仅进入一次冻结 cache；最终 correction campaign 只复用并校验该 cache，原始 test 读取次数为 0 |
| 核大小比较 | 同 source/target/seed 的 $3\times3\leftrightarrow7\times7$ paired bootstrap，10,000 次重采样，共 108 条 |
| 非配对退化 | 每个 checkpoint 相对自身 native cell，共 180 条 |
| 科学边界 | `exploratory_held_out_surrogate_only`；无训练调参、无 test-driven rerun、无真实求解器结论 |

上游实验合同见 [EXP-FWD-020 合同](../../../experiments/research/EXP-FWD-020-全局局部带隙算子桥接.md)。G2 的正式收敛门仍未通过；把未收敛的 epoch-300 validation-best 纳入本评估只用于检验探索性泛化，不构成对 G2 的追溯通过。

## 交叉域主结果

下表使用每个 cell 的 `bands_2_10_mre_pct`。跨域类别的绝对 MRE 已足够显示域偏移；相对退化百分比会受到极低 native 分母放大，故不单独作为主排名依据。

| source--target 关系 | cells | MRE 均值 | 最小--最大 MRE | 相对 native 的解释 |
|---|---:|---:|---:|---|
| native | 36 | 0.297% | 0.094--0.566% | checkpoint 的唯一原生目标 |
| 同数据集、跨 lane | 36 | 2.703% | 1.683--3.937% | 仅改变 Binary/Graded 路线已显著退化 |
| 跨数据集、同 lane | 72 | 3.295% | 0.673--9.667% | 数据集转移进一步放大误差 |
| 跨数据集且跨 lane | 72 | 5.409% | 1.954--12.471% | 两类 shift 叠加时最差 |

作为目标域时，36 个 source checkpoint 的平均 MRE 如下。这是“该 target 对当前冻结库存的总体表现”，不是独立的数据难度因果结论。

| target domain | cells | MRE 均值 |
|---|---:|---:|
| K5 Binary | 36 | 2.458% |
| K5 Graded | 36 | 2.937% |
| F50 Binary | 36 | 2.888% |
| F50 Graded | 36 | 2.960% |
| K20 Binary | 36 | 4.713% |
| K20 Graded | 36 | 4.452% |

## 全部 source--target 方向（36 组）

此前的主表按关系类别聚合，无法回答“哪个源域迁移到哪个目标域”的问题。下表完整列出 36 个有向组合；每行汇总 `2` 个 kernel × `3` 个 seed，即 6 个 cell 的 `bands_2_10_mre_pct`。因此它是全部 216 cell 的方向级视图；逐 seed 与逐 kernel 的原始汇总仍保留在证据目录的 `seed_summary.csv`、`kernel_direction_summary.csv` 和 `evaluation_cells.csv`，不应把方向均值误读为单一模型结果。

| source domain | target domain | 关系 | cells | MRE 均值 | MRE 标准差 |
|---|---|---|---:|---:|---:|
| F50 Binary | F50 Binary | native | 6 | 0.387% | 0.0019% |
| F50 Binary | F50 Graded | 跨 lane | 6 | 2.716% | 0.3177% |
| F50 Binary | K20 Binary | 跨数据集 | 6 | 4.078% | 0.6441% |
| F50 Binary | K20 Graded | 跨数据集且跨 lane | 6 | 5.033% | 1.4523% |
| F50 Binary | K5 Binary | 跨数据集 | 6 | 0.700% | 0.0246% |
| F50 Binary | K5 Graded | 跨数据集且跨 lane | 6 | 3.328% | 0.4049% |
| F50 Graded | F50 Binary | 跨 lane | 6 | 2.394% | 0.2641% |
| F50 Graded | F50 Graded | native | 6 | 0.0949% | 0.0008% |
| F50 Graded | K20 Binary | 跨数据集且跨 lane | 6 | 11.812% | 0.3944% |
| F50 Graded | K20 Graded | 跨数据集 | 6 | 9.062% | 0.3275% |
| F50 Graded | K5 Binary | 跨数据集且跨 lane | 6 | 5.919% | 0.4048% |
| F50 Graded | K5 Graded | 跨数据集 | 6 | 3.233% | 0.3463% |
| K20 Binary | F50 Binary | 跨数据集 | 6 | 4.881% | 1.1852% |
| K20 Binary | F50 Graded | 跨数据集且跨 lane | 6 | 7.319% | 2.1923% |
| K20 Binary | K20 Binary | native | 6 | 0.200% | 0.0041% |
| K20 Binary | K20 Graded | 跨 lane | 6 | 3.792% | 0.0872% |
| K20 Binary | K5 Binary | 跨数据集 | 6 | 2.516% | 0.4912% |
| K20 Binary | K5 Graded | 跨数据集且跨 lane | 6 | 5.479% | 0.9300% |
| K20 Graded | F50 Binary | 跨数据集且跨 lane | 6 | 6.426% | 0.6303% |
| K20 Graded | F50 Graded | 跨数据集 | 6 | 3.474% | 0.6510% |
| K20 Graded | K20 Binary | 跨 lane | 6 | 1.994% | 0.1814% |
| K20 Graded | K20 Graded | native | 6 | 0.141% | 0.0097% |
| K20 Graded | K5 Binary | 跨数据集且跨 lane | 6 | 3.066% | 0.1942% |
| K20 Graded | K5 Graded | 跨数据集 | 6 | 1.842% | 0.4029% |
| K5 Binary | F50 Binary | 跨数据集 | 6 | 0.845% | 0.0327% |
| K5 Binary | F50 Graded | 跨数据集且跨 lane | 6 | 3.207% | 0.6231% |
| K5 Binary | K20 Binary | 跨数据集 | 6 | 2.635% | 0.3931% |
| K5 Binary | K20 Graded | 跨数据集且跨 lane | 6 | 3.363% | 0.8546% |
| K5 Binary | K5 Binary | native | 6 | 0.555% | 0.0067% |
| K5 Binary | K5 Graded | 跨 lane | 6 | 3.336% | 0.2341% |
| K5 Graded | F50 Binary | 跨数据集且跨 lane | 6 | 2.397% | 0.2173% |
| K5 Graded | F50 Graded | 跨数据集 | 6 | 0.952% | 0.0733% |
| K5 Graded | K20 Binary | 跨数据集且跨 lane | 6 | 7.559% | 1.0822% |
| K5 Graded | K20 Graded | 跨数据集 | 6 | 5.321% | 1.4289% |
| K5 Graded | K5 Binary | 跨 lane | 6 | 1.990% | 0.1191% |
| K5 Graded | K5 Graded | native | 6 | 0.404% | 0.0155% |

## 跨数据集与跨 lane 的完整汇总

数据集迁移具有明显方向性：例如 K20→F50 的 24 个 cell 均值为 5.525%，而 F50→K20 为 7.496%；这说明不能用无方向的“F50--K20 迁移”掩盖 source 的差异。

| source dataset → target dataset | cells | MRE 均值 | MRE 标准差 |
|---|---:|---:|---:|
| K5 → K20 | 24 | 4.719% | 2.1627% |
| K5 → F50 | 24 | 1.850% | 1.0485% |
| K20 → K5 | 24 | 3.226% | 1.4854% |
| K20 → F50 | 24 | 5.525% | 1.9802% |
| F50 → K5 | 24 | 3.295% | 1.8756% |
| F50 → K20 | 24 | 7.496% | 3.2258% |

| lane 方向 | 数据集是否改变 | cells | MRE 均值 | MRE 标准差 |
|---|---|---:|---:|---:|
| Binary → Graded | 否 | 18 | 3.281% | 0.4991% |
| Graded → Binary | 否 | 18 | 2.126% | 0.2736% |
| Binary → Graded | 是 | 36 | 4.621% | 1.9367% |
| Graded → Binary | 是 | 36 | 6.197% | 3.1553% |

## 原生域的核大小结果

`MRE delta` 定义为 `7x7 - 3x3`，负值表示 $7\times7$ 较低。方向“一致”要求三个 seed 同向，paired bootstrap 结论同时使用配对的样本级误差，而不是把不同 test target 强行配对。

| native domain | 3x3 MRE 均值 | 7x7 MRE 均值 | 7x7--3x3 (百分点) | seed 方向 | paired-bootstrap 结论 |
|---|---:|---:|---:|---|---|
| F50 Binary | 0.388% | 0.386% | -0.0013 | 2 改善 / 1 退化 | mixed |
| F50 Graded | 0.0955% | 0.0943% | -0.0012 | 3 改善 / 0 退化 | candidate improves |
| K20 Binary | 0.2019% | 0.1985% | -0.0034 | 2 改善 / 1 退化 | mixed |
| K20 Graded | 0.1320% | 0.1500% | +0.0180 | 0 改善 / 3 退化 | candidate regresses |
| K5 Binary | 0.5521% | 0.5587% | +0.0067 | 1 改善 / 2 退化 | mixed |
| K5 Graded | 0.4062% | 0.4018% | -0.0044 | 2 改善 / 1 退化 | mixed |

因此，EXP-FWD-019 的“大核删除敏感性”信号不能直接推导为 EXP-FWD-020 中“大核训练后跨域普遍更优”。本实验更准确的结论是：核大小效应与 source domain、target domain 和 lane 有交互。

## 108 个配对比较的整体与代表方向

| paired-bootstrap 结论 | native | 跨 lane | 跨数据集 | 双重跨域 | 总计 |
|---|---:|---:|---:|---:|---:|
| 7x7 improves | 10 | 4 | 16 | 13 | 43 |
| 7x7 regresses | 8 | 14 | 18 | 23 | 63 |
| uncertain | 0 | 0 | 2 | 0 | 2 |

三个 seed 一致的代表性方向如下；这些是交互的例子，不能扩展为总体排序。

| source → target | 7x7--3x3 MRE (百分点) | 三 seed 方向 | 结论 |
|---|---:|---|---|
| K20 Binary → F50 Binary | -1.806 | 3 改善 / 0 退化 | 一致改善 |
| K20 Binary → K5 Graded | -1.546 | 3 改善 / 0 退化 | 一致改善 |
| K20 Binary → K5 Binary | -0.876 | 3 改善 / 0 退化 | 一致改善 |
| K5 Graded → K20 Binary | +1.809 | 0 改善 / 3 退化 | 一致退化 |
| F50 Binary → K20 Binary | +0.906 | 0 改善 / 3 退化 | 一致退化 |
| K5 Binary → K20 Binary | +0.667 | 0 改善 / 3 退化 | 一致退化 |

36 个方向的汇总为：6 个一致改善、13 个一致退化、17 个混合。故下一步若要解释机制，应预注册 domain-matched 或 parameter-matched 对照，而不是基于本次 held-out 表现继续调核或追调训练。

## 带隙、逐 band、逐 k 与资源数据

每个 cell 还保留完整的带隙决策、逐 band 与逐 k 数据，避免全谱 MRE 掩盖离散 band/k 的选择误差。

| 表 | 行数 | 内容 |
|---|---:|---|
| `gap_decision_metrics.csv` | 216 | 最大相邻 gap 的 band、upper-edge k、complete/pseudo 状态、宽度和中心误差 |
| `per_band_metrics.csv` | 2,160 | 216 cells $\times$ 10 bands |
| `per_k_metrics.csv` | 17,496 | 216 cells $\times$ 81 k points |
| `checkpoint_degradation.csv` | 180 | 同 checkpoint 相对自身 native 的 directed degradation |
| `kernel_paired_bootstrap.csv` | 108 | 每个配对的 CI、相对改善与结论 |

资源汇总为 216 次推理共 `151.29 s`，batch 固定为 64，最大记录显存约 `750.53 MB`。F50 target 每域 36 cell 的累计推理时间约 49.5 s；K5/K20 target 每域约 11.6--14.5 s，差异来自 target test size。

## 完整性、provenance 与原始证据

| 审计项 | 结果 |
|---|---|
| checkpoint / dataset / plan | 36 / 6 / 216，均已冻结 |
| shard / cell | 8 / 216，全部完成 |
| 数值与制品 | 必需表格存在，未检测到 `NaN` 或 `Inf` |
| cache | 只读符号引用最初通过 preflight 的六域 cache；哈希与 schema 已复核 |
| finalizer | 已输出 `evaluation_complete_pending_review` 和全部汇总表 |

机器可读证据位于 N56：`/data/run01/scxl892/runs/EXP-FWD-020/20260912/EXP-FWD-020_g3_cache_reuse_20260912123541/`。主要入口为 `results.json`、`metrics.json`、`checkpoint_manifest.csv`、`dataset_manifest.csv`、`evaluation_cells.csv`、`relationship_summary.csv`、`kernel_paired_bootstrap.csv` 和 `checkpoint_degradation.csv`。此前失败的预检、分片与部署目录保留为排除谱系，不计入该 final inventory。

## 决策与限制

1. 接受本页为完整的探索性六域 held-out surrogate 证据：其主发现是域偏移与核大小交互。
2. 不将 7x7 作为全局默认或 promotion 候选；K20 Graded native 的一致退化和多数 paired-bootstrap 退化必须一并保留。
3. 不基于 held-out 结果再调整学习率、epoch、seed、数据或核大小。本 campaign 已闭合；进一步训练须是新的预注册实验。
4. 若研究目标转向物理或 inverse，仍需独立的 gap-selection 审计及 real-MPB/Meep replay；本页不能代替这些证据。

本页为 AI 整理的 Draft，等待人类负责人审阅与接受。
