---
title: EXP-INV-016 独立来源与物理反馈结果
summary: 五个独立来源、多目标下纯代理与物理反馈终点的同口径物理比较。
page_type: evidence
status: draft
audience: 逆向设计研究者与证据审阅者
owner: fangtoast，导师审阅
last_reviewed: 2026-11-05
reading_time: 6 minutes
primary_layer: L3
evidence_scope: 五个测试来源、三个目标频率和配对种子的已登记终点
evidence_date: 2026-11-05
dataset_scope: K5 Binary 测试划分，TE B4–B5，目标频率0.55、0.57、0.60
evidence_level: solver_replay
evidence_status: completed
validation_backend: mixed
source_artifacts:
  - runs/20261105/EXP-INV-016_202611050714
---

<!--
  Independent inverse design and physics feedback comparison.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-INV-016 独立来源与物理反馈结果

**五个独立测试来源上，纯代理和三种物理反馈路线均未获得覆盖指定目标的名义或三场景共同带隙。** 残差校正降低了在线候选的双带边误差，但本批没有显示目标覆盖或共同裕量优于纯代理 Adam 的整体收益。结果保留失败设计，是所测来源和目标下的实证比较。

## 实验与验证

固定 Global–Local checkpoint 和方法后，从 K5 Binary 测试划分选择5个来源；每个来源、目标频率0.55／0.57／0.60和三个优化种子使用相同初始几何，分别比较纯代理 Adam、DE、Random，及 Adam 轨迹反馈、Adam 残差反馈、DE 残差反馈。每条代理轨迹60次目标评价；反馈路线共享五个初始物理观测，并经过三轮、每轮最多一个新增物理观测。纯代理在最终 MPB 前锁定，反馈终点是各自物理历史中已接受的结构。确定性 Adam 的重复种子不视为独立来源。

各路线终点使用名义及 $\pm1/512$ 周期边界扰动复核；两参数纯代理与反馈各135个角色均有数值稳定结果，分别对应51个和17个唯一硬几何，其中1个反馈几何复用纯代理三场景求解。四参数扩展有45个纯代理角色、25个唯一硬几何，均通过数值门。预登记的目标0.57、首个种子下5个纯代理 Adam 和5个反馈 Adam 终点均通过 M9／WSL 高精度复核；同81个 k 点、10条带的跨环境最大差为0，门限为 $10^{-6}$。[实验协议](../../../experiments/research/EXP-INV-016-独立来源与效率验证.md)、[纯代理对照](../../../runs/20261105/EXP-INV-016_202611050714/high_accuracy_pure_crosscheck.json)及[反馈对照](../../../runs/20261105/EXP-INV-016_202611050714/high_accuracy_feedback_crosscheck.json)记录细节。

来源抽样与目标选择未使用测试能带标签；一次探索命令曾在切出训练／验证行前将全量能带数组读入内存，统计和选择只使用训练／验证行。排除历史开发身份限于可认证记录，完整边界见实验协议。

## 质量与误差

| 方法 | 数值稳定角色 | 名义目标覆盖 | 三场景目标覆盖 | 共同目标裕量均值 |
|---|---:|---:|---:|---:|
| 纯代理 Adam | 45/45 | 0/45 | 0/45 | −0.243843 |
| 纯代理 DE | 45/45 | 0/45 | 0/45 | −0.244512 |
| 纯代理 Random | 45/45 | 0/45 | 0/45 | −0.243943 |
| Adam＋轨迹反馈 | 45/45 | 0/45 | 0/45 | −0.249961 |
| Adam＋残差反馈 | 45/45 | 0/45 | 0/45 | −0.247766 |
| DE＋残差反馈 | 45/45 | 0/45 | 0/45 | −0.249576 |

共同裕量为 $2\min(\nu_\star-L_{\rm common},U_{\rm common}-\nu_\star)/\nu_\star$。上述每组45个角色是5个来源、3个目标和3个种子的组合；所有组的共同带隙正宽度也为0。按来源内配对结果取均值后，残差 Adam 相对纯代理 Adam 的共同裕量平均变化为 −0.003922，相对仅轨迹反馈为 +0.002195；各来源结果方向不完全一致。[完整配对分析](../../../runs/20261105/EXP-INV-016_202611050714/paired_independent_analysis.json)保留目标、来源和交互项，不能把重复角色当作独立统计样本。

两参数纯代理终点同 k 全谱 MAE 为0.002254–0.005248、MRE 为0.445%–0.794%；四参数终点分别为0.002249–0.005276和0.444%–0.800%。频率为归一化量，MRE 使用 $10^{-8}$ 分母下限。[纯代理结果](EXP-INV-016-独立来源纯代理结果.md)给出详细口径。5个高精度反馈 Adam 终点的原始 Global–Local 同 k 全谱 MRE 为0.446%–0.788%；这些不是局部校正后的全谱误差。[原始谱比较](../../../runs/20261105/EXP-INV-016_202611050714/high_accuracy_feedback_physical_comparison.json)和[带边误差图](../../../runs/20261105/EXP-INV-016_202611050714/figures/high_accuracy_band_edge_error.svg)提供逐来源结果。

对于独立来源上每轮拟合前的新观测，残差 Adam 的平滑双带边 MAE 从0.016414降至0.000677，残差 DE 从0.016449降至0.000669，每组135个候选角色。这一误差包含软硬表示和平滑差，候选是在线自适应产生的，不能解释为独立静态测试集的全谱精度。[逐单元记录](../../../runs/20261105/EXP-INV-016_202611050714/independent_prefit_edge_errors.json)保留汇总及输入哈希。

## 成本与下一步

本独立批 N56 实际分配约6.165 GPU卡时；M9 实际分配约129.02 CPU核时，其中在线反馈26.51、纯代理三场景终点82.72、反馈终点15.43、高精度两组共4.36核时。WSL 纯代理与反馈复核进程分别约1.445和1.516 CPU核时，单独计算；不含排队与预训练。来源打包和跨方法共享求解使单方法 GPU／CPU费用不能直接从总账中准确拆分。[资源账目](../../../runs/20261105/EXP-INV-016_202611050714/m9_and_wsl_costs.json)、[GPU账目](../../../runs/20261105/EXP-INV-016_202611050714/n56_allocated_costs.json)和[在线预算曲线](../../../runs/20261105/EXP-INV-016_202611050714/figures/nominal_margin_observation_budget.svg)记录实际分配及角色预算。

本批支持继续研究局部带边误差校正，但不支持宣称现有方法已找到独立来源上的鲁棒目标设计或取得端到端效率优势。下一研究决策应针对目标可达性、结构参数空间和代理输入分布重新设计，而不是将本批失败终点表述为成功。[按来源的共同裕量图](../../../runs/20261105/EXP-INV-016_202611050714/figures/independent_common_margin.svg)和[主比较表](../../../runs/20261105/EXP-INV-016_202611050714/figures/independent_comparison_table.csv)保留来源差异。原始结果、角色清单和求解证据见[实验 run](../../../runs/20261105/EXP-INV-016_202611050714/)；[复现包](../../../runs/20261105/EXP-INV-016_202611050714/EXP-INV-016_reproducibility.zip)包含代码快照、配置、锁、图表与压缩原始证据，模型 checkpoint 和原数据集需按包内 SHA-256 单独取得。
