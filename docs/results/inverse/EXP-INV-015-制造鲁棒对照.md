---
title: EXP-INV-015 制造鲁棒逆向对照
summary: 比较三场景下纯代理与物理校正代理的共同带隙和成本。
page_type: evidence
status: draft
audience: 逆向设计研究者与证据审阅者
owner: fangtoast，导师审阅
last_reviewed: 2026-11-05
reading_time: 4 minutes
primary_layer: L3
evidence_scope: 三个开发来源及三个配对初值，有限制造场景
evidence_date: 2026-11-05
dataset_scope: K5 Binary，TE B4–B5，目标频率0.57
evidence_level: solver_replay
evidence_status: pilot
validation_backend: mixed
source_artifacts:
  - runs/20261105/EXP-INV-015_202611050412
---

<!--
  Manufacturing perturbation inverse design evidence.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-INV-015 制造鲁棒逆向对照

**纯代理的27个锁定终点均通过三场景数值门，但没有一个在名义及 $\pm1/512$ 几何偏移下保留覆盖0.57的共同带隙。** 物理残差校正路线和未见扰动独立复核正在进行，现阶段不能判断校正是否改善了鲁棒结果。

来源8、491、348各三个配对初值，每个初值分别用名义、三场景平均和三场景最坏目标优化60次。冻结 Global–Local，按硬几何的代理预测选 Top-1；所有 MPB 结果在选择锁定后获得。偏移以晶格常数为单位，是基于512主栅格的数值制造场景，不直接等同特定工艺误差。

| 纯代理目标 | 终点数值可判定 | 目标位于共同带隙 | 共同目标裕量均值 |
|---|---:|---:|---:|
| 名义 | 9/9 | 0/9 | −0.105902 |
| 三场景平均 | 9/9 | 0/9 | −0.103966 |
| 三场景最坏 | 9/9 | 0/9 | −0.104190 |

共同带隙的下带边取三个物理场景下带边的最大值，上带边取上带边的最小值。三场景共有的目标裕量为 $2\min(0.57-L_{\max},U_{\min}-0.57)/0.57$；正值才表示目标位于共同带隙。最佳纯代理终点的共同裕量为 −0.012611，三场景共同区间的边界为 $(0.572784,0.566406)$，区间已反向；它的名义目标裕量为 +0.012798。九次运行由三个开发来源及各自三个初值构成，不是九个独立来源。[完整场景与谱比较](../../../runs/20261105/EXP-INV-015_202611050412/pure_physical_comparison.json)保留所有角色，54个初始／终点角色去重为21个硬结构，均通过本批物理数值门。

初始训练 checkpoint 的256主栅格与逆向的512主栅格存在[已核实的表示差异](EXP-INV-012-来源8梯度与物理复核.md)。该差异、离散带边加密和有限扰动幅度共同界定本结果；其不说明所有制造扰动下都无法获得鲁棒结构。

45个三场景反馈样本均已完成且通过 M9 数值门。校正路线27个终点已锁定，其18个新增物理结构以及六个预选未见扰动角色对应的5个唯一结构正在 M9 复核；两条路线预选角色的 WSL 独立复核另计。完整比较、名义性能代价和成本将在这些结果齐备后更新。[实验协议](../../../experiments/research/EXP-INV-015-制造鲁棒对照.md)及[原始 run](../../../runs/20261105/EXP-INV-015_202611050412/)记录配置、锁和求解证据。
