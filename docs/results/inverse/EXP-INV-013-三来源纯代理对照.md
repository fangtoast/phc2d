---
title: EXP-INV-013 三来源纯代理对照
summary: 三来源配对初值的纯代理设计及锁定后的 MPB 复核。
page_type: evidence
status: draft
audience: 逆向设计研究者与证据审阅者
owner: fangtoast，导师审阅
last_reviewed: 2026-11-05
reading_time: 5 minutes
primary_layer: L3
evidence_scope: 三个开发来源、三个初值及三种方法的锁定候选
evidence_date: 2026-11-05
dataset_scope: K5 Binary 来源8、491、348，TE B4–B5，目标频率0.57
evidence_level: solver_replay
evidence_status: completed
validation_backend: mixed
source_artifacts:
  - runs/20261105/EXP-INV-013_202611050104
---

<!--
  Paired multi-source inverse design and locked physical replay.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-INV-013 三来源纯代理对照

**27 次纯代理设计的锁定 Top-1 中，Adam 有 3 个、DE 和 Random 各有 0 个在 MPB 加密带边口径下覆盖 0.57；3 个成功角色均来自来源8。** 三来源是方法开发样本，三个初值不是三个独立来源；本结果用于方法比较，不代表独立泛化性能。

来源8、491、348各使用三个共同初值，每个初值分别运行 Adam、DE、Random，每次60次软目标评价，最终按硬结构 Global–Local 精确裕量选出 Top-1。物理求解在选型锁定后进行，结果未用于改选。135个预登记角色中132个几何合格、3个不合格；合格角色去重为61个硬栅格，61个结果全部完成且数值稳定。六个预选角色对应的5个硬结构通过 WSL Ubuntu/meep_env 对照，同81个 k 点全部10条带的 M9–WSL 最大频率差均为0，门限为 $10^{-6}$。

## 选型与真实结果

目标裕量 $J=2\min(0.57-L,U-0.57)/0.57$，正值表示目标位于带隙内；MPB 值采用加密带边。每组9个角色由3个来源各3个初值组成。

| 方法 | MPB 目标覆盖角色 | MPB 裕量平均值 | 相对对应初始结构的真实裕量变化中位数 |
|---|---:|---:|---:|
| Adam | 3/9 | -0.05909 | 0.00000 |
| DE | 0/9 | -0.06733 | -0.00770 |
| Random | 0/9 | -0.06890 | -0.01089 |

三种方法在来源348上的真实裕量均明显为负，影响了平均值；来源491的9个终点也均未覆盖目标。来源8的3个 Adam 终点裕量分别为 0.003343、0.005749、0.000049；最后一个距离零很近，应与带边数值容差一起解读。27个 Top-1 中 Global–Local 判定3个覆盖目标，但它们的 MPB 结果均未覆盖；另外3个代理判定不覆盖的 Adam 角色在 MPB 加密口径下覆盖。固定81点的预测与加密 MPB 带边口径不同，不能把所有差异归因于代理误差。

按预登记的第1、20、40、60次候选及锁定 Top-1 做事后排序诊断，在27条轨迹中有12条的代理 Top-1 与这组候选中的最高 MPB 裕量并列；每条轨迹的物理最佳减去代理 Top-1 的差值中位数为 0.00569，最大为 0.02878。重复几何保留角色但只求解一次；诊断物理最佳没有替代原 Top-1。所有合格角色同 k 全谱 MRE 范围为 0.468%–0.876%，覆盖81点×10条带，分母下限 $10^{-8}$。

## 成本与后续

N56 的9个数组分片实际分配时间合计约0.605 GPU卡时；M9 的8个分片按调度器实际分配12 CPU计约21.31核时，物理候选自身墙钟时间之和约11.48小时，不应与并行墙钟或分配核时混同。WSL 独立复核另计。设计期间未调用 MPB；上述物理成本用于锁定后的诊断与复核。

这些对照将作为 EXP-INV-014 物理反馈的配对纯代理基线。物理求解、失败角色、预测和锁定身份见[机器可读比较](../../../runs/20261105/EXP-INV-013_202611050104/physical_comparison.json)、[跨环境对照](../../../runs/20261105/EXP-INV-013_202611050104/wsl_crosscheck_comparison.json)和[完整 run](../../../runs/20261105/EXP-INV-013_202611050104/)。
