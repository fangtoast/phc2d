---
title: EXP-INV-016 独立来源纯代理结果
summary: 五个独立测试来源上纯代理终点的同 k 频谱误差与 MPB 三场景结果。
page_type: evidence
status: draft
audience: 逆向设计研究者与证据审阅者
owner: fangtoast，导师审阅
last_reviewed: 2026-11-05
reading_time: 4 minutes
primary_layer: L3
evidence_scope: 五个测试来源的纯代理设计；物理反馈与最终方法比较进行中
evidence_date: 2026-11-05
dataset_scope: K5 Binary 测试划分，TE B4–B5，目标频率0.55、0.57、0.60
evidence_level: solver_replay
evidence_status: pilot
validation_backend: mixed
source_artifacts:
  - runs/20261105/EXP-INV-016_202611050714
---

<!--
  Independent-source pure surrogate inverse replay.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-INV-016 独立来源纯代理结果

**五个独立测试来源上的纯代理终点，同 k 全谱相对误差约为 0.44%–0.80%，但在本批目标和三场景口径下没有角色覆盖目标。** 这是纯代理阶段结果；物理反馈和完整质量—成本比较仍在进行。[实验协议](../../../experiments/research/EXP-INV-016-独立来源与效率验证.md)记录来源选择、目标和测试数据使用边界。

## 结果

| 设计变量 | 终选角色 | 同 k 全谱 MAE 范围 | 同 k 全谱 MRE 范围 | 名义 MPB 覆盖目标 | 三场景共同带隙覆盖目标 |
|---|---:|---:|---:|---:|---:|
| 两参数 | 135 | 0.002254–0.005248 | 0.445%–0.794% | 0/135 | 0/135 |
| 四参数 | 45 | 0.002249–0.005276 | 0.444%–0.800% | 0/45 | 0/45 |

全谱误差使用相同81个 k 点和10条能带，归一化频率的相对误差分母下限为 $10^{-8}$。两参数角色平均 MAE 为0.003615、平均 MRE 为0.584%；角色包含重复几何和重复种子，不能将180个角色当作独立样本。两参数为五来源、三个目标、三个种子和 Adam／DE／Random；四参数只使用首个种子。终选按冻结代理规则锁定，MPB 结果没有用于改选。

名义、过刻蚀和欠刻蚀分别使用偏移 $0$、$+1/512$、$-1/512$；共同带隙由三场景最大下带边和最小上带边确定。两参数51个唯一硬结构（其中1个复用四参数求解）和四参数25个唯一硬结构均完成 M9 复核，数值门通过。相同结构和求解合同跨角色去重，三场景结论不代表其他扰动幅度。

## 高精度复核与误差解释

预先指定的五个来源在目标0.57、首个种子下的纯代理 Adam 终点，均通过 M9 和 WSL `Ubuntu/meep_env` 的224／256分辨率与加密 k 点复核。同一81点、10条带的跨环境最大绝对频率差为0；同 k 代理 MAE 为0.002267–0.005179，MRE 为0.446%–0.791%。五个角色的代理与加密 MPB 均判断未覆盖目标。[跨环境数据](../../../runs/20261105/EXP-INV-016_202611050714/high_accuracy_pure_crosscheck.json)与[谱比较](../../../runs/20261105/EXP-INV-016_202611050714/high_accuracy_pure_physical_comparison.json)记录逐角色结果。

全谱平均误差小不能保证带边或目标判定准确：[来源8开发批](EXP-INV-012-来源8梯度与物理复核.md)的 Adam 终点同 k 相对谱误差为0.698%，上带边却低估约0.011398，代理与 MPB 对目标覆盖的判断相反。checkpoint 的训练主栅格为256，而本批逆向输入由512主栅格降采样；[表示敏感性诊断](../../../runs/20261105/EXP-INV-014_202611050146/raster_contract_probe.json)应与结果一起解释。

## 下一步与证据

三轮物理反馈的最后一轮 MPB 和反馈终点的高精度对照尚未完成，不能据此判断反馈方案的独立测试收益。全部原始轨迹、候选锁、角色映射和求解结果见[独立批 run](../../../runs/20261105/EXP-INV-016_202611050714/)，逐单元比较分别位于 `two_scenario_results/` 与 `four_scenario_results/`。资源成本与排队时间将在反馈路线完成后统一统计。
