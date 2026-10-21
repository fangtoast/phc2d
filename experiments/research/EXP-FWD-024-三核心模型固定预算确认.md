---
title: 三核心模型与分支消融固定预算确认
summary: 冻结六单域8k的54项核心比较与N40的30项分支消融，均从零训练300epoch。
page_type: plan
status: draft
audience: 正向代理研究者与论文审阅者
owner: fangtoast
last_reviewed: 2026-10-20
reading_time: 5 minutes
primary_layer: L3
plan_state: active
---

<!--
  Frozen supplemental experiment contract.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="budget-confirmation"></a>
# 三核心模型与分支消融固定预算确认

**已获用户授权执行：EXP-FWD-024 在 N56 对六种单域各取 8k，以三个核心模型、三个 seed 从零训练 300 epoch；EXP-FWD-025 在 N40 对既有分支能力矩阵全部 30 项从零重训 300 epoch。** 两项保留原 200 epoch 证据，不覆盖历史 run，不自动延长至 500。状态以 run 制品为准，本合同不表示已提交或已完成。

## 一、范围与比较合同

| 项目 | EXP-FWD-024 | EXP-FWD-025 |
|---|---|---|
| 问题 | 同样 8k、300 epoch 的原生测试表现 | 更长固定预算下组合能力优势是否保留 |
| 数据 | K5/K20/F50 × B/G 六个单域，每项仅 8,000 train | 复用 EXP-FWD-022 已冻结的两 lane mixed HDF5，各 16k train、2k validation |
| 模型 | Global–Local kernel 3、tfno2d_band_v3、convnext_band_v1 | Full、G-only、L-only、G-only cap、L-only cap |
| seeds / 数量 | 0/1/2，共 54 项 | 0/1/2，共 30 项 |
| 选择 | 原生 validation Bands 2–10 MRE 最优 | mixed validation Bands 2–10 MRE 最优 |
| 测试 | 完成全部训练并冻结 checkpoint 后，54 个 native idx_test cell | validation-only，未授权 test |
| 平台 | N56 gpu_5090，1 GPU＋8 CPU，默认内存，最多 7 并发 | N40 gpu_4090，1 GPU＋6 CPU，默认内存，最多 3 并发 |

优化沿用模型专属基础学习率、AdamW、FP32、AMP/TF32 off、effective batch 64，不进行额外调参。024 micro batch 64；025 沿用 micro batch 8。恒定学习率和固定轮数只回答预算内比较，不将 300 epoch 解释为充分优化。checkpoint 选择不读 test；train-only normalizer 与同域训练子集跨模型和 seed 固定。

原始数据、模型参数和学习率来自 EXP-FWD-021 冻结配置，025 的结构与容量参数来自 EXP-FWD-022 逐任务配置，读取并核验配置 SHA-256，不重做参数搜索。F50 仅从原 train split 哈希选 8k；历史 40k 专家不进入本次公平主排名。K8/K10/K15 不进入训练、验证、normalizer 或选择。

## 二、门与失败策略

1. 只读核验全部数据 SHA-256、字段、shape、split 范围与互斥，记录训练子集与 normalizer 哈希；不读 test 标签。
2. 在目标 GPU 平台对每个 domain×model / lane×variant 的 seed0 做一次单批前向、反向及完整 validation admission。
3. 全部 admission 审计通过后，冻结 manifest 执行正式训练。一个数组 slot 可顺序执行若干独立任务；54/30 项各有独立 run，数组不超过 16 项。
4. 完整性、300 行 history、有限指标、epoch300、model identity 和 checkpoint 证据全部闭合后才进入评估。
5. 收敛统计比较 281–290 与 291–300，阈值 0.005；未过门保持 `training_complete_convergence_not_met`。不依据 test 决定续训。
6. 错误、非有限值、hash 不符或既有输出立即停止该 slot，不自动重试、覆盖、改资源或降级。失败及未执行任务如实保留。

024 的既有 test 已在历史实验中查看：本次是预先冻结的新训练预算下后续验证，不称全新盲测。除 native 外的十二域全矩阵、本次新增随机盲测与 500 epoch 扩展均不在当前提交范围。025 的 test 继续锁定。

## 三、实现与验收

配置：[核心比较](../../configs/research/full_cell_core_confirmation_exp_fwd_024_v1.yaml)、[分支确认](../../configs/research/full_cell_branch_confirmation_exp_fwd_025_v1.yaml)。执行入口 `python -m phc2d.forward.budget_confirmation --help`，复用 `run_full_cell_forward` 和 native test evaluator。新代码只负责冻结、分片和阶段门，模型与训练循环不变。

机器制品位于 `runs/{date}/EXP-FWD-024_{timestamp}/` 与 `EXP-FWD-025_{timestamp}/`，包括 profile、数据记录、task manifest、admission、独立训练 run、审计，以及024冻结测试清单和逐样本/逐 band 指标。正式工程验收在 WSL Ubuntu + meep_env，Windows 检查单独标识。

本合同依据用户 2026-10-20 明确认可三个核心模型、六种8k与N40消融的指令冻结。人工后续审阅内容为结果、未收敛比例与主张边界，不重复请求既有提交授权。目标10月27日冻结实验结果，10月28–30日用于论文更新；排队及异常不承诺固定完工日。

## 四、实现边界审计

配置校验从既有Full-Wh runner抽离到独立合同模块，兼容导入保留；原runner约1190行降至1010行，569行训练函数本轮未增加逻辑。旧单元仍超结构警戒线，进一步拆训练循环会扩大checkpoint兼容范围，本轮停在配置边界。新campaign模块保持在800行内，函数均小于300行；模型factory、参数与state schema不改。

## 五、执行平台调整

N56隔离部署写入发生文件系统阻塞，GPU作业尚未提交。依据用户此前明确允许N40或N56，024改用[N40执行覆盖配置](../../configs/research/full_cell_core_confirmation_exp_fwd_024_n40_v1.yaml)：继承全部科学合同与数据SHA，仅替换同哈希数据路径、环境与资源。024最多4并发、6个数组slot；与025最多3并发合计7GPU，数组上限6+10=16。025既有作业不改资源。两平台环境不同，因此时间只作为执行成本，不从跨平台历史耗时解释架构因果。

024的最后一个成功训练bundle在全部bundle完成标记与完整训练审计闭合后，独占冻结54个validation-best checkpoint并执行54个native test；不用额外排队作业，不在其他训练尚未完成时提前读test。finalizer失败保留独占意向记录，不自动重试。测试输出逐样本、逐band、每seed cell和三seed均值/样本SD；不把样本P95或seed SD解释为置信区间。

## 六、测试展示口径修订

用户在训练开始后、test尚未执行时明确：测试主指标为Band 1–10 MRE；Band 2–10完整保留在原始制品供查询，默认主表不单列，Band 1也不单列。默认读取 `native_test_primary_summary.csv`，其每seed明细为 `native_test_primary_cells.csv`；原 `native_test_summary.csv`、`native_test_cells.csv` 与逐band/逐样本文件仍保留全部辅助指标。

本修订仅涉及L3测试展示，不修改已冻结训练配置、validation checkpoint选择、收敛判据、数据或运行资源。已经启动的Python进程保留原版本；对该campaign追加独立、版本记录的汇总步骤，等原测试全部完成后导出主表，不重跑训练或测试。
