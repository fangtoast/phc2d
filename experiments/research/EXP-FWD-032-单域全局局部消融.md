---
title: EXP-FWD-032 单域全局局部消融
summary: 四个K5与F50单域的分支、容量和局部核消融训练及六目标测试合同。
page_type: research_experiment
status: draft
audience: 正向代理研究者与实验审阅者
owner: fangtoast
last_reviewed: 2026-10-31
primary_layer: L3
---

<!--
  Planned single-domain Global/Local ablation campaign.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-FWD-032：单域全局局部消融

**结论：本实验用于补齐 K5/F50 单域分支独立能力证据；用户已明确要求开展双平台准入与训练，运行状态以 manifest 和 run 制品为准。** 原 K5+K20 mixed 消融不能代替单域训练；冻结关断和独立重训也回答不同问题。候选基线与既有 mixed 证据见 [EXP-FWD-022](../../docs/results/forward/EXP-FWD-022-分支依赖与独立能力.md) 和 [四域测试报告](../../docs/reports/EXP-FWD-031四域完整测试结果_20261029.md)。本协议为 Draft，待人类审阅。

## 比较合同

四个来源域为 K5 Binary/Graded 与 F50 Binary/Graded；每域使用与 EXP-FWD-024 同一 8k train 子集、validation split、train-only normalizer 和 seed 0/1/2。Full、Global-only、Local-only 与两种参数量对照各自独立训练，Local 分支采用 3×3 和 7×7。Global-only 不含 Local 核，不重复训练。

用当前模型 factory 和 [既有参数搜索](../../src/phc2d/forward/global_local_branch_capability.py) 在 `input_shape=(1,64,64)`、`output_shape=(81,10)` 上核算：

| 变体 | 3×3 Full 对照 | 7×7 Full 对照 | 参数量 |
|---|---|---|---:|
| Full | 是 | 是 | 2,571,978 / 2,587,338 |
| Global-only | 是 | 是 | 2,369,610 |
| Global-only 容量对照 | 是 | 是 | 2,572,395 |
| Local-only | 是 | 是 | 256,170 / 271,530 |
| Local-only 容量对照 | 是 | 是 | 2,614,890 / 2,668,650 |

Global-only 容量对照 width 64、dense head hidden 291，满足两个 Full 各 ±5%；这是容量对照，并非结构等价。Local-only 容量对照 width 224。参数元素计数、模型指纹及对照复用条件须在目标部署环境再次核验；固定同一 Global-only 容量模型用于两种核，不将它解释为核尺寸消融。

候选 Full 是 EXP-FWD-026 的 3×3 与 EXP-FWD-029 的 7×7 单域 checkpoint。逐项核验数据、split、子集、normalizer、架构、优化与 checkpoint 后才能复用；3×3 的 301 轮来源为 024 validation-best 热启动且重置优化器，须与从零固定 300 轮再重置的合同逐项对照。无法证明匹配时补训 Full。24 个 Full 全部可复用时，新训练为 4 域 × 6 个唯一变体 × 3 seed = **72 项**；若任一 Full 不合格，逐项补训。

初步只读检查确认 24 个候选 checkpoint 的文件哈希、核尺寸、micro/effective batch 和自适应起点；每个域的两种核与三个 seed 的 checkpoint normalizer、记录的数据身份和子集哈希一致。这不替代重新核算数据文件哈希及完整复用审计。历史 026 的终态为 `continuation_training_success`，现有 031 测试入口的终态白名单不接受这一状态，不能直接将该入口用于旧基线。

## 训练和测试门

训练配置继承 024 的单域数据与模型合同：AdamW、LR 5e-4、weight decay 1e-5、FP32、micro/effective batch 64、normalized full-matrix MSE。采用 [031 的自适应协议](EXP-FWD-031-官方算子与通道融合对比.md)：1–300 轮从零，301 从阶段内 validation-best 权重重置优化器及随机状态，累计最多 1000 轮；Band 2–10 validation MRE 选 checkpoint。停止门未通过保留预算耗尽状态，不称收敛。

N40 分配 K5 Binary、F50 Graded；N56 分配 K5 Graded、F50 Binary；各最多 8 个单 GPU 任务，分别 6/8 CPU。冻结前核验实际环境、数据哈希和可用资源；每域每唯一变体先完成 seed0 单批训练、完整 validation、有限梯度、显存及 checkpoint 往返准入。失败保留证据，不自动重试、迁移或改精度、batch。训练和测试共享平台 GPU 上限。

若全部 Full 复用，每机各 36 项新训练，上限共 72,000 单任务 epoch，另计准入与测试；epoch 不是 GPU 小时。准确工时须以各变体和平台的准入吞吐及实际停止轮次估算，不能从不同模型或硬件的历史时间直接外推。

每个新 checkpoint 分别评估四域 `idx_test`、EXP-DAT-007 12K 与 EXP-DAT-008 24K；72 checkpoint 对应 432 个唯一 cell，Full 补训将增加 cell。只在 checkpoint、目标身份和评价合同均匹配时复用 Full 测试。test 不用于停止、选模或调整。输出逐样本、逐 band、每 seed 与三 seed 均值和样本标准差；主指标 Band 1–10 MRE，Band 2–10 及耗时保留。native、跨域、12K、24K 分表，配对比较 Full 与单支及容量对照。

运行制品须保存源码与环境身份、数据和 split 哈希、manifest、准入、history、checkpoint、停止原因、预测和指标；核对身份、哈希、唯一 cell 和三 seed 完整性。已查看的测试集不是全新盲测；新几何代理迁移结果不是 WSL MPB 复算、分辨率收敛或物理验收。正式工程验收使用 WSL Ubuntu `meep_env`。

## 当前 gate

[实验配置](../../configs/research/full_cell_single_domain_branch_ablation_exp_fwd_032_v1.yaml) 不是已冻结的任务清单。用户于 2026-10-31 明确要求开展本实验直至训练完成，授权双平台准入与训练；仍须先核验基线、冻结 manifest 并通过准入。测试应按本协议另建独立冻结评价清单，不以训练配置的 `test_authorized: false` 替代评价授权或结果。

来源及数据审计在两机分别完成 12 个候选，共 24/24；四个训练域与对应来源的文件、子集、normalizer 和 k 点身份相符。每平台已冻结 36 项新训练配置及 216 个测试目标身份。由于初始提交指向缺失的共享启动脚本，两次 `sbatch` 均失败且未产生作业；脚本从原归档提取并校验后，用户明确授权各重提一次。N56 准入 `1754790` 经检查通过 12/12；N40 准入 `5524780` 排队中。完整运行记录分别在 N40 `/data/run01/scw6g7a/phc2d/runs/20261031/EXP-FWD-032_202610311238/` 与 N56 `/data/run01/scxl892/phc2d/runs/20261031/EXP-FWD-032_202610311238/`。这不是训练完成或测试完成。

用户于 2026-10-31 另行授权按平台独立放行：每机自己的 12 项准入、冻结清单和全部基线复用审计均通过后，才释放该机 36 项训练；N56 可以先执行，N40 不因另一平台通过而跳过本机准入。数据、优化、模型和每机 8 GPU 上限均不变，放行记录绑定实际清单、准入与审计哈希。
