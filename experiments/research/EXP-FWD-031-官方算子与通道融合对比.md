---
title: EXP-FWD-031 官方算子与通道融合对比
summary: 四域三模型三种子从零训练及双服务器准入合同。
page_type: research_experiment
status: draft
audience: 研究用户与复现执行者
owner: fangxiao
last_reviewed: 2026-10-27
primary_layer: L3
---

<!--
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-FWD-031 官方算子与通道融合对比

**结论：本协议冻结了训练与 validation 的比较口径；模型接入或任务提交均不构成性能结果。** 本次仅比较 surrogate validation，正式 held-out test 须在训练与 checkpoint 冻结后另行安排。

| 字段 | 冻结值 |
|---|---|
| 状态 | active，Draft，待人类接受；运行状态以 `runs/` 原始证据为准 |
| 生效日期 | 2026-10-27 |
| 科学问题 | 官方 FNO、官方 Tucker-TFNO 与每通道可学习融合 Global–Local 在四域中的 validation 表现和训练稳定性 |
| 基线 | 历史固定融合 Global–Local 仅作参照；不是本批同条件重训项 |
| 数据 | K5 Binary/Graded、F50 Binary/Graded；每域 train 8,000，K5 validation 1,000，F50 validation 5,000 |
| 模型与种子 | 三模型 × 四域 × seed 0/1/2 = 36 项，全部从零训练 |
| 执行 | N40 承担 K5-Binary、F50-Graded；N56 承担 K5-Graded、F50-Binary；各最多 8 项单卡并行 |

## 模型与可比边界

| 身份 | 主干 | 频谱与头 |
|---|---|---|
| `official_fno_band_v1` | NeuralOperator 2.0.0 稠密 FNO，宽 64、4 层 | `n_modes=(24,22)`；全局均池 + LayerNorm(64) + Linear(64,810) |
| `official_tucker_tfno_band_v1` | 同版 Tucker-TFNO，`rank=0.1`，宽 64、4 层 | 同上 |
| `global_local_channel_gate_v1` | 旧 GL 算子核心，宽 64、6 层、3×3 局部核；每层每通道 softmax 两分支权重，零 logits 初始等权 | 双侧 `modes=(12,12)`；同上 |

官方版本对应仓库提交 `b307030d9b7ee530823d6a05bddb8238ba22a743`。官方模型和 GL 的有效频谱覆盖均为纵向 -12 到 11、横向 0 到 11。官方模型为 4 层、GL 为 6 层；不得描述为严格同深度或同参数量比较。实现与数据身份见[主配置](../../configs/research/official_comparison_exp_fwd_031_v1.yaml)。

NeuralOperator 固定为 2.0.0，TensorLy 固定为兼容服务器 Python 3.10 的 0.9.0，`tensorly-torch` 固定为 0.5.0。两机的官方包安装在各自独立的覆盖目录；现有 Torch 环境保持原版本，作业记录实际组合与冻结源码哈希，依赖不兼容则停止准入。

## 训练与停止门

| 用途 | 指标或设置 |
|---|---|
| 损失 | Band 1–10 全矩阵、仅 train 子集拟合标准化的 MSE |
| validation-best 与降 LR | Band 2–10 validation MRE；checkpoint 只在严格改善时更新 |
| 主报告 / 辅助报告 | Band 1–10 MRE / Band 2–10、Band 1 与逐 band MRE |
| 优化 | AdamW、LR `5e-4`、weight decay `1e-5`、micro batch 16、累积 4 次、有效 batch 64；FP32，AMP/TF32 关闭 |
| 固定阶段 | 1–300 epoch，固定基础 LR，保留终态和阶段内 validation-best |
| 自适应阶段 | 301 epoch 从前 300 轮 Band 2–10 validation-best 权重启动，优化器与随机/数据顺序状态重置，LR `1e-4` |
| scheduler | ReduceLROnPlateau：factor 0.5、patience 30、relative threshold 0.001、cooldown 10、min LR `5e-6` |
| 停止 | 301 后至少 100 轮，每 20 轮检查；最低 LR 持续至少 80 轮；Band 1–10 和 Band 2–10 均须通过 80 轮改善、60 轮三窗口稳定及末窗口相对历史最优守卫，连续两次通过 |
| 上限 | 1,000 epoch；上限未过门标记预算耗尽，不称收敛 |

本批 micro batch 与部分历史任务不同，历史执行条件不得写成完全一致。完整停止阈值、模型参数和数据 SHA 以[主配置](../../configs/research/official_comparison_exp_fwd_031_v1.yaml)及冻结后的各项 `config.resolved.yaml` 为准。

## 准入、证据与失败门

两台服务器各先执行六项“模型 × 本机两域 × seed0”短准入，共 12 项。全部通过版本与数据 SHA、显存、有限损失和梯度、门控更新、checkpoint 往返及身份审计后，才释放 36 项正式训练。准入权重不参与正式训练。作业采用排他 claim，失败保留原始制品，不自动重试、迁移或改变 batch/精度。已存在任务和环境不得被覆盖。

每项正式运行保留配置、源码/依赖版本及哈希、数据/split/子集/标准化身份、模型指纹、参数元素与实自由度、逐轮 history、日志、validation-best、300 轮终态与阶段内最优、终态和每 20 轮完整训练状态，以及时间、吞吐、显存和门控权重。完成判断须逐项分为通过停止门、预算耗尽和失败。所有运行证据写入 `runs/{yyyymmdd}/EXP-FWD-031_{YYYYMMDDHHmm}/`，本页不复制原始指标。

逐 band validation MRE 在正式训练终态后，从冻结 validation-best checkpoint 只读重算并写入各 run 的 `per_band_validation.json`；该补充计算不访问 held-out test 标签，也不改写训练选模。其处理源码提交须随结果保留，与训练源码身份分开记录。

**下一决策门：**核查 12/12 准入，再核查 36/36 正式终态与 checkpoint 冻结；随后另行决定 held-out test。训练期间不得用 validation 单独宣称门控收益或物理验证通过。

## 冻结 checkpoint 的增量测试扩展（2026-10-28）

用户已授权在原训练继续期间，按完成状态逐 checkpoint 执行独立测试。原训练配置的 `test_authorized: false` 保持不变；测试入口及其执行源码单独记录。测试只读取 validation-best 冻结权重，不使用 test 调整训练或选模。

| 测试单元 | 目标与证据角色 |
|---|---|
| 四域各自 `idx_test` | 每个来源 checkpoint 均测 K5/F50 × Binary/Graded；同源域为 native held-out，其余为跨域 surrogate test |
| EXP-DAT-007 12k | 六族新几何直接迁移，使用冻结的 Binary 输入表示与现有真实 MPB 标签 |
| EXP-DAT-008 24k | 十二族新几何直接迁移，角色同上 |

预期为 36 个 checkpoint × 6 个目标＝216 个唯一测试单元；每个来源模型与目标组合齐备三个 seed 后，得到 72 组三 seed 汇总。新几何仍缺正式 WSL MPB 复算及分辨率收敛验收，不将代理误差表述为物理精度。

每项训练必须先有正式 `completed.json`，checkpoint SHA-256 与训练 `results.json` 一致，且逐 band validation 只读重算通过原聚合指标的 `1e-3` 百分点容差。目标数据需核验 SHA-256、输入/标签形状、k 点顺序和 split 不重叠。测试采用 FP32、batch 16，每机训练与测试合计最多八卡；只用训练释放的空闲额度，不自动重试失败单元。各测试输出保留来源与目标哈希、逐样本和逐 band 指标、推理耗时及正式运行日志。通过 216/216 完整性审计后才能称全批测试完成。
