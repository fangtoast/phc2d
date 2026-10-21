---
title: "EXP-FWD-021：8k 公平多域训练与六域验证"
summary: "在固定 8,000 条训练样本与 seed 0 下，比较九个 Full-Wh 模型的跨 K、跨路线和六域联合训练泛化。"
page_type: experiment
plan_state: active
status: draft
audience: "正向代理研究者、复现执行者与结果审阅者"
owner: "研究生维护，导师审阅研究主张"
last_reviewed: 2026-09-15
reading_time: 7 minutes
primary_layer: L3
---

<!--
  EXP-FWD-021 fair multidomain Full-Wh experiment contract.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-FWD-021：8k 公平多域训练与六域验证

## 结论与当前状态

本实验已经获得实现、N56 `gpu_5090` admission 与过门后分批执行授权，但在单卡 admission 闭合前保持 `authorized_pending_implementation_and_admission`。配置、脚本或冻结 manifest 不等于训练完成；只有 63 个训练任务和 378 个 held-out surrogate cell 的机器制品全部通过审计，才能写入结果页。

资源合同在 2026-09-15 19:03（UTC+8）经人工修订：首次 `hp_5090` 提交被集群拒绝，且该名称不是当前 Slurm 注册分区；随后人工明确授权改投 `gpu_5090` 或 `gpu_4090`。本次选择 `gpu_5090`，即使提交时无空闲卡也允许进入等待队列；不自动切换 `gpu_4090` 或 N40。

资源合同在第二次提交门后再次经人工修订：N56 `gpu_5090` 禁止显式 `--mem`，因此删除 60 GB 请求并接受调度器的单卡默认 126 GB。多卡加速采用任务级并行，每个训练任务仍使用 1 GPU、最多同时运行 7 个独立任务；禁止改为单任务 DDP，以保持 effective batch 64、epoch 采样和 checkpoint 选择语义不变。

研究问题是：在每个 checkpoint 的总训练样本严格固定为 8,000 时，跨 Fourier cutoff、跨 Binary/Graded 路线和六域联合训练，能否缓解既有单域模型的跨域退化？本实验只产生 `seed0_exploratory_surrogate` 证据，不涉及 real-MPB、Meep、物理结论或模型 promotion。

## 模型与优化合同

| 项目 | 冻结值 |
|---|---|
| 模型 | `mlp`、`res_mlp_wide`、`lifted_cnn`、`fno2d_band_v2`、`mlp_mixer_band_v1`、`res_mlp_gated_v2`、`convnext_band_v1`、`tfno2d_band_v3`、`global_local_band_operator_v1` |
| Global-Local | 本实验显式固定 `local_kernel: 3`；当前注册默认值为 7，本实验不修改注册表 |
| 排除 | component ablation 身份；需要 `lane_id` 的 `dual_lane_band_operator_v1` |
| 初始化 | 全部从零训练，seed 0，不加载历史 checkpoint |
| 训练 | AdamW、200 epochs、FP32、AMP/TF32 off、effective batch 64、现有模型专属学习率 |
| checkpoint | 训练源域等权宏平均 validation Bands 2--10 MRE 最优；另保留 epoch 200 |
| 收敛 | epoch 181--190 与 191--200 的宏平均 Bands 2--10 MRE 相对变化不超过 0.005 |

若固定 200 轮未通过收敛门，run 保持 `training_complete_convergence_not_met`，可以进入预注册的探索性评估，但不得改写为收敛成功或自动追加训练。

## 七种公平 8k 制度

| 阶段 | regime | 冻结训练构成 |
|---|---|---|
| 1 | `S1-B` | K5-B 4,000 + K20-B 4,000 |
| 1 | `S1-G` | K5-G 4,000 + K20-G 4,000 |
| 2 | `S2-K5-BG` | K5-B 4,000 + K5-G 4,000 |
| 2 | `S2-K20-BG` | K20-B 4,000 + K20-G 4,000 |
| 3 | `S3-B` | K5-B 2,667 + K20-B 2,667 + F50-B 2,666 |
| 3 | `S3-G` | K5-G 2,667 + K20-G 2,667 + F50-G 2,666 |
| 3 | `S3-BG` | K5-B 1,334 + K20-B 1,333 + F50-B 1,333 + K5-G 1,333 + K20-G 1,334 + F50-G 1,333 |

每种制度均为 8,000 条。第二阶段是两个独立 8k 任务，不使用 18k 合同。各域由同一 `sample_id` 哈希排序产生嵌套子集；同一制度的九个模型使用完全相同的样本身份，且所有选中样本与 validation/test 零重叠。

## 多域语义与指标

- 模型只读取结构网格，不输入 K 值或 Binary/Graded 域标识。
- 每个制度仅由其 8,000 条训练标签拟合一个逐坐标 pooled `Wh` mean/std；推理时禁止按目标域更换 normalizer。
- checkpoint 记录模型/factory 身份、架构参数、数据与 split 哈希、子集哈希、pooled normalizer 哈希和执行起始 commit。
- checkpoint 选择只读取训练源域 validation；各源域先独立计算指标，再等权宏平均，F50 的较大 validation 不获得额外权重。
- 每个冻结 checkpoint 在 K5/K20/F50 × Binary/Graded 六个 held-out test 域评估，共 `63 × 6 = 378` cells。
- 每个 cell 始终报告 Band 1、Bands 2--10、Bands 1--10 与十条逐 band MRE；不得因结果改变主口径。
- 跨六域主汇总使用等权 domain macro，不以样本数加权；同 regime/target 的模型比较可做同样本 paired bootstrap，但不能替代跨 seed 稳定性。

## 执行门与失败策略

1. 先冻结配置、六份数据身份、七种制度、63 项训练 manifest 与全部子集/normalizer。
2. N56 重新核验数据 SHA-256、字段、shape、split 和零重叠；held-out 标签在训练冻结阶段不得读取。
3. 在 `gpu_5090` 对九个模型分别执行一次 `S3-BG` admission，证明六域读取、前向、反向、有限 loss/gradient 和标准制品。
4. admission 全部通过后，按阶段 18、18、27 项执行；每个数组最多 16 项、并发不超过 7，前批审计通过后才提交下一批。
5. 63 个 checkpoint 冻结后，各 test target 只读取一次并进入只读 cache，再执行 378-cell 评估。
6. 任一作业失败、非有限值、制品缺失、身份冲突或 hash 不符时 fail closed；禁止自动重试、取消、续训、未经授权换分区、转移 N40 或修改科学判据。

资源合同为 N56 `gpu_5090`、每项 1 GPU、8 CPU、调度器默认 126 GB、最长 24 小时，沿用 CUDA 12.8 / PyTorch 2.7.1 环境。最多 7 项并发，从而在不改变单任务优化语义的前提下并行使用最多 7 张 GPU。允许在无空闲卡时进入 `gpu_5090` 等待队列；除非再次取得人工授权，否则不切换其他分区。

## 预期证据

机器证据写入 `runs/EXP-FWD-021/{date}/...`，包括冻结配置、dataset/subset/normalizer/task/checkpoint/test-cache manifests、63 个标准训练 run、378 个评估 cell、逐 band 指标、模型×目标域表、domain-macro 汇总、paired bootstrap 和资源汇总。

在 63/63 与 378/378 完整前不创建结果结论页。最终结果仍须标记为 `seed0_exploratory_surrogate_only`，并明确 `model_promotion=false`、`physical_claim=false`。
