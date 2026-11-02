---
title: EXP-FWD-033 K5与K20等量混合分支消融
summary: 每条材料路线使用K5 4k加K20 4k，比较Full、独立分支和容量对照。
page_type: research_experiment
status: draft
audience: 正向模型研究者与实验审阅者
owner: fangtoast
last_reviewed: 2026-11-02
reading_time: 5 minutes
primary_layer: L3
---

<!--
  Equal-count K5 and K20 mixed branch ablation protocol.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-FWD-033：等量混合分支消融

**问题：训练总量固定为 8,000 时，K5+K20 混合训练的 Full、独立分支和容量对照，与单域 8k 及历史混合 16k 的表现有何差异？** 本协议为 Draft；配置和来源选择审计不表示训练、测试或物理验收已完成。

## 数据与模型合同

Binary、Graded 分开训练；每条路线从 K5 与 K20 原域已审计的 8k train 子集各选 4k，同一选择用于全部模型和 seed。父 8k 按 EXP-FWD-026 的真实训练子集哈希核验；子集使用 `exp-fwd-021-domain-ranking-v1` 的 sample-ID SHA-256 嵌套排序。各来源保留原有 1k validation，合计 2k；normalizer 只由选定的 8k 混合 train 标签计算，不读取 test 标签。来源文件、split、原行号、选择规则、父子哈希和 normalizer 必须随 run 冻结。

| 每路线结构 | 唯一结构数 | 三 seed 训练数 |
|---|---:|---:|
| Full 3×3/7×7 | 2 | 6 |
| Global-only、Global-only 容量对照 | 2 | 6 |
| Local-only 3×3/7×7、对应容量对照 | 4 | 12 |
| **两路线合计** | **16** | **48** |

架构、参数元素计数和容量门继承 [EXP-FWD-032](EXP-FWD-032-单域全局局部消融.md)；冻结时重新计算参数量、factory、模型指纹与 state-dict schema。Global-only 不含局部核，不因核尺寸重复训练；其容量对照调整输出头，不能称结构等价。

## 训练、测试与对照

从零训练，FP32，AdamW，基础 LR 5e-4、weight decay 1e-5、micro/effective batch 64，normalized full-matrix MSE。1–300 epoch 后加载该阶段 validation-best，重置优化器与随机状态，进入 [EXP-FWD-031](EXP-FWD-031-官方算子与通道融合对比.md) 自适应协议；累计上限 1000 epoch，以 validation Band 2–10 MRE 选 checkpoint。未通过停止门者记录预算耗尽。

每个 checkpoint 测试 K5/F50 Binary/Graded 四个 `idx_test` 与 DAT-007 12K、DAT-008 24K，共 288 个唯一 cell、96 组三 seed 汇总。本轮不包含 K20 独立测试目标。主指标 Band 1–10 MRE；保留 Band 2–10、逐 band、逐样本、耗时与资源记录，分别报告 K5 已见来源 held-out、F50 跨域和两套新几何。

同一混合 8k 合同内按 seed 比较 Full 与普通单支、容量对照；另比较 [EXP-FWD-032 单域 8k](../../docs/results/forward/EXP-FWD-032-单域分支消融.md) 与历史混合 16k。数据分布、normalizer、validation 构成及历史优化合同可能不同；历史 3×3 逐 cell split 哈希缺口保留，未经复核不得称严格配对。不得用 test 调整训练、停止或选模。

## 准入与执行边界

N40 负责 Binary、N56 负责 Graded，每机 24 项，每项单 GPU；每机最多 8 GPU，分别每卡 6/8 CPU。各平台的 8 个 seed 0 结构均须通过单批训练、完整 validation、显存、梯度和 checkpoint 往返准入；资源报告区分探针测量和 GPU 小时预测。训练与测试共享 GPU 上限，失败保留证据，不自动重试或迁移。

实际来源审计记录为 `runs/20261102/EXP-FWD-033_202611021031/source_selection_audit.json`，其中记录四个来源各 4k 原行号。[冻结复核](../../runs/20261102/EXP-FWD-033_202611021031/freeze_review.json)确认两平台各 24 项训练、144 个测试目标，混合数据逐行和 normalizer 复核通过，8 个结构及六目标跨平台身份匹配。N40 准备阶段曾因登录环境 CUDA 库冲突在创建 suite 前退出；加载平台 CUDA 模块后完成准备，原错误记录保留，未创建作业。

[资源预算](../../runs/20261102/EXP-FWD-033_202611021031/resource_budget.json)的训练上限为每平台 24,000 单任务 epoch；每平台测试预测与标签 float64 数组估计约 14.93 GB，不含其他文件及文件系统开销。不据单批准入速度估算完整 GPU 小时。远程准入、训练和测试分别需要与冻结清单绑定的明确授权，尚未授权时不能提交作业。

WSL Ubuntu `meep_env` 为正式工程验收环境；远程准入及训练产物不能替代工程验收。已有 test 已被查看；新几何代理误差与 WSL MPB 复算、分辨率收敛和物理验收分开报告。
