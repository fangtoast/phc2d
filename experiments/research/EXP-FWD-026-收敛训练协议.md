---
title: 同条件三seed收敛训练
summary: 在保留固定预算结果的前提下，分别扩展核心模型、分支消融和六种其他模型的验证收敛训练。
page_type: plan
status: draft
audience: 正向代理研究者与实验审阅者
owner: fangtoast
last_reviewed: 2026-10-22
reading_time: 5 minutes
primary_layer: L3
plan_state: active
---

<!--
  Authorized bounded convergence campaign protocol.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="adaptive-convergence"></a>
# 同条件三seed收敛训练

**用户已授权按既定方案开始训练，N40与N56各最多8张GPU。** 本协议冻结新的自适应优化证据，保留全部200/300 epoch历史结果；同条件比较仍须区分固定预算与自适应优化，不能将验证平台期称为全局最优或物理验证。本文为Draft，执行状态以对应run为准。

## 一、任务与公平条件

| 实验 | 平台及任务 | 初始化 | 数据与边界 |
|---|---|---|---|
| EXP-FWD-026 | N40，三个核心模型×六单域×seed 0/1/2，54项 | EXP-FWD-024的300轮运行所选validation-best权重；新AdamW，重设数据顺序 | 每单域8k，沿用原split、训练子集、normalizer、架构、loss、FP32、effective batch 64及micro batch |
| EXP-FWD-027 | N40，Full与四种分支变体×两lane×三个seed，30项 | EXP-FWD-025的300轮validation-best权重；优化器重置 | 沿用mixed每lane16k train、2k validation；validation-only |
| EXP-FWD-028 | N56，六种其他模型×相同六单域×三个seed，108项 | 从零；前300轮保持模型专属基础LR | 与026相同单域8k、split与子集盐；不使用旧mixed单seed权重冒充同条件热续训 |

028模型为`mlp`、`res_mlp_wide`、`lifted_cnn`、`fno2d_band_v2`、`mlp_mixer_band_v1`、`res_mlp_gated_v2`；架构与基础LR核对EXP-FWD-021冻结配置。026/027记录来源best实际epoch、300轮终态、两者哈希、源码版本及优化器未恢复的事实。热启动后的301是累计预算编号，不声称从原300轮优化器状态无缝恢复。

数据合同继承[固定预算协议](EXP-FWD-024-三核心模型固定预算确认.md)。K8/K10/K15不进入训练、验证、normalizer、LR或停止决策。冻结数据SHA、互斥split、子集和train-only normalizer；只读取train与validation标签。

## 二、自适应阶段与停止门

全部任务累计预算上限为1000 epoch，在500轮作进展审阅；审阅不改变已冻结协议。到上限仍未过门标记预算耗尽，不自动追加。

| 合同项 | 冻结值 |
|---|---|
| 自适应阶段 | 301起；028先完成同基础LR的1–300；所有任务统一加载前300预算内validation-best权重，重置AdamW、随机状态和数据顺序generator，再开始新优化 |
| checkpoint选择 | validation Band2–10 MRE最优，精确数值改善才更新；不读test |
| 起始LR | 原基础LR×0.2 |
| 降LR | ReduceLROnPlateau，factor 0.5、patience 30、relative threshold 0.001、cooldown 10、下限原LR×0.01 |
| 收敛检查 | 每20轮；自适应阶段至少100轮；LR已到下限并持续至少80轮 |
| 最近80轮改善 | 相对80轮前的历史最优validation误差改善<0.1% |
| 最近60轮稳定 | 分为三个20轮窗口；窗口均值极差/总均值≤0.5% |
| 主指标保护 | validation Band1–10同时满足改善与稳定条件；末窗口均值不得比所选历史最优恶化超过10% |
| 确认 | 连续两次检查全部通过才停止；改变中的曲线重置计数 |

这些阈值是预先冻结的操作定义，不是普适收敛定理。稳定但显著过拟合的终态不能冒充所选模型收敛；持续改善、非有限值、预算耗尽和验证平台期分别保留原始证据。报告默认Band1–10 MRE三seed均值与样本标准差，Band2–10与逐band数据保留查询。不存在跨模型统一绝对MRE目标。

## 三、执行与资源

配置为[N40](../../configs/research/full_cell_convergence_n40_v1.yaml)与[N56](../../configs/research/full_cell_convergence_n56_v1.yaml)。共享训练入口复用既有Full-Wh runner；新控制器只拥有调度、停止与完整训练状态快照，不拥有模型结构或数据划分。

每平台使用8个单GPU worker，N40共享领取54+30项，N56领取108项。任务完成后立即领取下一个；输出和领取标记使用排他创建，失败保留证据且不自动重试。每项独立run，先对每个模型/域或lane/变体的seed0完成一批训练和完整validation准入，全部通过才放行对应平台训练。一个失败准入不得被Slurm其他worker成功掩盖。

分配每GPU N40 6 CPU、N56 8 CPU、调度默认内存；worker壁钟上限7天，任务队列耗尽立即退出释放GPU，不持续空等。该上限用于多个任务顺序执行，不将其当工期预测。数据仅将已选train与validation缓存到CPU内存，避免反复HDF5读取，不扩大batch、启用AMP/TF32或改训练样本。资源日志记录实际GPU利用率、显存、功耗与吞吐，允许显示小模型未满载的真实情况。N56共享JuiceFS不逐文件解包源码，而保存带SHA-256的归档，在每个计算节点的临时目录解包；日志与训练制品始终落到共享campaign，结束后移除节点临时源码。

每20轮和停止时保存模型、AdamW、scheduler、Python/NumPy/Torch/CUDA随机状态、DataLoader generator及控制器状态。原始来源没有AdamW状态；新快照为未来精确恢复保留信息，本轮没有自动恢复失败任务的入口。学习率、优化器、时间和源best epoch须随新证据记录。

## 四、验收与证据边界

机器制品位于`runs/{yyyymmdd}/EXP-FWD-026_{timestamp}/`、027与028对应目录；包括冻结profile、manifest、数据记录、来源哈希、准入、独立训练history/results/metrics、best/terminal/full-state checkpoint和资源观测。训练结果按任务审计连续历史、有限数值、实际停止原因、checkpoint哈希、模型身份、三seed及来源兼容；不能只凭Slurm COMPLETED。

本次只放行训练与validation。027继续validation-only；026/028的后续native测试须在全部训练与checkpoint冻结后单独决定，不用于LR、收敛或选模。历史测试已被查看，不能称全新盲测。正式工程验收在WSL Ubuntu + meep_env；GPU准入和远程产物不替代该验收，也不构成MPB/Meep物理证据。

L4实现将自适应控制、来源兼容与终态保存抽到独立模块。既有Full-Wh runner和函数仍超过架构警戒线；本次保留原checkpoint schema、历史实验与runner兼容，不重写整个历史训练/评估循环。剩余拆分边界是通用训练epoch与旧实验后处理，须由单独兼容回归支持。
