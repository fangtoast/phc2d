---
title: "EXP-FWD-007：K5 全模型训练与六域交叉验证"
summary: "冻结 K5 的 17 个训练任务，并以 K5/K20/F50 六个 held-out 数据域完整重推 510 个 surrogate cells。"
page_type: plan
plan_state: active
status: draft
audience: "研究执行者、导师与结果审阅者"
owner: "研究生维护"
last_reviewed: 2026-08-23
primary_layer: L3
---

<!--
  EXP-FWD-007 training and six-domain validation contract.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-FWD-007：K5 全模型训练与六域交叉验证

本实验将完成 K5 Binary/Graded 的 17 个 `seed=0`、dense-head、FP32 正向训练，并对 K5、K20、F50 的 B/G 六个 held-out 域重推 `85×6=510` 个 cells。训练与推理仅构成 held-out surrogate evidence，不构成 WSL real-MPB replay、物理验证或 inverse 成功。

| 冻结项 | 合同 |
|---|---|
| 训练 | 8 baseline B/G、8 single-lane upgrade B/G、1 joint；三波 `W1/W2/W3`，先完成 17 个 one-batch admission |
| 数据 | K5 B/G 各 `Wh[10000,81,10]`、`8000/1000/1000`、64×64；哈希见 K5 配置 |
| 评估 | 85 unique checkpoints、510 cells、10 N40 shards；joint 按目标 lane 选择其 B/G source normalizer 并传入 `lane_id` |
| 验收 | `17/17`、`85/85`、`510/510`、`10/10`，零 missing/duplicate/OOM/nonfinite；旧 272-cell 的数值回归必须通过 |
| 运行状态 | `user_authorized_surrogate_training_pending_wsl_acceptance`；用户已允许 surrogate 训练，但 WSL acceptance 仍未完成 |

## 入口与门

- K5 配置：`configs/research/full_cell_k5_forward_*_seed0_v1.yaml`；控制 profile：`configs/research/exp_fwd_007_k5_campaign_profile_v1.yaml`。
- 训练控制面：`src/phc2d/forward/full_cell_training_campaign.py`；六域控制面：`src/phc2d/forward/full_cell_six_domain_n40.py`。
- N40 脚本：`jobs/paracloud_submit_exp_fwd_007_k5.sh`、`jobs/paracloud_submit_exp_fwd_007_six_domain.sh` 与只读 `jobs/paracloud_exp_fwd_007_status.sh`。
- 仅在单独提供 `PHC2D_AUTHORIZE_EXP_FWD_007=YES` 和授权编号时，脚本才会调用 `sbatch`。本实验协议本身不授予提交权限。

每波结束必须先审计其结果，再由人工决定是否进入下一波。训练提交后每 30 分钟只读检查 Slurm、制品覆盖和 finalizer 状态；异常进入人工处理门，不自动重试、取消、重提或改参。

## 结果解释边界

K5/K20 是独立随机样本，K5↔K20 差异混合 cutoff 与分布变化，不构成 paired-K 因果效应。F50 的训练预算不同，跨 family 指标只用于迁移诊断，不作为公平训练排名。`dual_lane_band_operator_v1` 单列为 joint-source-family，不用于 B→G/G→B 泛化结论。

结果只在 [`../../docs/results/forward/EXP-FWD-007-K5训练与六域验证.md`](../../docs/results/forward/EXP-FWD-007-K5训练与六域验证.md) 更新一次；原始 510 行证据保留在 run 目录。
