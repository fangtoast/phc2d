<!--
  A2 objective-bridge seed-0 adjustment experiment.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# A2 Objective-Bridge Seed0 调整实验

## 目的与边界

本实验只修复 A2 生成器的训练目标与候选塌缩问题。A1 seed0 forward/query
checkpoint、retrieval 和 direct inverse 均冻结；原 A1 `formal_completed` 与原 A2
`single_seed_formal_replay` 结果不覆盖、不回填。

新证据标签固定为 `single_seed_adjustment_pilot`。PINO、field inverse、B4 checkpoint
与 physics-aware rerank 不进入本实验。

## 配置与入口

- 配置：[`../../configs/research/a2_inverse_objective_bridge_seed0_v1.yaml`](../../configs/research/a2_inverse_objective_bridge_seed0_v1.yaml)
- 统一入口：`phc2d inverse benchmark`
- 正式环境：WSL `Ubuntu` + conda `meep_env`
- 固定 seed：`0`
- 派生 run：`runs/{yyyymmdd}/a2_inverse_objective_bridge_seed0_v1_{YYYYMMDDHHmm}/`

```bash
PYTHONPATH=src python -m phc2d.cli inverse benchmark \
  --config configs/research/a2_inverse_objective_bridge_seed0_v1.yaml \
  --stage preflight
```

后续阶段必须显式恢复同一 `--run-dir`：

```bash
PYTHONPATH=src python -m phc2d.cli inverse benchmark \
  --config configs/research/a2_inverse_objective_bridge_seed0_v1.yaml \
  --stage train \
  --run-dir runs/YYYYMMDD/a2_inverse_objective_bridge_seed0_v1_YYYYMMDDHHmm
```

支持阶段为 `preflight`、`fake-smoke`、`train`、`sample`、`real-smoke`、
`pilot-replay`、`pilot-aggregate`、`replay` 和 `aggregate`。

## Objective-Bridge

single-mode 与 mixture 从各自原 A2 validation-selected checkpoint warm start，
固定 LR `1e-4`，训练最多 50 epochs，在 epoch `10/30/50` 保存候选 checkpoint。
每个训练 condition 生成 16 个候选，mixture 的每个 condition 均覆盖全部 4 个 mode。

forward FNO 对全部 16 个候选计算 bands 2–10 主 objective。query FNO 的逐 k 点计算
采用固定 seed 的组内候选抽样，每个 condition 每步抽 1 个候选，并随 epoch/batch
轮换；这是 cross-surrogate consistency 的确定性随机估计，不改变 forward 主目标、
16 候选多样性损失或完整 validation 门禁。

checkpoint 先通过 candidate-health 硬门禁，再按 validation surrogate MRE 选择。
任何方法没有合格 checkpoint 时写入 `blocked_candidate_health`，不得启动 MPB。

## 晋级顺序

1. preflight 校验 dataset、split、manifest、A1/warm-start checkpoint、k-path 与 solver hash；
2. fake-MPB smoke；
3. 串行训练 single-mode、mixture；
4. 完整 validation 选型与固定 100 test targets 的 `K=128` 候选冻结；
5. `2 targets × 2 candidates × 2 methods` real-MPB smoke；
6. 四个 strata 各前 5 个 target 的 20-target pilot；
7. 只有 mixture pilot 通过才允许完整 100-target replay；
8. aggregate 复用冻结的 retrieval/direct，并与原 A2 single/mixture 做 paired 对比。

20-target pilot 与完整 replay 共用结构 SHA-256 cache；恢复时只补缺失 candidate shard。
训练与 MPB 不得并发。

## 状态判定

本文件不声明运行已通过。当前状态必须分别从以下产物判定：

- `preflight_decision.json`
- `train/candidate_health_decision.json`
- `candidates/candidate_health_decision.json`
- `replay_smoke/metrics.json`
- `pilot/promotion_decision.json`
- `aggregate/results.json`

原结果与 checkpoint 只读保留；新 run outputs/checkpoints 不进入 Git。
