<!--
  A2 multi-surrogate inverse-design ablation profile.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# A2 多 Surrogate 逆向设计消融

## 研究问题与证据边界

本实验判断 A2 的 FNO ranker 是否应被其他频率 surrogate 替换。统一逆向链路为：

`Wh target → reconstruction-trained mixture generator → K=128 rho_wedge → frozen surrogate ranking → Top-M=5 → real-MPB replay`

阶段 A 只比较 7 个可恢复完整 `Wh` 的频率 surrogate。`unet_field_v2`、
`fno_field_v2` 和 adapted PINO 不进入该矩阵；它们继续受 B4 admission gate 约束。
adapted PINO 是 `fno_field_v2` 主干上的训练制度，不作为独立架构计数。

当前代码和配置状态为 `config_ready`。本说明不表示 20-target real-MPB pilot、
3 个 objective arm 训练或 2 模型 × 3 seeds 正式 replay 已经完成。

## 固定输入

- 阶段 A 配置：
  [`../../configs/research/a2_inverse_surrogate_screen_v1.yaml`](../../configs/research/a2_inverse_surrogate_screen_v1.yaml)
- 阶段 B/C 配置：
  [`../../configs/research/a2_inverse_surrogate_objective_v1.yaml`](../../configs/research/a2_inverse_surrogate_objective_v1.yaml)
- validation 目标：
  [`../../configs/research/target_manifests/inverse_surrogate_val100_v1.json`](../../configs/research/target_manifests/inverse_surrogate_val100_v1.json)
- 正式 test100：
  [`../../configs/research/target_manifests/inverse_paired_test100_v1.json`](../../configs/research/target_manifests/inverse_paired_test100_v1.json)
- 正式环境：WSL `Ubuntu` + conda `meep_env`
- GPU：`cuda:0`，禁止静默回退 CPU

validation manifest 从 `idx_val` 按模型无关的 bands 2–N 相邻 k 点变化量分四分位，
每层确定性抽取 25 个目标。固定 pilot 位置为每层前 5 个，共 20 个。test100 不参与
surrogate 或 checkpoint 选择。

## 阶段 A：固定候选池筛选

preflight 校验 dataset、split、validation manifest、7 个 seed-0 checkpoint、
reconstruction generator、physical k-path、重复 Gamma 端点和 solver protocol 的
SHA-256。query adapter 只允许真实 k 坐标，不允许 index-query 回退，并严格恢复
`[batch, k_points, n_bands]`。

```bash
PYTHONPATH=src python -m phc2d.cli inverse benchmark \
  --config configs/research/a2_inverse_surrogate_screen_v1.yaml \
  --stage preflight
```

记下返回的 run 目录。其余阶段均显式复用它：

```bash
RUN_DIR=runs/YYYYMMDD/a2_inverse_surrogate_screen_v1_YYYYMMDDHHmm

PYTHONPATH=src python -m phc2d.cli inverse benchmark \
  --config configs/research/a2_inverse_surrogate_screen_v1.yaml \
  --stage sample --run-dir "$RUN_DIR"

PYTHONPATH=src python -m phc2d.cli inverse benchmark \
  --config configs/research/a2_inverse_surrogate_screen_v1.yaml \
  --stage screen --run-dir "$RUN_DIR"

PYTHONPATH=src python -m phc2d.cli inverse benchmark \
  --config configs/research/a2_inverse_surrogate_screen_v1.yaml \
  --stage fake-smoke --run-dir "$RUN_DIR"

PYTHONPATH=src python -m phc2d.cli inverse benchmark \
  --config configs/research/a2_inverse_surrogate_screen_v1.yaml \
  --stage real-smoke --run-dir "$RUN_DIR"

PYTHONPATH=src python -m phc2d.cli inverse benchmark \
  --config configs/research/a2_inverse_surrogate_screen_v1.yaml \
  --stage pilot-replay --run-dir "$RUN_DIR"

PYTHONPATH=src python -m phc2d.cli inverse benchmark \
  --config configs/research/a2_inverse_surrogate_screen_v1.yaml \
  --stage promote --run-dir "$RUN_DIR"
```

`sample` 只用 reconstruction checkpoint 生成一次 100 × 128 候选；`screen` 令 7 个
策略读取同一个 `candidates.json`。`fake-smoke` 和 `real-smoke` 固定为 2 targets ×
2 candidates × 7 strategies。20-target pilot 对 7 个 Top-5 并集按结构哈希去重，
理论求解上限 700。

阶段 A 关键产物为：

- `surrogate_manifest.json`
- `candidate_pool_manifest.json`
- `screen/strategy_metrics.csv`
- `screen/selection_overlap.csv`
- `pilot_replay_metrics.csv`
- `promotion_decision.json`

`promote` 只有在 replay backend 为 `real_mpb` 且 validation boundary 为正式物理
回放时才放行。晋级组固定为 FNO 参考、最佳非 FNO Full-Wh、最佳 query。置信区间
重叠时依次比较绝对 replay gap、推理时间和稳定的策略 ID。

## 阶段 B：三个 Primary-Only Objective Arm

阶段 A promotion 完成后派生三个公平训练配置：

```bash
PYTHONPATH=src python -m phc2d.cli inverse benchmark \
  --config configs/research/a2_inverse_surrogate_objective_v1.yaml \
  --stage derive-objective --run-dir "$RUN_DIR"
```

派生配置位于 `$RUN_DIR/objective/configs/`。每个 arm 均从同一 reconstruction
checkpoint warm start，使用 seed 0、50 epochs、LR `1e-4`、每条件 16 候选，并
固定 `cross_surrogate_weight=0`。其他 surrogate 只能作为训练后只读诊断器。

每个 arm 的执行入口仍为已有命令：

```bash
PYTHONPATH=src python -m phc2d.cli inverse objective-train \
  --config "$RUN_DIR/objective/configs/SURROGATE_ID.yaml"
```

训练完成后必须另行完成 validation checkpoint 选择、candidate-health gate、
100-target K=128 冻结候选和 20-target real-MPB pilot，才能生成
`objective_promotion_decision.json`。历史 FNO + query 双-surrogate bridge 只作为
单列控制组，不能与 primary-only 公平臂合并。

## 阶段 C：两模型三种子正式验证

只有 `objective_promotion_decision.json` 选出一个非 FNO challenger 后，才允许派生：

```bash
PYTHONPATH=src python -m phc2d.cli inverse benchmark \
  --config configs/research/a2_inverse_surrogate_objective_v1.yaml \
  --stage derive-formal --run-dir "$RUN_DIR"
```

该阶段先写出 `formal/formal_execution_manifest.json`，所有条目初始状态均为
`blocked_pending_seed_specific_checkpoint`。seed 1/2 必须分别完成 reconstruction
pretrain 和 objective finetune；checkpoint 只按 validation 选择。冻结 test100
只能用于最后一次等预算 replay。

正式主指标为 replay MRE、surrogate MRE、replay gap、validity、effective K、
unique rate、Hamming diversity、训练/采样/排序/MPB 时间和峰值显存。统计采用
同目标配对 bootstrap 95% CI、配对效应量以及三种子 `mean ± std`。只完成 seed 0
时状态必须为 `single_seed_pilot`。

## 阻断与完成判定

- GPU 训练回退 CPU：`blocked`
- query 使用非物理 k 或丢失重复 Gamma：`blocked`
- checkpoint、candidate pool、target manifest、solver protocol 或 k-path 哈希不符：`blocked`
- real-MPB 失败或覆盖不完整：`blocked`
- candidate-health gate 未通过：`blocked`
- 只降低 surrogate loss 或 Maxwell residual：不构成 inverse 成功证据
- 未完成 2 模型 × 3 seeds × test100 real-MPB：不得声明跨种子优势

run outputs、checkpoint 和原始 gallery 保留在 ignored `runs/` 下，不覆盖已有 A1/A2
结果。阶段 A、B、C 的代码/配置变更应分别做范围严格的提交。
