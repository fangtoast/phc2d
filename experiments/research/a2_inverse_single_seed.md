<!--
  A2 single-seed inverse benchmark experiment entry.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# A2 单 Seed 逆向公平对照

## 入口

配置为
[`configs/research/a2_inverse_single_seed_v1.yaml`](../../configs/research/a2_inverse_single_seed_v1.yaml)，
统一入口为：

```bash
PYTHONPATH=src python -m phc2d.cli inverse benchmark \
  --config configs/research/a2_inverse_single_seed_v1.yaml \
  --stage preflight
```

后续阶段必须显式传入 preflight 返回的同一 `--run-dir`：

```bash
PYTHONPATH=src python -m phc2d.cli inverse benchmark \
  --config configs/research/a2_inverse_single_seed_v1.yaml \
  --stage fake-smoke \
  --run-dir runs/YYYYMMDD/a2_inverse_single_seed_v1_YYYYMMDDHHmm
```

支持的阶段为 `preflight`、`fake-smoke`、`real-smoke`、`train`、`sample`、
`replay` 和 `aggregate`。所有正式测试与执行均在 WSL `Ubuntu` 的 conda
`meep_env` 中运行。

## 恢复语义

- 候选 manifest 冻结时同时保存 SHA-256；replay 前重新核验。
- 每个唯一结构以结构 hash 为 `sample_id`。
- `structure_cache/` 中已完成的 candidate shard 会被恢复，退出后只补缺失项。
- direct inverse 的 128 次重复输出在二值去重后只保留一个物理解，不重复调用 MPB。

## 当前状态

本文件只描述实验入口，不承诺当前运行状态。状态必须从具体 run 目录的
`preflight_decision.json`、smoke `metrics.json`、candidate manifest、replay
明细和 `aggregate/results.json` 判定。
