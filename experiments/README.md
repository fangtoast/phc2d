<!--
  Experiment-profile index for reproduction and research workflows.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# Experiments

`experiments/` 是科研实验协议层：定义研究问题、冻结输入与比较边界、指定配置和接受门槛。统一入口是 [`registry.yaml`](registry.yaml)，它只登记当前活跃、仍有科研价值或预计会被论文引用的正式实验。

## 目录职责

| 位置 | 唯一职责 | 不应存放 |
|---|---|---|
| [`registry.yaml`](registry.yaml) | Experiment ID 与 spec、config、run、result、report、decision 的映射 | 原始指标、训练日志、逐 seed 实验副本 |
| [`research/`](research/README.md) | 后续研究的正式协议与 gate | Python 执行代码、模型实现、运行产物 |
| [`reproduction/`](reproduction/wang2026/README.md) | 文献复现协议与证据边界 | 后续研究的通用实现 |

边界必须保持清楚：

- `experiments/` 不是运行产物层；日志、checkpoint、resolved config 和机器指标属于 [`../runs/`](../runs/)。
- `experiments/` 不是 Python 实验执行代码；可复用实现属于 [`../src/phc2d/`](../src/phc2d/)，薄入口按项目规则放置。
- 配置实例属于 [`../configs/research/`](../configs/research/)；当前人类可读结论属于 [`../docs/结果/`](../docs/结果/README.md)，详细证据与历史兼容页属于 [`../docs/results/`](../docs/results/README.md)。
- 一个正式研究问题使用一个稳定 Experiment ID；seed、shard 与同协议重跑都属于该 ID 下的 run，不创建新 ID。

## Experiment ID

| 前缀 | 任务 |
|---|---|
| `EXP-FWD-xxx` | 正向预测 |
| `EXP-QRY-xxx` | 连续查询 |
| `EXP-INV-xxx` | 逆向设计 |
| `EXP-PHY-xxx` | 物理验证与场问题 |
| `EXP-DAT-xxx` | 数据生产与接受 |

新增正式实验前，先在 Registry 中确认是否应扩展已有 ID。只有研究问题或冻结协议发生实质变化时才创建新 ID；日期不能作为主要身份。

## 使用顺序

1. 从 [`registry.yaml`](registry.yaml) 找到 Experiment ID、当前状态与 decision。
2. 阅读 `spec`，确认研究问题、数据合同、比较边界和 gate。
3. 使用登记的 `config` 启动 run；不要从结果文档反推配置。
4. 把原始证据留在 `runs/`，把详细证据写入 `docs/results/`，把 1–3 屏当前结论写入 `docs/结果/`，并回填 Registry 链接。

Registry 中的 `mapping_status: partial` 表示存在已确认缺口，不授权 Agent 猜测关系。`referenced_not_local` 表示结果页引用了该 run，但当前工作区没有对应目录；它不等同于本地已核验。

执行前同时阅读 [分级复现指南](../docs/reproducibility.md) 和 [`AGENTS.md`](../AGENTS.md)。
