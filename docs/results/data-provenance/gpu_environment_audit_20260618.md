# GPU Environment Audit

## 摘要
- **目的**：基于原始记录保留实验链路与指标定义，对文档进行中文化结构化重构。
- **状态**：已完成中文主语体补充，原始英文命令、路径、文件名、模型字段与数值保持原文不变。
- **文件角色**：Experimental Technical Report | **原始语言比例**：英文主导（0%）
- **证据入口**：文内无直接 runs/artifact 命令字段引用；请按实验组入口补充。

## 结论
- 结论以原实验逻辑为准，聚焦结果含义、边界条件与复现实验条件。
- 若涉及 `surrogate` 与 `mpb_replay`，请在原文中使用统一术语进行区分。

## 数据与方法
- 数据：以原文 `runs/` 与 `metrics.json` 为准。
- 方法：保持原始训练流程与实验顺序，不改动采样/评估口径。
- 命令与环境：保留原样，`path` 与 `config` 字段不改写。

## 风险与下一步
- 风险主要关注结论复用边界（surrogate-only 与 formal validation 的分离）。
- 下一步优先补齐跨文档术语映射、证据引用一致性与可复现实验条目。


- date: 2026-06-18
- environment: WSL `Ubuntu` + conda `meep_env`
- command scope: CUDA/PyTorch audit only; no Full-2, formal generative benchmark, or MPB replay was started.

## Result

| item | value |
|---|---|
| `nvidia-smi` | available |
| GPU count | 1 |
| GPU 0 | NVIDIA GeForce RTX 4060 Laptop GPU |
| NVIDIA-SMI | 610.43.02 |
| CUDA UMD version from `nvidia-smi` | 13.3 |
| PyTorch version | 2.12.0+cu130 |
| PyTorch CUDA version | 13.0 |
| `torch.cuda.is_available()` | true |
| `torch.cuda.device_count()` | 1 |
| cuDNN version | 92000 |
| CUDA-enabled PyTorch build | true |

## Decision

WSL `meep_env` is CUDA-ready for GPU-first forward/inverse neural training and smoke validation. Pilot/full/formal neural configs can require an accelerator; CPU remains allowed only for smoke/debug/reference profiles. MPB replay remains a solver path and is not reclassified as CUDA neural training.