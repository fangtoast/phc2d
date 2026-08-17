<!--
  Executable contract for the Xingyi full-cell MPB resolution Stage-A pilot.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="full-cell-resolution-convergence-xingyi"></a>

# 星仪全胞 MPB 分辨率收敛阶段 A

本文是[MPB 分辨率收敛与耗时验证计划](../../docs/planning/数据验证/MPB分辨率收敛与耗时验证计划.md)的阶段 A 可执行合同。主层级为 **L3 实验与证据**；本轮只授权 16 个结构的计费资源 pilot，不授权 100 个结构的阶段 B。

## 一、冻结矩阵

| 项目 | 合同 |
|---|---|
| 输入 | Binary 8 个、Graded 8 个，按 geometry hash 与 stratum 确定性抽取 |
| MPB resolution | `32/48/64/96/128` |
| k 点与能带 | full-BZ `81` 点，TE，10 bands；headline 使用 bands 2–10 |
| 物理材料 | 解析 `FullCellStructureSpec.epsilon_at`，不读取模型栅格作为材料真值 |
| 并发 | 星仪 `xyfree` 单个 64 CPU 节点，16 个单线程 worker |
| 数值参考 | 本轮以 128 为参考；它不是解析真值 |
| 状态边界 | `stage_a_resource_pilot_complete_pending_review`，待 WSL 抽查与人工资源审阅 |

配置为 [`full_cell_resolution_convergence_xingyi_stage_a_v1.yaml`](../../configs/research/full_cell_resolution_convergence_xingyi_stage_a_v1.yaml)。作业只允许在固定部署 commit、Meep `1.33.0-beta`、MPB `1.12.0` 和两份冻结 HDF5 输入齐备时运行。

## 二、提交与停止条件

提交入口是 [`xingyi_submit_full_cell_resolution_stage_a.sh`](../../jobs/xingyi_submit_full_cell_resolution_stage_a.sh)，默认仅输出 dry-run 命令。真实提交同时要求：

- `PHC2D_SUBMIT_MODE=submit`；
- `PHC2D_AUTHORIZE_RESOLUTION_STAGE_A=YES`；
- 用户队列、磁盘、部署 commit 和输入 SHA-256 预检通过；
- run 目录不存在旧 `results.json` 或 `worker_shards/`。

任一环境版本、输入、有限值、band order、矩阵完整性或 worker 失败即 fail-closed。脚本不自动取消、重启、改参数、恢复或进入阶段 B。

## 三、产物与下一 gate

阶段 A 除标准 run 产物外，必须写出 `sample_manifest.csv`、`job_manifest.csv`、`timing.csv`、`convergence_bands.npz`、`convergence_metrics.json`、`failed_case_ledger.jsonl`、`resource_usage.txt` 和 `figures/`。

完成后先审阅实际 CPU-hours、wall time、峰值 RSS、各 resolution 耗时和 provisional 精度门；再以本机 WSL `Ubuntu` + conda `meep_env` 固定小样本复算。只有资源报告获人工接受并再次获得计费授权，才能实现并提交 100 结构阶段 B。
