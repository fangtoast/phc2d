<!--
  Executable contract for the Xingyi full-cell MPB resolution study.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="full-cell-resolution-convergence-xingyi"></a>

# 星仪全胞 MPB 分辨率收敛实验

本文是[MPB 分辨率收敛与耗时验证计划](../../docs/planning/数据验证/MPB分辨率收敛与耗时验证计划.md)的可执行合同。主层级为 **L3 实验与证据**，依次执行 16 结构资源 pilot 与 100 结构正式确认；两阶段都不能替代本机 WSL 跨环境抽查。

## 一、冻结矩阵

| 项目 | 合同 |
|---|---|
| Stage A 输入 | Binary 8 个、Graded 8 个，按 geometry hash 与 stratum 确定性抽取 |
| Stage B 输入 | Binary `13/12/25`，Graded `25/25`，共 100 个结构 |
| MPB resolution | `32/48/64/96/128` |
| k 点与能带 | full-BZ `81` 点，TE，10 bands；headline 使用 bands 2–10 |
| 物理材料 | 解析 `FullCellStructureSpec.epsilon_at`，不读取模型栅格作为材料真值 |
| 并发 | 星仪 `xyfree` 单个 64 CPU 节点，16 个单线程 worker |
| 数值参考 | 本轮以 128 为参考；它不是解析真值 |
| 推荐候选 | 32/48 只作诊断；按计划只在 64/96 中选择，128 是数值参考 |
| 状态边界 | Stage A 是资源证据；Stage B 集群结果仍需 WSL 跨环境验收 |

配置分别为 [`full_cell_resolution_convergence_xingyi_stage_a_v1.yaml`](../../configs/research/full_cell_resolution_convergence_xingyi_stage_a_v1.yaml) 和 [`full_cell_resolution_convergence_xingyi_stage_b_v1.yaml`](../../configs/research/full_cell_resolution_convergence_xingyi_stage_b_v1.yaml)。作业只允许在固定部署 commit、Meep `1.33.0-beta`、MPB `1.12.0` 和两份冻结 HDF5 输入齐备时运行。

## 二、提交与停止条件

提交入口是 [`xingyi_submit_full_cell_resolution_stage_a.sh`](../../jobs/xingyi_submit_full_cell_resolution_stage_a.sh) 与 [`xingyi_submit_full_cell_resolution_stage_b.sh`](../../jobs/xingyi_submit_full_cell_resolution_stage_b.sh)，默认仅输出 dry-run 命令。真实提交同时要求：

- `PHC2D_SUBMIT_MODE=submit`；
- `PHC2D_AUTHORIZE_RESOLUTION_STAGE_A=YES`；
- Stage B 使用独立的 `PHC2D_AUTHORIZE_RESOLUTION_STAGE_B=YES`；
- 用户队列、磁盘、部署 commit 和输入 SHA-256 预检通过；
- run 目录不存在旧 `results.json` 或 `worker_shards/`。

任一环境版本、输入、有限值、band order、矩阵完整性或 worker 失败即 fail-closed。脚本不自动取消、重启、改参数或恢复。

## 三、产物与下一 gate

每阶段除标准 run 产物外，必须写出 `sample_manifest.csv`、`job_manifest.csv`、`timing.csv`、`convergence_bands.npz`、`convergence_metrics.json`、`failed_case_ledger.jsonl`、`resource_usage.txt` 和 `figures/`。

Stage B 完成后以本机 WSL `Ubuntu` + conda `meep_env` 固定小样本复算，并比较同一结构、k 点、band 和 resolution 的集群/WSL 数值。只有跨环境门通过后，才能冻结推荐 resolution。
