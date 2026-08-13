<!--
  Xingyi execution contract for the binary and graded full-cell 50k datasets.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

> 2026-08-07 后的新 campaign 应使用 [`../../docs/hpc_physics_campaigns.md`](../../docs/hpc_physics_campaigns.md) 的通用 plan → solve array → finalize 合同。本文件中的单作业 64-worker 命令保留为已完成 50k campaign 的历史执行证据，不再作为后续规模化任务的默认架构。

<a id="full-cell-dual-lane-50000-xingyi"></a>

# Xingyi 全胞双路线 50k 数据生成合同

本文冻结二值与连续渐变路线各 50,000 条 real-MPB 数据在 Xingyi 超算上的准备、提交、恢复与验收边界。它是 **L3 实验/evidence 合同**，不表示 50k 数据已经开始或完成。

**目录：** [目标与边界](#一目标与边界) · [当前资源证据](#二当前资源证据) · [执行顺序](#三执行顺序) · [完成与验收](#四完成与验收)

---

## 一、目标与边界

| 项目 | 二值路线 | 连续路线 |
|---|---|---|
| 配置 | [`../../configs/research/full_cell_binary_bz9_50000_xingyi_v1.yaml`](../../configs/research/full_cell_binary_bz9_50000_xingyi_v1.yaml) | [`../../configs/research/full_cell_graded_bz9_50000_xingyi_v1.yaml`](../../configs/research/full_cell_graded_bz9_50000_xingyi_v1.yaml) |
| 样本数 | 50,000 | 50,000 |
| split | 40,000 / 5,000 / 5,000 | 40,000 / 5,000 / 5,000 |
| 标签 | 9×9 full BZ、10 TE bands | 9×9 full BZ、10 TE bands |
| 求解器 | real MPB，resolution 64，mesh 3，tolerance $10^{-7}$ | 同左 |
| 正式验收 | Xingyi 生成后由 WSL `Ubuntu` + `meep_env` replay | 同左 |

两份 50k 数据均从新的确定性结构计划开始，不合并现有 10k HDF5 或任何旧 worker shard。结构计划、Xingyi 求解和 WSL replay 是三个独立 gate。

---

## 二、当前资源证据

2026-08-06 的只读快照显示，`xyfree` 有 32 个节点，每节点 64 CPU、约 500 GiB，分区为 exclusive；当时 2 个节点 idle，用户队列为空，账户最多可提交 16 个作业。容量是瞬时状态，正式提交前必须重查。

历史二值 10k 作业 `6850058` 使用一个 64 核节点，状态 `COMPLETED/0:0`，墙钟 1:54:29；求解阶段吞吐 1.663 samples/s，失败 0。按该证据线性估算，二值 50k 单节点约 8.4 小时。连续路线必须单独做小样本预检/benchmark，不能把 M9 的 256-worker 吞吐直接移植到 Xingyi。

当前策略是二值和连续各占一个 64 核节点并行。结构计划阶段在每个节点使用 64 个确定性进程，计划冻结后同一作业自动切换为 64 个 MPB worker；两条路线合计使用 128 核。跨更多节点的同路线分片与合并尚未实现，不得让多个作业同时写同一 run 目录。

---

## 三、执行顺序

1. 在非 MPB 环境生成并冻结两份 50,000 行结构计划：

   ```bash
   python -m phc2d.cli data prepare-full-cell-plan \
     --config configs/research/full_cell_binary_bz9_50000_xingyi_v1.yaml \
     --run-dir runs/20260806/full_cell_binary_bz9_50000_xingyi_v1_plan_20260806HHMM \
     --workers 64

   python -m phc2d.cli data prepare-full-cell-plan \
     --config configs/research/full_cell_graded_bz9_50000_xingyi_v1.yaml \
     --run-dir runs/20260806/full_cell_graded_bz9_50000_xingyi_v1_plan_20260806HHMM \
     --workers 64
   ```

2. 默认核对两份 `sample_plan_manifest.json`、行数、SHA-256、split 和 strata，并把 `sample_plan.jsonl` 放入两个固定正式 run 目录。若用户明确要求停止本地计划生成并直接使用超算，可设置 `PHC2D_PLAN_MODE=generate-on-node`；每个正式作业会以 `--plan-workers 64` 在自己的全新 run 目录生成并冻结 50,000 行计划，再由同一作业自动进入 `--workers 64` 的 MPB 阶段。`sample_plan_progress.json` 每完成约 1% 更新一次，只表示结构计划进度，不是 MPB 分片进度。两种模式都不得复用半成品或混合旧 shard。
3. 部署范围化 Git commit 到 Xingyi 的新版本目录，验证 `DEPLOYED_GIT_COMMIT`、Meep `1.33.0-beta`、MPB `1.12.0` 和 shell 语法。
4. 提交前重查 `sinfo`、用户队列、文件系统空间；取得单独的短预检授权后，通过 [`../../jobs/xingyi_full_cell_50000_preflight.sh`](../../jobs/xingyi_full_cell_50000_preflight.sh) 执行 solver oracle 与每路线 4 样本预检。任何环境、有限值、频带升序或 failure gate 不通过即停止。
5. 提交器默认 `dry-run`。只有用户明确批准计费运行后，才可同时设置 `PHC2D_SUBMIT_MODE=submit` 与 `PHC2D_AUTHORIZE_50K=YES`：

   ```bash
   PHC2D_CAMPAIGN_ID=full_cell_dual_lane_50000_xingyi_YYYYMMDDHHMM \
   PHC2D_PLAN_MODE=generate-on-node \
   PHC2D_SUBMIT_MODE=submit \
   PHC2D_AUTHORIZE_50K=YES \
   bash jobs/xingyi_submit_full_cell_50000.sh
   ```

6. 只读监控 completed/failed shard、吞吐、ETA、磁盘、Slurm 状态和标准产物。不得自动取消、重启、恢复或改代码。

恢复只能复用相同 commit、配置、计划、run 目录和 shard；审计后显式设置 `PHC2D_ALLOW_RESUME=1`。不得重采样结构或混合其他环境的 shard。

---

## 四、完成与验收

集群完成必须同时存在 HDF5、`dataset_manifest.yaml`、`metrics.json`、`results.json`、`history.csv`、`solver_provenance.json`、零失败账本和 50,000 条 metadata；仅看到 50,000 个 shard 仍不能宣称完成。

下载后先核对远端/本地 SHA-256、schema、`Wh=[50000,81,10]`、split、sample ID 唯一性、NaN/Inf 和 band order。随后分别使用：

- [`../../configs/research/full_cell_binary_50000_xingyi_wsl_acceptance_v1.yaml`](../../configs/research/full_cell_binary_50000_xingyi_wsl_acceptance_v1.yaml)
- [`../../configs/research/full_cell_graded_50000_xingyi_wsl_acceptance_v1.yaml`](../../configs/research/full_cell_graded_50000_xingyi_wsl_acceptance_v1.yaml)

在固定 WSL `Ubuntu` + conda `meep_env` 执行 8 个结构 × 9 个代表性 k 点、bands 2–10 replay。median MRE 必须不高于 0.5%，P95 必须不高于 1%。通过前的最高状态只能是 `cluster_generated_pending_wsl_acceptance`。
