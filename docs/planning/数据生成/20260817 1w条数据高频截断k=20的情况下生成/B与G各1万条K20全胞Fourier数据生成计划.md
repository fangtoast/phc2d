---
title: "B 与 G 各 1 万条 K=20 全胞 Fourier 数据生成计划"
summary: "统计 Xingyi CPU 资源后，使用 Binary 8 节点与 Graded 8 节点并行生成各 10000 条 K=20 real-MPB 数据，每 30 分钟只读检查，完成后下载最终 HDF5 与 provenance。"
page_type: plan
status: draft
plan_state: active
effective_date: 2026-08-17
superseded_by: null
related_results:
  - ../../../results/data-provenance/全胞Fourier数据重构与星仪部署_20260817.md
  - ../../../results/data-provenance/全胞Fourier_64加64星仪试点_20260817.md
  - ../../../results/data-provenance/全胞Fourier高频截断配对试验_20260817.md
  - ../../../results/data-provenance/全胞MPB分辨率收敛验证_20260817.md
audience: "研究用户、超算执行者与数据使用者"
owner: "研究生维护，人工批准正式 Xingyi 计费作业"
last_reviewed: 2026-08-17
reading_time: 10 minutes
primary_layer: L3
---

<!--
  Parallel Xingyi plan for 10,000 binary and 10,000 graded K=20 samples.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# B 与 G 各 1 万条 K=20 全胞 Fourier 数据生成计划

## 一、任务与结论

本次生成：

- Binary：10,000 条；
- Graded：10,000 条；
- 合计：20,000 条；
- Fourier 高频截断固定为 `K=20`；
- real-MPB 计算、分片和最终 HDF5 汇总均在 Xingyi 完成；
- 每 30 分钟只读检查一次，不实时追踪；
- 两路完成后把最终 HDF5 与必要 provenance 下载到本机；
- 本次跳过本机 WSL real-MPB 复算、图件审阅和模型训练。

原来的“两条单节点作业”只会使用 128 CPU，确实没有充分利用 Xingyi。修订后的正式求解采用：

```text
Binary：8 个 solve shard 节点 × 64 CPU
Graded：8 个 solve shard 节点 × 64 CPU
峰值：16 个节点、1024 CPU
```

这是当前项目 Xingyi profile 允许的最高档位：每个 campaign 最多 8 节点。它使用整个 `xyfree` 分区 50% 的 CPU，而不是原方案的 6.25%。若要使用全部 32 节点，必须另行修改平台资源上限并确认共享分区/QoS，不在本计划中直接扩张。

`K=20` 是本次指定运行条件，不表述为全局最优。由于跳过本机物理复算，最终数据状态保持 `cluster_generated_pending_wsl_acceptance`，但“Xingyi 完成生成、远端审计通过并成功下载”仍是本次任务的完成条件。

本文件是 Draft 计划，不自动授权 `sbatch`。

## 二、Xingyi CPU 与调度统计

2026-08-17 23:25（Asia/Shanghai）的只读统计为：

| 项目 | 统计值 |
|---|---:|
| partition | `xyfree` |
| 节点范围 | `cnode2432–cnode2463` |
| 总节点 | 32 |
| 每节点 CPU | 64 |
| 总 CPU | 2,048 |
| 每节点内存 | 512,000 MiB，约 500 GiB |
| 总内存 | 16,000 GiB |
| 调度方式 | `OverSubscribe=EXCLUSIVE`，整节点独占 |
| 分区状态 | `UP` |
| 当时 CPU 状态 | `2048 allocated / 0 idle / 0 other / 2048 total` |
| 本用户当时队列 | 空 |
| 共享存储 | 约 7.9 PB 可用 |

全部 32 个节点在该快照时刻都处于 `allocated`，因此 16 个 solve task 提交后很可能先排队。这里的“最大化性能”指资源一旦获得就以 16 节点并行求解，不代表能够立即启动，也不承诺排队时间。

仓库当前 [`hpc_platform_xingyi_v1.yaml`](https://github.com/fangtoast/phc2d/blob/master/configs/research/hpc_platform_xingyi_v1.yaml) 规定：

```text
cores_per_node: 64
max_nodes: 8
max_array_tasks: 16
allowed_node_tiers: [1, 2, 4, 8]
```

两条 campaign 各选择 8 节点，既达到当前 profile 的单 campaign 上限，也没有超过 array task 上限。

## 三、冻结的数据合同

| 项目 | Binary | Graded |
|---|---:|---:|
| lane | `binary_fourier` | `graded_fourier` |
| sample count | 10,000 | 10,000 |
| master seed | `2026081703` | `2026081704` |
| train / val / test | 8,000 / 1,000 / 1,000 | 8,000 / 1,000 / 1,000 |
| strata | 4 层 × 2,500 | 16 层 × 625 |
| generator family | `unified_fourier_binary` | `unified_fourier_graded` |
| Fourier cutoff | 20 | 20 |
| 独立半平面模态 | 628 | 628 |
| algorithm version | `phc2d.unified_fourier.v1` | `phc2d.unified_fourier.v1` |

结构与求解器沿用 64+64 试点：

| 合同 | 冻结值 |
|---|---|
| raster | 256 母版直接 block mean 到 128/64/32 |
| MPB 材料真值 | 解析 `epsilon_at` |
| solver resolution | 64 |
| Meep / MPB | `1.33.0-beta` / `1.12.0` |
| polarization | TE |
| k points / bands | 81 个 full-BZ k 点、10 个保留频带、4 个 guard bands |
| mesh / tolerance | `mesh_size=3`、`1e-7` |
| band order | `frequency_ascending` |
| worker | 每个 solve shard 64 process workers；OMP/BLAS/MKL 各 1 thread |

新增配置：

```text
configs/research/full_cell_binary_fourier_bz9_10000_xingyi_v1.yaml
configs/research/full_cell_graded_fourier_bz9_10000_xingyi_v1.yaml
```

两份配置只在现有 64 条配置上修改 experiment identity、output path、sample count、split、seed 和 HPC identity，不改变 K、材料范围、raster、MPB 或 k-grid。

## 四、为什么需要 shard array

20,000 个样本是样本级独立 MPB 求解，适合横向分片。每条路线只生成一个冻结 `sample_plan.jsonl`，再按全局 ordinal 切成 8 个互斥 shard：

| 路线 | shard 数 | 每 shard 样本 | 每 shard CPU | 峰值节点/CPU |
|---|---:|---:|---:|---:|
| Binary | 8 | 1,250 | 64 | 8 / 512 |
| Graded | 8 | 1,250 | 64 | 8 / 512 |
| 合计 | 16 | 20,000 | — | 16 / 1,024 |

每个 shard 只写自己的 `partition.h5` 和 manifest。8 个 shard 全部成功后，由该路线唯一的 finalizer 检查 plan SHA-256、覆盖范围、重复、失败数和 partition hash，再原子写出一份最终 HDF5。不得让多个任务共同写最终 HDF5，也不得把 8 个独立随机数据集事后拼接冒充一个冻结数据集。

## 五、正式前唯一必要的实现门

仓库已有 `plan → solve-shard → finalize` 的 `full_cell_band` 控制面，但当前 [`full_cell_hpc.py`](https://github.com/fangtoast/phc2d/blob/master/src/phc2d/data/full_cell_hpc.py) 的 plan/finalizer 仍把部分 schema/protocol 固定为 `full_cell_v1`。直接对 Fourier v2 使用会产生错误元数据。

为获得 8+8 节点性能，只做以下最小改造：

1. `output_schema` 和 finalizer `metadata.protocol` 从配置读取，对 Fourier 数据写为 v2；
2. 保留 Fourier v2 的 raster contract、semantic fields、parameter/geometry/content hash 和 collision ledger；
3. 为提交入口增加显式的 8-shard 资源档位；dry-run 必须显示 B/G 均为 `array=0-7`，不能让默认 3 小时目标把 Binary 自动降到 4 节点；
4. 在 Xingyi 做一次 B/G 各 64 条、每路 2 shard 的集群 smoke，检查 4 个 partition 和两份最终 HDF5；
5. smoke 通过后再提交 8+8 正式 array。

本次可以跳过本机 WSL 验证，但不能跳过这个 Xingyi 多 shard smoke；否则无法证明 finalizer 能正确合并 Fourier v2 数据。

## 六、作业编排与预计耗时

每条路线使用独立依赖链：

```text
plan job
  → solve array[0-7]（8 个节点并行）
  → afterok finalizer
```

两条路线合计：2 个 plan job、16 个 solve array task、2 个 finalizer job。实际峰值是 16 个 solve 节点，不再只是两个作业。

### 6.1 资源

| 阶段 | 数量 | 单任务资源 | time limit | 说明 |
|---|---:|---:|---:|---|
| plan | B/G 各 1 | 64 CPU / 128 GB | 1 h | 冻结 10k plan、split 与 hash |
| solve | B/G 各 8 | 64 CPU / 128 GB | 6 h | 每 task 1,250 个 real-MPB 样本 |
| finalize | B/G 各 1 | 64 CPU / 128 GB | 2 h | 单写者检查并发布最终 HDF5 |

虽然节点有约 500 GiB 内存，项目 profile 当前请求 128 GB；64+64 试点峰值约 7.75 GiB，内存余量充足。由于节点为 exclusive，减少内存请求不会让第二个用户共享同一节点，主要性能由 CPU 并发决定。

### 6.2 预计 solve 时间

| 路线 | 64-worker 单节点吞吐 | 1 节点 10k 外推 | 8 节点外推 |
|---|---:|---:|---:|
| Binary | 0.257736 条/s | 约 10.8 h | 约 1.35 h |
| Graded | 0.175984 条/s | 约 15.8 h | 约 1.97 h |

加上 plan、调度启动和 finalizer，资源到位后的目标总 wall time 为约 2–3 小时。该数值是线性外推，正式报告必须记录实际 queue wait、solve wall time、throughput 和 MaxRSS。

当前 32 节点全部占用时，两个 8-task array 会等待资源。不得因为排队而自动取消并退回两个单节点；若人工希望改用 4+4 节点以更早启动，必须在提交前重新确认，并保留同一 sample plan。

## 七、每 30 分钟只读检查

两个 campaign 提交后，以以下固定输入创建 30 分钟周期监控：

```text
binary_plan_job_id
binary_solve_array_job_id
binary_finalize_job_id
binary_run_dir
graded_plan_job_id
graded_solve_array_job_id
graded_finalize_job_id
graded_run_dir
```

现在没有正式 job ID，因此计划阶段不创建空监控。实际提交后立即创建。

每次检查只执行：

1. `squeue` / `sacct` 汇总 B/G plan、array 和 finalizer 状态；
2. 只读运行 `phc2d hpc status --run-dir <run_dir>`；
3. 汇总 8 个 shard 中 pending/running/completed/failed 数；
4. 进入 finalizer 后只判断最终 HDF5 和标准 manifest 是否出现，不读取大数组。

每次只返回：

```text
检查时间 | B: 队列/8 shard完成数/失败数/finalizer | G: 队列/8 shard完成数/失败数/finalizer | 人工处理
```

监控规则：

- 固定每 30 分钟一次，不使用 `watch`、持续 `tail`、分钟级轮询或常驻 SSH；
- 只读，不自动取消、重提、恢复、改资源、改参数或改代码；
- 连续两次无进度但任务仍 `RUNNING` 时只报告停滞风险；
- 任一 array task 为 `FAILED`、`CANCELLED`、`TIMEOUT` 或出现失败样本时立即报告并等待人工决定；
- 一路完成后继续监控另一路；
- 两路都进入终态后停止监控，成功则进入远端审计和下载，失败则停止在故障门。

## 八、远端完成检查与下载

### 8.1 远端完成门

每条路线必须满足：

1. plan job、8 个 solve task 和 finalizer 均为 `COMPLETED/0:0`；
2. `sample_plan.jsonl` 恰好 10,000 行且 SHA-256 与 physical plan manifest 一致；
3. 8 个 shard 无遗漏、无重叠，全部 `partition_complete=true`、failed count 为 0；
4. 最终 HDF5、`dataset_manifest.yaml`、`metrics.json`、`results.json`、`solver_provenance.json` 和 run meta 齐全；
5. `Wh_shape=[10000,81,10]`，split 为 `8000/1000/1000`；
6. collision/failed ledger 无未处理记录；
7. `phc2d hpc audit --run-dir <run_dir>` 通过；
8. 在 Xingyi 保存最终 HDF5 的 SHA-256。

不能仅因 8 个 shard 完成就宣布完成；finalizer 和最终标准产物必须存在。

### 8.2 下载到本机

下载：

```text
最终 *.h5
run.log
run.meta.json
config.resolved.yaml
sample_plan_manifest.json
physical_plan_manifest.json
physical_campaign_manifest.json
submission_manifest.json
dataset_manifest.yaml
solver_provenance.json
hash_ledger.csv
collision_ledger.csv
failed_case_ledger.jsonl
sample_metadata.jsonl
results.json
metrics.json
history.csv
远端 SHA-256 清单
```

默认不下载 `shards/`、`partition.h5` 和完整图件，减少传输量；远端完整 run 目录保留。优先使用支持断点续传的方式，下载后只比对远端与本机 SHA-256。

本机目标：

```text
data/full_cell_binary_fourier_bz9_10000_xingyi_v1.h5
data/full_cell_graded_fourier_bz9_10000_xingyi_v1.h5
runs/{yyyymmdd}/full_cell_fourier_k20_10000_xingyi_{YYYYMMDDHHmm}/
```

HDF5 和 run 产物保持 Git ignored。SHA-256 比对只证明传输完整，不是本机物理验证。本次不运行 WSL replay、全量科学审计或画图。

## 九、完成定义与边界

满足以下条件即完成：

- B/G 各 10,000 条均由 8 个 Xingyi solve shard 生成，并由单写者 finalizer 发布最终 HDF5；
- 两条路线均 0 失败、远端 HPC audit 通过；
- 每 30 分钟监控持续到两路终态并停止；
- 两份最终 HDF5 与必要 provenance 已下载到本机；
- 本机 SHA-256 与 Xingyi 一致；
- 记录 CPU/节点、commit、job ID、run 路径、排队时间、计算耗时、MaxRSS、吞吐和 hash。

本次完成不等于数据通过 WSL 跨环境物理验收。后续若要将数据标记为 accepted 或用于正式论文证据，再单独安排 WSL 复算；该工作不阻塞本次“Xingyi 并行生成并下载”。

## 十、人工理解记录

```text
解决了什么问题：统计 Xingyi 32×64 CPU 后，把 B/G 各 1 万条从两个单节点改为 8+8 节点并行，并保留 30 分钟监控和下载闭环。
为什么采用这个方案：2 个节点只使用 6.25% 分区 CPU；8+8 达到当前项目 profile 的最高档位，峰值使用 1024 CPU，同时用冻结 plan、互斥 partition 和单写者 finalizer 保证数据身份。
改变了项目哪一层/哪一模块：主层级 L3；后续最小 L4 工作是 Fourier v2 HPC schema 修正、Xingyi 两 shard smoke、两份 10k 配置和提交入口。
使用什么证据：Xingyi 实时节点统计、平台 profile、统一 Fourier v2 部署、resolution=64、64+64 吞吐和现有 full_cell_band HPC 控制面。
最大边界：当前 32 节点全部占用，作业可能排队；本机 WSL 物理复算仍跳过，数据保持 cluster_generated_pending_wsl_acceptance。
```
