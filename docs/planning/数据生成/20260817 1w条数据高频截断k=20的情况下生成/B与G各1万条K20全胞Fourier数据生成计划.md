---
title: "B 与 G 各 1 万条 K=20 全胞 Fourier 数据生成计划"
summary: "在星仪分别用一个 64 CPU 节点生成 Binary 10000 条和 Graded 10000 条 K=20 real-MPB 数据，每 30 分钟只读检查一次进度，完成后下载最终 HDF5 与必要 provenance。"
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
owner: "研究生维护，人工批准正式星仪计费作业"
last_reviewed: 2026-08-17
reading_time: 8 minutes
primary_layer: L3
---

<!--
  Simple Xingyi plan for 10,000 binary and 10,000 graded K=20 samples.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# B 与 G 各 1 万条 K=20 全胞 Fourier 数据生成计划

## 一、任务定义

本次在已部署的 `full_cell_fourier_v2` 逻辑上生成：

- Binary：10,000 条；
- Graded：10,000 条；
- 合计：20,000 条；
- Fourier 高频截断固定为 `K=20`；
- 计算在星仪完成；
- 计算完成后，把两份最终 HDF5 和必要运行记录下载到本机；
- 本次跳过本机 WSL real-MPB 复算、图件审阅和模型训练。

`K=20` 是本次指定条件，不表述为全局最优。由于跳过本机物理复算，下载后的数据状态保持 `cluster_generated_pending_wsl_acceptance`，但这不妨碍把“星仪生成完成并成功下载”作为本次任务的完成条件。

本文件仍是 Draft 计划，不自动授权 `sbatch`。

## 二、最简单的执行流程

```text
准备 B/G 两份 10k 配置
  → 部署固定 Git commit 到星仪
  → 同时提交 B、G 两个单节点作业
  → 用 job ID 和 run 目录设置每 30 分钟一次的只读监控
  → 每个作业自行完成 sample plan、64-worker MPB 和最终 HDF5
  → 远端检查数量、失败数、最终文件和 SHA-256
  → 下载两份 HDF5 与必要 provenance 到本机
  → 比对下载前后 SHA-256，结束
```

不使用多节点 shard array，也不引入新的多节点 finalizer。原因是现有 64+64 Fourier 单节点链路已经完成真实 MPB 试点，而 B/G 各 10,000 条按实测吞吐仍可在单节点 24 小时窗口内完成。这样只需要两个计算作业，恢复和下载路径最直接。

## 三、冻结的数据合同

| 项目 | Binary | Graded |
|---|---:|---:|
| lane | `binary_fourier` | `graded_fourier` |
| sample count | 10,000 | 10,000 |
| master seed | `2026081703` | `2026081704` |
| train / val / test | 8,000 / 1,000 / 1,000 | 8,000 / 1,000 / 1,000 |
| generator family | `unified_fourier_binary` | `unified_fourier_graded` |
| Fourier cutoff | 20 | 20 |
| 独立半平面模态 | 628 | 628 |
| algorithm version | `phc2d.unified_fourier.v1` | `phc2d.unified_fourier.v1` |

Binary 的 4 个材料占比分层各 2,500 条。Graded 的 16 个“材料均值 × 温度”分层各 625 条，因此不需要不等长补齐。

结构与求解器继续沿用已完成 64+64 试点的合同：

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
| workers | 每个作业 64 process workers，OMP/BLAS/MKL 各 1 thread |

需要新增两份配置：

```text
configs/research/full_cell_binary_fourier_bz9_10000_xingyi_v1.yaml
configs/research/full_cell_graded_fourier_bz9_10000_xingyi_v1.yaml
```

配置从现有 64 条配置复制，只修改 experiment identity、output path、sample count、split、seed 和 plan worker 数；不得改变 K、材料范围、raster、MPB 或 k-grid 合同。

## 四、星仪作业安排

### 4.1 两个独立单节点作业

| 作业 | 资源 | time limit | 实测吞吐外推 | 预计最终 HDF5 |
|---|---:|---:|---:|---:|
| Binary 10k | 1 node、64 CPU、64 GB | 24 h | 约 10.8 h | 约 1.53 GB |
| Graded 10k | 1 node、64 CPU、64 GB | 24 h | 约 15.8 h | 约 3.72 GB |

估算来自现有 64+64 试点的 worker throughput：Binary `0.257736` 条/s，Graded `0.175984` 条/s。64+64 两路峰值 RSS 均约 7.75 GiB，因此 64 GB 已留有充足余量。小样本外推只用于设置资源和超时，不是完成承诺。

两条路线写入完全独立的 run 目录：

```text
runs/{yyyymmdd}/full_cell_fourier_k20_10000_xingyi_{YYYYMMDDHHmm}/
  binary/
  graded/
```

每个作业直接调用现有 `phc2d data generate` 单节点链路，在同一作业内依次完成 sample plan、real-MPB worker shards、HDF5 汇总、manifest 和 metrics。两个作业之间没有文件依赖，可以同时排队和运行。

### 4.2 最小提交前检查

只在星仪侧检查：

1. `DEPLOYED_GIT_COMMIT` 与本次配置/作业脚本 commit 一致；
2. 两份配置能被 CLI 解析，sample count 均为 10,000，`fourier_cutoff=20`；
3. Meep/MPB 版本与 64+64 试点一致；
4. `xyfree` 在线、磁盘空间足够、本用户没有冲突中的同名作业；
5. dry-run 只产生两个 `sbatch` 命令，run 目录不同。

本次不要求先跑新的本机测试或本机 real-MPB smoke。

### 4.3 提交后每 30 分钟检查

两个 `sbatch` 成功返回 job ID 和固定 run 目录后，立即在当前 Codex 任务中创建一个 **每 30 分钟运行一次**的低频监控。创建时必须把以下四项写成固定输入：

```text
binary_job_id
graded_job_id
binary_run_dir
graded_run_dir
```

现在尚无正式 job ID，因此本计划阶段不提前创建空监控。实际提交后再创建，避免监控错误目录或历史作业。

每次检查只读取：

| 检查项 | 排队中 | 运行中 | 终态 |
|---|---|---|---|
| Slurm | job state、queue reason | job state、elapsed、node | `sacct` state、exit code、elapsed、MaxRSS |
| 进度 | 不扫描 run 目录 | B/G completed shard 数、失败数、最近更新时间 | 最终 completed/failed 数 |
| 最终产物 | 不检查 | 只判断 HDF5 是否已出现，不读取大数组 | 检查 HDF5、manifest、metrics、results、provenance 是否齐全 |

每次只返回一条简短状态：

```text
检查时间 | B 状态/完成数/失败数 | G 状态/完成数/失败数 | 是否需要人工处理
```

监控规则：

- 固定 30 分钟周期，不使用 `watch`、持续 `tail`、分钟级轮询或常驻 SSH；
- 只读检查，不自动取消、重提、恢复、改参数或改代码；
- 某一路连续两次检查完成数不变但 Slurm 仍为 `RUNNING` 时，只报告“进度可能停滞”，不自行处理；
- 某一路为 `FAILED`、`CANCELLED`、`TIMEOUT` 或出现失败样本时，立即报告具体状态并等待人工决定；
- 一路完成而另一路仍运行时，继续按 30 分钟周期检查未完成路线；
- 两路都进入终态后停止周期监控；若两路均成功，则进入远端完成检查和下载；否则停在故障报告，不下载不完整数据。

### 4.4 运行与恢复

- 不在运行中修改 K、resolution、worker 数、seed、split 或材料范围；
- 不自动取消、重提或切换资源；
- 若作业异常退出，保留原 run 目录和 `sample_plan.jsonl`；
- 人工检查已有 shard、failed ledger 和退出原因后，才可在同一 run 目录设置 `PHC2D_ALLOW_RESUME=1` 恢复；
- 不创建替代 run 后混合两批 shard。

## 五、远端完成检查

每条路线只做下列必要检查：

1. Slurm 状态为 `COMPLETED`，退出码 `0:0`；
2. `sample_plan.jsonl` 恰好 10,000 行；
3. `metrics.json` 中 `completed_count=10000`、`failed_count=0`；
4. `results.json` 表明最终 dataset 已写出；
5. `dataset_manifest.yaml` 中 sample count 为 10,000，`Wh_shape=[10000,81,10]`；
6. `failed_case_ledger.jsonl` 为空；
7. collision ledger 无未处理碰撞；
8. 最终 HDF5、manifest、provenance、metrics 和 run meta 均存在；
9. 在星仪计算并保存最终 HDF5 的 SHA-256。

不能仅因 worker shards 达到 10,000 就宣布完成；最终 HDF5 与标准记录必须全部存在。

## 六、下载到本机

### 6.1 下载内容

每条路线下载：

```text
最终 *.h5
run.log
run.meta.json
config.resolved.yaml
sample_plan_manifest.json
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

默认不下载 `worker_shards/` 和完整图件，以减少传输量；远端 run 目录继续保留，便于以后恢复或补充验收。

本机目标建议为：

```text
data/full_cell_binary_fourier_bz9_10000_xingyi_v1.h5
data/full_cell_graded_fourier_bz9_10000_xingyi_v1.h5
runs/{yyyymmdd}/full_cell_fourier_k20_10000_xingyi_{YYYYMMDDHHmm}/
```

HDF5 和 run 产物保持 Git ignored，不提交仓库。

### 6.2 下载完成条件

优先使用支持断点续传的传输方式；若环境中只有 `scp`，则分别下载两份 HDF5 和小型 provenance 文件。下载后只比对远端与本机 SHA-256，确认传输没有损坏。

SHA-256 比对属于文件传输完整性检查，不是本机物理验证。本次不再执行 WSL replay、HDF5 全量科学审计或画图。

## 七、本次完成定义

满足以下条件即完成本次任务：

- Binary 10,000 条和 Graded 10,000 条都在星仪生成最终 HDF5；
- 两个作业均 0 失败，样本数和 `Wh` shape 正确；
- 两份 HDF5 与必要 provenance 已下载到本机；
- 本机文件 SHA-256 与星仪一致；
- 提交后已按 30 分钟周期完成只读进度检查，并在两路进入终态后停止监控；
- 记录星仪 commit、Slurm job ID、run 路径、远端/本机路径、耗时和 hash。

本次完成不等于数据已通过 WSL 跨环境物理验收。若后续要把数据标记为 accepted 或用于正式论文证据，再单独安排 WSL 复算；该工作不阻塞本次“超算生成并下载”的任务。

## 八、人工理解记录

```text
解决了什么问题：用最短流程在星仪生成 B/G 各 1 万条 K=20 数据，每 30 分钟只读检查一次，并下载到本机。
为什么采用这个方案：两个单节点 runner 已有 64+64 real-MPB 运行证据，预计都能在 24 小时内完成，无需增加多节点分片和 finalizer 改造。
改变了项目哪一层/哪一模块：主层级 L3；后续只需新增两份配置和一个简单的星仪双作业入口。
使用什么证据：统一 Fourier v2 部署、resolution=64 冻结结果、64+64 试点吞吐与 K=5–20 配对试验。
最大边界：跳过本机 WSL 物理复算，因此数据保持 cluster_generated_pending_wsl_acceptance。
```
