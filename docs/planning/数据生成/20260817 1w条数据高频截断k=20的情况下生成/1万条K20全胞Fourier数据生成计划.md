---
title: "1 万条 K=20 全胞 Fourier 数据生成与星仪算力计划"
summary: "在统一 Fourier v2 数据合同上冻结 K=20，按 Binary 5000 条与 Graded 5000 条组织可恢复的星仪多节点 real-MPB 生成、汇总与 WSL 验收。"
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
audience: "研究用户、数据实现者、超算执行者与人工验收者"
owner: "研究生维护，导师确认 1 万条数据合同与正式计费提交"
last_reviewed: 2026-08-17
reading_time: 14 minutes
primary_layer: L3
---

<!--
  Plan for a 10,000-sample K=20 full-cell Fourier campaign on Xingyi.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 1 万条 K=20 全胞 Fourier 数据生成与星仪算力计划

## 一、执行结论

本计划把“1 万条数据”冻结为 **总计 10,000 条**：Binary 5,000 条、Graded 5,000 条。两条路线继续使用已实现的 `full_cell_fourier_v2` 逻辑，并把 Fourier 高频截断固定为 `K=20`。若实际需求是“Binary 10,000 条 + Graded 10,000 条”，则属于 20,000 条扩容分支，必须另行修改样本数、资源预算、超时和验收范围，不能静默套用本计划。

推荐星仪正式求解采用非对称资源：

| 路线 | 样本数 | solve shard | 每个 shard | 预计 solve wall time |
|---|---:|---:|---:|---:|
| Binary | 5,000 | 2 | 64 CPU、约 2,500 条 | 约 2.7 小时 |
| Graded | 5,000 | 3 | 64 CPU、约 1,667 条 | 约 2.6 小时 |

即同时使用 5 个 64 CPU 求解节点，使两条路线大致同时结束。该估算来自 64+64 试点的 worker throughput，仍须由提交前的多 shard smoke 更新；正式时间上限建议设为 6 小时，不把小样本线性外推当作完成承诺。

本文件只授权计划、配置与 dry-run 准备，**不授权 `sbatch`、自动恢复、自动取消或任何计费 real-MPB 正式作业**。

## 二、当前证据与尚未关闭的门

### 2.1 已具备的基础

| 证据 | 当前事实 | 对本计划的作用 |
|---|---|---|
| [统一 Fourier 数据实现与部署](../../../results/data-provenance/全胞Fourier数据重构与星仪部署_20260817.md) | `full_cell_fourier_v2`、256 母版、版本化哈希和 B/G 生成器已经实现 | 不重新设计生成方式 |
| [64+64 星仪试点](../../../results/data-provenance/全胞Fourier_64加64星仪试点_20260817.md) | B/G 各 64 条、0 失败、`resolution=64`、真实 MPB | 作为吞吐、内存、文件体量与故障率基线 |
| [K=5–20 配对试验](../../../results/data-provenance/全胞Fourier高频截断配对试验_20260817.md) | 每路一个固定系数库完成 K=5–20 real-MPB 扫描 | 证明 K 会显著影响结构与频带，但不证明 K=20 全局最优 |
| [MPB 分辨率验证](../../../results/data-provenance/全胞MPB分辨率收敛验证_20260817.md) | `resolution=64` 已在既定门限下通过并冻结 | 本轮不再追加 r96/r128 全量生成 |

`K=20` 是本次任务的**用户指定运行条件**，不是从单一配对试验推导出的最优值。正式数据必须在 manifest 中记录 `fourier_cutoff: 20`、`algorithm_version: phc2d.unified_fourier.v1` 与 628 个独立半平面模态，避免后续将“指定条件”改写成“统计最优”。

### 2.2 实测资源基线

64+64 试点的求解侧实测为：

| 路线 | worker solve time | worker wall time | worker throughput | 总 wall time | 峰值 RSS | HDF5 大小 |
|---|---:|---:|---:|---:|---:|---:|
| Binary 64 | 13,736.62 s | 248.32 s | 0.257736 条/s | 388.70 s | 约 7.75 GiB | 9.76 MB |
| Graded 64 | 22,326.84 s | 363.67 s | 0.175984 条/s | 500.96 s | 约 7.75 GiB | 23.81 MB |

按 worker throughput 线性估算，单节点生成 5,000 条约需 B 5.4 小时、G 7.9 小时；B 2 节点与 G 3 节点后分别约为 2.7 小时和 2.6 小时。按单样本 CPU 时间估算，本轮求解总量约为 B 298 core-hours、G 485 core-hours。

按 64 条试点文件线性外推，最终两份 HDF5 约为 B 0.76 GB、G 1.86 GB。考虑 sample plan、partition、临时文件、最终文件和日志同时存在，正式 campaign 根目录至少预留 20 GB，建议预留 50 GB 并在提交前重新检查配额。

2026-08-17 23:00（Asia/Shanghai）的只读快照显示：星仪 `xyfree` 为 32 个 64 CPU 节点，分区在线，但当时 32 个节点均已分配；本用户队列为空，`DEPLOYED_GIT_COMMIT=51244ec`，共享存储可用约 7.9 PB。该快照只说明当时状态，提交前必须刷新，不能据此承诺排队时间。

### 2.3 必须先修正的扩展风险

现有 64+64 入口是单节点 runner；1 万条不得通过让多个作业写入同一个 run 目录来扩展。仓库已有 `plan → solve-shard → finalize` 的 `full_cell_band` HPC 控制面，但当前 [`full_cell_hpc.py`](https://github.com/fangtoast/phc2d/blob/master/src/phc2d/data/full_cell_hpc.py) 的 finalizer 仍把部分元数据固定写成 `full_cell_v1`，且现有两 shard 自动测试没有覆盖 `full_cell_fourier_v2` 的语义字段。

因此，正式提交前必须完成以下最小修正：

1. finalizer 从配置继承 `dataset.protocol`，为 Fourier v2 写出正确的 output schema、raster contract、内容哈希字段和 dataset identity；
2. 增加 Fourier v2 两 shard 测试，检查 `material_binary_grid_256`、`material_area_fraction_grid_*` 或 `material_weight_grid_*`、split、hash、分区覆盖与最终 HDF5；
3. 扩展 WSL acceptance 入口，使其识别 `unified_fourier_binary` 与 `unified_fourier_graded`；
4. 只有这些检查通过后，才能把 64+64 的状态从 `cluster_generated_pending_wsl_acceptance` 推进到人工可接受的物理数据门，再申请 1 万条正式运行。

## 三、冻结的数据合同

### 3.1 样本与 split

| 项目 | Binary | Graded |
|---|---:|---:|
| lane | `binary_fourier` | `graded_fourier` |
| sample count | 5,000 | 5,000 |
| master seed | `2026081703` | `2026081704` |
| train / val / test | 4,000 / 500 / 500 | 4,000 / 500 / 500 |
| generator family | `unified_fourier_binary` | `unified_fourier_graded` |
| Fourier cutoff | 20 | 20 |
| algorithm version | `phc2d.unified_fourier.v1` | `phc2d.unified_fourier.v1` |

Binary 的 4 个材料占比分层各 1,250 条。Graded 的 16 个“材料均值 × 温度”分层采用 8 层 313 条、8 层 312 条的确定性轮转，任意两层计数差不超过 1；具体分配必须写入冻结 `sample_plan_manifest.json`，不得在求解完成后重新平衡。

split 在冻结 plan 阶段按 strata 与 hash group 确定。任何 parameter、geometry 或内容碰撞组不得跨 train/val/test；发现重复时先写入 collision ledger 并 fail closed，不得静默补采后继续沿用旧 plan hash。

### 3.2 结构、栅格与求解器

| 合同 | 冻结值 |
|---|---|
| 潜在场 | 硬圆盘 Fourier 截断，`K=20`，float64 求值 |
| Binary 映射 | 目标材料占比分位数阈值，256 母版严格 0/1 |
| Graded 映射 | logistic bias/T，目标均值容差 `1e-5` |
| 模型表示 | 256 母版直接 block mean 到 128/64/32，不级联降采样 |
| MPB 真值 | 解析 `epsilon_at`，不读取栅格作为 solver 材料真值 |
| MPB | Meep `1.33.0-beta`、MPB `1.12.0`、`resolution=64`、`mesh_size=3` |
| 频带 | TE、81 个 full-BZ k 点、10 个保留频带、4 个 guard bands |
| 数值策略 | `frequency_ascending`、eigensolver tolerance `1e-7` |
| worker | 每求解节点 64 process workers；OMP/BLAS/MKL 各 1 thread |

建议新增配置：

```text
configs/research/full_cell_binary_fourier_bz9_5000_xingyi_v1.yaml
configs/research/full_cell_graded_fourier_bz9_5000_xingyi_v1.yaml
```

两份配置从已验收的 64 条配置显式复制合同，只改 experiment/dataset identity、sample count、split、seed 与多 shard 运行字段，不引用本文文字作为隐式默认值。

## 四、分阶段执行

### Phase 0：实现与 WSL 工程门

1. 修正 Fourier v2 多 shard finalizer 与 provenance；
2. 新增两份 5,000 配置、对应实验说明和 Xingyi dry-run 提交入口；
3. 增加配置解析、两 shard、finalizer fail-closed、v2 HDF5 round-trip 和 acceptance family 测试；
4. 在 WSL `Ubuntu` + conda `meep_env` 运行 compileall、定向测试、完整测试、CLI help 和 shell syntax；
5. 形成范围化 commit，部署到 Xingyi，核对 `DEPLOYED_GIT_COMMIT`、配置 hash、Meep/MPB 版本和 dry-run 输出。

Phase 0 只证明实现可运行，不授权正式数据生成。

### Phase 1：冻结 5,000 + 5,000 sample plan

每条路线独立生成一次完整 plan，并冻结：

- `sample_plan.jsonl` 与 SHA-256；
- 5,000 个唯一 sample ID、parameter/geometry hash 与 strata；
- 4,000/500/500 split；
- 2 个 B shard 和 3 个 G shard 的无遗漏、无重叠范围；
- 每个 shard 的 strata 计数，任意可分层计数差不超过 1；
- 估算磁盘、最长结构参数行和 finalizer 内存。

冻结后不允许同名 run 目录重新生成 plan。若计划合同变化，创建新 experiment version 和新 run 目录。

### Phase 2：多 shard packed smoke

在一个 64 CPU allocation 内运行 B 32 + G 32 的 packed smoke：每路 2 个 shard、每 shard 16 workers，写四个独立 `partition.h5`，随后分别执行 finalizer。该 smoke 复核的是新 HPC 控制面、共享目录隔离、v2 字段、恢复语义和单写者合并，不重复宣称 64+64 数据分布已验证。

通过条件：64/64 完成、0 失败、分区无重叠/遗漏、最终 HDF5 hash 稳定、字段与单节点 runner 一致。失败时停止，不提交 1 万条。

### Phase 3：正式 1 万条求解

两条路线使用独立 run 目录和独立依赖链：

```text
Binary plan → B solve array[0..1] → B finalize
Graded plan → G solve array[0..2] → G finalize
```

求解 array 可以并行，finalizer 只能在本路线全部 shard `afterok` 后启动。任何 shard 失败都阻断 finalizer；只读审计失败 shard、plan hash 和 partition 后，才可由人工授权该 shard 的 `--allow-resume`。不得删除成功 partition、改变 K、改变 resolution 或以新 plan 覆盖旧目录。

### Phase 4：下载、审计与 WSL 物理验收

1. 远端先完成 counts、shape、有限值、band order、hash、collision 与标准产物审计；
2. 下载 manifest、最终 HDF5、provenance、ledger、metrics、日志与精选图件，逐文件核对 SHA-256；
3. 本机 WSL `Ubuntu` + `meep_env` 对每路至少 8 个分层结构、每个 9 个代表 k 点复算 bands 2–10；
4. 固定门限为 solver failure 0、median replay MRE `≤0.5%`、P95 replay MRE `≤1.0%`；
5. 人工查看每路至少 100 个分层样本的结构图册、频谱、材料分布、异常样本和能带包络。

通过 WSL 复算前，状态保持 `cluster_generated_pending_wsl_acceptance`。本地内容审计、星仪成功退出或代理模型指标都不能替代此门。

## 五、星仪资源编排

### 5.1 推荐资源

| 阶段 | 并发 | 单任务资源 | time limit | 说明 |
|---|---:|---:|---:|---|
| plan | 2 路可并行 | 16 CPU / 32 GB | 30 min | 不调用 MPB，只冻结结构与 hash |
| packed smoke | 1 | 64 CPU / 64 GB | 1 h | 四个 16-worker shard 进程隔离输出 |
| B solve | 2 array tasks | 64 CPU / 64 GB | 6 h | 每 shard 约 2,500 条 |
| G solve | 3 array tasks | 64 CPU / 64 GB | 6 h | 每 shard 约 1,666–1,667 条 |
| finalize | 每路 1 | 32 CPU / 64 GB | 2 h | 单写者流式合并、栅格化与 hash |

64+64 试点峰值 RSS 约 7.75 GiB，正式 solve 申请 64 GB 已有约 8 倍余量；除非新 smoke 显示显著增长，不继续沿用 128 GB 的保守请求。每个 solve 任务占满 64 CPU，因此不与其他用户任务在同一节点内自行超卖。

### 5.2 资源档位与选择规则

| 档位 | B / G 节点 | 预计最长 solve | 使用条件 |
|---|---:|---:|---|
| 资源效率优先（推荐） | 2 / 3 | 约 2.7 h | 5 个节点能在可接受窗口内启动 |
| 排队友好 | 1 / 1 | 约 7.9 h | 5 节点等待显著更长，且人工接受较慢完成 |
| 最短 wall time | 3 / 5 | 约 1.8 h | 8 节点同时空闲且人工明确优先 wall time |

提交前刷新 `sinfo`、本用户 `squeue`、关联/QoS 限制、磁盘和部署 commit。资源档位只能在提交前由人工选择；不得在排队或运行中自动取消并换档。若采用非推荐档位，必须在 `submission_manifest` 记录原因和重新计算的 ETA。

### 5.3 效率与故障保护

- B/G 分开建 array，避免 Graded 较慢导致 Binary 节点空等；
- shard 只写自己的 `partition.h5`、progress 和 manifest，最终 HDF5 只由 finalizer 写；
- array 并发上限与实际 shard 数相同，不额外提交空任务；
- 每 10–15 分钟做一次只读状态汇总，不高频扫描全部 shard 文件；
- 记录 `worker_solve_time_s`、`worker_wall_time_s`、`wall_time_total_s`、throughput、queue wait、MaxRSS 与失败原因；
- 不因短时吞吐波动修改样本、K、resolution、worker 数或 split；
- finalizer 必须验证 plan SHA-256、完整覆盖、零重复、零失败和 partition hash 后再原子发布。

## 六、接受门与标准产物

### 6.1 数据接受门

- B/G 各恰好 5,000 条，`Wh` 均为 `(5000,81,10)`；
- 所有频率有限且按 band 维升序，无 solver failure；
- 每个样本的 generator family、`fourier_cutoff=20`、seed derivation、解析参数和 solver provenance 完整；
- 32/64/128/256 表示齐全，且来自同一 256 母版；
- Binary 256 母版严格 0/1，Graded 权重位于 `[0,1]`；
- parameter/geometry/content collision ledger 经过审计，碰撞组不跨 split；
- shard 覆盖完整、无重叠，final dataset hash 与下载后 hash 一致；
- WSL real-MPB 分层复算通过冻结门限；
- 人工审阅图件后，才能把数据标记为 accepted，并决定是否进入模型训练。

### 6.2 每条路线的标准产物

```text
run.log
run.meta.json
config.resolved.yaml
sample_plan.jsonl
sample_plan_manifest.json
physical_plan_manifest.json
physical_campaign_manifest.json
submission_manifest.json
shards/shard_*/partition.h5
shards/shard_*/shard_manifest.json
dataset_manifest.yaml
solver_provenance.json
hash_ledger.csv
collision_ledger.csv
failed_case_ledger.jsonl
sample_metadata.jsonl
results.json
metrics.json
history.csv
figures/
data/*.h5
```

原始 HDF5、partition、完整图件和运行目录不提交 Git。完成后新增 `docs/results/data-provenance/` 日期化 Draft 报告，记录配置、commit、Slurm job、plan/data SHA-256、吞吐、MaxRSS、失败账本、WSL acceptance 和证据边界。

## 七、人工授权点与非目标

需要四次独立人工确认：

1. 接受“总计 10,000 = B 5,000 + G 5,000”及两份冻结配置；
2. 接受 Fourier v2 多 shard 实现、WSL 工程门和 packed smoke；
3. 刷新星仪资源后，批准具体档位与正式 `sbatch`；
4. 接受下载审计、WSL real-MPB 复算和图件，再授权模型训练。

本计划不包含正向模型训练、inverse 训练、surrogate 排名、real-MPB inverse replay，也不证明 K=20 对所有结构分布最优。1 万条生成成功只关闭本数据合同的生成与验收门。

## 八、人工理解记录

```text
解决了什么问题：把统一 Fourier v2 的 K=20 条件扩展为总计 1 万条的可恢复、多节点、可验收数据 campaign。
为什么采用这个方案：B/G 实测耗时不同，2+3 节点能以较少空等实现接近同步完成；plan、partition、finalize 分离可避免多作业共享写入。
改变了项目哪一层/哪一模块：主层级 L3；后续 L4 修改只服务于本实验合同，包括 v2 HPC finalizer、配置、作业入口和 acceptance。
使用什么证据验证：现有实现/部署、64+64 real-MPB 试点、K=5–20 配对试验、resolution=64 收敛结果和提交前星仪只读资源快照。
最大风险、未知或下一 gate：64+64 仍缺新 family 的 WSL acceptance，多 shard finalizer 仍需 v2 专项测试；二者通过后才可申请正式计费作业。
```
