<!--
  Architecture and operating contract for portable HPC physics campaigns.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 超算物理数据生成统一框架

| Metadata | Value |
|---|---|
| Status | Draft implementation contract |
| Audience | 物理数据生成维护者、超算作业提交者与 AI 协作者 |
| Owner | 研究生维护；正式资源规模和物理结论由人类负责人接受 |
| Last reviewed | 2026-08-07 |
| Reading time | 8 minutes |

本文规定 Xingyi 与 Parallel Computing Cloud M9 上 MPB/Meep 数据任务的公共执行架构。它覆盖计划、资源选择、分片求解、汇总、状态、下载前审计和 WSL 验收边界；不覆盖神经网络 GPU 训练，也不表示尚未迁移的任务已经具备多节点执行能力。

## 执行流水线

```text
plan -> benchmark/admission -> solve array -> finalize -> download/audit -> WSL acceptance
```

各阶段是独立 Slurm 作业。`plan` 和 `finalize` 只占一个节点；`solve array` 根据基准证据选择 1、2、4 或 8 个节点。正式提交脚本使用 `afterok`，任一求解分区失败时不会启动 finalizer。

| 阶段 | 主要产物 | 完成含义 |
|---|---|---|
| Plan | `sample_plan.jsonl`、`physical_plan_manifest.json` | 输入、顺序、split 与哈希冻结 |
| Solve | `shards/shard_NNN/partition.h5`、`progress.json` | 分区内所有 case 成功，不代表最终数据完成 |
| Finalize | `data/*.h5` 与标准 run 产物 | 超算生成完成，仍等待下载与跨环境验收 |
| Accept | 哈希/内容审计与 WSL replay | 达到项目规定的跨环境证据门 |

分区求解采用“多 worker 计算、一个协调进程写入”的模式。每个节点只写自己的 HDF5 分区，使用完成位图支持同一计划、同一配置和同一 commit 下的显式恢复，不再创建每样本 JSON/NPY 文件。

## 公共接口

```bash
phc2d hpc adapters
phc2d hpc plan --adapter full_cell_band --config CONFIG --run-dir RUN --workers 64 --shards 4 --platform xingyi
phc2d hpc benchmark --platform-profile PLATFORM --evidence BENCHMARK.json --case-count 50000
phc2d hpc solve-shard --adapter full_cell_band --config CONFIG --run-dir RUN --shard-index 0 --shard-count 4 --workers 64
phc2d hpc finalize --adapter full_cell_band --config CONFIG --run-dir RUN --workers 64
phc2d hpc status --run-dir RUN
phc2d hpc audit --run-dir RUN
```

`status` 只读取 campaign manifest、每个分区的一个进度文件和最终产物清单，开销随节点数增长，不遍历样本文件。`audit` 只检查文件与哈希完整性，不会把超算产物提升为跨环境物理验收通过。

## 资源策略

节点数由实测单 case 中位时间、节点核数、目标墙钟时间和 0.75 保守并行效率计算。默认目标为三小时，选择满足目标的最小已准入档位。四节点必须达到单节点至少三倍吞吐；八节点必须有额外强缩放证据。

平台合同：

- [`../configs/research/hpc_platform_xingyi_v1.yaml`](../configs/research/hpc_platform_xingyi_v1.yaml)：64 核/节点，`xyfree`。
- [`../configs/research/hpc_platform_paracloud_m9_v1.yaml`](../configs/research/hpc_platform_paracloud_m9_v1.yaml)：256 核/节点，`amd_m9_768`。

通用提交入口是 [`../jobs/hpc_submit_physics_campaign.sh`](../jobs/hpc_submit_physics_campaign.sh)。它默认只输出 dry-run；正式提交同时要求 `PHC2D_SUBMIT_MODE=submit` 和 `PHC2D_AUTHORIZE_FORMAL=YES`。

## Finalizer 合同

finalizer 必须先验证分区范围无遗漏、无重叠、plan hash 一致且失败数为零。full-cell 参考实现按块读取 bands，使用节点内多进程计算栅格；每个分辨率只计算一次 material weight，epsilon 从该结果派生。单写者把数据写入 run 目录中的隐藏 incomplete 文件，全部检查通过后原子发布到 `data/`。

部署目录只保存不可变代码与环境。正式数据、分区、manifest、metrics 和 provenance 必须属于 `runs/{date}/{campaign}/`，避免不同 campaign 使用相同输出名时互相覆盖。

## 适配器状态

| Adapter | 当前状态 | 说明 |
|---|---|---|
| `full_cell_band` | Reference adapter implemented | plan、HDF5 分区、finalize、status、audit 已接入 |
| `query_k_path` | Registered for migration | 旧脚本仍是执行入口，不得宣称已多节点迁移 |
| `field_eigen` | Registered for migration | 等待把 field result 接入公共分区容器 |
| `resolution_convergence` | Registered for migration | 等待拆分 case plan 与 reducer |
| `inverse_mpb_replay` | Registered for migration | 等待冻结候选计划和公共分区输出 |

注册只固定任务身份、输出类型和迁移边界。除 `full_cell_band` 外，其余任务在迁移状态变为可执行前，`phc2d hpc plan/solve-shard/finalize` 会明确拒绝，不允许静默回落到单机流程。

## 证据边界

- 本地 Windows 测试只证明 Python 合同与非物理流程可执行。
- WSL `Ubuntu` + `meep_env` 测试是项目实现验收环境。
- 正式超算 benchmark 才能准入具体节点档位。
- 超算 HDF5、manifest 和零失败只证明集群生成完成。
- 下载后的 SHA-256、内容审计和固定 WSL replay 通过后，才能标记 `cross_environment_accepted`。

禁止自动重启、恢复、取消、删除分区或扩大节点数。恢复、清理和计费正式提交都需要独立人工授权。

## 人类理解记录

```text
解决了什么问题：把超算物理数据任务的计划、分片、资源选择、状态、汇总和审计从 full-cell 专用脚本中抽出。
为什么采用这个方案：独立物理 case 适合跨节点分区，但共享 HDF5 仍应保持单写者。
改变了哪一层：L2 超算物理计算执行架构，以及必要的 L4 CLI、Slurm 和 full-cell 适配器。
使用什么证据验证：自动测试、WSL meep_env 测试、dry-run 提交合同和后续强缩放 benchmark。
最大风险和下一 gate：其余四类适配器尚未执行迁移；full-cell 还需要真实 1/2/4 节点 benchmark 才能准入正式新 campaign。
```
