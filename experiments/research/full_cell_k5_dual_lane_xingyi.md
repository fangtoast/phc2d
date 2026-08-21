<!--
  EXP-DAT-005 K=5 binary/graded Xingyi dataset production contract.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-DAT-005：K=5 双路线数据生产与双端交付

本文是 `L3 实验与证据` 合同，冻结在 Xingyi 独立生成 binary/graded unified-Fourier `K=5` 数据各 `10,000` 条，并完成 Xingyi 审计、本地/N40 双端交付。配置、运行产物和哈希是机器证据；本文不承载 raw metrics、日志或求解器输出。

## 研究问题与边界

| 项目 | 冻结内容 |
|---|---|
| Experiment ID | `EXP-DAT-005` |
| 协议 | `full_cell_fourier_v2`；K=5 与 K=20 同合同、不同独立随机样本 |
| 路线 | binary 与 graded，各 `10,000`；不合并数组、不复用 K=20 sample plan |
| seed | binary `2026082105`；graded `2026082106` |
| split | `8000/1000/1000` |
| 输入与标签 | full-cell raster contract；`Wh[10000,81,10]`；81 个 full-BZ k 点、10 个 TE 频带、4 个 guard bands |
| 求解 | real MPB，resolution `64`，mesh `3`，tolerance `1e-7`；Xingyi native Linux conda `meep_env` |
| 输出 | `data/full_cell_binary_fourier_k5_bz9_10000_xingyi_v1.h5` 与 `data/full_cell_graded_fourier_k5_bz9_10000_xingyi_v1.h5` |
| 状态边界 | finalizer 后为 `cluster_generated_pending_wsl_acceptance`；本轮不执行 WSL replay、K=5 训练或 K=5/K=20 优劣结论 |

两路均不设置 `paired_coefficient_seed`，不把本轮误记为 paired cutoff study。结构计划必须由各自 seed 独立产生，且 plan hash、geometry/content hash 和 collision ledger 分路保存。

## 控制面与资源合同

复用现有 `full_cell_band` adapter 与通用 `phc2d hpc` 控制面：

```text
plan job → solve array[0-7] → afterok finalizer → hpc audit
```

每条路线独立使用 `8` 个 exclusive Xingyi `xyfree` 节点；每个 job 请求 `1` 节点、`64` CPU、`128 GB`，每个 solve shard 覆盖连续且互斥的 `1,250` 条样本。两条链可并行，峰值为 `16` 节点、`1,024` CPU。排队时保持该资源合同，不自动降档、取消、重提、改参数或重新采样。

K=20 仅作为资源预算参考；本轮实际墙钟、CPU 时间和 MaxRSS 必须来自本轮 Slurm `sacct`/run 证据，不用外推值替代。

## 执行顺序

1. 本地完成 YAML/schema 校验、CLI help、HPC dry-run、定向测试和无界面 Fourier/HPC 回归。
2. 用第一个配置就绪 scoped commit 部署 Xingyi；远端只读核对 checkout commit、`DEPLOYED_GIT_COMMIT`、环境版本、磁盘、队列和 shell/Python 语法。
3. 在计算节点执行 solver environment/oracle 预检，并对 binary/graded 各执行 4 条 real-MPB smoke；任一路失败即停在人工处理门。
4. 两条独立链分别执行 `hpc plan`、`solve-shard --array=0-7` 和 `afterok` finalizer。finalizer 是唯一正式 HDF5 写者，禁止 shard 直接写最终文件。
5. 每 30 分钟只读查看 `squeue/sacct`、8/8 shard 覆盖、失败数和 finalizer 状态。异常只进入人工处理门，不自动重试、取消、重提或改资源。

## Xingyi 通过门

每条路线必须同时满足：

- plan、8 个 solve task、finalizer 均 `COMPLETED/0:0`；
- frozen plan hash 一致，覆盖 `10,000/10,000`，无重复/遗漏，collision ledger 为空；
- 所有 `Wh` 有限且频带升序，形状为 `[10000,81,10]`，split 为 `8000/1000/1000`；
- partition hash、dataset hash、solver provenance、results/metrics/history 和日志齐全；
- `phc2d hpc audit --run-dir ...` 通过，最终状态仍为 `cluster_generated_pending_wsl_acceptance`。

任一环境/commit 漂移、plan/hash 漂移、结构碰撞、非有限或乱序频带、shard 非零退出、覆盖不完整、finalizer/audit 失败，都阻断正式完成。

## 双端交付门

Xingyi 冻结最终 HDF5 SHA-256 后，下载到本地只取最终 HDF5 及必要 provenance、config、manifest、ledger、metrics、results、history 和日志；不下载 partitions、shards 或完整图件。传输使用临时 `.part` 文件和可续传方式，完成后核对哈希。

再从本地以临时 `.part` 文件上传 N40 `$HOME/run/phc2d/data`，同步精简 provenance 到标准 run 目录；N40 哈希通过后原子改为正式文件名。只有 Xingyi、本地、N40 三端大小和 SHA-256 均一致，才把交付记录为完成。传输中断只允许续传同一不可变文件，不触发计算重跑。

成功后的结果记录必须回链本协议、两个配置、两条 run、三端哈希和 `phc2d hpc audit` 输出，并更新 Registry、结果索引和 `PROJECT_MAP.md`。在这些证据出现前，Registry 保持 `planned/pending`，不得升级为 `completed` 或 `physically_verified`。
