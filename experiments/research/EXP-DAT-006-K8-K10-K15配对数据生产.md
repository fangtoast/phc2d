<!--
  EXP-DAT-006 paired K8/K10/K15 dataset production protocol.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-DAT-006：K8/K10/K15 双路线配对数据生产

## 结论与状态

`EXP-DAT-006` 已完成实现与配置准备，当前状态是 `planned / config_ready`。本页冻结 K8、K10、K15 在 Binary 与 Graded 两条 lane 中的 60,000 条 real-MPB 数据合同，但不代表已授权或已执行正式计费求解。

| 属性 | Binary | Graded |
|---|---:|---:|
| K | 8 / 10 / 15 | 8 / 10 / 15 |
| 每 K 样本数 | 10,000 | 10,000 |
| train / val / test | 8,000 / 1,000 / 1,000 | 8,000 / 1,000 / 1,000 |
| `dataset.seed` | `2026091501` | `2026091502` |
| `sample_id_prefix` | `EXP-DAT-006-B` | `EXP-DAT-006-G` |
| `paired_cohort_seed` | `2026091551` | `2026091552` |
| 材料比例 | 0.50–0.90 | 0.60–0.78 |
| 温度 | 不适用 | 0.08–0.16 |

## 研究问题与边界

本实验只回答：在样本身份、目标材料比例、Graded 温度和共享低频 Fourier 系数固定时，不同径向截断 K 能否产生完整、可追溯且通过跨环境验收的全胞数据。`K` 是 Fourier 径向高频截断，不是 BZ k-grid 尺寸。

非目标包括训练模型、挑选“最优 K”、上传其他训练集群，以及将 Xingyi 生成或完整性审计表述为物理验收。历史 K5/K20 不重生成，只作非配对参考。

## 冻结生产合同

- 六份输出命名为 `full_cell_{binary|graded}_fourier_k{8|10|15}_bz9_10000_xingyi_v1.h5`。
- 表示合同为 `full_cell_fourier_v2`；raster 为 `[32,64,128,256]`；求解为 real MPB、resolution 64、9×9 full-BZ、10 个 TE bands 加 4 个 guard bands。
- 同 lane 的样本使用共享前缀与样本序号形成稳定配对键。每个 `(lane, pairing_key, mx, my)` 的 Fourier 系数由 `paired_cohort_seed` 独立派生，因此新增高频模态不改变旧模态。
- K8/K10/K15 分别含 98/158/354 个独立半平面模态。同 lane 内共享模态系数、split、stratum、目标材料比例和 Graded 温度必须逐样本精确一致。
- `generator.paired_cohort_seed` 与历史 `paired_coefficient_seed` 互斥。未设置新字段时，样本 ID 与随机行为保持旧合同。

## Xingyi 分段编排

| 阶段 | 任务形状 | 上限 | 继续条件 |
|---|---|---:|---|
| plan | 六个独立 plan job | 1 h | 六份 `sample_plan.jsonl` 完整 |
| pairing-audit | 一个跨 K 流式审计 | 1 h | 两 lane 各 10,000 个配对键通过 |
| smoke | 一个独占节点并行六配置 | 2 h | commit/版本/磁盘/64 CPU/real-MPB 全部通过 |
| solve K8 | Binary + Graded 各 `array[0-7]` | 6 h | 新的正式计费授权 |
| finalize K8 | 两个 finalizer | 2 h | K8 的 16 个 shard 全部成功 |
| wave-audit K8 | 一个只读审计 job | 1 h | Binary/Graded 的 `phc2d hpc audit` 都通过 |
| solve/finalize K10 | 同上 | 6 h / 2 h | K8 关闭后手动继续 |
| solve/finalize K15 | 同上 | 6 h / 2 h | K10 关闭后手动继续 |

`solve` 阶段只保留两组 8-task array，刚好占用账号 `MaxSubmitJobs=16` 上限，且不预占 finalizer 额度。队列无空闲节点时保持排队，不自动降规、取消、重采样、重试或恢复。每 30 分钟只读查看队列、shard 覆盖、失败数和 finalizer；任何失败、超时、哈希漂移或额度错误都停在当前 gate。

## 完成与证据 gate

1. 六份远程数据分别通过 `phc2d hpc audit`：10,000/10,000、8/8 shards、零失败/碰撞、无 NaN/Inf、频带升序、split 正确，且 plan/partition/dataset/provenance 哈希完整。
2. 下载只包含最终 HDF5、冻结配置、manifest、ledger、metrics、results、history 和 Slurm 日志；先写 `.part`，再核对 SHA-256，不下载 partition/shard 中间文件。
3. 在 WSL Ubuntu `meep_env` 中对六份数据分别运行 `phc2d data accept-cluster`；固定 8 个结构、9 个代表 k 点、Bands 2–10，median MRE ≤ 0.5%，P95 MRE ≤ 1%。
4. 只有六份数据全部通过 WSL 验收，才能把实验更新为 `completed / accepted`，写入数据 provenance 结果页、结果索引和 `PROJECT_MAP.md`。

## 当前人工决策

正式 K8 求解前，需要一次明确覆盖六份数据和 K8 → K10 → K15 三个 solve wave 的计费授权及授权 ID。授权不改变科学合同，也不授权任何自动恢复或资源变更。
