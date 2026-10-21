---
title: "MPB 分辨率收敛与耗时验证计划"
summary: "在星仪超算上用 100 个分层全胞结构比较 MPB resolution 的能带差异、耗时与推荐值。"
page_type: plan
status: draft
plan_state: active
effective_date: 2026-08-17
superseded_by: null
related_results: []
audience: "研究用户、超算执行者与导师"
owner: "研究生维护，导师确认样本量、容差与算力"
last_reviewed: 2026-08-17
reading_time: 12 minutes
primary_layer: L3
---

<!--
  Plan for MPB resolution convergence and runtime validation on Xingyi.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# MPB 分辨率收敛与耗时验证计划

## 一、目标与非目标

本计划在星仪超算上对同一批解析全胞结构重复运行 real MPB，量化不同 `resolution` 的能带差异、墙钟时间、吞吐和失败率，并据此冻结后续数据生成的 MPB 求解分辨率。计划本身不授权连接、提交、恢复或计费运行，也不把任一 smoke 结果升级为正式收敛结论。

本计划不研究神经网络输入采用 32、64、128 还是 256；模型输入栅格与 MPB `resolution` 是两个独立变量。除 `resolution` 外，晶格、材料、结构、k 点、极化、能带数、`mesh_size`、容差、软件版本、线程数和节点类型全部保持不变。

## 二、已有能力与本轮增量

仓库已有 [`full_cell_resolution_convergence_v1.yaml`](https://github.com/fangtoast/phc2d/blob/main/configs/research/full_cell_resolution_convergence_v1.yaml) 和 [`full_cell_convergence.py`](https://github.com/fangtoast/phc2d/blob/main/src/phc2d/data/full_cell_convergence.py)，当前合同使用 B/G 各 8 个结构、9 个代表性 k 点比较 64/96/128。它适合作为小型准入门，但没有回答以下问题：

- 100 个结构上的误差尾部、结构族差异和失败分布；
- 各 resolution 在同一星仪节点上的可比计算时间；
- 全部 81 个 full-BZ k 点上 bands 2–10 的收敛；
- binary 尖锐界面与 graded 平滑介质是否需要不同的推荐 resolution。

因此本轮不覆盖旧证据，而是新增“资源 pilot → 100 结构正式确认”的扩展协议。

## 三、实验问题与预注册决策

1. 以最高分辨率结果为参考时，64/96 是否已经满足 bands 2–10 的误差容限？
2. 计算时间随 resolution 增长的实际曲线是什么，binary 与 graded 是否不同？
3. 误差是否集中在特定结构族、k 点、band、近简并或小尺度结构？
4. 是否能够为 B/G 冻结一个共同 resolution；若不能，是否值得维护分路线求解合同？

默认优先选择满足精度门的最低共同 resolution，以保持标签合同简单。只有当共同值造成显著且可量化的资源浪费时，才提交“分路线 resolution”供人工决定。

## 四、样本数与分层

正式规模冻结为 **100 个结构：binary 50 个、graded 50 个**。100 足以覆盖当前主要结构层，同时仍能在星仪上采用受控的单批或分片运行；它是工程与统计折中，不代表总体误差的最终极限证明。

| 路线 | 分层 | 数量 | 选择规则 |
|---|---|---:|---|
| B | `binary_polar_holes_1` | 13 | 按几何哈希排序后等距抽取 |
| B | `binary_polar_holes_2` | 12 | 同上 |
| B | `binary_spectral_threshold` | 25 | 同上，并覆盖材料占比与高频能量分位数 |
| G | `graded_core_1` | 25 | 按几何哈希排序后覆盖尺度、温度和材料均值分位数 |
| G | `graded_core_2` | 25 | 同上 |

固定样本清单必须记录 dataset SHA-256、sample index、sample ID、geometry hash、generator stratum、关键形态统计和 split。不得按运行结果重新抽样。若新生成器在计划 2 中采用新的 family 名称，应保持 50/50 路线总数，并在每条路线内按最终冻结 strata 重新等额或近等额分配。

## 五、resolution 梯度与求解合同

资源 pilot 先比较：

```text
resolution = [32, 48, 64, 96, 128]
```

正式 100 样本默认保留同一梯度，以同时获得低分辨率失真、64 的现有基准以及 96/128 的收敛趋势。若 16 样本 pilot 证明 32 或 48 必然失败且其耗时信息已稳定，可在人工确认后把正式矩阵缩减为 `[64, 96, 128]`；不得由脚本自动删减。

共同冻结项：

| 项目 | 合同 |
|---|---|
| backend | real `meep.mpb.ModeSolver`，禁止 proxy |
| geometry | `FullCellStructureSpec.epsilon_at` 解析周期材料函数 |
| lattice | square，$a=1$ |
| material | 与样本结构保存的 `epsilon_air/epsilon_material` 一致 |
| polarization | TE |
| k points | 全部 9×9 full BZ，共 81 点 |
| bands | 求解 10 条并保留 guard；headline 比较 bands 2–10 |
| mesh/tolerance/order | `mesh_size=3`、`1e-7`、`frequency_ascending` |
| parallelism | 每个 MPB worker 单线程；固定节点类型和 worker 数 |
| reference | 本轮最高 resolution 128；它是数值参考，不是解析真值 |

若 96→128 仍未收敛，停止推荐，不自动接受 128；另行追加 160/192 小样本复核。

## 六、两阶段星仪执行

本机只使用已有 SSH 别名 `xingyi`。账号、IP、端口和私钥位置保留在用户本机 SSH 配置，不复制进版本库。星仪部署遵守 [`xingyi_common.sh`](https://github.com/fangtoast/phc2d/blob/main/jobs/xingyi_common.sh) 的当前根目录、环境、`DEPLOYED_GIT_COMMIT`、Meep/MPB 版本和单线程约束。

### 6.1 阶段 A：16 个结构资源 pilot

- B/G 各 8 个，覆盖全部 strata；
- 5 个 resolution、81 个 k 点、10 bands；
- 使用一个与正式任务相同类型的 64 CPU 节点；
- 固定并记录 worker 数，禁止让不同 resolution 使用不同总 CPU 配额；
- 记录每个 `sample × resolution` 的 wall time、CPU time、峰值 RSS、solver 状态和重试状态；
- 输出预计正式 100 样本的 CPU-hours、wall time、磁盘量和推荐分片数。

阶段 A 只有在用户明确批准计费 pilot 后才能提交。任一环境、版本、材料函数、band order 或有限值检查失败即停止，不进入阶段 B。

### 6.2 阶段 B：100 个结构正式确认

- 使用阶段 A 冻结的 commit、配置、样本清单、节点类型和并发；
- 按 sample ID 或 resolution 分片，但一个原子任务必须写入独立 shard；
- finalizer 只读 shard，并 fail-closed 检查完整矩阵；
- 只读监控 Slurm 状态、完成数、失败数、吞吐、ETA 和磁盘；
- 不自动取消、重启、修改参数或重采样；恢复须单独授权并复用相同 commit/config/manifest。

阶段 B 必须在阶段 A 资源报告获人工接受、并再次获得计费授权后才能提交。

## 七、需要实现的代码与配置

正式运行前先完成一个 L4 实现任务，建议最小范围为：

1. 扩展 `src/phc2d/data/full_cell_convergence.py`：resolution 列表、全 81 k 点、样本数和执行环境由配置驱动，移除对 `[64,96,128]`、8 样本和 WSL-only 字符串的硬编码，同时保留 WSL 正式验收入口。
2. 新增 Xingyi 专用研究配置，显式冻结样本 manifest、resolution 梯度、计时字段、worker/thread 合同和门限。
3. 新增 Xingyi plan/solve-array/finalize 作业脚本，复用 `jobs/xingyi_common.sh`，默认 dry-run，并要求显式授权环境变量。
4. 增加 contract tests：配置拒错、分层选择稳定性、矩阵完整性、重复 shard、计时有限性、失败账本、finalizer fail-closed 和 CLI help。
5. 不把数据、shard 或运行输出加入 Git；代码、配置和测试必须单独提交后再部署固定 commit。

## 八、指标、统计与图件

对每个较低 resolution $R$ 和参考 128，计算：

$$
e_{s,k,b}^{R}=
\frac{|W_{s,k,b}^{R}-W_{s,k,b}^{128}|}
{\max(|W_{s,k,b}^{128}|,10^{-8})}.
$$

至少报告：

- B、G、combined 的 median、P90、P95、P99、max MRE；
- 按 stratum、band、k 点和材料统计量分层的 MRE；
- 每个 resolution 的总 wall time、每样本 wall time、CPU-hours、吞吐与峰值 RSS；
- 相对 64 的耗时倍率，以及精度改善/耗时增量；
- 失败率、超时率和所有失败案例。

必需图件：

1. resolution–median/P95/P99 MRE 曲线，B/G 分面；
2. resolution–每样本 wall time 与 CPU-hours 曲线；
3. 精度–耗时 Pareto 图；
4. `sample × resolution` 耗时箱线图；
5. 按 band 和 k 点的误差热图；
6. 最大误差样本的 64/128 能带叠加图与结构图。

## 九、接受门与推荐规则

门限在提交前人工冻结。默认候选门为：

- bands 2–10 combined median MRE `≤0.10%`；
- combined P95 MRE `≤0.50%`；
- B、G 各自 P95 MRE `≤0.75%`；
- 任一失败、NaN/Inf、矩阵缺失或 band order 违规均 fail-closed；
- 选中的 resolution 还必须在最坏 stratum 与最大误差样本上通过人工图件审查。

若 64 通过，则优先推荐 64；若 64 不通过而 96 通过，则推荐 96；若 96 不通过，不能仅因为 128 是本轮最高值就宣称 128 已收敛，必须追加更高参考。

## 十、标准产物与完成定义

运行目录遵守 `runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/`，至少包含：

```text
run.log
run.meta.json
config.resolved.yaml
sample_manifest.csv
job_manifest.csv
timing.csv
convergence_bands.npz
convergence_metrics.json
failed_case_ledger.jsonl
results.json
metrics.json
history.csv
figures/
```

完成条件是：代码/配置/测试通过、星仪两阶段运行均有完整制品、100 样本矩阵完整、统计与图件生成、集群结果经本机 WSL `Ubuntu` + `meep_env` 固定小样本交叉复算，并由人工接受推荐 resolution。任何较早阶段只能标记为 pilot、cluster result 或 pending acceptance。
