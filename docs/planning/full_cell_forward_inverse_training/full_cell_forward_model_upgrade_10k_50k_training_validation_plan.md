<!--
  Draft 10k/50k validation plan for upgraded full-cell forward models on N40.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="full-cell-forward-model-upgrade-training"></a>

# 全胞正向升级模型 10k/50k 训练与验证方案

| Metadata | Value |
|---|---|
| Status | Draft experiment plan |
| Primary layer | L3 experiment/evidence |
| Authority | [生成网络调整研究计划](../fno_generative_network_adjustment_plan_20260617.md) |
| Implementation prerequisite | [全胞正向模型升级代码实现规格](full_cell_forward_model_upgrade_implementation_plan.md) |
| Audience | 项目负责人、实验执行者、N40 运维人员、结果审阅者与 AI 协作者 |
| Owner | 研究生维护，导师审阅 promotion 与科学主张 |
| Last reviewed | Pending human review |
| Reading time | 20 minutes |

本文只定义升级模型在 10k/50k full-cell 数据上的训练、验证、N40 调度和证据边界。模型源码当前是 `implemented / no_result`；本文不是付费训练授权，也不证明任何新模型优于当前 FNO。

## 一、目标、前置门与禁止提升

### 1.1 主要研究问题

在冻结数据、split、损失、精度和三 seed 条件下回答四个问题：

1. 每个升级模型是否优于其直接旧基线？
2. 显式 `k_points` 的 `k_conditioned` 输出头是否优于固定 dense head？
3. 是否至少有一个升级模型在 binary 与 continuous 两条 native lane 上同时优于当前 `fno2d_band_v2`？
4. `dual_lane_band_operator_v1` 能否在一个共享 checkpoint 中保持两条 native lane 的精度，同时降低总参数与存储？

### 1.2 正式执行前置门

以下任一项缺失，campaign 状态必须保持 `blocked_before_training`：

- [实现规格](full_cell_forward_model_upgrade_implementation_plan.md) 已由人工接受；
- 五个模型、两种输出头、registry/runner 和 checkpoint 已实现；
- WSL `Ubuntu` + conda `meep_env` 的 unit/integration/legacy regression tests 通过；
- N40 部署 commit 固定，`DEPLOYED_GIT_COMMIT` 与本地计划 commit 一致；
- 数据文件、schema、dataset SHA-256、split hash 和 `k_points` hash 通过；
- 每个 model×head×lane 至少完成一批单卡 admission probe；
- 用户对具体 campaign、任务数和预算给出新的明确授权 ID。

“后续训练使用 N40 的 8 张 RTX4090”只冻结平台与并发方式，不等于本轮已经授权提交。

### 1.3 禁止提升

- admission probe、单 seed 或 validation 改善不得写成正式模型胜出。
- 10k 输出头选择不得读取 10k test；50k 模型选择不得读取 50k test。
- cross-lane、threshold 和 final-fit monitor 不得替代 native held-out test。
- 64 checkpoint 接受 32/128 shape 不等于分辨率不变。
- surrogate MRE 改善不等于 inverse replay 或 real-MPB 物理设计改善。

## 二、模型、输出头与基线身份

### 2.1 一对一升级关系

| 直接基线 | 升级模型 | 两种输出头 | 当前状态 |
|---|---|---|---|
| `mlp` | `mlp_mixer_band_v1` | dense / k-conditioned | planned, no_result |
| `res_mlp_wide` | `res_mlp_gated_v2` | dense / k-conditioned | planned, no_result |
| `lifted_cnn` | `convnext_band_v1` | dense / k-conditioned | planned, no_result |
| `fno2d_band_v2` | `tfno2d_band_v3` | dense / k-conditioned | planned, no_result |
| 两条独立 native 模型 | `dual_lane_band_operator_v1` | dense / k-conditioned | planned, no_result |

旧四模型、旧 checkpoint 和现有 10k/50k 报告保留为比较基线，不因新 campaign 改名、重训或覆盖。

### 2.2 当前冻结证据

| 报告 | 可作为的基线 | 必须保留的边界 |
|---|---|---|
| [10k 双路线](../../results/forward/full_cell_dual_lane_forward_training_20260806.md) | 102/102 run、FNO 七波次最低均值、跨数据与独立分辨率重训 | 32/64/128 是各自重训，不是零样本迁移 |
| [50k 双路线](../../results/forward/full_cell_dual_lane_50000_forward_training_20260808.md) | 204/204 run、906 eval cells、40k held-out test、300 组 paired bootstrap、64→32/128 零样本输入 | threshold/final-fit 是训练重叠 monitor；无 M9 real-MPB 结论 |

当前 50k held-out test 的主要 FNO 锚点为：binary 100/200 epoch `0.62965%`，continuous 100 epoch `0.16441%`、200 epoch `0.16016%`；binary→continuous 100 epoch `1.71709%`，continuous→binary 100 epoch `2.09599%`。这些数字只用于预注册比较，不允许从测试集反向调参。

## 三、数据、split 与归一化

### 3.1 10k 合同

| Lane | Dataset | Split | 主输入 | Target |
|---|---|---|---|---|
| binary | `full_cell_binary_bz9_10000_xingyi_v1.h5` | 8k/1k/1k | `material_weight_grid_64` | `Wh` |
| graded | `full_cell_graded_bz9_10000_paracloud_m9_v1.h5` | 8k/1k/1k | `material_weight_grid_64` | `Wh` |

两套本地 10k 数据当前 `k_points` 均为 `[81,2]` 且字节 hash 一致。正式 run 仍必须重新记录 dataset/split/k-point hash，不能用本文叙述替代机器审计。

### 3.2 50k 合同

| Lane | Dataset | Train | Validation | Test |
|---|---|---:|---:|---:|
| binary | `full_cell_binary_bz9_50000_xingyi_v1.h5` | 40,000 | 5,000 | 5,000 |
| graded | `full_cell_graded_bz9_50000_xingyi_v1.h5` | 40,000 | 5,000 | 5,000 |

50k 文件当前不在本机数据目录；N40 预检必须使用现有冻结配置记录的 dataset SHA-256、`idx_train/idx_val/idx_test` hash，并额外比较两 lane 的 target shape、band 语义与 `k_points` hash。任一不一致都阻断联合模型。

### 3.3 归一化与泄漏门

- 单 lane 模型只使用本 lane train split 计算输入/target statistics。
- 联合模型保持两个 target normalizer；`lane_id` 选择对应 normalize/denormalize，不把 validation/test 纳入统计。
- 所有 normalization statistics、hash、样本数与字段名写入 checkpoint metadata 和 `config.resolved.yaml`。
- binary 与 graded batch 可以混合，但不能通过对齐 sample index 假设两数据集结构一一对应。

## 四、共同训练协议

为了把主要差异限制在网络和输出头，主比较冻结为：

| 项目 | 固定值 |
|---|---|
| optimizer | AdamW |
| weight decay | `1e-5` |
| loss | normalized full-matrix MSE |
| checkpoint selection | validation bands 2–10 MRE |
| micro/effective batch | 64 / 64 |
| runtime precision | FP32 |
| AMP / TF32 | disabled / disabled |
| seeds | `0,1,2` |
| data workers | 4 |
| gradient clipping | disabled；非有限梯度直接失败 |

固定学习率如下，不在正式矩阵内追加模型专属搜索：

| Model | Learning rate |
|---|---:|
| `mlp_mixer_band_v1` | `3e-4` |
| `res_mlp_gated_v2` | `3e-4` |
| `convnext_band_v1` | `1e-3` |
| `tfno2d_band_v3` | `5e-4` |
| `dual_lane_band_operator_v1` | `5e-4` |

若任一 architecture 的 seed0 admission 出现非有限 loss/gradient，统一将该 architecture 学习率减半并重新执行所有 head/lane admission；不得只修复表现不佳的 seed。仍失败则标记 `blocked_unstable`，不进入正式矩阵。

## 五、10k 输出头与架构筛选

### 5.1 冻结任务矩阵

单 lane 四模型：

$$
4\ \text{models}
\times 2\ \text{heads}
\times 2\ \text{lanes}
\times 3\ \text{seeds}
=48\ \text{jobs}.
$$

联合模型：

$$
1\ \text{model}
\times 2\ \text{heads}
\times 3\ \text{seeds}
=6\ \text{jobs}.
$$

10k 正式总数固定为 **54 个单 GPU jobs**。每个 job 最多 100 epochs，early-stopping patience 15；联合模型每个 batch 固定 32 binary + 32 graded，并分别记录两 lane validation metric。

### 5.2 只使用 validation 的 head 选择

对每个单 lane architecture 独立选择输出头；联合模型以两 lane validation MRE 的等权平均选择。规则按顺序执行：

1. 若 `k_conditioned` 三 seed validation mean 比 dense 至少低 1% relative，选择 `k_conditioned`；
2. 若两者差距在 ±1% 内，且 `k_conditioned` 的 32/128 零样本输入平均 MRE 更低、两项 paired CI 至少一项完全低于 0、native validation 不退化超过 1%，选择 `k_conditioned`；
3. 其余情况选择 dense。

选择冻结后才读取 10k test。若模型不支持对应零样本输入，第二条不得用缺失值偏向查询头。

### 5.3 10k 架构准入 50k

每个升级模型与其直接旧基线比较。进入 50k 至少满足：

- 两条 native lane 中至少一条的三 seed test mean relative MRE 改善 5%；
- 另一条 native lane 不退化超过 5%；
- 三个 seed 均可复现，未出现 checkpoint、梯度或数据谱系失败；
- 参数量在 2–8M，峰值显存低于单卡可用预算；
- 输出头选择全程没有读取 test。

联合模型只要两条 native test 相对对应 direct baseline 都不退化超过 5%，即可进入 50k；这只是共享建模可行性门，不是最终 promotion。

## 六、50k 确认矩阵

### 6.1 任务数

每个单 lane architecture 只保留 10k 选中的 head：

$$
4\ \text{models}
\times 2\ \text{lanes}
\times 3\ \text{seeds}
=24\ \text{jobs}.
$$

联合模型使用选中的一个 head 和三个 seed，共 3 jobs。50k 正式总数为 **27 个单 GPU jobs**。若某个模型未通过 10k 准入，对应 50k 行保留在冻结计划中并标为 `not_admitted`，不得用其他模型补足任务数。

### 6.2 Epoch 100 继续训练门

所有 50k jobs 预注册最大 200 epochs。在 epoch 100 比较 validation bands 2–10 MRE：

$$
r=
\frac{\operatorname{mean}(m_{81:90})-
      \operatorname{mean}(m_{91:100})}
     {\operatorname{mean}(m_{81:90})}.
$$

- `r < 0.005`：在 epoch 100 停止；
- `r >= 0.005`：继续到最多 epoch 200，并启用 patience 15；
- epoch 100 前已产生非有限值或 artifact 不完整：失败，不进入继续门。

100/200 的 checkpoint 角色、实际 stop epoch 和 best-validation epoch 必须分别写入结果；不能把同一 best checkpoint 重复计为独立训练证据。

## 七、N40 8×RTX4090 调度

### 7.1 资源合同

| 项目 | 固定值 |
|---|---|
| platform | `paracloud_n40` |
| partition | `gpu_4090` |
| GPU per task | 1 |
| CPU per task | 6 |
| array concurrency | `%8` |
| environment | `phc2d_gpu_cu118` / CUDA 11.8 |
| precision | FP32 formal accuracy run |

8 张 GPU 用于同时运行 8 个独立 model/head/lane/seed 任务。2–8M 参数模型不采用 8 卡 DDP；这样可以最大化矩阵吞吐、隔离失败并保持单卡 checkpoint 身份。BF16 只允许在最终获胜模型上另建吞吐 profile，不与 FP32 精度结果合并。

### 7.2 排队顺序与失败隔离

任务清单按预计耗时排序：

1. `tfno2d_band_v3` 与 `dual_lane_band_operator_v1`；
2. `res_mlp_gated_v2`；
3. `convnext_band_v1`；
4. `mlp_mixer_band_v1`。

Slurm array 每行只负责一个独立 run directory。单任务失败不取消其他已运行任务；重提只允许使用相同 config/hash/seed 和新的 retry lineage，不覆盖旧目录。

### 7.3 授权与 dry-run 门

未来 submitter 必须默认 dry-run，并在提交前输出：

- campaign id、部署 commit 与授权 ID；
- 54 或 27 行冻结任务清单；
- 两套数据与 split/k-point hashes；
- 每行模型、head、lane、seed、资源和 run directory；
- admission probe 汇总；
- 预计 GPU-job 数和并发上限。

只有显式 submit mode 与独立 `PHC2D_AUTHORIZE_UPGRADE_CAMPAIGN=YES` 同时存在才允许 `sbatch`。本文及其人工接受本身均不是授权。

## 八、评估矩阵与统计

### 8.1 Native 主评估

主指标为 held-out test 的 bands 2–10 MRE。每个模型同时报告：

- 三 seed mean ± std；
- bands 1–10 per-band MRE；
- MAE、RMSE、full-matrix MRE 与 ordering violations；
- 参数量、checkpoint bytes、peak GPU memory；
- `train_time_s`、`eval_time_s`、`inference_time_per_sample_ms`、throughput；
- 10,000 次 paired sample bootstrap 的差值、95% CI 和方向。

### 8.2 诊断评估

| 诊断 | 范围 | 证据身份 |
|---|---|---|
| binary→continuous / continuous→binary | 所有 50k admitted 模型 | cross-lane domain-shift diagnostic |
| 64→32/128 direct input | Mixer、ConvNeXt、TFNO、联合模型 | zero-shot input-resolution diagnostic；不是 MPB 收敛 |
| threshold `0.3–0.7` | 只对冻结胜者执行 | representation proxy；不具物理等价性 |
| k permutation/subset | k-conditioned head | interface/consistency check；不是 unseen-k 物理精度 |

ResMLP 因固定展平输入不参与 64→32/128 direct input 排行；缺失项标为 `not_applicable`，不得按失败或零误差汇总。

## 九、预注册 promotion 标准

### 9.1 直接升级模型

相对对应旧基线，只有同时满足下列条件才写为 `promoted_over_direct_baseline`：

- 至少一条 native lane 的 50k test mean MRE relative 改善不低于 10%；
- 改善 lane 的 paired bootstrap 95% CI 完全低于 0；
- 另一条 native lane relative 退化不超过 3%；
- 三 seed、artifact、运行时和参数预算全部合格。

否则分别记录 `local_improvement_only`、`no_clear_gain` 或 `regressed`。

### 9.2 相对当前最佳 FNO

只有至少一个新模型在 binary 与 continuous 两条 native lane 上都相对 `fno2d_band_v2` 改善不低于 5%，且两条 paired 95% CI 均完全低于 0，才能声明“总体优于当前最佳 FNO”。任何单 lane 胜出都只能写局部改进。

### 9.3 联合模型

联合模型只有同时满足以下条件才可替代两个独立模型：

- 两条 native lane 相对各自最佳独立升级模型的 MRE 退化均不超过 3%；
- 单 checkpoint 参数量与存储不超过两个独立模型总和的 60%；
- 两个 lane 的 worst-seed 均未发生超过 5% 的 relative regression；
- 不以 cross-lane 结果替代 native 结果。

## 十、运行制品与恢复

每个 run 必须遵循：

```text
runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/
```

最小制品包括：

- `run.log`
- `run.meta.json`
- `config.resolved.yaml`
- `results.json`
- `metrics.json`
- `history.csv`
- `checkpoint.pt` 及 SHA-256
- `model_summary.json`
- `n40_profile.json`
- `figures/`（存在曲线或诊断图时）

campaign 顶层必须有任务清单、提交清单、状态快照、失败/重试谱系、聚合配置、完整性审计和机器可读 leaderboard。聚合器只能读取完成且 hash/shape/provenance 通过的 run。

## 十一、状态与人工验收

| 状态 | 含义 |
|---|---|
| `planned` | 只有本文和实现规格 |
| `implemented` | 代码与 WSL tests 通过，但没有正式训练 |
| `admission_pass` | N40 一批 probe 通过 |
| `pilot_pass` | 10k 三 seed/两 head 矩阵完成且准入 50k |
| `formal_completed` | 50k admitted 矩阵、held-out test、统计与制品审计完成 |
| `physically_verified` | 本 forward campaign 不授予；需要独立 real-MPB/replay 任务 |

人工验收必须确认：

1. 10k head 选择没有读取 test；
2. 50k split/hash 与旧报告一致；
3. 8 卡是八个独立单卡作业而非 DDP；
4. final-fit、threshold、cross-lane、zero-shot resolution 的证据标签正确；
5. promotion 由预注册门得出，没有事后降低阈值；
6. 是否授权后续 inverse 使用新 surrogate，必须另开任务决定。

## 十二、人工理解记录

- 本文解决的问题：把五个 planned 模型的 10k 输出头筛选、50k 确认和 N40 八卡执行拆成可审计实验合同。
- 关键选择：正式精度保持 FP32 与旧损失；8 卡并行独立任务；validation 选择 head；test 只在冻结后使用。
- 当前没有解决的问题：没有模型代码、N40 admission、训练 job、checkpoint 或新结果。
- 最近的下一 gate：先实现并通过 [代码实现规格](full_cell_forward_model_upgrade_implementation_plan.md) 的 WSL 测试，再由用户决定是否授权 54 个 10k jobs。
