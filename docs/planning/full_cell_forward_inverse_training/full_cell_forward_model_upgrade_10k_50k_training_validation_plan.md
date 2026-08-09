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
| Reading time | 28 minutes |

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
| `mlp` | `mlp_mixer_band_v1` | dense / k-conditioned | implemented, no_result |
| `res_mlp_wide` | `res_mlp_gated_v2` | dense / k-conditioned | implemented, no_result |
| `lifted_cnn` | `convnext_band_v1` | dense / k-conditioned | implemented, no_result |
| `fno2d_band_v2` | `tfno2d_band_v3` | dense / k-conditioned | implemented, no_result |
| 两条独立 native 模型 | `dual_lane_band_operator_v1` | dense / k-conditioned | implemented, no_result |

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

### 7.2 GPU 槽位、任务编号与单任务资源

本文中的 `S0`–`S7` 是一个 wave 内的八个**并发槽位**，不是预先绑定的物理 GPU 编号。每个槽位提交一个单卡 Slurm task；调度器可以把它分配到任意一张可用 RTX4090。作业内通常只看到重新映射后的 `CUDA_VISIBLE_DEVICES=0`，因此每个 `run.meta.json` 必须额外记录 `SLURM_ARRAY_TASK_ID`、物理 GPU UUID、GPU 名称和节点名，不能用进程内的逻辑编号冒充物理卡身份。

冻结缩写如下：

| 缩写 | 含义 |
|---|---|
| `MM` | `mlp_mixer_band_v1` |
| `RM` | `res_mlp_gated_v2` |
| `CX` | `convnext_band_v1` |
| `TF` | `tfno2d_band_v3` |
| `DL` | `dual_lane_band_operator_v1` |
| `D` / `Q` | dense / k-conditioned head |
| `B` / `G` / `J` | binary / graded / joint dual-lane |
| `s0`–`s2` | seed 0–2 |

单 lane 正式任务 ID 固定为 `{stage}_{model}_{head}_{lane}_{seed}`，例如 `10k_TF_Q_B_s1`；联合任务固定为 `{stage}_DL_{head}_J_{seed}`。每行任务资源均为 `1×RTX4090 + 6 CPU + FP32`，不允许在同一 GPU 上再启动第二个训练进程，也不允许一个任务跨两张卡。

### 7.3 Admission：正式训练前的三波探针

Admission 不计入 54 个正式 10k jobs。每项只执行一批 forward/backward、一次 optimizer step、validation forward 和 checkpoint save/load；禁止用 probe 指标选择模型或输出头。三波固定如下：

| Admission wave | S0 | S1 | S2 | S3 | S4 | S5 | S6 | S7 |
|---|---|---|---|---|---|---|---|---|
| `A0` | `TF-D-B-s0` | `TF-D-G-s0` | `TF-Q-B-s0` | `TF-Q-G-s0` | `DL-D-J-s0` | `DL-Q-J-s0` | `MM-D-B-s0` | `MM-D-G-s0` |
| `A1` | `MM-Q-B-s0` | `MM-Q-G-s0` | `RM-D-B-s0` | `RM-D-G-s0` | `RM-Q-B-s0` | `RM-Q-G-s0` | `CX-D-B-s0` | `CX-D-G-s0` |
| `A2` | `CX-Q-B-s0` | `CX-Q-G-s0` | idle | idle | idle | idle | idle | idle |

`A0` 先覆盖最接近当前强基线的 TFNO、联合模型和 Mixer dense；`A1` 覆盖剩余 Mixer、完整 ResMLP 和 ConvNeXt dense；`A2` 完成 ConvNeXt 查询头。只有 18 个组合的 shape、有限 loss/gradient、显存、checkpoint 和 artifact 均通过，才能创建正式 `10K-W1`。若某个 architecture 触发学习率减半，必须重跑该 architecture 的全部 admission 组合，不能只重跑失败槽位。

### 7.4 10k 正式训练：七波 54 个任务

以下是**全部模型均通过 admission**时的冻结波次。每个非 idle 槽位都是一个最多 100 epoch、patience 15 的独立单卡训练；同一波通过一个 `0-7%8` array 提交。第一波优先完成 TFNO dense 并启动联合模型 dense，第二波完成 TFNO query 和联合模型 dense，第三波补齐联合模型 query，随后依次完成 Mixer、ResMLP 和 ConvNeXt。

| 10k wave | S0 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | Jobs |
|---|---|---|---|---|---|---|---|---|---:|
| `10K-W1` | `TF-D-B-s0` | `TF-D-B-s1` | `TF-D-B-s2` | `TF-D-G-s0` | `TF-D-G-s1` | `TF-D-G-s2` | `DL-D-J-s0` | `DL-D-J-s1` | 8 |
| `10K-W2` | `DL-D-J-s2` | `TF-Q-B-s0` | `TF-Q-B-s1` | `TF-Q-B-s2` | `TF-Q-G-s0` | `TF-Q-G-s1` | `TF-Q-G-s2` | `DL-Q-J-s0` | 8 |
| `10K-W3` | `DL-Q-J-s1` | `DL-Q-J-s2` | `MM-D-B-s0` | `MM-D-B-s1` | `MM-D-B-s2` | `MM-D-G-s0` | `MM-D-G-s1` | `MM-D-G-s2` | 8 |
| `10K-W4` | `MM-Q-B-s0` | `MM-Q-B-s1` | `MM-Q-B-s2` | `MM-Q-G-s0` | `MM-Q-G-s1` | `MM-Q-G-s2` | `RM-D-B-s0` | `RM-D-B-s1` | 8 |
| `10K-W5` | `RM-D-B-s2` | `RM-D-G-s0` | `RM-D-G-s1` | `RM-D-G-s2` | `RM-Q-B-s0` | `RM-Q-B-s1` | `RM-Q-B-s2` | `RM-Q-G-s0` | 8 |
| `10K-W6` | `RM-Q-G-s1` | `RM-Q-G-s2` | `CX-D-B-s0` | `CX-D-B-s1` | `CX-D-B-s2` | `CX-D-G-s0` | `CX-D-G-s1` | `CX-D-G-s2` | 8 |
| `10K-W7` | `CX-Q-B-s0` | `CX-Q-B-s1` | `CX-Q-B-s2` | `CX-Q-G-s0` | `CX-Q-G-s1` | `CX-Q-G-s2` | idle | idle | 6 |

第一波 `10K-W1` 的明确目标是得到 TFNO dense 的 binary/graded 三 seed 完整结果，并启动 dual-lane dense 的 seed0/1；它不执行 test、不做 head 选择。`10K-W2` 先补齐 dual-lane dense seed2，再一次性完成 TFNO query 的两 lane 三 seed，并启动 dual-lane query seed0。`10K-W3` 的前两个槽位补齐 dual-lane query，至此 TFNO 与联合模型形成完整 head 对比；余下六个槽位开始并完成 Mixer dense。

每波结束必须生成 `wave_status.json` 和 artifact audit。只有本波所有非 idle 任务处于 `completed` 或具有已登记的 `failed_with_retry_lineage`，下一波才可提交。失败任务可以使用原 config/hash/seed 在本波末尾补提；不得把下一波的任务提前塞入空槽，也不得覆盖原 run directory。

### 7.5 10k 结束后的固定评估顺序

`10K-W7` 完成后按以下顺序处理，防止提前读取 test：

1. 聚合 54 个任务的 validation、资源和完整性结果；
2. 冻结四个 architecture×lane 和联合模型的输出头选择；
3. 写入 `head_selection.lock.json`，包含输入 run hashes 和选择理由；
4. 只对冻结 head 的 checkpoint 执行 10k held-out test；
5. 根据第 5.3 节形成 `50k_admission.lock.json`；
6. 未通过的模型保留为 `not_admitted`，不得进入 50k 训练。

### 7.6 50k 正式训练：最多四波 27 个任务

下表中的 `H_B`、`H_G` 分别表示该模型在 binary、graded validation 上冻结的胜出 head，取值只能是 `D` 或 `Q`；同一模型的两条 lane 可以选择不同 head。`H_J` 表示联合模型按两 lane 等权 validation 冻结的胜出 head。该矩阵是假定五个模型全部通过 50k admission 时的**最大矩阵**；生成机器清单时必须把 `H_*` 替换为 `head_selection.lock.json` 中的实际 `D/Q`，未通过的任务原位标为 `SKIP_NOT_ADMITTED`，任务编号和后续槽位不重新排序，避免事后改变比较结构。

| 50k wave | S0 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | Max jobs |
|---|---|---|---|---|---|---|---|---|---:|
| `50K-W1` | `TF-H_B-B-s0` | `TF-H_B-B-s1` | `TF-H_B-B-s2` | `TF-H_G-G-s0` | `TF-H_G-G-s1` | `TF-H_G-G-s2` | `DL-H_J-J-s0` | `DL-H_J-J-s1` | 8 |
| `50K-W2` | `DL-H_J-J-s2` | `MM-H_B-B-s0` | `MM-H_B-B-s1` | `MM-H_B-B-s2` | `MM-H_G-G-s0` | `MM-H_G-G-s1` | `MM-H_G-G-s2` | `RM-H_B-B-s0` | 8 |
| `50K-W3` | `RM-H_B-B-s1` | `RM-H_B-B-s2` | `RM-H_G-G-s0` | `RM-H_G-G-s1` | `RM-H_G-G-s2` | `CX-H_B-B-s0` | `CX-H_B-B-s1` | `CX-H_B-B-s2` | 8 |
| `50K-W4` | `CX-H_G-G-s0` | `CX-H_G-G-s1` | `CX-H_G-G-s2` | idle | idle | idle | idle | idle | 3 |

`50K-W1` 先确认最可能挑战当前 FNO 的 TFNO 和共享双路线模型；`50K-W2` 完成联合 seed2 并训练 Mixer；`50K-W3` 完成 ResMLP 和 ConvNeXt binary；`50K-W4` 完成 ConvNeXt graded。每个任务先执行到 epoch 100，再按第 6.2 节的预注册斜率门决定是否在**原作业 lineage**内继续到最多 200 epoch；“继续训练”不是新 seed，也不计作额外模型任务。

若部分模型未获准进入 50k，实际提交任务数为 `27 - skipped`。可以减少 array 长度，但机器清单必须保留全部 27 行及其 `RUN/SKIP_NOT_ADMITTED` 状态，不能用新模型、额外 seed 或学习率搜索填补空位。

### 7.7 波次级提交门、失败隔离与 GPU 利用率

每波提交前必须同时满足：上一波终态冻结、artifact audit 通过、队列中不存在同 campaign 的重复 task ID、数据和部署 commit 未变化。若部署 commit 或 dataset hash 改变，后续波次整体阻断，不能在同一 campaign 内混合代码或数据身份。

GPU 利用率按以下规则记录：

- 每 60 秒采样 GPU utilization、显存、功耗和温度，写入 `n40_profile.json`；
- 每个任务记录实际 GPU UUID；一张卡同时出现两个本 campaign 训练 PID 视为资源合同失败；
- `10K-W7` 和 `50K-W4` 的 idle 槽位保持空闲，不自动运行 threshold、cross-lane、测试或其他项目任务；
- 单任务失败不取消同波其他任务；重提沿用 task ID 并增加 `retry01`，保留失败目录；
- 任何 OOM 先标记失败并审计，禁止在单个 seed 上临时降低 batch size 后混入主比较。

### 7.8 授权与 dry-run 门

未来 submitter 必须默认 dry-run，并在提交前输出：

- campaign id、部署 commit 与授权 ID；
- admission 18 行、10k 54 行和 50k 最大 27 行冻结任务清单；
- 两套数据与 split/k-point hashes；
- 每行模型、head、lane、seed、资源和 run directory；
- admission probe 汇总；
- 预计 GPU-job 数和并发上限。

只有显式 submit mode 与独立 `PHC2D_AUTHORIZE_UPGRADE_CAMPAIGN=YES` 同时存在才允许 `sbatch`。授权还必须明确覆盖哪些波次；例如只授权 `A0-A2` 不允许继续提交 `10K-W1`，只授权 10k 不允许提交任何 `50K-W*`。本文及其人工接受本身均不是授权。

## 八、评估矩阵与统计

### 8.1 Native 主评估

主指标为 held-out test 的 bands 2–10 MRE。每个模型同时报告：

- 三 seed mean ± std；
- bands 1–10 per-band MRE；
- MAE、RMSE、full-matrix MRE 与 ordering violations；
- 参数量、checkpoint bytes、peak GPU memory；
- `train_time_s`、`eval_time_s`、`inference_time_per_sample_ms`、throughput；
- 10,000 次 paired sample bootstrap 的差值、95% CI 和方向。

在任何升级模型 50k test 被读取前，比较 checkpoint 冻结为：

- 升级模型使用 `50K-W1-W4` 各任务的单一 `best-validation` checkpoint；epoch 100 快照只用于继续训练门记录，不作为第二个独立结果；
- 四个直接旧基线统一使用既有 50k campaign 中 `source_group=current40k`、`epoch_budget=200`、`training_resolution=64` 的 `best-validation` checkpoint，每模型、lane、seed 各一项，共 24 项；
- “相对当前最佳 FNO”同样使用上述 `fno2d_band_v2 / current40k / epoch_budget=200` checkpoint，不在读取 test 后从 20/100/200 epoch 预算中择优；
- 第 2.2 节的 FNO 100 epoch 数字继续作为历史锚点汇报，但不参与本轮 promotion 的 paired comparator 选择。

冻结评估清单必须包含 27 个升级 checkpoint、24 个旧基线 checkpoint 和 checkpoint SHA-256。原生、cross-lane 与 32/128 输入分辨率合计 126 个可执行 evaluation cells；另保留 18 个 `not_applicable` cells（ResMLP 的 32/128 固定展平形状 12 项、联合模型无单源 cross-lane 方向 6 项），不得删除或按零误差汇总。

上述 126 cells 聚合并冻结 native graded 胜者后，才生成独立 threshold 清单：只对该胜者的 3 个 seed 执行 `0.3/0.4/0.5/0.6/0.7`，共 15 cells、最多 3 个并行单 GPU shards。threshold 结果不得回写 native promotion，也不得据此改选胜者。

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

- 本文解决的问题：把五个 implemented/no_result 模型的 admission、10k 七波输出头筛选、50k 最多四波确认和 N40 八卡槽位拆成可审计实验合同。
- 关键选择：正式精度保持 FP32 与旧损失；8 卡并行独立任务；validation 选择 head；test 只在冻结后使用。
- 当前没有解决的问题：没有 N40 admission、训练 job、checkpoint 或新结果；本文新增的波次表也不是执行制品。
- 最近的下一 gate：先人工审阅 `A0-A2`、`10K-W1-W7` 和 `50K-W1-W4` 的任务分配，再由用户决定是否单独授权 admission 或 54 个 10k jobs。
