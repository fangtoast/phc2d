---
title: "全胞双路线 50k 训练、迁移评估与混合算力计划"
summary: "研究计划或执行协议；实施前需与当前代码和证据重新核对。"
page_type: plan
status: draft
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 31 minutes
primary_layer: L0
plan_state: completed
effective_date: 2026-08-07
superseded_by: null
related_results:
  - docs/results/forward/full_cell_dual_lane_50000_forward_training_20260808.md
---

<!--
  Full-cell dual-lane 50k training, transfer evaluation, and hybrid-compute plan.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="full-cell-dual-lane-50000-training-plan"></a>

# 全胞双路线 50k 训练、迁移评估与混合算力计划

| Metadata | Value |
|---|---|
| Status | Draft；N40 training/evaluation completed on 2026-08-08，pending human review |
| Primary layer | L3 experiment/evidence；L4 仅包含执行本计划所需的接口门 |
| Authority | [生成网络调整研究计划](../fno_generative_network_adjustment_plan_20260617.md) |
| 10k reference | [全胞双路线训练与混合算力计划](full_cell_dual_lane_hybrid_compute_plan.md) |
| 10k evidence | [全胞双路线正向训练完整实验报告](../../results/forward/full_cell_dual_lane_forward_training_20260806.md) |
| 50k data contract | [Xingyi 全胞双路线 50k 数据生成合同](https://github.com/fangtoast/phc2d/blob/main/experiments/research/full_cell_dual_lane_50000_xingyi.md) |
| 50k evidence | [50k 正向训练与冻结评估完整报告](../../results/forward/full_cell_dual_lane_50000_forward_training_20260808.md) |
| Audience | 项目负责人、研究人员、N40/M9 执行人员与 AI 协作者 |
| Owner | 研究生维护，导师审阅科学主张、资源预算与 promotion gate |
| Created | 2026-08-07 |
| Last reviewed | Pending human review |
| Reading time | 24 minutes |

本文定义 binary/graded 两套 50k full-cell 数据上的神经网络训练、旧 10k checkpoint 迁移评估、epoch 预算比较、跨分辨率零样本测试和 N40/M9 混合算力边界。2026-08-07，项目负责人已授权使用 N40 启动正式训练，并明确免除本轮“本地 WSL 物理 replay”训练前置门；该豁免只改变训练准入，不把数据状态提升为跨环境物理验收完成。

2026-08-07 的负责人修订将训练矩阵冻结为 **17 波、204 个独立作业**：binary native 与 graded continuous 分别执行 40k/50k × 20/100/200；graded t0.3–t0.7 只执行 50k/100 epoch。40k 使用互斥的 5k validation 选择 best checkpoint；50k 使用现有全部样本做 final-fit，保存最后 epoch checkpoint，原 `idx_val`/`idx_test` 只允许作为明确标记为训练重叠的 monitor，不得用于模型选择或泛化主张。8k/20k 训练和 threshold 的 20/200 epoch 已从本轮矩阵删除。

2026-08-08 执行记录：训练 campaign `17/17 waves`、`204/204 runs`、`0 failed`；冻结评估 campaign 完成 `906/906` 个实际推理 cell、`156` 个架构不适用项、`300` 组 paired bootstrap 和 `0 failed`。旧 10k Stage 4 因执行顺序偏差在新模型训练之后补齐，因此报告明确记录该偏差；评估入口未使用新模型结果筛选旧 checkpoint，仍覆盖预注册的全部 102 个旧 checkpoint。

**目录：** [一、目标与边界](#fc50k-goal) · [二、当前证据与准入](#fc50k-evidence) · [三、数据与训练边界](#fc50k-data) · [四、实验矩阵](#fc50k-matrix) · [五、epoch 协议](#fc50k-epoch) · [六、跨分辨率测试](#fc50k-resolution) · [七、N40/M9 资源编排](#fc50k-compute) · [八、执行阶段](#fc50k-stages) · [九、统计与报告](#fc50k-report) · [十、制品与恢复](#fc50k-artifacts) · [十一、验收](#fc50k-acceptance) · [十二、人工理解记录](#fc50k-human-record)

---

<a id="fc50k-goal"></a>

## 一、目标与边界

### 1.1 主要研究问题

本计划回答七个问题：

1. 10k 数据训练得到的冻结模型，在新的 50k test split 上是否保持精度？
2. 40k 独立验证训练与 50k final-fit 的训练轨迹、成本和 checkpoint 角色有何差异？
3. epoch `20/100/200` 对 binary native 与 graded continuous 的影响是什么？
4. threshold t0.3–t0.7 在固定 50k/100-epoch final-fit 下呈现什么训练重叠 monitor 行为？
5. binary-trained 模型在 graded continuous test 上、graded-trained 模型在 binary test 上是否具有跨路线泛化能力？
6. 64 分辨率训练的空间模型能否不微调地直接读取同一结构的 32/128 栅格？
7. 怎样使用 N40 GPU 池和 M9 CPU 超算缩短墙钟时间，同时保留公平性、provenance 和物理证据边界？

### 1.2 范围

本轮包含：

- binary native 与 graded continuous/t0.3–t0.7；
- `mlp`、`res_mlp_wide`、`lifted_cnn`、`fno2d_band_v2`；
- seeds `0/1/2`；
- 40k 独立 validation 训练与 50k 全数据 final-fit；
- binary native/graded continuous 的 epoch 20/100/200 独立预算；
- graded t0.3–t0.7 的 50k/100-epoch 固定预算；
- 旧 10k checkpoint 在新 50k test split 上的无微调推理；
- binary-trained → graded continuous/threshold 与 graded-trained → binary 的双向跨路线测试；
- 64-trained spatial checkpoint 的 32/64/128 零样本分辨率测试；
- validation-only 选择、共同 test 对照、训练曲线和资源效率报告；
- 阈值代理值得继续时，由 M9 执行的独立 real-MPB 验证。

### 1.3 非目标与禁止提升

- 不改变结构生成分布、MPB 参数、`Wh` 定义、k 点、band 数或验收阈值。
- 不把 M9 CPU 用于正式神经网络训练；N40 负责 CUDA 训练/推理，M9 负责 MPB/Meep。
- 不使用 test split 调学习率、挑模型、挑阈值、挑 epoch 或决定是否补跑。
- 不把 50k final-fit 的训练重叠 monitor 写成 validation/test，也不从中产生 validation winner。
- 不把 threshold proxy 误写成 threshold structure 的真实物理响应。
- 不把 64→32/128 输入测试写成“跨分辨率重新训练”。
- 不把 50k 数据生成、GPU 训练完成或 M9 shard 完成提升为模型物理验证完成。
- 不自动取消、重启或恢复计费作业；本轮 N40 正式训练已获项目负责人明确授权，后续改变科学矩阵或 M9 物理作业仍需单独授权。

[↑ 回顶部](#full-cell-dual-lane-50000-training-plan)

---

<a id="fc50k-evidence"></a>

## 二、当前证据与准入

| 对象 | 当前证据状态 | 本计划中的使用边界 |
|---|---|---|
| 10k Forward | 102/102 正式训练已完成；84-run 主矩阵、18-run 分辨率重训和冻结 test 已有报告 | checkpoint 作为冻结旧基线，不重新选择或微调 |
| 50k 数据配置 | binary/graded 均为 50,000 samples 和冻结的 `40000/5000/5000` split；训练时另建 50k final-fit 角色 | 40k checkpoint 用互斥 val/test；50k final-fit 不得形成无偏 test |
| 50k HDF5 | binary/graded 已导入 N40 与 M9；N40 目标文件与 Xingyi 源文件的大小和 SHA-256 完全一致 | 本轮训练按负责人豁免无需等待本地 WSL replay；仍不得称为跨环境物理验收完成 |
| N40 | 17 波训练以单任务 1×RTX4090、6 CPU、60,000 MB 完成 204/204 runs；评估以 8 张 RTX4090 并行完成 906 cells | 实测并发受账号最多 8 张 GPU 同时运行约束；其余任务由 Slurm 排队，不误报为资源故障 |
| M9 | `amd_m9_768` CPU 超算可承担授权后的 real-MPB | 不承担神经网络训练，不因 N40 开始训练而自动提交 |
| 50k 执行代码 | runner、17-wave orchestrator、sharded evaluation 与 finalizer 均已实现、测试并部署 | 训练 commit `7498511`；评估/制品合同最终 commit `31d7ff0` |

50k 训练的硬数据门为：

```text
remote final artifacts complete
-> Xingyi/N40 SHA-256 equal
-> N40 file size and split contract equal
-> owner waives local WSL replay as training prerequisite
-> 50k training admission allowed
```

本轮已满足上述训练准入。数据 provenance 仍保留 `cluster_generated_pending_wsl_acceptance` 的历史事实，但该状态不再阻塞神经网络训练；报告必须同时记录“训练已授权”和“本地 WSL 物理 replay 未执行”，不得把两者混为一谈。

[↑ 回顶部](#full-cell-dual-lane-50000-training-plan)

---

<a id="fc50k-data"></a>

## 三、数据、共同 test 与训练边界

### 3.1 50k 数据合同

| 路线 | 计划文件 | 预期本地训练文件 | 固定 split |
|---|---|---|---|
| binary | [`full_cell_binary_bz9_50000_xingyi_v1.yaml`](https://github.com/fangtoast/phc2d/blob/main/configs/research/full_cell_binary_bz9_50000_xingyi_v1.yaml) | `data/full_cell_binary_bz9_50000_xingyi_v1.h5` | 40,000 / 5,000 / 5,000 |
| graded | [`full_cell_graded_bz9_50000_xingyi_v1.yaml`](https://github.com/fangtoast/phc2d/blob/main/configs/research/full_cell_graded_bz9_50000_xingyi_v1.yaml) | `data/full_cell_graded_bz9_50000_xingyi_v1.h5` | 40,000 / 5,000 / 5,000 |

训练前必须冻结：

- dataset SHA-256；
- `idx_train`、`idx_val`、`idx_test` 的内容、长度与 SHA-256；
- sample ID、plan hash、solver fingerprint 和 HDF5 schema；
- `Wh`、k points、`material_weight_grid_32/64/128` 的 shape、dtype 和 finite audit；
- binary/graded 与 40k/50k 各自独立且只由对应训练集合计算 target normalization；normalization sidecar 必须绑定 subset hash。

两条路线不合并训练集。不同路线的 normalization 不共享。

### 3.2 旧 10k checkpoint 的共同 50k test

旧 checkpoint 只允许在以下条件全部满足时加载：

1. checkpoint SHA-256、原训练 dataset hash、split hash、model identity 和输入表征完整；
2. 10k 与 50k 的 `Wh` shape、k points、频率单位、band order 和 solver contract 一致；
3. 使用 checkpoint 内保存的 **10k train normalization**；不得用 50k train/test 标签重新归一化旧模型；
4. 不进行 fine-tune、test-time adaptation、阈值重选或 calibration；
5. 推理目标固定为对应路线的新 50k `idx_test` 5,000 samples。

10k→50k test 回答的是旧模型对新样本分布的迁移能力。它与“同一 50k train pool 内的样本量消融”分表报告，不能混成一个数据规模结论。

### 3.3 40k validation 与 50k final-fit 双合同

40k 合同保持原始互斥划分：

```text
train_40k == HDF5 idx_train
val_5k == HDF5 idx_val
test_5k == HDF5 idx_test
```

50k final-fit 合同则显式使用 HDF5 全部 50,000 行：

```text
train_50k == all sample indices
selection_split == none_final_fit
checkpoint_role == final_epoch_final_fit
validation_metrics_admissible == false
```

原 5k validation/test 都包含在 50k train 内，故只允许输出 `training-overlap monitor` 诊断，不允许输出 validation leaderboard、validation winner 或无泄漏 test 指标。40k 与 50k checkpoint 必须使用不同 run identity、subset hash、normalization 和 checkpoint role。

[↑ 回顶部](#full-cell-dual-lane-50000-training-plan)

---

<a id="fc50k-matrix"></a>

## 四、训练与推理矩阵

### 4.1 表征与模型

| 路线 | 表征 | 数量 |
|---|---|---:|
| binary | native binary | 1 |
| graded | continuous、t0.3、t0.4、t0.5、t0.6、t0.7 | 6 |
| 合计 |  | 7 |

每个表征均训练：

- `mlp`
- `res_mlp_wide`
- `lifted_cnn`
- `fno2d_band_v2`

固定 seeds 为 `0/1/2`，主训练输入分辨率固定为 64。

### 4.2 旧 10k checkpoint → 新 50k test

覆盖 10k 的全部 102 个正式 checkpoint：

```text
84 main-matrix checkpoints at native resolution 64
+ 18 resolution-retraining checkpoints at native resolution 32 or 128
= 102 legacy checkpoint evaluations
```

18 个额外 checkpoint 只按其实际训练分辨率读取对应 32/128 grid，不冒充零样本分辨率迁移。每个 checkpoint 独立输出一行指标及 checkpoint/dataset lineage；同一 checkpoint 的推理可以在一个 GPU 作业中批量完成，禁止为每个 test sample 建作业。

### 4.3 冻结的 17 波训练主矩阵

每个训练规模和 epoch 预算都是独立作业：

```text
12 primary waves + 5 threshold waves = 17 waves
17 waves × 4 models × 3 seeds = 204 independent training runs
```

波次定义：

```text
binary_native:       40k/50k × 20/100/200 = 6 waves
graded_continuous:   40k/50k × 20/100/200 = 6 waves
graded_t03-t07:      50k × 100 only        = 5 waves
```

每个 epoch 预算均从冻结初始化协议独立开始。40k 作业保存 best-validation checkpoint；50k 作业关闭 validation 选择与 early stopping，保存最后 epoch checkpoint。前 3 波 binary 40k 的 20/100/200 结果允许按相同 dataset/split/config 合同直接纳入新 campaign。

### 4.4 必做交叉表征和双向跨路线评估

交叉评估是后续独立证据门。每次推理必须使用**源 checkpoint 的 target normalization**，不得使用目标路线 train/validation/test 标签重新估计 normalization。只有 40k validation-selected checkpoint 可进入无泄漏 test 和模型比较；50k final-fit checkpoint 只能按预先冻结角色用于部署或外部独立数据，不能在原 50k HDF5 上产生泛化结论。

交叉方向固定为：

- binary-trained → graded continuous test；
- binary-trained → graded t0.3–t0.7 test；
- graded continuous-trained → binary native test；
- graded t0.3–t0.7-trained → binary native test；
- graded 内部 continuous/t0.3–t0.7 的 `6×6`。

binary-trained → graded continuous 是直接把连续 `[0,1]` material-weight grid 输入原 binary-trained checkpoint，不做 threshold、resize、fine-tune 或目标域 calibration。它回答的是强 domain shift 下的泛化能力，不能解释为“binary 与 graded 物理等价”。

#### 4.4.1 旧 10k 主矩阵在新 50k test 上

仅使用 84 个 64-resolution 主矩阵 checkpoint；18 个分辨率重训 checkpoint 保留原生路线测试，不加入此矩阵：

```text
graded internal: 4 models × 3 seeds × 6 sources × 6 targets = 432 cells
binary -> graded: 4 models × 3 seeds × 6 targets = 72 cells
graded -> binary: 4 models × 6 sources × 3 seeds = 72 cells
total = 576 cross-evaluation cells
```

#### 4.4.2 40k validation checkpoint 的 binary ↔ continuous 双向交叉验证

这一组是 50k 新模型训练完成后的独立必做结果，不使用旧 10k checkpoint：

| 方向 | 源 checkpoint | 目标 test | 单个 epoch 预算 |
|---|---|---|---:|
| binary → continuous | binary 40k-train、64-resolution、4 models × 3 seeds | graded continuous `idx_test` 5,000 samples | 12 cells |
| continuous → binary | graded-continuous 40k-train、64-resolution、4 models × 3 seeds | binary native `idx_test` 5,000 samples | 12 cells |
| 双向合计 |  |  | **24 cells** |

对 epoch 20/100/200 三组独立训练 checkpoint 全部执行：

```text
24 cells × 3 epoch budgets = 72 required 40k validation-selected cross-lane cells
```

这 72 cells 回答 binary 模型能否泛化到 continuous test，以及反方向是否成立。50k final-fit checkpoint 不进入这张表。

两个方向的目标 test sample ID 不同，因此不能把正反方向样本强行配对。paired bootstrap 只在同一方向、同一目标 test IDs 上比较模型、seed 聚合或 epoch；方向间比较报告独立 CI 和 effect，不伪造逐样本配对。

#### 4.4.3 40k 三个独立 epoch 预算

对 train size 40k 的 epoch 20/100/200 独立作业执行相同的预注册交叉评估。这三个预算必须完整报告，不得只保留 test 上最好的预算。

#### 4.4.4 50k final-fit 边界

50k final-fit checkpoint 只保留训练曲线、训练重叠 monitor、final-epoch checkpoint 和资源证据。它们不得在原 HDF5 上产生 validation winner、test 泛化或 40k-vs-50k 精度提升结论。未来若取得独立外部 test，可另建评估合同；不得把本计划中的 monitor 直接升级为该证据。

[↑ 回顶部](#full-cell-dual-lane-50000-training-plan)

---

<a id="fc50k-epoch"></a>

## 五、20/100/200 epoch 协议

### 5.1 独立预算、独立作业

20、100、200 epoch 分别从头启动独立作业。相同 `(train_size, representation, model, seed)` 的三条作业使用相同：

- 初始化 seed；
- DataLoader seed 和样本顺序；
- optimizer 与学习率策略；
- micro/effective batch size；
- precision 和 GPU 型号；
- train subset、validation split 和 normalization。

每条作业保存：

```text
model_best_validation.pt
model_last.pt
history.csv
figures/training_curve.{png,pdf}
```

run identity 必须包含 train size 和 max epochs；不同预算不得写入同一目录或覆盖 checkpoint。

### 5.2 为保证可比性的固定规则

- 正式 epoch 对比关闭 early stopping；否则 200-epoch 组可能在 100 前终止，无法回答用户问题。
- 学习率策略固定且不得根据 test 结果为某个预算单独调参。
- 若采用 scheduler，必须显式记录其是否依赖 `max_epochs`；主矩阵默认不使用依赖 horizon 的 scheduler。
- 默认 effective batch size 保持 64；显存不足时只降低 micro-batch 并用梯度累积保持 64。
- FP32、AMP 和 TF32 口径保持与 10k 主报告一致；若后续增加 AMP，只能作为单独性能实验，不混入主比较。
- 每个 epoch 记录完整 train/validation 曲线；不得只保存三个终点。
- 每个独立作业同时记录 optimizer steps、samples seen、数据遍历次数、wall time 和 GPU-hours。

### 5.3 两类公平比较

本计划同时报告但不混淆：

1. **相同 epoch：** 比较不同 train size 在 20/100/200 次数据遍历后的效果；大 train size 使用更多 optimizer steps。
2. **近似 compute-matched：** 依据实际 optimizer steps/samples seen，从现有曲线选取预注册的最邻近观察点进行辅助比较；不能把它写成完全相同计算量，除非 step 数严格相同。

10k 的 8,000-train、100 epoch 约对应 `12,500` 个 batch-64 optimizer steps；50k 的 40,000-train 在约 20 epoch 达到相近 step 数。因此 40k/20-epoch 是旧 10k/100-epoch 的重要 compute-matched 近似对照，但最终报告必须使用实际 batch/step 计数而不是只用公式估计。

[↑ 回顶部](#full-cell-dual-lane-50000-training-plan)

---

<a id="fc50k-resolution"></a>

## 六、64-trained checkpoint 的 32/128 零样本分辨率测试

### 6.1 测试定义

同一个 64-trained checkpoint 不做 fine-tune，分别读取同一 test sample 的：

```text
material_weight_grid_32
material_weight_grid_64
material_weight_grid_128
```

目标 `Wh`、sample ID、split 和 checkpoint normalization 均保持不变。结果命名为：

```text
zero_shot_input_resolution_transfer
```

不得命名为 resolution retraining、physical convergence 或跨分辨率物理验证。

### 6.2 模型兼容性

| 模型 | 64→32/128 直接输入 | 原因与处理 |
|---|---|---|
| `mlp` | 不适用 | 第一层输入维度固定为 `64×64`；不得通过重建第一层、补零或插值伪装成同一 checkpoint |
| `res_mlp_wide` | 不适用 | 同样固定展平输入维度；记录 `not_applicable_architecture_shape_mismatch` |
| `lifted_cnn` | 允许进入兼容性测试 | 卷积加 adaptive pooling 可接受可变空间尺寸；仍需 checkpoint exact-load 和 shape smoke |
| `fno2d_band_v2` | 允许进入兼容性测试 | spectral modes 运行时按输入尺寸截取并使用 adaptive pooling；需分别报告有效 mode 数和分辨率变化 |

禁止为了让 MLP 工作而把 32/128 图像 resize 到 64；这会变成另一种 preprocessing 实验。

### 6.3 必做范围

1. 全部 10k 主矩阵中的 CNN/FNO 64-trained checkpoint：

   ```text
   2 models × 7 representations × 3 seeds = 42 checkpoint bundles
   ```

2. 50k `train_size=40000` 下 binary native 与 graded continuous、epoch 20/100/200 的 CNN/FNO checkpoint：

   ```text
   2 models × 2 representations × 3 seeds × 3 independent epoch budgets
   = 36 checkpoint bundles
   ```

每个 bundle 在一个推理作业内依次测试 32/64/128，避免创建样本级或分辨率级碎片作业。报告以 64 为基准给出 32/128 的 MRE、MAE、RMSE 变化及 paired bootstrap CI。

[↑ 回顶部](#full-cell-dual-lane-50000-training-plan)

---

<a id="fc50k-compute"></a>

## 七、N40 GPU 与 M9 CPU 资源编排

### 7.1 平台职责

| 平台 | 负责 | 不负责 |
|---|---|---|
| N40 `gpu_4090` | 204 条独立训练作业、旧 checkpoint 推理、零样本分辨率测试、指标汇总前的 GPU 推理 | real-MPB、跨站点 DDP、多个任务写同一 run |
| M9 `amd_m9_768` | 经授权的 threshold analytic MPB preflight/pilot、物理 replay 和 solver 证据 | 正式神经网络训练、替代 N40 checkpoint test |
| 本地 WSL `Ubuntu + meep_env` | 可选的开发测试和 checkpoint 小规模复核 | 本轮训练前置物理验收、正式 50k 计划生成、批量 GPU 训练、批量 MPB 数据生成 |

### 7.2 N40 单任务资源

既有平台证据对应每张 RTX4090：6 CPU、60,000 MB 内存。50k admission 的初始资源合同为：

```text
1 GPU: RTX4090
6 CPU cores
60,000 MB host memory
1 isolated Python process
4 HDF5 DataLoader workers
effective batch size: 64
```

这些是 admission 起点，不是永久常数。正式矩阵前对每个模型至少测量：

- epoch time 与 samples/s；
- GPU utilization、峰值显存和 host memory；
- HDF5 data wait；
- `num_workers=2/4/8` 的吞吐；
- 40k/200-epoch 预计 wall time；
- checkpoint 保存和恢复时间。

根据同一 50k HDF5 的实测选择 workers 和 time limit，不得用 10k 读取吞吐直接外推。

### 7.3 17 个训练波次

每个 `(train_size, max_epochs, representation)` 形成一个完整波次：

```text
4 models × 3 seeds = 12 independent one-GPU tasks
12 primary waves + 5 fixed threshold waves = 17 waves
```

默认并发上限为 12 张 GPU。历史 18-task 提交曾触发账号提交上限，因此在新的 QoS/admission 证据出现前不得把并发上限写成 18 或 144。

推荐采用 rolling array/queue：

```text
204 tasks, account maximum 8 concurrently active
```

调度器可在一个任务 terminal 后立即补入下一任务，以减少整波等待；但波次 aggregation 必须等该波次 12 个任务均 terminal。并发度只影响墙钟时间，不得改变 seed、数据顺序、batch 或训练协议。

### 7.4 长作业处理

若独立 200-epoch 作业预计超过分区 wall-time，优先为该作业增加合理 wall-time。分区硬限制不允许时，才采用同一 run identity 的依赖式恢复段：

```text
segment A: epoch 1-N
segment B: epoch N+1-200, afterok(A)
```

分段恢复必须保存并恢复 model、optimizer、scheduler、gradient scaler（如有）、epoch、global step、Python/NumPy/Torch/CUDA RNG 和 DataLoader sampler state。只有相同 commit、config hash、dataset hash、subset hash 和 run ID 才允许显式恢复。20/100/200 三个预算之间不得相互续训；它们始终是独立实验。

### 7.5 HDF5 与共享存储

- 12 个 GPU 作业只读共享 HDF5，各自独占 run 目录。
- 不为每个作业复制一份 50k HDF5。
- 若共享盘 probe 显示 GPU 长时间等待，可使用“每节点每 dataset hash 一份”的只读节点缓存；通过原子锁和 SHA-256 防止并发复制或脏缓存。
- normalization 应分块计算一次、保存带 train subset hash 的 sidecar；12 个作业复用，不让每个作业重复扫描全部 `Wh`。
- 汇总、曲线和 leaderboard 由依赖式单写者 aggregation 作业生成。

### 7.6 M9 资源

M9 只在 threshold proxy 的 validation 结果通过人工科学 gate 后启用。物理任务使用 [通用 HPC campaign 合同](../../hpc_physics_campaigns.md)：

```text
plan -> benchmark/admission -> solve array -> finalize
-> download/hash audit -> WSL acceptance
```

节点数由 M9 profile、单 case 实测耗时和强缩放证据从 `{1,2,4,8}` 中选择。不得因为 M9 核数多就直接占用 8 节点；也不得把 solver shards complete 写成 final dataset complete。

[↑ 回顶部](#full-cell-dual-lane-50000-training-plan)

---

<a id="fc50k-stages"></a>

## 八、实现与执行阶段

### Stage 0：L4 实现门

正式运行前必须实现并测试：

1. full-cell runner 不再强制 exactly 100 epochs；支持独立的 `max_epochs=20/100/200`；
2. 支持关闭 early stopping，每个独立作业保存 best/last checkpoint 和完整训练曲线；
3. run identity 包含 train size 和 epoch budget，禁止不同实验覆盖；
4. 支持 40k 互斥 validation 与 50k final-fit 双合同，并核对 source split/subset hash；
5. 旧 checkpoint 可在新 dataset test split 推理，且强制使用 source normalization；
6. CNN/FNO 可变分辨率直接推理；MLP/ResMLP 必须明确拒绝 shape mismatch；
7. 支持训练曲线、独立预算 metrics、resource metrics 和固定大小 campaign status；
8. N40 submit/status/aggregate 默认 dry-run，正式提交要求显式人工授权。

自动测试至少覆盖：

- 40k train/validation/test 互斥、50k final-fit 重叠边界和 hash 确定性；
- 20/100/200 独立作业 identity、终态与曲线完整性；
- 分段恢复与连续训练在容差内一致；
- source normalization 不被 50k test 替换；
- CNN/FNO 64→32/128 shape smoke；
- MLP/ResMLP 直接分辨率迁移失败关闭；
- status 不扫描 sample 级文件；
- dry-run 不调用 `sbatch`。

### Stage 1：50k 数据门和 checkpoint inventory

- 核对 Xingyi 与 N40 两套 50k HDF5 的文件大小和 SHA-256；本轮按负责人豁免不要求本地 WSL replay；
- 在 N40 复核文件 SHA-256；
- 枚举 84 个主矩阵和 18 个分辨率重训 checkpoint，生成 102 行 `legacy_checkpoint_manifest.csv`；
- 缺 checkpoint、重复 identity、hash 不符或 normalization 缺失即 blocked。

### Stage 2：40k/50k 合同与单卡 benchmark

- 冻结 binary/graded 的 40k split hash 与 50k all-indices subset hash；
- 每路线选择代表任务执行短 I/O benchmark；
- 四个模型分别测 40k 的 epoch time、显存和 samples/s；
- 计算 20/100/200 预计 wall time、GPU-hours 和是否需要分段。

### Stage 3：12-job admission

选择一个 40k/64 表征波次，提交 4 models × 3 seeds 的短任务：

- 实际获得 12 张 RTX4090；
- 每任务 6 CPU/60,000 MB 或平台实配资源；
- 共享 HDF5 无错误；
- GPU utilization 和 I/O wait 可接受；
- 所有任务原子写入最小制品并 terminal。

首波运行异常时降并发为 6 或 3，但不改变科学矩阵。项目负责人已于 2026-08-07 授权正式训练提交。

### Stage 4：旧 10k checkpoint → 50k test

先执行 102 个原生路线无微调推理，并用其中 84 个 64-resolution 主矩阵 checkpoint 完成 576-cell 跨表征/跨路线评估。binary-trained → graded continuous 必须单独列出。该阶段在任何新 50k 模型训练前冻结结果，以避免根据新训练结果选择性报告旧 checkpoint。

### Stage 5：204 条独立正式训练作业

按资源估计和已获授权执行 17 个波次。优先顺序：

1. 40k binary 与 graded continuous 的 20/100/200；
2. 50k binary 与 graded continuous 的 20/100/200 final-fit；
3. 50k t0.3–t0.7 的固定 100-epoch final-fit；
4. 失败审计后的显式人工补跑。

一个任务失败不允许覆盖成功任务，也不允许自动改变 batch、precision、epoch 或学习率。修复若改变代码/配置，必须创建新 campaign version，并明确旧 run 是否排除。

### Stage 6：共同 test、epoch 与数据规模比较

- 对所有预注册独立预算 checkpoint 执行 native-64 共同 test；
- test 结果全部报告，不用其反向改变训练；
- 完成 40k × 20/100/200 的 paired comparison；
- 训练完成后对 40k validation-selected checkpoint 完成 binary → continuous 与 continuous → binary 双向交叉验证；
- 50k final-fit checkpoint 不在原 HDF5 上执行无泄漏 test 或 validation 排名；
- 冻结用于后续交叉评估和 inverse 的 checkpoint 角色。

### Stage 7：零样本分辨率与必做交叉评估

- 执行第六节的 10k 和 50k CNN/FNO 32/64/128 bundle；
- 记录 MLP/ResMLP 为不适用而不是失败样本；
- 对 40k train 的 20/100/200 独立 checkpoint执行预注册交叉矩阵；
- 50k final-fit checkpoint 只在未来外部独立数据合同下评估；
- 核对 source/target dataset、representation、checkpoint role 和 normalization lineage。

### Stage 8：有条件的 M9 物理验证

只有 threshold proxy 在预注册 validation gate 上表现出继续研究价值，且解析 threshold codec 已实现，才允许另行申请 M9 MPB preflight/pilot。M9 结果通过下载、hash 和 WSL acceptance 前，不进入 threshold physical conclusion。

[↑ 回顶部](#full-cell-dual-lane-50000-training-plan)

---

<a id="fc50k-report"></a>

## 九、统计、训练曲线与报告

### 9.1 主要指标

- bands 2–10 MRE（主指标）；
- MAE、RMSE、完整 band 1–10 诊断；
- per-band、per-k 和 sample-level error；
- band ordering violation 与 NaN/Inf；
- train/validation loss 与 validation bands 2–10 MRE 曲线；
- best epoch、configured max epoch 和 terminal epoch；
- train/eval/inference time、samples/s、optimizer steps、samples seen；
- GPU-hours、峰值显存、host memory、CPU 配额和 DataLoader workers；
- failed/interrupted/resumed runs 和错误分类。

### 9.2 预注册主比较

1. 旧 10k checkpoint 与 50k/40k-trained checkpoint 在同一 50k test 上的差异；
2. 同一 `(lane, representation, model, seed)` 下 40k 的 epoch 20 vs 100 vs 200；
3. 50k final-fit 的 train curve、重叠 monitor 和 GPU-hours，但不做泛化排名；
4. CNN/FNO 同一 checkpoint 下 resolution 32 vs 64 vs 128；
5. binary-trained → graded continuous 与 graded continuous-trained → binary 的方向性差异。

使用相同 sample ID 的比较必须使用 paired bootstrap 95% CI。三 seed 报告每 seed 独立值与 mean ± std。多模型、多表征和多 epoch 的 secondary comparisons 使用 Holm 校正，同时保留未校正 effect size 和 CI。

### 9.3 选择边界

- validation 可用于冻结模型和后续交叉评估 checkpoint；
- test 上三个预注册独立 epoch 预算可以全部报告，但不得根据 test 继续调参或只发布最好一个；
- 10k→50k 是 domain/generalization comparison；
- binary-trained → graded continuous 和反方向均属于跨路线 OOD，不证明两种介质或标签物理等价；
- 40k validation 与 50k final-fit 是不同 checkpoint 角色，不能作无泄漏精度比较；
- 同 epoch 与 compute-matched 近似分别报告；
- 如果 50k 没有显著优于 10k，必须报告“未观察到稳定收益”，不能以样本数更大替代证据。

### 9.4 必须生成的图

- 40k 轨迹的 train/validation loss 和 validation MRE 曲线；50k 轨迹明确标为 training-overlap monitor；
- 每模型三 seed 均值曲线与标准差带；
- 40k validation 与 50k final-fit 的训练成本/轨迹对照，不绘制误导性的泛化误差差值；
- epoch 20/100/200—误差与 GPU-hours 曲线；
- 10k checkpoint 与 50k checkpoint 的共同 test paired difference 图；
- binary-trained → graded continuous、graded continuous-trained → binary 的双向对照图；
- graded continuous/t0.3–t0.7 的 `6×6` cross-representation heatmap；
- CNN/FNO 的 32/64/128 零样本分辨率迁移图；
- 模型×表征×训练规模×epoch 的紧凑 heatmap；
- 失败、长尾和最差样本案例图。

曲线必须从 `history.csv` 或等价机器源生成，禁止只保存图片。

[↑ 回顶部](#full-cell-dual-lane-50000-training-plan)

---

<a id="fc50k-artifacts"></a>

## 十、制品、状态与恢复

### 10.1 每条训练轨迹

run 目录遵循 `runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/`，至少包含：

| 制品 | 必须内容 |
|---|---|
| `run.log` | admission、epoch、恢复和 terminal 摘要 |
| `run.meta.json` | commit、配置、dataset/split/subset hash、Slurm/GPU/CPU/软件环境 |
| `config.resolved.yaml` | train size、representation、model、seed 和独立 max-epoch 合同 |
| `history.csv` | 完整配置预算的 train 与 monitor、learning rate、steps、samples seen、时间 |
| `model.pt` | 40k 的 best-validation 或 50k 的 final-epoch checkpoint，角色写入 metadata |
| `metrics.json` | configured-budget/best 指标、runtime、throughput、GPU/CPU/memory |
| `results.json` | success/failed/interrupted、completed epoch、下一 gate |
| `figures/` | 训练曲线 PNG/PDF |

### 10.2 Campaign 顶层

每个阶段维护固定大小的状态和索引：

```text
campaign_status.json
submission_manifest.csv
legacy_checkpoint_manifest.csv
trajectory_manifest.csv
epoch_budget_manifest.csv
cross_evaluation_manifest.csv
cross_evaluation_cells.csv
failed_tasks.csv
seed_summary.csv
resource_summary.csv
```

状态查询读取这些摘要，不递归扫描所有 checkpoint 或样本。204 条训练作业达到 terminal 不等于 test、aggregation 或物理验证完成。

### 10.3 状态词

```text
data_pending_acceptance
data_ready
legacy_inference_running
legacy_inference_complete
training_admission_pass
training_running
training_trajectories_complete
evaluating_epoch_budgets
resolution_transfer_complete
neural_evidence_complete
physical_validation_pending
cross_environment_accepted
```

`neural_evidence_complete` 不得提升为 `physically_verified`。

[↑ 回顶部](#full-cell-dual-lane-50000-training-plan)

---

<a id="fc50k-acceptance"></a>

## 十一、完成判据与人工验收

### 11.1 自动完成判据

- 两套 50k 数据在 Xingyi 与 N40 的大小和 SHA-256 一致；本轮本地 WSL replay 已由负责人明确豁免且证据边界已记录；
- 40k split 证明 train/val/test 互斥；50k subset hash 证明覆盖全部样本并标记 val/test overlap；
- 102 个旧 10k checkpoint 在 50k test 的原生训练分辨率推理结果完整；
- 40k validation-selected checkpoint 的预注册交叉评估完整；50k final-fit 不混入无泄漏 test 表；
- 204 条独立训练作业均有明确 terminal 状态；
- 每条成功作业的 best/last checkpoint、history、曲线、metrics 和 provenance 齐全；
- 40k checkpoint 有 validation 记录；50k checkpoint 有明确的 training-overlap monitor 与 final-epoch role；
- 所有规定的 CNN/FNO 32/64/128 bundle 完成；
- MLP/ResMLP 跨分辨率被正确标记为不适用，没有静默 resize；
- 三 seed 汇总、paired CI、资源表、失败表和最终报告完整；
- 相关实现测试通过；本轮不把本地 WSL 物理 replay 设为训练完成条件；
- 文档与索引通过 `phc2d report docs-audit --root README.md --strict`。

### 11.2 人工必须确认

1. 204 条独立训练作业及其预计 GPU-hours 是否接受（已于 2026-08-07 授权启动）；
2. 10k checkpoint inventory 和 source-normalization 边界是否正确；
3. 40k validation 与 50k final-fit 的 checkpoint 角色和证据边界是否被正确理解；
4. test 上公开三个预注册独立 epoch 预算、但不据此继续调参的统计口径；
5. 64→32/128 只适用于 CNN/FNO 的架构边界；
6. 是否在 neural evidence 完成后授权 M9 threshold real-MPB 验证。

计划文件、dry-run、smoke 或 admission pass 均不构成正式训练授权。

[↑ 回顶部](#full-cell-dual-lane-50000-training-plan)

---

<a id="fc50k-human-record"></a>

## 十二、人工理解记录

```text
解决了什么问题：
把原 10k 双路线训练扩展为 40k validation 与 50k final-fit 双合同，加入旧 checkpoint 共同 test、20/100/200 独立 epoch 预算、双向跨路线/跨表征评估和 64→32/128 零样本分辨率测试。

为什么采用这个方案：
runner 不具备完整 milestone/resume 合同时，使用 204 条彼此独立的训练作业最清晰可靠；它们组织为 17 个 12-job 波次，在账号允许的 8 张 GPU 上滚动执行，并为每个训练规模和 epoch 预算保留独立曲线与 checkpoint。

改变了项目哪一层/哪一模块：
主要改变 L3 实验/证据协议；未来需要 L4 runner、resume、evaluation 和 Slurm orchestration 支持，但不改变模型架构或物理数据定义。

使用什么证据验证：
10k 102-run 结果、N40 RTX4090/6 CPU/60,000 MB 单卡证据、12-job 历史训练波次、50k 数据配置、当前模型输入形状实现与通用 M9 HPC campaign 合同。

最大风险、未知或下一 gate：
两套 50k HDF5 已在 N40 与 Xingyi 源文件完成大小和 SHA-256 对齐；负责人明确豁免本地 WSL replay 训练前置门。binary 40k 前两波已完成，第三波正在运行；最近 gate 是部署 17 波 v2 controller、纳入前三波并滚动执行剩余波次。跨 dataset checkpoint 推理和最终汇总仍是后续独立证据门。
```

[↑ 回顶部](#full-cell-dual-lane-50000-training-plan)
