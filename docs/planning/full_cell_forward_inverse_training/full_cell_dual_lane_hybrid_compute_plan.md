---
title: "全胞双路线训练与混合算力计划"
summary: "研究计划或执行协议；实施前需与当前代码和证据重新核对。"
page_type: plan
status: draft
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 20 minutes
primary_layer: L0
---

<!--
  Three-seed full-cell dual-lane forward training and hybrid compute plan.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="full-cell-dual-lane-hybrid-compute-plan"></a>

# 全胞双路线训练与混合算力计划

| Metadata | Value |
|---|---|
| Status | Draft pending human review |
| Primary layer | L3 experiment/evidence；L4 仅提供必要接口支持 |
| Authority | [生成网络调整研究计划](../fno_generative_network_adjustment_plan_20260617.md) |
| Replaces | [旧版单 seed 正向计划](archive/full_cell_forward_single_seed_legacy.md) |
| Audience | 项目负责人、研究人员、GPU/超算执行人员与 AI 协作者 |
| Owner | 研究生维护，导师审阅科学主张与 promotion gate |
| Created | 2026-08-06 |
| Environment snapshot | 2026-08-06，执行前必须刷新 |
| Last reviewed | Pending human review |
| Reading time | 18 minutes |

本文固定 **连续/五阈值表征、三 seed、模型 × seed 独占单卡训练和超算阈值物理验证** 的 Full-Wh 前向实验协议。主矩阵按表征分波次执行；每个波次同时运行 `4 models × 3 seeds = 12` 个独立单卡作业，使每个“模型 + seed”组合独占一张 GPU。M9 超算只执行 real-MPB，不参与跨站点神经网络训练。

本文仍是待人工审计的 Draft。2026-08-06 的现场证据已经确认两套数据门、N40 三卡分配与单卡训练预检通过，因此可以进入 **12 个独立单卡作业的并发扩容测试**；这不表示 12/18 卡并发已被账号 QoS 实测接受，也不表示 102 个正式训练 run 或新的 M9 任务已经启动。

同日稍后，原先缺失于 M9 的 binary 10k HDF5 已补传并通过远端 SHA-256，M9 与 N40 代码静态部署统一到 `b93cc2d`。这解除文件交接缺口，但 packed Meep/MPB 计算节点 preflight 和任何新 solver 作业仍需单独授权。

**目录：** [一、目标与边界](#forward-goal) · [二、数据门禁](#forward-data) · [三、表征方案](#forward-representation) · [四、模型与训练矩阵](#forward-matrix) · [五、混合算力](#forward-compute) · [六、训练协议](#forward-protocol) · [七、交叉评估](#forward-cross-eval) · [八、超算物理验证](#forward-mpb) · [九、制品与验收](#forward-acceptance) · [十、人工理解记录](#forward-human-record)

---

<a id="forward-goal"></a>

## 一、研究目标与边界

本阶段只完成：

```text
全胞材料表征 -> Wh [81,10]
```

包含：

- 二值全胞数据训练；
- 渐进全胞连续表征训练；
- 渐进数据五种确定阈值二值表征训练；
- continuous/threshold 之间的交叉测试；
- binary/graded 之间的跨路线测试；
- 三 seed 稳定性统计；
- 胜出方案的 `32/64/128` 分辨率消融；
- 有条件启动的阈值结构真实 MPB 验证。

暂不包含 Query、Field/PINO、inverse generator 或大规模 inverse candidate replay。后续 inverse 工作由 [逆向设计扩展计划](full_cell_inverse_design_extension_plan.md) 单独管理。

[↑ 回顶部](#full-cell-dual-lane-hybrid-compute-plan)

---

<a id="forward-data"></a>

## 二、数据门禁

### 2.1 二值路线

使用 [`data/full_cell_binary_bz9_10000_xingyi_v1.h5`](https://github.com/fangtoast/phc2d/blob/master/data/README.md) 对应的本地数据。当前状态为已通过 WSL real-MPB 跨环境验收，可以进入训练准备。

### 2.2 渐进路线

使用 `data/full_cell_graded_bz9_10000_paracloud_m9_v1.h5`。该数据已通过 [固定 WSL acceptance 配置](https://github.com/fangtoast/phc2d/blob/master/configs/research/full_cell_graded_paracloud_m9_wsl_acceptance_v1.yaml)，状态为：

```text
wsl_cross_environment_acceptance_pass
```

验收 run 为 `runs/20260806/full_cell_graded_paracloud_m9_wsl_acceptance_v1_202608061417/`。如果未来数据 hash、schema 或 solver contract 改变，必须重新执行该验收；新验收失败时只允许继续 HDF5 读取、shape smoke 和配置检查，不得生成或发布相应版本的正式渐进训练结果。

### 2.3 分割与 provenance

- 两条路线始终使用各自冻结的 `8000/1000/1000` split，不合并训练集。
- 训练前核验 HDF5 schema、文件 SHA-256、split hash、`Wh`、k 点和 `material_weight_grid_32/64/128`。
- 任何数据 hash、split 或 solver contract 变化都创建新 experiment id，不覆盖既有结果。

[↑ 回顶部](#full-cell-dual-lane-hybrid-compute-plan)

---

<a id="forward-representation"></a>

## 三、连续与阈值表征

设主分辨率材料权重网格为：

$$
u=\texttt{material\_weight\_grid\_64},\qquad
u\in[0,1]^{64\times64}.
$$

### 3.1 连续表征

```python
continuous_flat = u.astype("float32").reshape(4096, order="C")
continuous_image = u.astype("float32")[None, :, :]
```

### 3.2 确定阈值表征

```python
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
binary_t = (u >= threshold).astype("float32")
threshold_flat_t = binary_t.reshape(4096, order="C")
threshold_image_t = binary_t[None, :, :]
```

固定规则：

- 展平统一使用 C-order/row-major，并在 metadata 中保存原始 `grid_shape`；
- 本计划不使用随机 Bernoulli 采样；等于阈值的像素按 `>=` 规则确定为 1；
- 二值原始数据在五个阈值下完全相同，只训练原生版本；
- 渐进阈值输入继续使用原渐进介质的 `Wh`；
- 阈值实验只检验信息压缩和模型鲁棒性，不代表阈值图具有原连续结构的真实物理响应。

每个阈值 run 必须记录：

```yaml
label_semantics: original_continuous_medium_Wh
representation_status: thresholded_proxy_input
physical_equivalence: false
threshold: 0.3 | 0.4 | 0.5 | 0.6 | 0.7
```

### 3.3 表征接口

`DataRepresentationSpec` 至少支持：

```yaml
source: material_weight_grid
layout: flat | image
sampling: continuous | deterministic_threshold
threshold: null | 0.3 | 0.4 | 0.5 | 0.6 | 0.7
flatten_order: row_major
resolution: metadata
physical_equivalence: true | false
```

所有 shape 从 HDF5 metadata 推导，禁止在数据适配器、模型、loss、metrics 或 plot 中硬编码 `64`、`81` 或 `10`。

[↑ 回顶部](#full-cell-dual-lane-hybrid-compute-plan)

---

<a id="forward-matrix"></a>

## 四、模型与三 seed 训练矩阵

### 4.1 模型范围

| 模型 | 输入 | 角色 |
|---|---|---|
| `mlp` | 展平数组 | 非残差向量基线 |
| `res_mlp_wide` | 展平数组 | 残差向量模型 |
| `lifted_cnn` | 二维图像 | 局部空间基线 |
| `fno2d_band_v2` | 二维图像 | 频谱/全局空间模型 |

不进入主排行榜：

- `pod_res_mlp`：仍为 placeholder；
- Query 模型：当前目标不是 band-query；
- Field/U-Net/PINO：当前 HDF5 没有本征场标签；
- 历史轻量模型：只允许 shape smoke。

### 4.2 `64x64` 主实验

固定 `seeds=[0,1,2]`。

| 路线 | 计算 | runs |
|---|---|---:|
| binary | 4 models × 1 representation × 3 seeds | 12 |
| graded | 4 models × 6 representations × 3 seeds | 72 |
| 合计 |  | **84** |

### 4.3 分辨率消融

三 seed 主结果汇总后，只依据 validation 选择：

1. binary 胜出模型；
2. graded continuous 胜出模型；
3. graded threshold 胜出的“模型 + threshold”组合。

三种胜出配置分别补做 `32x32` 和 `128x128`：

```text
3 winning configurations × 2 resolutions × 3 seeds = 18 runs
```

最终正式训练总数：

```text
84 main + 18 resolution = 102 runs
```

分辨率胜者选择前不得提前启动 18 个消融 run，避免使用单 seed 或未完整 validation 的临时领先结果。

[↑ 回顶部](#full-cell-dual-lane-hybrid-compute-plan)

---

<a id="forward-compute"></a>

## 五、租用 GPU 与混合算力编排

### 5.1 当前实验环境状态

以下是 2026-08-06 的现场快照；队列空闲量会变化，正式提交前必须重新运行只读资源检查。详细证据见 [N40/N50/M9 全链路预检报告](../../results/data-provenance/paracloud_gpu_preflight_20260806.md)。

| 环境 | 当前状态 | 已有证据 | 当前边界 / 下一步 |
|---|---|---|---|
| N40 `gpu_4090` | **可用于扩容测试**；队列共 144 张 RTX4090，快照时 44 张空闲 | 作业 `5390073` 实际同时分配 3 卡；作业 `5390162` 单卡训练预检通过 | 3 卡可用已实测；12/18 个独立单卡作业仍需短并发探针确认账号 QoS、排队和共享存储吞吐 |
| N40 软件 | **就绪** | 部署 commit `05f5776`；PyTorch `2.1.2+cu118`、CUDA `11.8`、h5py `3.10.0`、NumPy `1.26.4` | 正式 run 必须记录实际节点、driver、CUDA、PyTorch 和依赖锁文件 hash |
| N40 数据 | **就绪** | binary/graded 两份 10k HDF5 已到位，远端 SHA-256 与本机一致 | 每个波次启动前只读复核 dataset/split hash；不得复制成每个作业一份临时训练数据 |
| N50R5 `gpu` | **blocked** | 平台网关拒绝 `loginCluster` | 未安装环境、未分配 RTX5090、未通过 preflight，不计入本计划可用卡数 |
| M9 `amd_m9_768` | **静态交接 ready，可用于经授权的 MPB 测试** | SSH、packed Meep/MPB 环境、作业脚本及 binary/graded 10k 数据存在且 hash 一致；代码为 `b93cc2d` | 所有新计算节点 environment preflight、MPB preflight/pilot 仍需单独授权，不因 GPU 训练启动而自动提交 |
| 本机 WSL `meep_env` | **正式验收环境就绪** | binary/graded 10k 均已通过固定 real-MPB 跨环境验收 | 用于 checkpoint 回收复核和 MPB acceptance；Windows `.venv` 不算正式验收 |

N40/N50 的平台约束分别以 [N40 文档](https://kbs.paratera.com/info/1503) 和 [N50R5 文档](https://kbs.paratera.com/info/1661) 为准。上表中的“44 张空闲”和“M9 5,633 CPU 空闲”只是检查时快照，不能写入正式结果作为长期容量承诺。

### 5.2 模型 × seed 独占单卡映射

每个表征波次创建 12 个相互独立的单卡作业：

| 逻辑任务槽 | 模型 | seed | 作业数 |
|---|---|---|---:|
| GPU-00～GPU-02 | `mlp` | 0、1、2 | 3 |
| GPU-03～GPU-05 | `res_mlp_wide` | 0、1、2 | 3 |
| GPU-06～GPU-08 | `lifted_cnn` | 0、1、2 | 3 |
| GPU-09～GPU-11 | `fno2d_band_v2` | 0、1、2 | 3 |

“GPU-00～GPU-11”只是编排槽位，不绑定物理卡号。每个 Slurm 作业只申请 1 张 GPU，并在自己的 allocation 内使用 `cuda:0`；不得手工共享 `CUDA_VISIBLE_DEVICES`，也不使用单机或跨站点 DDP。这样一个模型的慢速或失败不会占用其他模型的进程，也不会改变 seed 对齐。

### 5.3 分波次并发计划

| 阶段 | 波次 | 每波任务 | 并发上限 | 训练 run |
|---|---:|---|---:|---:|
| binary native | 1 | 4 models × 3 seeds | 12 | 12 |
| graded continuous | 1 | 4 models × 3 seeds | 12 | 12 |
| graded t0.3～t0.7 | 5 | 每阈值 4 models × 3 seeds | 12 | 60 |
| 主矩阵合计 | **7** | 每个表征完成一个完整 12-job 波次 | **12** | **84** |

默认只让一个表征波次占用 GPU 池，即最多 12 张卡并行，而不是把 84 个主 run 一次性全部提交。一个波次的 12 个 terminal 状态和核心制品齐备后，再开放下一个波次；科研矩阵、随机种子和模型选择规则均不因并发扩容而改变。

分辨率胜者冻结后，18 个消融 run 仍采用“一配置 × 一 seed × 一 GPU”。若当时队列与账号 QoS 允许，可一次并发 18 个独立单卡作业；否则按 `12 + 6` 或 `6 + 6 + 6` 节流。并发度只影响墙钟时间，不得改变配置、epoch、batch、seed 或最终统计口径。

### 5.4 并发扩容 gate 与调度规则

现有三卡探针和单卡训练预检不能替代 12/18 卡并发验证。正式训练授权前新增以下最小 gate：

1. 提交 12 个短时独立单卡任务，完整覆盖 4 models × 3 seeds 的启动、HDF5 读取、一次 forward/backward 和原子写盘；
2. 核对 12 个任务的 Slurm `AllocTRES`、节点、退出码、实际 GPU 名称与 driver，不以 `PENDING` 任务冒充已分配 GPU；
3. 观察共享 HDF5 在 12 并发下是否出现读取错误、异常等待或 GPU 长时间空转，据实选择正式并发上限 `12`、`6` 或 `3`；
4. 在 128 分辨率消融前再做 18 个短任务的 admission/吞吐探针；若无法同时获配 18 卡，自动降为 `12 + 6`，不更改实验矩阵；
5. 每个任务使用唯一 run 目录和唯一 checkpoint 路径；共享 HDF5 只读，汇总文件只能由波次结束后的单独聚合作业写入；
6. 失败任务保留 terminal 状态和最小错误证据，修复后使用新 run id 重跑；不得覆盖成功 run，也不得让自动重试改变配置。

并发探针通过只表示可以请求正式训练授权，不等于自动提交 84/102 个训练 run。正式作业仍以当时 `sinfo`、账号 QoS、预计 GPU-hours 和人工授权为准。

### 5.5 选卡与环境分工

- 当前正式候选固定为 N40 RTX4090；N50R5 在登录、PyTorch ≥2.7/cu128 和单卡/并发 preflight 全部通过前只作备选，不允许混入同一三 seed 对照。
- 首轮不使用 A100/H100；只有 24 GB 卡在 `128x128` 最大任务上无法达到合理吞吐，或其他卡型的实测总费用更低时才重新审议。
- 同一个可比较组的三个 seed 必须使用同一 GPU 型号、同一软件锁文件和同一训练配置；不得让 seed 0 使用 4090、seed 1/2 使用 5090。

| 环境 | 职责 | 禁止事项 |
|---|---|---|
| 本机 RTX 4060 Laptop 8 GB | 开发、shape smoke、WSL 验收、checkpoint 复核和报告 | 不承担 102 个正式 run；local smoke 不算正式证据 |
| N40 GPU 池 | 12-job 主矩阵波次、最多 18-job 消融波次、推理与指标汇总 | 不运行 real-MPB，不跨站点 DDP，不让多个任务写同一 run 目录 |
| M9 CPU 超算 | 阈值结构 real-MPB 物理验证 | 不训练神经网络，不与 N40 共享分布式进程 |

N40 与 M9 通过冻结配置、manifest、checkpoint/hash 和原子 shard 异步交接，不共享可变工作目录。

[↑ 回顶部](#full-cell-dual-lane-hybrid-compute-plan)

---

<a id="forward-protocol"></a>

## 六、统一训练协议

- precision：FP32；
- optimizer：AdamW，`weight_decay=1e-5`；
- MLP/ResMLP 学习率：`3e-4`；
- CNN/FNO 学习率：`1e-3`；
- 最大 100 epochs；
- early-stopping patience 15；
- 有效 batch size 64；显存不足时按 `64 -> 32 -> 16 -> 8` 降低 micro-batch，并用梯度累积保持有效 batch；
- 每个 seed 使用相同配置，不允许独立调参；
- 输入材料权重保持 `[0,1]` 或阈值后的 `{0,1}`；
- `Wh` 使用各数据路线 train split 的逐 `(k,band)` mean/std 标准化；
- 训练损失为标准化后的全矩阵 MSE；
- checkpoint 选择指标为 validation bands 2–10 MRE；
- 模型和 threshold 最终选择依据为三 seed validation 均值，不查看 test split；
- 每个正式 run 由 `(lane, representation, model, seed, resolution)` 唯一标识，并独占一个标准 run 目录；
- 同一波次的 12 个作业读取同一冻结 HDF5 和 split，但不得共享可写 checkpoint、history、日志或临时目录；
- 并发度变化只允许改变调度节奏，不得改变训练超参数、early-stopping、有效 batch size 或数据顺序合同。

同一 seed 的所有模型使用相同 data-loader seed 和 batch 顺序。三 seed 结果完成前不按单 seed 临时结果改变超参数。

[↑ 回顶部](#full-cell-dual-lane-hybrid-compute-plan)

---

<a id="forward-cross-eval"></a>

## 七、交叉评估

### 7.1 渐进内部交叉表征

每个模型、每个 seed 构造 `6x6` 矩阵：

- 训练：continuous、t0.3、t0.4、t0.5、t0.6、t0.7；
- 测试：continuous、t0.3、t0.4、t0.5、t0.6、t0.7。

```text
4 models × 3 seeds × 6 train representations × 6 test representations
= 432 inference cells
```

这些仅增加推理与指标任务，不增加训练 run。

### 7.2 跨数据路线

- binary-trained：4 models × 3 seeds，分别测试 graded continuous 和五种 threshold，共 72 个 inference cells；
- graded-trained：4 models × 6 representations × 3 seeds，测试真实 binary test split，共 72 个 inference cells；
- 跨路线合计 144 个 inference cells。

始终使用源模型自己的 `Wh` normalization，不读取目标路线 train/val 标签重新校准。跨路线结果只属于 OOD/generalization 诊断，不参与模型或阈值选择。

### 7.3 报告指标

- 每个 run 的逐 epoch train/validation 曲线，以及同一“模型 + 表征”的三 seed 均值曲线与标准差带；
- best epoch、early-stopping epoch、最终 train loss 与最佳 validation 指标；
- 三 seed mean ± std 与每 seed 独立结果；
- paired bootstrap 95% CI；
- bands 2–10 MRE、MAE、RMSE；
- per-band、per-k 指标和 band 1 诊断；
- threshold sensitivity 与 continuous ↔ threshold 迁移；
- binary ↔ graded 泛化；
- 参数量、训练时间、推理吞吐、峰值显存；
- failed runs、异常样本和非有限值计数。

[↑ 回顶部](#full-cell-dual-lane-hybrid-compute-plan)

---

<a id="forward-mpb"></a>

## 八、超算阈值物理验证

本轮神经网络训练不需要超算参与。只有阈值代理实验在 validation 上显示继续研究价值后，才允许进入独立物理验证。

### 8.1 先决实现门

当前 full-cell solver 只接受已有解析 `FullCellStructureSpec`，不能把任意 raster 直接当作正式几何。阈值 MPB 前必须增加一个可追溯的解析包装合同，例如：

```yaml
generator_family: thresholded_graded_rbf_fourier
source_structure_hash: <graded structure hash>
threshold: 0.3 | 0.4 | 0.5 | 0.6 | 0.7
material_kind: binary
```

其材料函数必须在任意 solver 坐标点重新计算源 `graded_rbf_fourier` 权重，再按 `weight >= threshold` 返回 0/1。不得用 `64x64` raster 插值替代解析阈值，也不得复用原连续结构的 `Wh` 作为阈值物理真值。

该接口须先在 WSL `Ubuntu` + `meep_env` 完成 codec、geometry hash、周期边界、raster 对照和最小 real-MPB smoke。

### 8.2 M9 preflight

```text
8 graded structures × 5 thresholds = 40 thresholded physical structures
```

每个结构重新运行 81 个 k 点、10 bands 的 MPB。40 个结构全部成功、频率有限有序、manifest/hash 完整后，才允许扩展。

### 8.3 M9 pilot

```text
64 stratified graded structures × 5 thresholds = 320 real-MPB structures
```

64 个源结构按 core count、material fraction 和 forward difficulty 分层冻结。比较：

- 阈值结构真实 `Wh_threshold_MPB`；
- 原连续结构 `Wh_graded_MPB`；
- threshold proxy surrogate prediction；
- bands 2–10 MRE；
- bandgap 改变量；
- surrogate–replay gap；
- solver success rate 和 runtime。

该阶段回答“阈值结构是否具有接近的真实物理响应”，与前面的表征消融严格分开。M9 结果在固定 WSL cross-environment replay 通过前只能标记为 `cluster_generated_pending_wsl_acceptance`。

所有 M9 preflight、pilot 和后续扩展都需要单独授权；不得因 GPU 训练完成而自动提交超算任务。

[↑ 回顶部](#full-cell-dual-lane-hybrid-compute-plan)

---

<a id="forward-acceptance"></a>

## 九、制品、回收与验收

### 9.1 每个训练 run 的最小证据集

run 目录遵循 `runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/`。102 个正式训练 run 中，每个 run 只强制保存下列核心制品：

| 制品 | 必须内容 | 用途 |
|---|---|---|
| `run.log` | 启动、关键 gate、best epoch、terminal 状态和异常摘要 | 故障审计，不记录每个 batch 的重复信息 |
| `run.meta.json` | git commit、config/dataset/split hash、model、representation、threshold、resolution、seed、Slurm job/node、GPU/driver/CUDA/PyTorch | 复现与环境追踪 |
| `config.resolved.yaml` | 本 run 完整冻结配置 | 防止并发任务配置串写 |
| `history.csv` | `epoch`、`train_loss`、`val_loss`、validation bands 2–10 MRE、learning rate、epoch time、累计 wall time | 训练曲线的唯一数值源 |
| `figures/training_curve.png`、`figures/training_curve.pdf` | train/validation loss 与主 validation 指标，标出 best/stop epoch | 人工核对收敛、过拟合和异常中断 |
| `model.pt` | best-validation checkpoint | 回收复核与冻结后 test；SHA-256 写入 `results.json` |
| `metrics.json` | MRE/MAE/RMSE、per-band/per-k 摘要、train/eval/inference time、throughput、峰值显存 | 单 run 核心数值 |
| `results.json` | success/failed、best epoch、checkpoint hash、选择 split、是否执行 test、下一 gate | 机器可读 terminal 结论 |

必须从 `history.csv` 生成曲线，禁止只保存图片而丢失逐 epoch 数值。若某 run 失败，也必须保留已写入的 history、错误摘要、退出码和资源元数据；失败 run 不要求伪造完整曲线或指标。

### 9.2 波次与项目级汇总

每个 12-job 主波次结束后生成一次聚合目录，至少包含：

- `run_manifest.csv`：12 个 `(model, seed)` 对应的 run id、状态、checkpoint hash 和核心路径；
- `seed_summary.csv`：每个模型的三 seed 独立值、mean、std 和成功数；
- `leaderboard.csv`：只基于 validation 的冻结排序；
- `figures/training_curves_by_model.png` 与 `.pdf`：每模型三 seed 曲线及均值 ± 标准差带；
- `figures/validation_comparison.png` 与 `.pdf`：四模型的核心 validation 对比；
- `wave.meta.json`：数据/配置 hash、计划任务数、完成/失败/缺失数和并发环境快照。

三 seed 聚合曲线只在三个 seed 都存在的共同 epoch 区间计算均值与标准差；early stopping 后的缺失 epoch 不做前向填充。共同区间之外保留各 seed 独立曲线，避免用重复末值伪造稳定性。

只有三 seed 完整且 validation 选择已经冻结的最终胜出配置，才必须保存 test split 的紧凑 `predictions_test.npz`、sample id 和 per-sample 误差；其余非胜出训练 run 不保存全量预测张量。

以下杂项默认不进入长期证据集：per-batch loss 流、逐步 GPU telemetry、profiler trace、中间 activation/gradient、每个 epoch checkpoint、重复 HDF5/阈值数据副本、临时 cache、debug preview 和重复 stdout。它们可在定位故障时短期保留，但不得替代上面的核心制品，也不得进入 Git。

### 9.3 云端训练回收

1. 汇总三 seed validation 结果；
2. 冻结胜出配置；
3. 统一运行 test split；
4. 下载 best checkpoint、resolved config、history、曲线、核心 metrics/results 和聚合表，不要求回收上述临时杂项；
5. 校验文件 SHA-256；
6. 在本机 WSL `Ubuntu` + `meep_env` 重新评估胜出 checkpoint；
7. 核对预测与指标一致性；
8. 生成 leaderboard、`6x6` 热图、置信区间和研究报告。

### 9.4 最终 gate

- 102 个训练 run 全部具有明确 success/failed 状态；
- 三 seed 不缺失，不以两个 seed 冒充正式三 seed；
- 每个成功 run 的 `history.csv`、训练曲线、best checkpoint、核心 metrics/results 和 provenance 齐全；
- 每个 12-job 波次都有 run manifest、三 seed 汇总和完成/失败/缺失计数；
- 选择过程只访问 validation，test 只在冻结后运行；
- graded 数据已经通过 WSL acceptance；
- 云端 checkpoint 在本机 WSL 可复核；
- threshold proxy 与 threshold real-MPB 指标分表报告；
- 未运行 M9 时明确写“物理阈值验证未执行”，不得把 proxy 指标提升为物理结论；
- 人工负责人审阅后，才能把本文从 Draft 改为 Reviewed/Canonical 或授权执行。

[↑ 回顶部](#full-cell-dual-lane-hybrid-compute-plan)

---

<a id="forward-human-record"></a>

## 十、本次调整的人工理解记录

```text
解决了什么问题：把原“三张卡分别承包一个 seed”的串行编排，调整为每个表征波次 4 models × 3 seeds 的 12 个独立单卡作业，并明确训练曲线与核心结果的最小保存合同。
为什么采用这个方案：N40 当前容量足以考虑超过三卡的并发；按模型 × seed 隔离可以缩短墙钟时间，同时不改变 102-run 科研矩阵和三 seed 公平性。
改变了项目哪一层/哪一模块：主要改变 L3 实验/证据编排；没有改变模型架构、损失、数据 split 或 L0/L1 科学主张。
使用什么证据验证：N40 三卡硬件作业、单卡四模型训练预检、两套 HDF5 hash、graded WSL acceptance，以及 2026-08-06 的 N40/M9 队列与环境只读快照。
最大风险、未知或下一 gate：12/18 个独立单卡作业尚未做账号 QoS 与共享 HDF5 并发探针；N50 登录仍 blocked；正式训练与任何新 M9 作业仍需人工授权。
```

[↑ 回顶部](#full-cell-dual-lane-hybrid-compute-plan)
