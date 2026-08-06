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
| Last reviewed | Pending human review |
| Reading time | 15 minutes |

本文固定 **连续/五阈值表征、三 seed、三张租用 GPU 并行训练和超算阈值物理验证** 的 Full-Wh 前向实验协议。核心变化是把旧版单 seed 计划扩展为三 seed：每张云 GPU 固定负责一个 seed，M9 超算只执行 real-MPB，不参与跨站点神经网络训练。

本文只是一份待人工审计的计划，不表示数据门禁已全部通过、GPU 已租用、超算任务已提交或任何正式结果已经产生。

**目录：** [一、目标与边界](#forward-goal) · [二、数据门禁](#forward-data) · [三、表征方案](#forward-representation) · [四、模型与训练矩阵](#forward-matrix) · [五、混合算力](#forward-compute) · [六、训练协议](#forward-protocol) · [七、交叉评估](#forward-cross-eval) · [八、超算物理验证](#forward-mpb) · [九、制品与验收](#forward-acceptance)

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

使用 [`data/full_cell_binary_bz9_10000_xingyi_v1.h5`](../../../data/README.md) 对应的本地数据。当前状态为已通过 WSL real-MPB 跨环境验收，可以进入训练准备。

### 2.2 渐进路线

使用 `data/full_cell_graded_bz9_10000_paracloud_m9_v1.h5`。训练前必须执行 [固定 WSL acceptance 配置](../../../configs/research/full_cell_graded_paracloud_m9_wsl_acceptance_v1.yaml)，并取得：

```text
wsl_cross_environment_acceptance_pass
```

如果验收失败，允许继续开发 HDF5 读取、shape smoke 和配置检查，但不得生成或发布正式渐进训练结果。

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

### 5.1 三卡任务映射

推荐租用三张同型号 GPU，每张固定负责一个 seed：

| 实例 | 任务 | 主矩阵 | 消融 | 总训练数 |
|---|---|---:|---:|---:|
| GPU-0 | 所有 seed 0 训练 | 28 | 6 | 34 |
| GPU-1 | 所有 seed 1 训练 | 28 | 6 | 34 |
| GPU-2 | 所有 seed 2 训练 | 28 | 6 | 34 |

不使用跨卡或跨站点 DDP。每个模型在单卡独立训练，卡内串行、三 seed 并行，以保证故障隔离、制品管理和 seed 对齐。

### 5.2 选卡规则

- 若 RTX 5090 租价不超过同平台 RTX 4090 的 `1.30x`，且 CUDA/PyTorch preflight 通过，选择 3×RTX 5090 32 GB；否则选择 3×RTX 4090 24 GB。
- 只有当 RTX 3090 价格低于 4090 的约 55%，且实测单位完成任务成本更低时才考虑 3090。
- 首轮不租 A100/H100；只有 24/32 GB 卡在 `128x128` 最大任务上无法达到合理吞吐，或数据中心卡的实测总费用更低时才升级。
- 5090/4090 的显存与带宽差异参考 [NVIDIA RTX 5090](https://www.nvidia.com/en-eu/geforce/graphics-cards/50-series/rtx-5090/) 和 [NVIDIA RTX 4090](https://www.nvidia.com/en-gb/geforce/graphics-cards/40-series/rtx-4090/)，实际训练选择只依据租用平台 preflight。

### 5.3 单卡 preflight

正式三卡租用前，只授权一张候选卡完成：

1. GPU、driver、CUDA、PyTorch 和架构检查；
2. 四模型 forward/backward；
3. FNO 10 epochs 小基准；
4. batch 64 峰值显存检查；
5. HDF5 multi-worker 吞吐检查；
6. checkpoint 写入、恢复和结果 hash 检查。

根据实测 ETA 计算三卡总费用后，单独请求正式租用授权。preflight 通过不等于自动启动三卡。

### 5.4 本机、云 GPU、M9 分工

| 环境 | 职责 | 禁止事项 |
|---|---|---|
| 本机 RTX 4060 Laptop 8 GB | 开发、shape smoke、WSL 验收、checkpoint 复核和报告 | 不承担 102 个正式 run；local smoke 不算正式证据 |
| 三张租用 GPU | 神经网络训练、推理和指标汇总 | 不运行 real-MPB，不跨站点 DDP |
| M9 CPU 超算 | 阈值结构 real-MPB 物理验证 | 不训练神经网络，不与云 GPU 共享分布式进程 |

云端与 M9 通过冻结配置、manifest、checkpoint/hash 和原子 shard 异步交接，不共享可变工作目录。

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
- 模型和 threshold 最终选择依据为三 seed validation 均值，不查看 test split。

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

### 9.1 每个 run 必须记录

```text
git commit
config hash
dataset SHA-256
split hash
model
representation
threshold
resolution
seed
normalization
checkpoint
metrics
history
runtime
GPU / driver / CUDA / PyTorch
```

run 目录遵循 `runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/`，包含 `run.log`、`run.meta.json`、`config.resolved.yaml`、`results.json`、`metrics.json`、`history.csv` 和必要图件。

### 9.2 云端训练回收

1. 汇总三 seed validation 结果；
2. 冻结胜出配置；
3. 统一运行 test split；
4. 下载 checkpoint、resolved config 和全部运行制品；
5. 校验文件 SHA-256；
6. 在本机 WSL `Ubuntu` + `meep_env` 重新评估胜出 checkpoint；
7. 核对预测与指标一致性；
8. 生成 leaderboard、`6x6` 热图、置信区间和研究报告。

### 9.3 最终 gate

- 102 个训练 run 全部具有明确 success/failed 状态；
- 三 seed 不缺失，不以两个 seed 冒充正式三 seed；
- 选择过程只访问 validation，test 只在冻结后运行；
- graded 数据已经通过 WSL acceptance；
- 云端 checkpoint 在本机 WSL 可复核；
- threshold proxy 与 threshold real-MPB 指标分表报告；
- 未运行 M9 时明确写“物理阈值验证未执行”，不得把 proxy 指标提升为物理结论；
- 人工负责人审阅后，才能把本文从 Draft 改为 Reviewed/Canonical 或授权执行。

[↑ 回顶部](#full-cell-dual-lane-hybrid-compute-plan)
