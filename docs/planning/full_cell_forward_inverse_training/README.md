<!--
  Execution plan for binary and graded full-cell forward surrogates and inverse design.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="full-cell-forward-inverse-training-plan"></a>

# 二值/渐进全胞正向替代与解析参数逆向设计训练计划

| Metadata | Value |
|---|---|
| Status | Draft pending human review |
| Primary layer | L3 experiment/evidence；L4 仅提供必要接口支持 |
| Authority | [生成网络调整研究计划](../fno_generative_network_adjustment_plan_20260617.md) |
| Supersedes | [旧版单 seed 正向训练草案](archive/full_cell_forward_single_seed_legacy.md) |
| Audience | 项目负责人、研究人员、GPU/超算执行人员与 AI 协作者 |
| Owner | 研究生维护，导师审阅科学主张与 promotion gate |
| Created | 2026-08-06 |
| Last reviewed | Pending human review |
| Reading time | 15 minutes |

本文固定二值与渐进全胞数据上的 **Full-Wh 正向替代、解析参数逆向生成、双替代筛选和 real-MPB replay** 协议。它回答训练什么、如何表征、怎样分配本机/租用 GPU/M9 超算以及何时可以提升证据状态；它不证明任何训练或物理验证已经完成。

**目录：** [一、目标与边界](#plan-goal) · [二、数据与接口](#plan-data) · [三、正向训练](#plan-forward) · [四、逆向设计](#plan-inverse) · [五、资源编排](#plan-resources) · [六、验证与交付](#plan-validation)

---

<a id="plan-goal"></a>

## 一、目标、现状与证据边界

- 使用两套 `phc2d.full_cell.v1` 10k 数据，固定原始 `8000/1000/1000` 划分，完成：
  1. `material_weight_grid -> Wh` 正向替代；
  2. `目标 Wh -> 解析结构参数 -> 候选筛选 -> real-MPB replay` 逆向闭环。
- 当前模型注册表包含 17 个模型：8 个 Full-Wh、5 个 Query、4 个场模型；非 registry 的 inverse 方法继续单独盘点。本轮 Full-Wh 只训练 `mlp`、`res_mlp_wide`、`lifted_cnn`、`fno2d_band_v2`。
- Query、场预测、adapted PINO、扩散模型和任意栅格直接求解合同不进入首轮。它们不得凭历史结果进入新全胞结论。
- 二值数据集已经通过 WSL 验收。渐进数据集当前只可标记为 `cluster_generated_pending_wsl_acceptance`，必须先通过 [固定 8 结构 WSL 验收配置](../../../configs/research/full_cell_graded_paracloud_m9_wsl_acceptance_v1.yaml)，随后才能启动渐进正式训练。
- 计划、配置就绪、smoke、pilot、GPU 正式训练、M9 bulk replay 和 WSL acceptance 是不同状态，必须分别记录。

[↑ 回顶部](#full-cell-forward-inverse-training-plan)

---

<a id="plan-data"></a>

## 二、数据表征与公共接口

### 2.1 正向输入表征

基础输入统一使用 HDF5 中的 `material_weight_grid_{resolution}`：0 表示空气，1 表示介质，网格尺寸从 schema/metadata 读取。

| 表征 | 张量形状 | 使用模型 | 规则 |
|---|---|---|---|
| `image` | `[B,1,H,W]` | `lifted_cnn`、`fno2d_band_v2` | 保留周期二维拓扑 |
| `flat` | `[B,H*W]` | `mlp`、`res_mlp_wide` | C-order/row-major 展平，并保存原始 `grid_shape` |
| `continuous` | 值域 `[0,1]` | 渐进物理主路线 | 不阈值化、不添加噪声 |
| `bernoulli_density` | 值域 `{0,1}` | 渐进表征消融 | 每像素按介质权重作为概率采样 |

渐进随机二值表征固定为：

$$
x_{ij}^{(s,e)} \sim \operatorname{Bernoulli}(\rho_{ij}),
$$

其中 $\rho_{ij}=0.5$ 时等概率得到 0 或 1。训练时每个 epoch 重采样，随机键由 `sample_id + epoch + seed` 决定；同一 seed 下不同模型看到相同采样序列。验证/测试冻结 8 组 Monte Carlo draw bank，报告预测均值及 draw 间方差。

Bernoulli 图继续使用原渐进结构的 `Wh` 标签，因此必须记录：

```yaml
label_semantics: original_continuous_medium_Wh
representation_status: bernoulli_density_proxy_input
physical_equivalence: false
```

它只用于表征鲁棒性研究，不得作为逆向候选或 real-MPB 物理结构。二值数据保持原生精确 0/1，不重复随机采样。

### 2.2 `DataRepresentationSpec` 扩展

新增或扩展以下字段，所有形状均从数据元信息派生：

```yaml
source: material_weight_grid
layout: image | flat
sampling: continuous | bernoulli_density
resolution: metadata
flatten_order: row_major
sampling_seed: 0
eval_mc_draws: 8
physical_equivalence: true | false
```

### 2.3 逆向解析参数合同

逆向不得直接把任意 `64x64` 数组交给现有 full-cell solver。新增解析参数 codec，分别处理五个结构变体：

| 变体 | 原参数宽度 | 材料路线 |
|---|---:|---|
| `binary_polar_holes_1` | 16 | binary |
| `binary_polar_holes_2` | 31 | binary |
| `binary_spectral_threshold` | 580 | binary |
| `graded_rbf_fourier_1` | 106 | graded |
| `graded_rbf_fourier_2` | 112 | graded |

codec 只学习中心、半径、角度、傅里叶系数、阈值等可变字段；`hole_count`、傅里叶阶数、`mx` 和 `my` 等常量由变体模板恢复。角度使用正余弦编码，其余字段使用训练集统计量归一化并按合法范围反变换。

解码结果必须能构造现有 `FullCellStructureSpec`，并输出 `family_id`、`variant_id`、有效参数 mask、`geometry_hash`、精确 raster 和候选有效性。`CandidateBatch` 与 replay manifest 传递解析结构和 provenance，不把扁平向量冒充 solver 几何。

[↑ 回顶部](#full-cell-forward-inverse-training-plan)

---

<a id="plan-forward"></a>

## 三、正向替代训练

### 3.1 训练矩阵

主分辨率为 `64x64`，seeds 固定为 `[0,1,2]`。

| 路线 | 模型数 | 表征数 | seeds | 训练数 |
|---|---:|---:|---:|---:|
| binary 原生 | 4 | 1 | 3 | 12 |
| graded continuous | 4 | 1 | 3 | 12 |
| graded Bernoulli | 4 | 1 | 3 | 12 |
| 主矩阵 |  |  |  | **36** |

主矩阵完成后，每条表征路线选择一个胜者，在 `32x32` 和 `128x128` 上补训：3 条表征路线 × 2 个额外分辨率 × 3 seeds = 18。正向总计 **54 次训练**。

### 3.2 固定训练协议

- precision：FP32，首轮关闭 AMP；
- optimizer：AdamW，`weight_decay=1e-5`；
- MLP/ResMLP 学习率：`3e-4`；
- CNN/FNO 学习率：`1e-3`；
- 最大 100 epochs，early-stopping patience 15；
- 有效 batch size 64；显存不足时降低 micro-batch，并用梯度累积保持有效 batch；
- `Wh` 只用训练集按 `(k,band)` 标准化；
- 训练损失为标准化 `Wh` 全矩阵 MSE；
- checkpoint 选择指标为验证集 bands 2–10 MRE，band 1 只作诊断。

输出形状来自数据集 `Wh` metadata，禁止在模型、loss、metrics 和 plot 中硬编码 `64`、`81` 或 `10`。

### 3.3 主辅替代模型选择

每条物理路线只从 `continuous/native` 输入模型中选择 inverse ranker；Bernoulli 模型不参与逆向。

1. 主替代模型：三 seed 验证集 bands 2–10 MRE 均值最低者。
2. 辅替代模型：从另一架构组中取验证 MRE 最低者。
   - 向量组：`mlp`、`res_mlp_wide`；
   - 空间组：`lifted_cnn`、`fno2d_band_v2`。
3. 测试集只在架构、超参数和 checkpoint 冻结后使用。
4. binary ↔ graded 交叉评估只作为 OOD 诊断，不参与 promotion。

[↑ 回顶部](#full-cell-forward-inverse-training-plan)

---

<a id="plan-inverse"></a>

## 四、解析参数逆向设计

### 4.1 方法矩阵

五个结构变体分别比较：

| 方法 | 候选合同 | 是否训练 |
|---|---|---|
| `train_only_nearest_neighbor` | 从 train library 返回 128 个唯一解析结构 | 否 |
| `direct_inverse_mlp` | 每个变体输出一个候选 | 是 |
| `single_mode_generator` | `mode_count=1`，每目标合计 K=128 | 是 |
| `mixture_conditional_generator` | `mode_count=4`，每目标合计 K=128 | 是 |

三个可训练方法 × 5 个变体 × 3 seeds = **45 次逆向训练**；与正向合计为 **99 次神经网络训练**。生成器沿用“条件编码器 + mode embedding + latent noise + MLP decoder”，不得称作 cVAE、flow 或 diffusion。

每次逆向训练包含：

1. 最多 200 epochs 的参数监督预训练，patience 25；
2. 最多 50 epochs 的冻结双替代模型微调，patience 10；
3. AdamW、batch 64；直接模型学习率 `3e-4`，生成器学习率 `1e-4`。

解析结构提供可微软栅格化以支持 surrogate gradient；二值边界训练时采用退火软阈值，导出、筛选和 MPB replay 始终使用精确解析结构。

### 4.2 候选配额与健康门

- 每个目标、每个生成方法取 `K=128`。
- binary 三变体配额为 `43/43/42`；graded 两变体为 `64/64`。
- 必须检查解析参数范围、结构统计、有限值、geometry hash 唯一性和精确 raster 合法性。
- 正式门限：
  - `validity_rate=1.0`；
  - `effective_k_mean>=64`；
  - `effective_k_p10>=32`；
  - 四模态覆盖率为 1；
  - 主辅 surrogate 预测差异均值 `<=0.02`。
- 通过门限后，按主辅 surrogate MRE 均值排序，并用 raster 距离贪心去重后选择 Top-5。
- direct 模型每个变体只有一个候选，单独报告 `K=1/variant`，不得包装成 best-of-128。
- graded inverse 只生成连续 `graded_rbf_fourier` 参数；Bernoulli 表征不进入 inverse。

### 4.3 real-MPB replay 阶梯

目标 manifest 必须在运行前冻结并记录 SHA-256：

| 阶段 | split | 每路线目标数 | seeds | 方法 |
|---|---|---:|---|---|
| Smoke | val | 8 | 0 | 全部四种方法 |
| Pilot | val | 20 | 0 | 全部四种方法 |
| Formal | test | 100 | 0/1/2 | retrieval、direct、pilot 胜出生成器 |

Formal 的 100 个目标按结构变体及正向替代难度分层抽取，test 不得用于调参。跨方法、跨 seed 使用 `geometry_hash` 全局去重；理论 MPB 求解上限约 6,480 个结构，实际只重放唯一结构。

solver 协议固定为全布里渊区 81 个 k 点、TE、10 bands、resolution 64、mesh 3、guard bands 4、eigensolver tolerance `1e-7`。

必须分别报告：

- `best_of_k_surrogate_mre`；
- `best_of_k_replay_mre`；
- `surrogate_replay_gap`；
- bands 2–10 MRE、success@5%、solver success rate；
- 主辅 surrogate disagreement、rank correlation；
- candidate sampling、ranking、MPB replay 和 wall-clock runtime。

只有当生成器在每路线 100 targets、三 seed 上相对最强基线的配对 bootstrap 95% CI 完全低于 0，同时 solver success 不下降且 cross-surrogate gate 通过，才允许 promotion；否则保留基线并报告未晋级。

[↑ 回顶部](#full-cell-forward-inverse-training-plan)

---

<a id="plan-resources"></a>

## 五、本机、租用 GPU 与 M9 超算编排

### 5.1 环境职责

| 环境 | 职责 | 边界 |
|---|---|---|
| 本机 RTX 4060 Laptop 8 GB | 开发、单 batch/单 epoch smoke、WSL 测试、报告汇总 | 不承担 99 次正式训练；local smoke 不属于正式结果 |
| 同型号租用 GPU ×3 | 正向/逆向训练、候选生成、双 surrogate 排序 | 一卡固定一个 seed；卡内串行、三 seed 并行；不跨机器 DDP |
| M9 CPU 超算 | 批量 real-MPB replay | 沿用已验证的 `amd_m9_768` profile，提交前重新核验配额和环境 |
| 本机 WSL `Ubuntu` + `meep_env` | 数据验收、M9 跨环境复核、项目测试 | M9 结果在 WSL 复核前保持 pending acceptance |

### 5.2 GPU 选择规则

- RTX 5090 为 32 GB GDDR7，RTX 4090 为 24 GB GDDR6X。若 5090 实时报价不超过 4090 的 1.3 倍，并通过 Blackwell 环境预检，则租 3×5090；否则租 3×4090。参见 [NVIDIA RTX 5090](https://www.nvidia.com/en-eu/geforce/graphics-cards/50-series/rtx-5090/) 与 [NVIDIA RTX 4090](https://www.nvidia.com/en-gb/geforce/graphics-cards/40-series/rtx-4090/)。
- 5090 镜像必须实际完成最大模型 forward/backward。Blackwell 工具链和 PyTorch/CUDA 组合必须以租用当天预检为准，不能只依据平台标签。
- RTX 3090 仅在小时价不高于 4090 的约 55%，且实测每个完成任务的成本更低时采用。
- A100/H100 首轮不选；只有 `128x128` 最大任务在 24/32 GB 上无法达到合理吞吐，或数据中心卡实测总成本更低时升级。

租用前必须在候选卡上运行同一 15 分钟 preflight，记录环境版本、最大模型完整 step、峰值显存、samples/s、checkpoint 写入和恢复。三台机器必须使用相同容器/依赖锁和代码 commit。

### 5.3 M9 提交与交接

- 默认资源基线：`amd_m9_768`、256 CPU、96 GiB、24 h；正式提交前先用小批候选重新测量吞吐和内存。
- 上传包只包含冻结配置、代码 commit、候选 manifest 和 SHA-256，不混入本机 run 垃圾或其他环境数据。
- 每个候选写原子 shard；完成后合并并下载 `results/metrics/history/provenance`，本机逐项复核 hash。
- 任何租卡或 M9 `sbatch` 都是计费动作：只完成 preflight 不自动提交；每个计费波次必须取得明确授权，不自动取消、重启或续跑。

### 5.4 固定执行波次

1. 本机完成渐进数据 WSL acceptance，同时完成云 GPU image preflight。
2. 三张 GPU 按 seed 并行完成 36 个主正向任务。
3. 冻结主辅 surrogate，再完成 18 个 resolution ablation。
4. 三张 GPU 按 seed 并行完成 45 个 inverse 任务。
5. seed 0 candidate manifest 冻结后，可在授权下让 M9 跑 smoke/pilot，同时 GPU 继续其他 seeds。
6. pilot 通过后再次取得授权，提交 Formal M9 replay。
7. 每路线固定抽取 8 个 M9 replay candidate 在 WSL 重放；要求零失败、跨环境 median MRE `<=0.5%`、p95 `<=1%`，通过后才形成正式结论。

不得跨站点做 DDP。所有环境仅通过只读数据包、manifest、hash 和原子 shard 异步交接。

[↑ 回顶部](#full-cell-forward-inverse-training-plan)

---

<a id="plan-validation"></a>

## 六、测试、产物与人工验收

### 6.1 自动测试

- 表征测试：任意 metadata 尺寸的 flatten/unflatten 恒等；Bernoulli 在 $\rho=0/1$ 时确定，在 $\rho=0.5$ 的多 seed 统计接近 50%；固定 validation draw bank 可复现。
- 参数测试：五个变体完成 encode/decode、常量恢复、geometry hash 和 raster round-trip；非法参数必须拒绝。
- 梯度测试：五类软栅格化梯度有限，二值软表示退火后接近精确 raster。
- 集成测试：tiny HDF5 完成四个正向模型一轮训练、三个 inverse 模型一轮训练、K=128 采样、Top-5、去重 manifest 和最小 replay 链路。
- WSL 工程门：相关 `compileall`、`pytest`、CLI help 和严格 docs audit 通过。

### 6.2 运行产物

每个 run 必须遵循 `runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/`，至少包含：

```text
run.log
run.meta.json
config.resolved.yaml
results.json
metrics.json
history.csv
figures/
```

checkpoint、normalization、dataset/split hash、模型身份、训练/推理时间、峰值显存和候选 provenance 必须进入 run 制品，但不得直接提交到 Git。

### 6.3 人工理解记录

每个 substantive gate 完成后，按 [协作治理](../../collaboration.md) 记录：

```text
解决了什么问题：
为什么采用这个方案：
改变了项目哪个层级和模块：
使用什么证据验证：
最大风险、未知或下一 gate：
```

模型盘点最终必须重新对齐 `phc2d models list`、[模型 inventory](../../reports/phc2d_model_inventory_v1.yaml) 和非注册 inverse 方法。模型注册、实现、训练、formal validation、historical evidence、placeholder 和 blocked 状态必须分列，不得互相替代。

[↑ 回顶部](#full-cell-forward-inverse-training-plan)
