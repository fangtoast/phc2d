---
title: "全胞正向模型升级代码实现规格"
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
  Draft implementation specification for upgraded full-cell forward models.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="full-cell-forward-model-upgrade-implementation"></a>

# 全胞正向模型升级代码实现规格

| Metadata | Value |
|---|---|
| Status | Draft implementation record / implemented / no_result |
| Primary layer | L4 implementation |
| Authority | [生成网络调整研究计划](../fno_generative_network_adjustment_plan_20260617.md) |
| Audience | 模型实现者、代码审阅者、测试维护者与 AI 协作者 |
| Owner | 研究生维护，导师审阅模型身份与科学边界 |
| Last reviewed | Pending human review |
| Reading time | 24 minutes |

本文既冻结全胞 Full-Wh 网络升级的**代码实现合同**，也记录 2026-08-08 的 L4 实现：新增哪些模型、每个模型如何计算、输入输出如何表示、怎样注册以及代码测试覆盖什么。`implemented` 只表示代码门通过，不证明这些模型优于当前基线；10k/50k 训练矩阵、N40 调度和性能 promotion 见 [10k/50k 训练验证方案](full_cell_forward_model_upgrade_10k_50k_training_validation_plan.md)。

## 一、目标、证据输入与非目标

### 1.1 唯一目标

在不替换现有 `mlp`、`res_mlp_wide`、`lifted_cnn`、`fno2d_band_v2` 的前提下，为五个新增 canonical model 定义可直接实现的网络、接口和测试规格：

- `mlp_mixer_band_v1`
- `res_mlp_gated_v2`
- `convnext_band_v1`
- `tfno2d_band_v3`
- `dual_lane_band_operator_v1`

所有新名称当前均为 `implemented / no_result`。它们已注册并通过代码级 shape、梯度、参数、lane、checkpoint 和 runner 检查，但没有训练 checkpoint、held-out 指标或 performance promotion。

### 1.2 当前证据账本

| 证据 | 当前可支持的判断 | 不能支持的判断 |
|---|---|---|
| [10k 双路线结果](../../results/forward/full_cell_dual_lane_forward_training_20260806.md) | FNO 在七个主波次均为最低三 seed 均值；binary/continuous 冻结 test 分别约为 `0.80394%`、`0.21579%`；跨路线存在明显 domain shift | 不能证明 64 checkpoint 对 32/128 零样本分辨率不变；该报告中的 32/64/128 是独立重训 |
| [50k 双路线结果](../../results/forward/full_cell_dual_lane_50000_forward_training_20260808.md) | FNO 在 12 个方向×epoch 组均为四模型最低三 seed 均值；binary 100 epoch 约 `0.62965%`，continuous 200 epoch 约 `0.16016%` | 不能把 threshold 或 final-fit monitor 当成无偏物理结论；更多 epoch 没有修复跨路线泛化 |
| 当前注册表与 [`baselines.py`](https://github.com/fangtoast/phc2d/blob/master/src/phc2d/models/baselines.py) | 四个基线均可由 shape 合同动态构建 | 代码中出现模型名不等于已有正式训练结果 |

按 64×64 输入和 81×10 输出的当前全胞合同动态构建时，实时参数量为：

| Baseline | Parameters | 实现诊断 |
|---|---:|---|
| `mlp` | 2,046,506 | 容量不低，但深层 block 无归一化、layer scale 或 gating，10k 中 seed 波动明显 |
| `res_mlp_wide` | 2,046,506 | identity residual 明显优于 plain MLP，但仍是固定展平输入 |
| `lifted_cnn` | 11,958 | 只有两层 12-channel 卷积和 GAP，容量远低于其余基线 |
| `fno2d_band_v2` | 406,218 | 当前误差最优，但频谱容量较小，且 64→32/128 零样本输入明显退化 |

参数量取决于数据提供的输入和输出 shape；上述数字只是当前 64×64/81×10 合同下的实现证据，不得硬编码回模型。

### 1.3 非目标

- 不修改或重命名现有四个基线及其历史 checkpoint 身份。
- 不在本文定义新的训练损失、优化器搜索或数据生成协议。
- 不把 `k_conditioned` 输出头写成已经证明可在任意未见物理 k 点上泛化。
- 不把跨 lane 推理、threshold 输入或 final-fit monitor 写成 real-MPB 物理验收。
- 不复制 MLP-Mixer、ResMLP、ConvNeXt 或 NeuralOperator 项目的源代码到 `src/phc2d`。

## 二、共享数据与公共接口合同

### 2.1 Shape 与语义

| 名称 | Shape | 来源 | 语义 |
|---|---|---|---|
| `x_image` | `[B,C,H,W]` | HDF5 grid field + representation spec | 周期材料权重或等价图像表征 |
| `x_flat` | `[B,C*H*W]` | runner 按 `input_layout=flat` 展开 | 固定长度 ResMLP 输入 |
| `k_points` | `[Nk,2]` | HDF5 `k_points` / target metadata | 真实二维波矢坐标，不是 k-index |
| `lane_id` | `[B]` | 双数据集 batch assembler | `0=binary`、`1=graded` |
| `Wh` | `[B,Nk,Nb]` | HDF5 target + `TargetSpec` | 完整能带矩阵 |

`C`、`H`、`W`、`Nk` 和 `Nb` 必须来自 HDF5 schema inspection、`DataRepresentationSpec`、`TargetSpec` 或解析后的配置。模型、runner、loss、metric 和 plot 均不得写死 `16×16`、`31`、`81` 或 `10`。

### 2.2 注册表扩展

未来实现必须向 `ModelSpec` 增加三个向后兼容字段，并为所有现有模型显式赋值：

```python
input_layout: Literal["flat", "image"]
accepts_k_points: bool = False
accepts_lane_id: bool = False
```

五个新模型的固定注册身份如下：

| Model | family | input_layout | variable grid | k points | lane id |
|---|---|---|---|---|---|
| `mlp_mixer_band_v1` | `mixer` | `image` | yes，先自适应到配置 token grid | optional | no |
| `res_mlp_gated_v2` | `resmlp` | `flat` | no | optional | no |
| `convnext_band_v1` | `convnext` | `image` | yes | optional | no |
| `tfno2d_band_v3` | `neural_operator` | `image` | yes | optional | no |
| `dual_lane_band_operator_v1` | `conditional_neural_operator` | `image` | yes | optional | required |

低层 full-cell runner 应从配置列出的 registered model 与 `ModelSpec.input_layout` 解析输入，不再以模型名集合推断 flat/image。历史 `full_cell_wave.MODELS`、旧四模型正式配置和历史聚合器继续冻结，不把新模型悄悄加入旧 12-run 协议。

### 2.3 统一 forward 接口

所有新模型公开相同调用形式：

```python
forward(
    x: Tensor,
    *,
    k_points: Tensor | None = None,
    lane_id: Tensor | None = None,
) -> Tensor  # [B, Nk, Nb]
```

约束如下：

- 普通四模型收到非空 `lane_id` 时抛出 `ValueError`，避免静默忽略条件。
- 联合模型缺少 `lane_id`、出现非法值或 batch 长度不一致时抛出 `ValueError`。
- `head_kind=dense` 时 `k_points` 只用于 shape/provenance 校验，不参与数值计算。
- `head_kind=k_conditioned` 时优先使用调用方传入的 `k_points`；否则使用构造时注册的默认 k-grid buffer；两者都不存在时失败。
- 输出必须保持 `[B,Nk,Nb]`，禁止在 runner 外留下扁平输出。

## 三、共享 Full-Wh 输出头

### 3.1 `dense` 对照头

编码器产生结构向量 `z∈R^D` 后执行：

```text
z [B,D]
  -> LayerNorm(D)
  -> Linear(D, Nk*Nb)
  -> reshape [B,Nk,Nb]
```

`Nk` 和 `Nb` 在构造时由目标 shape 注入。该头维持当前 Full-Wh 固定输出拓扑，是 `k_conditioned` 的必要对照；它不宣称支持运行时改变 k-grid。

### 3.2 `k_conditioned` 查询头

固定默认参数：

| 参数 | 默认值 |
|---|---:|
| `head_hidden_dim` | 256 |
| `band_embedding_dim` | 32 |
| `fourier_frequencies` | `(1,2,4,8)` |
| `head_depth` | 2 |
| `dropout` | 0.05 |

二维 k 坐标编码为：

$$
\phi(\mathbf k)=
[k_x,k_y,
\sin(2\pi f k_x),\cos(2\pi f k_x),
\sin(2\pi f k_y),\cos(2\pi f k_y)]_{f\in\{1,2,4,8\}}.
$$

对每个 `(k,b)`，将 `φ(k)` 与可学习 band embedding `e_b` 拼接并投影为查询向量 `q_{k,b}∈R^{256}`；结构向量另经 `Linear(D,256)` 投影。随后执行：

$$
h_{i,k,b}=\operatorname{LayerNorm}
\left(W_z z_i+q_{k,b}\right),
\qquad
\hat W_{i,k,b}=W_2\operatorname{Dropout}
\left(\operatorname{GELU}(W_1h_{i,k,b})\right).
$$

计算必须在 `Nk×Nb` 维度向量化，不允许 Python 逐 k/逐 band 循环。band 数来自 target spec；k-grid 可以按调用方输入执行重排或子集查询，但在没有新 k 真值前只属于接口能力，不属于泛化证据。

## 四、`mlp_mixer_band_v1`

### 4.1 默认结构

| 参数 | 默认值 |
|---|---:|
| `token_grid` | `(16,16)`，由配置注入而非代码常量 |
| `embed_dim` | 256 |
| `depth` | 8 |
| `token_mlp_dim` | 128 |
| `channel_mlp_dim` | 512 |
| `dropout` | 0.05 |
| `drop_path` | 0.10，随 block 深度线性增加 |
| `layer_scale_init` | `1e-4` |
| `activation` | GELU |

### 4.2 张量流与 block

```text
x [B,C,H,W]
  -> adaptive area pooling to configured Ph×Pw
  -> flatten spatial cells [B,T,C], T=Ph*Pw
  -> Linear(C,256) + learned token-position parameter [1,T,256]
  -> 8 × MixerBlock
  -> LayerNorm
  -> token mean [B,256]
  -> selected Full-Wh head
```

每个 Mixer block 固定为两条 Pre-LN residual：

1. token mixing：转置为 `[B,D,T]`，执行 `Linear(T,128) -> GELU -> Dropout -> Linear(128,T)`；
2. channel mixing：执行 `Linear(256,512) -> GELU -> Dropout -> Linear(512,256)`。

两条 residual 都使用独立 layer-scale 和 stochastic depth。token-position 参数必须随配置 token grid 构造；输入分辨率变化时先通过 area pooling 保持 token 数，不对参数做插值。

### 4.3 身份与边界

该模型是 Full-Wh 的 MLP-Mixer-style 网格模型，不是对原 `mlp` 文件的加深覆盖。它以自适应 tokenization 换取跨输入 shape 的接口兼容，但是否改善 64→32/128 必须由独立实验验证。

## 五、`res_mlp_gated_v2`

### 5.1 默认结构

| 参数 | 默认值 |
|---|---:|
| `hidden_dim` | 320 |
| `depth` | 8 |
| `expansion` | 2 |
| `dropout` | 0.05 |
| `drop_path` | 0.10，随 block 深度线性增加 |
| `layer_scale_init` | `1e-4` |
| `activation` | GELU/GEGLU gate |

### 5.2 张量流与 block

```text
x [B,C*H*W]
  -> Linear(input_dim,320)
  -> GELU
  -> 8 × GatedResidualMLPBlock
  -> LayerNorm(320)
  -> selected Full-Wh head
```

block 的固定计算为：

$$
u=\operatorname{LayerNorm}(h),
\qquad
g=\operatorname{GELU}(W_g u)\odot W_vu,
\qquad
h'=h+\alpha\,\operatorname{DropPath}(W_og),
$$

其中 `W_g` 与 `W_v` 输出 `640` 维，`W_o` 投回 `320` 维，`α` 为每通道 layer-scale。该模型不接受运行时变化的 `H×W`；分辨率必须与 checkpoint 构造时的 `input_shape` 一致。

## 六、`convnext_band_v1`

### 6.1 默认结构

| 参数 | 默认值 |
|---|---|
| `stage_dims` | `(48,96,192)` |
| `stage_depths` | `(2,3,6)` |
| `stem_stride` | 4 |
| `kernel_size` | 7 |
| `channel_expansion` | 4 |
| `drop_path` | 0.10，跨全部 block 线性增加 |
| `layer_scale_init` | `1e-6` |
| `activation` | GELU |

### 6.2 周期卷积数据流

```text
x [B,C,H,W]
  -> periodic stem Conv(C,48,k=4,s=4)
  -> stage 1: 2 × ConvNeXtBlock(48)
  -> anti-aliased periodic downsample -> 96
  -> stage 2: 3 × ConvNeXtBlock(96)
  -> anti-aliased periodic downsample -> 192
  -> stage 3: 6 × ConvNeXtBlock(192)
  -> global mean pool + LayerNorm [B,192]
  -> selected Full-Wh head
```

每个 block 为 `circular pad(3) -> depthwise Conv7×7 -> channels-last LayerNorm -> Linear(D,4D) -> GELU -> Linear(4D,D) -> layer-scale -> drop-path -> residual`。stem 与 downsample 同样执行周期扩展，不允许默认零填充在晶胞边缘引入人工界面。

输入 `H/W` 必须能够通过 stem 和两次下采样形成非空网格；不满足时在 schema admission 阶段失败，不做静默裁剪。

## 七、`tfno2d_band_v3`

### 7.1 默认结构

| 参数 | 默认值 |
|---|---:|
| `width` | 64 |
| `depth` | 6 |
| `modes` | `(12,12)`，由配置注入 |
| `factorization` | Tucker |
| `factorization_rank` | 0.75 |
| `norm_groups` | 8 |
| `channel_expansion` | 2 |
| `local_kernel` | 3，circular padding |
| `activation` | GELU |

### 7.2 双侧频谱与 residual block

```text
x [B,C,H,W]
  -> Conv1×1(C,64) lifting
  -> 6 × FactorizedOperatorBlock
  -> GroupNorm + GELU
  -> global spatial mean [B,64]
  -> selected Full-Wh head
```

每个 operator block 包含：

1. Pre-GroupNorm；
2. `rfft2` 后分别处理第一频轴的正侧和负侧 modes，第二频轴使用有效非负 modes；
3. 复权重使用项目原生 Tucker factors/core 表达，rank 0.75，不能依赖外部 NeuralOperator 运行时；
4. `irfft2(..., s=(H,W))` 明确恢复原 shape；
5. circular 3×3 local bias branch；
6. `1×1 -> GELU -> 1×1` 的 2× channel MLP；
7. 可学习 sigmoid soft gates 融合 spectral/local/channel 三支路，再与 block 输入 residual 相加。

实际 `modes` 不能超过当前 FFT 网格允许范围；实现应裁到有效范围并把解析值写入 run metadata，而不是越界或改变 checkpoint 权重 shape。周期晶胞默认不添加绝对 `(x,y)` 坐标和非周期 domain padding。

### 7.3 模型身份

`tfno2d_band_v3` 是项目原生 Tucker-factorized FNO-style Full-Wh encoder，不得描述为官方 NeuralOperator TFNO 的直接代码复现，也不得在跨分辨率验证前称为 resolution-invariant operator。

## 八、`dual_lane_band_operator_v1`

### 8.1 条件编码

该模型复用 7 节的 width-64、depth-6 TFNO encoder，并增加：

| 参数 | 默认值 |
|---|---:|
| `num_lanes` | 2 |
| `lane_embedding_dim` | 16 |
| `lane_mapping` | `{binary: 0, graded: 1}` |
| `film_location` | 每个 operator block 的 Pre-GroupNorm 之后 |

对 lane embedding `e_l`，每个 block 独立生成通道调制量：

$$
\operatorname{FiLM}(u,e_l)
=
u\odot(1+\gamma_l)+\beta_l,
\qquad
[\gamma_l,\beta_l]=W_l e_l.
$$

`γ/β` shape 为 `[B,64,1,1]`。binary 与 graded 共享 encoder 和输出头参数，但 target normalization 必须分别只由各自 train split 统计，并随 checkpoint 保存；推理时由 `lane_id` 选择对应反归一化参数。

### 8.2 不得形成的错误声明

- 联合训练不等于已经解决 binary↔graded domain shift。
- lane embedding 不把 threshold proxy 变成物理等价数据。
- 一个共享 checkpoint 只有在两条 native lane 都通过独立 test promotion 后才能替代两个独立模型。

## 九、参数预算与初始化

在当前 64×64、81×10 合同下，五个模型的 dense 与 k-conditioned 两种变体都必须满足：

$$
2{,}000{,}000 \le N_{\mathrm{params}} \le 8{,}000{,}000.
$$

预期区间如下；实际数字只能由实现后的自动测试与 `model_summary.json` 给出：

| Model | 设计区间 |
|---|---:|
| `mlp_mixer_band_v1` | 2.8–3.2M |
| `res_mlp_gated_v2` | 6–7M |
| `convnext_band_v1` | 2–3M |
| `tfno2d_band_v3` | 2.5–4M |
| `dual_lane_band_operator_v1` | 3–5M |

Linear/Conv 使用 PyTorch 默认兼容的 truncated-normal 或 Kaiming 初始化；layer-scale 按各模型表中常量初始化；所有 bias 为零。不得为了通过参数测试临时添加未使用参数。

## 十、未来代码落点与兼容边界

### 10.1 允许修改的模块

未来代码任务的主实现文件为：

- `src/phc2d/models/full_cell_upgrades.py`：五个模型、共享 block 与输出头；
- `src/phc2d/models/registry.py`：注册与 `ModelSpec` 元数据；
- `src/phc2d/forward/full_cell.py`：按注册规格解析 layout、k-points 与 lane；
- 独立 upgrade campaign/aggregator；不得改写旧 `full_cell_wave` 的四模型身份。

### 10.2 配置合同

每个新模型配置至少声明：

```yaml
model:
  name: tfno2d_band_v3
  params:
    head_kind: k_conditioned
    head_hidden_dim: 256
    fourier_frequencies: [1, 2, 4, 8]
```

模型专属默认参数来自 registry，但 run 必须在 `config.resolved.yaml` 中写出完整解析值。动态 `input_shape`、`output_shape`、`k_points` 和 lane normalization 由 runner 在 schema inspection 后注入，不能要求用户把真实数组复制进 YAML。

### 10.3 Optional dependency

新模块只能在 factory 实际构建时导入 torch；`phc2d --help`、文档审计和不涉及模型的 CLI 不得因 torch、h5py、MPB 或 Meep 不可用而失败。

## 十一、代码测试流程

测试必须在正式 WSL `Ubuntu` + conda `meep_env` 环境运行。Windows 本地结果只能用于编辑期诊断。

| 测试组 | 必测内容 | 通过条件 |
|---|---|---|
| Registry | 五个名称、family、layout、k/lane flags、defaults | `phc2d models list` 唯一出现一次；旧 canonical name 不变 |
| Shape | 32/64/128 image、flat mismatch、动态 `Nk/Nb` | 合法输入输出 `[B,Nk,Nb]`；非法 shape 明确失败 |
| Heads | dense、k-conditioned、k 重排、k 子集 | dense shape 正确；query 对 k 重排保持等变且无逐点 Python loop |
| Gradient | 每个模型两种 head 的一批 forward/backward | loss 与所有应训练参数梯度均有限 |
| Parameter budget | 64×64/81×10 两种 head | 每个变体都在 2–8M，未使用参数为零 |
| Periodicity | ConvNeXt/TFNO circular roll 与边界 impulse | padding 不引入零边界；频谱输出为实数且 shape 保持 |
| Spectral modes | 正/负第一频轴、mode clipping、Tucker factors | 两侧权重均得到梯度；小网格不越界 |
| Lane | binary/graded FiLM、非法/缺失 lane、双 normalizer | 条件生效；错误输入失败；统计只来自对应 train split |
| Checkpoint | state dict、默认 k-grid buffer、normalizer metadata | 保存/加载后同输入数值一致 |
| Legacy regression | 旧四模型构建、runner probe、旧 wave aggregate | 旧输出 shape、参数量、模型顺序和配置语义不变 |
| Lazy import | 无 torch 环境下基本 CLI/import | 基本命令不加载重依赖 |

计划新增的测试文件应按职责拆成模型单元测试、full-cell runner 集成测试和 N40 submitter 合同测试；不得用一个 GPU smoke 代替上述单元与回归覆盖。

## 十二、实验验证边界

本文到此只定义代码与测试。以下内容全部属于另一份 L3 文档：

- 10k 的 54 个模型×输出头×lane×seed 任务；
- 50k 的 27 个确认任务；
- validation-only 输出头选择；
- 32/128 零样本输入、跨 lane 与 threshold 诊断；
- N40 8×RTX4090 调度、付费授权和正式 promotion。

实现文档通过、代码合入、WSL tests 通过和 N40 admission probe 通过，是训练方案的前置门，不是训练成功。

## 十三、外部来源与代码来源治理

| 参考 | 只借鉴的思想 | 官方来源/许可 |
|---|---|---|
| MLP-Mixer | token/channel 两类 MLP residual | [Google Research repository](https://github.com/google-research/vision_transformer) · [LICENSE](https://github.com/google-research/vision_transformer/blob/main/LICENSE) |
| ResMLP | 深层残差 MLP、layer scale | [Meta DeiT ResMLP implementation](https://github.com/facebookresearch/deit/blob/main/resmlp_models.py) · [LICENSE](https://github.com/facebookresearch/deit/blob/main/LICENSE) |
| ConvNeXt | depthwise 7×7、channels-last LN、4× channel MLP | [Meta ConvNeXt](https://github.com/facebookresearch/ConvNeXt) · [LICENSE](https://github.com/facebookresearch/ConvNeXt/blob/main/LICENSE) |
| FNO/TFNO/UNO | 双侧谱模态、factorization、局部支路和多尺度算子设计 | [FNO](https://neuraloperator.github.io/dev/modules/generated/neuralop.models.FNO.html) · [TFNO](https://neuraloperator.github.io/dev/modules/generated/neuralop.models.TFNO.html) · [UNO](https://neuraloperator.github.io/dev/modules/generated/neuralop.models.UNO.html) · [NeuralOperator LICENSE](https://github.com/neuraloperator/neuraloperator/blob/main/LICENSE) |

若后续下载外部代码或材料，只能进入 `external_references/`，并记录 `SOURCE.md`、许可链接、commit/hash、下载时间和参考模块；禁止直接复制到公共源码。

## 十四、实现完成门

只有同时满足下列条件，后续代码任务才能把模型从 `planned` 提升为 `implemented`：

1. 五个新模型和两种输出头均按本文接口注册并可构建；
2. 现有四模型代码与历史协议未被替换；
3. WSL `meep_env` 相关 unit/integration/regression tests 全部通过；
4. 参数量、shape、k-grid、lane、checkpoint、optional dependency 门全部通过；
5. `phc2d models list`、模型 inventory、项目总介绍和实现状态同步；
6. 代码和文档形成独立、可审阅的 scoped commit。

即使以上全部通过，证据状态仍为 `no_result`，直到 10k/50k 独立 test 产出完成并通过人工审阅。

## 十五、人工理解记录

### 15.1 2026-08-08 实现与代码验收记录

- 新增实现：[`src/phc2d/models/full_cell_upgrades.py`](https://github.com/fangtoast/phc2d/blob/master/src/phc2d/models/full_cell_upgrades.py)，包含五个模型、dense/k-conditioned 输出头、周期 ConvNeXt block、双侧 Tucker 频谱卷积和双 lane FiLM。
- 注册与 runner：`ModelSpec` 已增加 `input_layout`、`accepts_k_points`、`accepts_lane_id`；Full-Wh runner 从注册规格解析 layout，并从 HDF5 注入动态 `k_points` 和 lane id。
- 默认参数量（dense / k-conditioned）：Mixer `2,918,186 / 2,921,025`；gated ResMLP `6,508,010 / 6,475,265`；ConvNeXt `2,298,138 / 2,336,561`；TFNO `2,690,652 / 2,800,243`；dual-lane operator `2,703,740 / 2,813,331`。
- WSL `Ubuntu` + conda `meep_env`：升级模型、registry、runner 与报告审计定向测试 `62 passed`；完整 `tests/` 回归 `431 passed`；`compileall`、基础 CLI 与 22-model registry 检查通过。
- 未执行事项：没有训练、checkpoint、admission probe、Slurm job、MPB/Meep 或新性能结果；10k/50k campaign 仍需独立授权。

因此，本实现可标记为 `implemented / code-tested / no_result`，不能标记为 `trained`、`validated` 或 `promoted`。

- 本文解决的问题：把“加深网络”拆成五个具体、可测试、可注册且不覆盖旧基线的模型实现合同。
- 关键设计：容量统一到约 2–8M；CNN 引入周期 ConvNeXt；FNO 引入双侧 Tucker 频谱；所有编码器都对比 dense 与显式 k-conditioned head；联合模型通过 lane FiLM 共享编码。
- 当前没有解决的问题：没有模型代码、测试结果、checkpoint、10k/50k 指标或 N40 正式训练。
- 人工需审阅：模型名称、默认深度/宽度、输出头公式、参数预算、联合模型的 lane 语义以及是否允许进入代码实现。
