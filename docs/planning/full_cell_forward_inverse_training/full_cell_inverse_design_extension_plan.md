---
title: "全胞解析参数逆向设计扩展计划"
summary: "研究计划或执行协议；实施前需与当前代码和证据重新核对。"
page_type: plan
status: draft
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 31 minutes
primary_layer: L0
plan_state: active
effective_date: 2026-08-06
superseded_by: null
related_results:
  - docs/results/inverse/full_cell_inverse_stage_a_cpu_preflight_20260806.md
---

<!--
  Deferred analytic-parameter inverse-design extension after full-cell forward training.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="full-cell-inverse-design-extension-plan"></a>

# 全胞解析参数逆向设计扩展计划

| Metadata | Value |
|---|---|
| Status | Draft；Stage-A codec/preflight/epoch runner 已实现，正向 surrogate、N40 正式训练、Stage B 与 inverse replay 均未开始 |
| Primary layer | L3 experiment/evidence；L4 仅实现本文列出的 codec、训练与 replay 接口 |
| Authority | [生成网络调整研究计划](../fno_generative_network_adjustment_plan_20260617.md) |
| Prerequisite | [全胞双路线训练与混合算力计划](full_cell_dual_lane_hybrid_compute_plan.md) |
| Evidence snapshot | 2026-08-06，正式执行前必须刷新环境与 checkpoint 状态 |
| Audience | 逆向设计研究人员、GPU/超算执行人员与项目负责人 |
| Owner | 研究生维护，导师审阅 inverse promotion |
| Last reviewed | Pending human review |
| Reading time | 22 minutes |

本文规定 **解析参数 inverse Stage-A 监督训练 → 冻结正向 surrogate 的 Stage-B objective bridge → candidate health → 双 surrogate 筛选 → real-MPB replay** 协议。Stage A 直接使用 HDF5 中的真实 `Wh → analytic parameters` 配对，不依赖已经训练好的正向 surrogate，因此可与 102 个 Full-Wh 前向 run 分开授权、并行推进；Stage B、候选 surrogate 排序和 inverse MPB replay 则必须等正向主/辅 surrogate 按三 seed 训练、验收并冻结后才可进入。

本次明确采用 **每个解析变体 × 每个训练方法 × 每个 seed 一个独立单卡作业**。MPB replay 采用“每路线 8-target smoke → 整个 validation split → 整个 test split”三级边界，取消原来的 20-target 中间 pilot。完整 test 保留全部预注册模型/方法，而不是只回放 validation 胜出 arm；validation 仍负责冻结 primary comparison，test 不得用于重新选模或调参。

**目录：** [一、当前状态](#inverse-current-status) · [二、准入条件](#inverse-admission) · [三、参数 codec](#inverse-codec) · [四、方法与训练](#inverse-training) · [五、GPU 编排](#inverse-gpu) · [六、候选与筛选](#inverse-candidates) · [七、MPB replay](#inverse-replay) · [八、制品](#inverse-artifacts) · [九、授权与结论](#inverse-authorization) · [十、人工理解记录](#inverse-human-record)

---

<a id="inverse-current-status"></a>

## 一、当前状态与证据边界

| 项目 | 当前状态 | 可得出的结论 |
|---|---|---|
| binary/graded full-cell 10k HDF5 | **ready**；均为 `[10000,81,10]`，split 均为 `8000/1000/1000`，WSL acceptance 已通过 | 数据可以进入 inverse 实现与训练准备 |
| full-cell forward surrogate | **not trained**；截至 2026-08-06 没有 102 个正式训练 run、三 seed 冻结 checkpoint 或 validation 排名 | Stage A 可先做；Stage B、候选排序和 inverse MPB replay 均 blocked |
| 解析参数 | **data ready**；HDF5 含 `params`、`param_mask`、`family_id` 和 `structure_json` | 可以构建变体专用 codec，但不能直接把 padding 后数组当作统一输出 |
| 通用 inverse runner | **部分可复用**；已有 retrieval、direct、single-mode、mixture、candidate health 和 replay 基础设施 | 当前只接受 `rho_wedge/rho_grid`，不能直接训练 full-cell 解析参数 |
| full-cell codec / 可微解析解码器 | **core implemented**；五变体 hard/soft codec、真实 10k train-only 宽度检查与专用 Stage-A epoch runner 已通过实现测试 | Stage B、candidate/ranker 和 replay 尚未接入；N40 gate 前不得提交 45 个正式训练 run |
| full-cell Stage-A implementation | **runner/orchestrator implemented / CPU smoke passed**；真实 10k 上 15/15 single-batch 合同通过，小数据 WSL epoch/resume 与 18-task 波次 dry-run 通过 | N40 batch-64 单卡/并发 preflight 与任何正式 epoch 训练均未执行 |
| N40 GPU | **环境 ready**；三卡分配与单卡四模型预检已通过 | 可做 full-cell inverse 单卡与并发 preflight；尚无 inverse 正式 run |
| N50R5 | **blocked**；`loginCluster` 权限仍被拒绝 | 不计入可用 GPU，也不能与 N40 混合 seed |
| M9 MPB | **static handoff ready**；binary/graded 10k 与 N40 hash 一致，代码同为 `b93cc2d`，既有 packed MPB 环境已保留 | 仍须经授权的计算节点 environment preflight；当前没有合法 candidate manifest 或本计划的新 replay 作业 |
| full-cell inverse 8-target smoke | **not run** | 不能把历史 V2 inverse replay 或数据集 WSL acceptance 当作本计划 smoke |

此前完成的“8 个结构”是 graded 10k 数据的 WSL 跨环境验收，用于确认集群生成的原始标签可复现。它没有生成 inverse candidates，也没有比较 `MPB(candidate)` 与目标 `Wh`，因此 **不等于本计划的 8-target inverse smoke**。历史 A2/V2 的 smoke、20-target 或 test625 replay 使用不同数据、参数空间与 checkpoint，同样只能作为方法经验，不能顶替 full-cell gate。

[↑ 回顶部](#full-cell-inverse-design-extension-plan)

---

<a id="inverse-admission"></a>

## 二、执行准入条件

各阶段按下列门禁进入，不能用后续依赖反向阻塞本来独立的 Stage A，也不能用 Stage-A 成功越过正向模型门：

| 阶段 | 必须已经满足 | 当前状态 |
|---|---|---|
| Stage-A 正式训练 | 双路线 HDF5/split/哈希冻结；codec hard/soft/round-trip 通过；N40 单卡 batch-64 preflight 通过；GPU training 单独授权 | **blocked on N40 preflight authorization**；不依赖 forward checkpoint |
| Stage-B objective bridge | 对应路线 Stage-A best checkpoint 完整；binary/graded 主、辅 Full-Wh surrogate 三 seed validation 冻结；checkpoint/config/SHA-256 齐全；Stage B 单独授权 | **blocked: forward surrogate not trained** |
| Candidate generation/ranking | Stage-B checkpoint 过 candidate-health；主/辅 ranker 身份冻结；threshold proxy 不进入 objective/ranker/物理结论 | **blocked by Stage B and forward** |
| 8-target inverse smoke | candidate manifest 冻结；M9 环境再次预检；预算核算并取得 M9 smoke 授权 | **not run** |
| Full validation / full test-1000 | 前一级 replay 成功、失败账本完整、费用重新核算并逐级授权 | **not admitted** |

两套数据继续使用各自冻结的 `8000/1000/1000` split；test split 不参与 codec 统计、Stage A/B 训练、checkpoint 选择或 validation replay。单卡训练 preflight 与 18 个独立单卡任务的并发 admission/共享 HDF5 探针也必须分别留证。

主、辅 surrogate 都使用各自三个 seed checkpoint 的预测均值；三个 seed 的预测标准差作为不确定性诊断。任何一个 seed 缺失时，不得把两 seed ensemble 写成正式三 seed ranker。

[↑ 回顶部](#full-cell-inverse-design-extension-plan)

---

<a id="inverse-codec"></a>

## 三、解析参数 codec 与可微解码

### 3.1 五个变体的真实输出合同

HDF5 的 `params` 是按全数据最大宽度 padding 的原始数值表，不是网络应直接学习的统一向量。正式实现按变体拆分：

> inverse 网络应该输出什么，以及如何把网络输出安全地还原成一个真正可计算的 full-cell 结构。

### 为什么不能直接预测 HDF5 的 `params`？

HDF5 为了把不同结构放进一个规则数组，对参数进行了 padding。

例如 binary 文件里的数组是：

```text
params.shape = (10000, 580)
```

但这不代表每个 binary 样本都有 580 个需要预测的自由参数。

例如：

```text
binary_polar_holes_1
HDF5 实际字段：16 个
真正学习：11 个
```

剩下的 5 个字段是：

```text
hole_count
4 个 mode order
```

它们描述结构模板，不是连续的设计自由度。

因此不能让网络统一输出 580 个数：

```text
错误思路：

Wh → 网络 → 580 个参数
```

因为对 `binary_polar_holes_1` 来说，大部分位置根本没有意义；不同结构族中，同一个数组位置也未必具有同样的物理含义。

正确方案是每种变体单独建立 codec 和模型输出：

```text
binary_polar_holes_1       → 学习 11 个物理标量
binary_polar_holes_2       → 学习 22 个物理标量
binary_spectral_threshold  → 学习 289 个物理标量
graded_rbf_fourier_1       → 学习 57 个物理标量
graded_rbf_fourier_2       → 学习 63 个物理标量
```

| 变体 | 样本数 | HDF5 原宽度 | 固定/辅助字段 | 学习的几何标量 |
|---|---:|---:|---:|---:|
| `binary_polar_holes_1` | 2,500 | 16 | 5：`hole_count` 与 4 个 mode order | 11 |
| `binary_polar_holes_2` | 2,500 | 31 | 9：`hole_count` 与 8 个 mode order | 22 |
| `binary_spectral_threshold` | 5,000 | 580 | 291：288 个固定 `(mx,my)` 与 3 个生成辅助字段 | 289：288 个 cos/sin 系数与 threshold |
| `graded_rbf_fourier_1` | 5,000 | 106 | 49：48 个固定 `(mx,my)` 与 `target_material_fraction` 辅助字段 | 57 |
| `graded_rbf_fourier_2` | 5,000 | 112 | 49：48 个固定 `(mx,my)` 与 `target_material_fraction` 辅助字段 | 63 |

“学习的几何标量”是物理参数数目，不等于经过周期编码后的网络张量宽度。`target_material_fraction`、`target_air_fraction`、`threshold_quantile` 和 `cutoff_scale` 只作生成 provenance/诊断；候选解码后从 hard raster 重算可重算字段，不把它们作为独立几何自由度。

### 3.2 编码、约束与解码规则

- `Wh` 条件保留完整 `[81,10]`，按对应路线 train split 的逐 `(k,band)` mean/std 标准化后展平为 810 维；headline objective 与排名只使用 bands 2–10。
- 所有参数 transform 只从该变体 train split 拟合，并写入 `codec.resolved.json`；val/test 不参与 min/max、分位数或 normalization。
- 正值参数（radius、sigma、temperature、幅度绝对尺度）在受限区间内用 log/logit 变换；中心坐标按周期单胞编码；angle 用归一化 `(sin,cos)` 对编码。
- Fourier cos/sin 系数使用 train-only robust scale；hard decode 后必须通过冻结的物理范围和 morphology gate，不允许用无界输出直接进入 MPB。
- 固定 `hole_count`、mode order 和 `(mx,my)` 从变体模板恢复，不由神经网络预测。
- hard decoder 必须重建可序列化的 `FullCellStructureSpec`，保存 `variant_id`、parameter names/vector、codec version、geometry hash 和 `64/128` raster hash。
- binary objective 微调使用可微 soft raster：polar holes 对有符号边界距离使用退火 sigmoid，spectral threshold 使用 `sigmoid((field-threshold)/tau)`；candidate freeze 和 MPB 一律使用 hard analytic decoder，不能把 soft raster 当作物理几何。
- graded decoder 直接使用解析 RBF/Fourier/logistic 公式；不得用预测的 `64x64` raster 反推参数或替代解析材料函数。

### 3.3 codec 实现门

每个变体至少验证：

1. 32 个 train/val 结构执行 `encode → decode → encode`，活动参数误差在浮点容差内；
2. 原始与重建结构的 `geometry_hash`、`64/128` raster 和周期边界一致；
3. 所有固定字段逐项相同，辅助字段的重算规则可追溯；
4. soft decoder 对活动参数梯度有限且非零，hard decoder 输出全部合法；
5. 随机越界网络输出被 fail-closed 拒绝或映射回冻结物理区间；
6. full-cell 专用配置必须显式声明 `output_repr=full_cell_analytic_params`；不得伪装成 `rho_grid` 或静默落回旧 `GenerativeInverseConfig` 路线。

截至 2026-08-06，以上六项的 codec/表示核心、独立 Stage-A 单 batch CPU 合同和可断点续训 epoch runner 的 WSL 小数据测试已通过；证据见 [full-cell inverse codec 预检](../../results/inverse/full_cell_inverse_codec_preflight_20260806.md)、[Stage-A CPU 预检](../../results/inverse/full_cell_inverse_stage_a_cpu_preflight_20260806.md) 和 [Stage-A runner smoke](../../results/inverse/full_cell_inverse_stage_a_runner_smoke_20260806.md)。N40 单卡与正式 validation checkpoint 选择仍未执行，因此尚未达到 GPU training admission。

[↑ 回顶部](#full-cell-inverse-design-extension-plan)

---

<a id="inverse-training"></a>

## 四、方法、网络与训练协议

### 4.1 比较方法

| 方法 | 训练 | 输出与候选角色 |
|---|---|---|
| `train_only_nearest_neighbor` | 无 | 在同变体 train library 中按标准化 `Wh` bands 2–10 距离检索，`K=1`；CPU baseline |
| `direct_inverse_mlp` | 有 | 确定性 `Wh → analytic parameters`，每目标一个候选；一对一 baseline |
| `single_mode_generator` | 有 | `mixture_conditional_generator` 且 `mode_count=1`，latent one-to-many；每目标 `K=128` |
| `mixture_conditional_generator` | 有 | `mode_count=4`、mode embedding、round-robin modes 与 latent noise；每目标 `K=128` |

生成器是项目原生 mode-conditioned latent MLP，不得称作 cVAE、flow 或 diffusion。Field/PINO 当前仍未获 inverse admission，不进入 loss、rerank 或 replay。

### 4.2 冻结网络参数

| 项目 | Direct | Single-mode | Mixture |
|---|---:|---:|---:|
| condition width | 810 | 810 | 810 |
| hidden width | 512 | 512 | 512 |
| hidden depth | 固定两层 GELU | 4 层 GELU | 4 层 GELU |
| latent width | 0 | 32 | 32 |
| mode count | 1 | 1 | 4 |
| output | 变体 codec tensor | 变体 codec tensor | 变体 codec tensor |

每个变体独立训练，不把 11/22/289/57/63 维输出 padding 成一个混合模型，也不让 `family_id` 由网络猜测。跨变体能力通过候选池合并和 replay 比较，而不是一个不透明的多头输出。

### 4.3 两阶段训练

每个 trainable `(variant, method, seed)` 在同一标准 run 内依次执行：

**Stage A：参数监督预训练**

- 最多 200 epochs；checkpoint epochs `50/100/150/200`；early-stopping patience 20；
- Direct：AdamW，learning rate `3e-4`；Single/Mixture：AdamW，learning rate `1e-4`；
- `weight_decay=1e-5`、有效 batch size 64、FP32、AMP/TF32 关闭；
- 主损失为 codec-space reconstruction，加 decoded raster 一致性；角度/周期中心使用周期距离，不用裸欧氏差；
- checkpoint 只依据 val composite reconstruction/raster loss 选择，test 不可见。

**Stage B：冻结双 surrogate objective bridge**

- 当前正向替代模型尚未训练，故 Stage B 的实现/执行门均保持关闭；Stage-A checkpoint 不得被写成 Stage-B ready。
- 从 Stage-A best checkpoint warm start，最多 50 epochs；checkpoint epochs `10/30/50`；early-stopping patience 10；learning rate `1e-4`；
- 每个条件在训练时生成 16 个 objective candidates；主、辅 Full-Wh surrogate 均冻结，只允许梯度穿过 surrogate 和 soft analytic decoder 回到 generator；
- checkpoint 先过 candidate-health 硬门，再按 val primary bands 2–10 MRE 选择；不能按 test 或 MPB replay 反向选 checkpoint。

默认 objective 权重为：

| 分量 | Direct | Single/Mixture |
|---|---:|---:|
| primary surrogate bands 2–10 | 1.0 | 1.0 |
| cross-surrogate consistency | 0.2 | 0.2 |
| bandgap descriptor | 0.2 | 0.2 |
| parameter/raster reconstruction | 0.03 | 0.03 |
| continuous diversity | 0 | 0.2 |
| binary raster diversity | 0 | binary 变体为 0.5；graded 为 0 |
| latent response | 0 | 0.25 |
| collision penalty | 0 | 0.9 |

这些值是从当前 A2 objective-bridge 合同迁移的 **预注册起点**，不是已经在 full-cell 上验证过的最优权重。若 codec preflight 暴露量纲失衡，只能在固定 validation 子集上形成一个新版本配置，并完整重跑三 seed；不得按 replay/test 临时调权。

### 4.4 训练 run 数

```text
5 variants × 3 trainable methods × 3 seeds = 45 GPU training runs
```

Retrieval 不计训练 run。Stage A/B 在一个 run 内分阶段记录，不拆成 90 个独立正式 run；任何恢复必须复用同一 run id、核对 checkpoint/config hash，并保留 resumed provenance。

[↑ 回顶部](#full-cell-inverse-design-extension-plan)

---

<a id="inverse-gpu"></a>

## 五、GPU 资源与并发编排

### 5.1 单变体 9 卡映射

每个变体同时运行 9 个独立单卡作业：

| 逻辑槽 | 方法 | seeds | 作业数 |
|---|---|---|---:|
| GPU-00～GPU-02 | `direct_inverse_mlp` | 0、1、2 | 3 |
| GPU-03～GPU-05 | `single_mode_generator` | 0、1、2 | 3 |
| GPU-06～GPU-08 | `mixture_conditional_generator` | 0、1、2 | 3 |

每个 Slurm 作业只申请 1 张 N40 RTX4090，并在 allocation 内使用 `cuda:0`。不使用 DDP，不让多个 seed 共享一个进程，也不让 N40/N50 混合构成同一三 seed 结果。

### 5.2 三轮训练调度

在 18-job 并发探针通过后，推荐：

| 轮次 | 同时训练的变体 | GPU 上限 | training runs |
|---|---|---:|---:|
| R1 | `binary_polar_holes_1` + `graded_rbf_fourier_1` | 18 | 18 |
| R2 | `binary_polar_holes_2` + `graded_rbf_fourier_2` | 18 | 18 |
| R3 | `binary_spectral_threshold` | 9 | 9 |
| 合计 | 5 variants |  | **45** |

这样每个 HDF5 同时最多承担 9 个训练读取任务。若账号 QoS、排队或 HDF5 吞吐不支持 18 并发，则降为单变体 9 卡；仍不稳定时降为每方法三 seed 的 3 卡波次。节流只改变墙钟时间，不改变训练、seed 或选择规则。

### 5.3 正式训练前 GPU gate

1. 每个变体 × 方法完成一轮 batch-64 forward/backward；binary soft decoder 梯度有限非零；
2. Direct、Single、Mixture checkpoint 均可原子写入并精确恢复；
3. 18 个独立单卡短任务实际进入 `RUNNING`/`COMPLETED`，逐项核对 `AllocTRES`，不以 dry-run manifest 或排队数量证明可用卡数；
4. 记录共享 HDF5 的并发吞吐、GPU utilization、峰值显存和启动等待，据此冻结 `max_concurrent_jobs=18|9|3`；
5. 候选生成与 surrogate ranking 继续在 N40 GPU 上执行，可复用相同 18-job 上限，但必须写独立 candidate run；
6. M9 只运行 hard analytic candidates 的 MPB，不参与训练、反向传播或跨站点 DDP。

N40 当前环境为 PyTorch `2.1.2+cu118` / CUDA `11.8`；N50R5 在登录权限和独立 preflight 通过前保持 blocked。环境事实与瞬时队列容量见 [GPU 全链路预检报告](../../results/data-provenance/paracloud_gpu_preflight_20260806.md)。

Stage-A 的 CPU 合同已在真实双路线 10k 数据上完成 15/15 cases；N40 preflight 冻结入口为 [`configs/research/full_cell_inverse_stage_a_n40_preflight_v1.yaml`](https://github.com/fangtoast/phc2d/blob/master/configs/research/full_cell_inverse_stage_a_n40_preflight_v1.yaml) 与 [`jobs/paracloud_gpu_full_cell_inverse_stage_a_preflight.sh`](https://github.com/fangtoast/phc2d/blob/master/jobs/paracloud_gpu_full_cell_inverse_stage_a_preflight.sh)。正式单任务训练入口为 [`configs/research/full_cell_inverse_stage_a_train_n40_v1.yaml`](https://github.com/fangtoast/phc2d/blob/master/configs/research/full_cell_inverse_stage_a_train_n40_v1.yaml) 与 [`jobs/paracloud_gpu_full_cell_inverse_stage_a_train.sh`](https://github.com/fangtoast/phc2d/blob/master/jobs/paracloud_gpu_full_cell_inverse_stage_a_train.sh)，每次只接收一个 variant、method、seed 和稳定 run dir，并校验 dataset/config/codec/code identity 后恢复 optimizer。

波次入口为 [`jobs/paracloud_submit_full_cell_inverse_stage_a_wave.sh`](https://github.com/fangtoast/phc2d/blob/master/jobs/paracloud_submit_full_cell_inverse_stage_a_wave.sh)，支持 1–2 个 variant，即 9/18 个任务，并只允许 `3|9|18` 并发节流；`probe` 使用单 case batch-64 gate，`train` 使用正式 epoch runner。每次提交先验证完整配置与双数据集 SHA-256，再写 `submission_manifest.csv`；[`jobs/paracloud_full_cell_inverse_stage_a_status.sh`](https://github.com/fangtoast/phc2d/blob/master/jobs/paracloud_full_cell_inverse_stage_a_status.sh) 只读汇总 Slurm 状态、artifact status、epoch、validation loss 和 forward gate。`PHC2D_SUBMIT_DRY_RUN=1` 仅生成 manifest，不调用 `sbatch`、不写 active-wave 指针。2026-08-06 的 18-task/9-concurrency dry-run 已通过，但两类 N40 配置仍未在云端执行，不能据 dry-run 或 CPU 结果推断 9/18 卡容量、batch-64 显存、吞吐或正式收敛通过；证据见 [Stage-A wave dry-run](../../results/inverse/full_cell_inverse_stage_a_wave_dryrun_20260806.md)。

[↑ 回顶部](#full-cell-inverse-design-extension-plan)

---

<a id="inverse-candidates"></a>

## 六、候选生成、健康门与筛选

### 6.1 候选预算

- retrieval：每目标、每变体 `K=1`；
- direct：每目标、每变体、每 seed `K=1`；
- single/mixture：每目标、每变体、每 seed `K=128`；Mixture 按 `32/32/32/32` 覆盖四个 mode；
- generator 内部先 hard decode、合法性过滤和 geometry-hash 去重，再进入 surrogate；
- 每个 generator checkpoint 按主 ranker选择 Top-5；retrieval/direct 保留唯一候选；
- MPB 前对同一批任务做全局 geometry-hash 去重与缓存，但每个 target/method/seed 的成员关系必须保留，不能因缓存复用破坏 paired 统计。

### 6.2 Candidate-health 硬门

生成器必须满足：

- `analytic_decode_success_rate=1.0`；
- `validity_rate=1.0`，所有候选通过对应 generator family 的 morphology/材料范围规则；
- `effective_k_mean>=64`、`effective_k_p10>=32`；
- `mode_coverage_fraction=1.0`；
- latent gradient、latent weight change、candidate std 均有限且非零；
- hard-raster exact duplicate、nearest-neighbour distance 和最大 duplicate cluster 单独记录；
- 主辅 surrogate bands 2–10 disagreement mean `<=0.02`；
- candidate lineage、codec version、hard geometry hash 与 raster hash 完整。

Direct/retrieval 因 `K=1` 不适用 effective-K/mode coverage，但必须满足解析解码、几何合法性、provenance 和 solver-ready gate。任一生成器硬门失败时，不得以较低 surrogate MRE 为由进入 MPB。

### 6.3 Ranker 与指标边界

候选排名使用对应路线冻结的三 seed primary surrogate ensemble；辅助 ensemble 只作 consistency gate 与诊断，不把两个模型不加区分地写成“双代理平均”。报告必须分开：

- `best_of_k_surrogate_mre_bands_2_10_pct`；
- `best_of_k_replay_mre_bands_2_10_pct`；
- `surrogate_replay_gap_bands_2_10_pct`；
- candidate sampling、surrogate ranking、MPB replay 和总 wall time；
- legality、diversity、mode/collision 与 solver success。

Real replay 比较 `MPB(decoded_candidate)` 与目标 `Wh`。Inverse 是多解问题，candidate 与原始样本参数或 raster 不相同本身不是失败；结构重建误差只作训练/诊断，不是最终物理成功标准。

[↑ 回顶部](#full-cell-inverse-design-extension-plan)

---

<a id="inverse-replay"></a>

## 七、8-target smoke → full validation → full test-1000

### 7.1 阶段与最大求解预算

“8 targets”按原计划解释为 **每路线 8 个**：binary 8 个、graded 8 个，共 16 个目标。Smoke 使用 seed 0；完整 validation 与完整 test 使用全部三个训练 seed。

本节所有正式 hard-candidate real-MPB 均在 M9 CPU 超算 `amd_m9_768` 分区执行；N40 只负责候选生成和 surrogate ranking，本机 WSL 不承担批量 replay。M9 的 smoke、full-validation 和 full-test 分别使用冻结 manifest、独立授权和独立 run 目录，前一阶段授权不得自动扩展到后一阶段。

截至 2026-08-06，正向 surrogate 尚未训练，因而本节三个 inverse replay 阶段均未获准进入。此前完成的 8 结构 WSL replay 只是 graded 数据集跨环境验收；不能跳过本节 smoke，也不能现在直接扩展到整个 validation。

| 阶段 | split | 目标 | 进入的 arm | hard-MPB 最大候选数（去重前） |
|---|---|---:|---|---:|
| Smoke | val | 每路线 8 | 全变体、全部方法、seed 0 | binary 288 + graded 192 = **480** |
| Full validation | val | 每路线完整 1,000 | 全变体、全部方法、seeds 0/1/2 | binary 102,000 + graded 68,000 = **170,000** |
| Full test | test | 每路线完整 1,000 | 全变体、全部方法、seeds 0/1/2 | binary 102,000 + graded 68,000 = **170,000** |

预算计算规则：seed-0 smoke 中每变体每目标为 `retrieval 1 + direct 1 + single Top-5 + mixture Top-5 = 12`；full validation 和 full test 中均为 `retrieval 1 + direct 3 seeds + single 3×Top-5 + mixture 3×Top-5 = 34`。实际求解数以全局 geometry-hash 去重后的 manifest 为准，但授权前按上限估算 CPU-hours、存储和 wall time。按去重前上限，validation 与 test 合计最多约 **340,000** 次 hard-MPB；两阶段必须分别基于 smoke 实测单候选 runtime、失败率、缓存命中率和存储量重新报价。

原来的 20-target pilot 已取消。Smoke 只验证整条物理链与明显 surrogate exploit；通过后可以直接申请 full-validation replay，但 validation 和 test 两个各 **170,000** 上限的正式 M9 阶段都必须重新报价和单独授权，不能由 smoke 或 validation 授权自动扩张。

仓库要求正式 MPB/Meep 验收经过 WSL `Ubuntu` + conda `meep_env`。为避免对 M9 大规模结果做重复批量计算，本计划把该要求收缩为 **一次性跨环境 solver sentinel**：首次 M9 inverse smoke 前，五个解析变体各冻结一个 hard candidate，在 M9 与本机 WSL 分别求解并核对 geometry/raster hash、81×10 频率和 band order；bands 2–10 的跨环境 median MRE 必须 `<=0.5%`、P95 必须 `<=1%`。只要 decoder、MPB 配置、M9 packed environment、WSL environment 与代码 hash 均未改变，full-validation 和 full-test 不再抽代表子集到 WSL 复算；任一身份变化才使 sentinel 失效并要求重跑。

### 7.2 8-target smoke

两个 smoke manifest 都从各自 `idx_val` 分层冻结：覆盖全部解析变体，并按材料分数、形态复杂度和 forward-surrogate difficulty 选取。Smoke 必须：

1. 使用 hard analytic decoder；soft raster 不得进入 MPB；
2. 所有候选都有 target/method/variant/seed/latent/rank/geometry hash 谱系；
3. solver success 为 1.0，频率有限且 band ordered，失败 ledger 为空；
4. mean absolute surrogate–replay gap 不超过 5 percentage points；
5. 首次 M9 smoke 前完成上述五变体一次性 WSL solver sentinel；后续阶段只审计 sentinel/environment/code hash，不重复批量复算；
6. 只产生 `full_cell_inverse_smoke_pass|failed`，不在 8 targets 上选择胜出模型或宣称优越性。

截至 2026-08-06，该 smoke **尚未执行**。

### 7.3 整个 validation split

Smoke 通过后，为 binary/graded 分别冻结完整 1,000-target validation manifest。Full-validation replay：

- 覆盖全部五个变体、retrieval/direct/single/mixture 和三个训练 seed；
- 使用原子 shard、fail ledger、geometry-hash cache 和可重复 resume；
- target、candidate、ranker、codec、MPB config 和所有 checkpoint hash 在首个 shard 前冻结；
- 只在 validation 上比较 arm、校准 surrogate–replay gap、选择 route-level `(variant, method, checkpoint policy)`；
- 报告 per-target、per-band、per-k、失败案例、runtime 和 paired bootstrap 95% CI；
- 不把 validation replay 写成最终 test 结论，也不根据 replay 结果回头调训练 loss；任何调参都创建新协议版本并从训练阶段重来。

### 7.4 Full test-1000、全模型结果与 promotion

Full validation 为每路线冻结一个 primary generator arm、最强 retrieval/direct baseline 和 primary comparison；同时冻结全部五个变体、全部方法、三个 seed checkpoint、Top-5 预算与 test 分析代码。随后对 binary/graded 未使用的完整 `idx_test` 各 1,000 个目标执行正式 M9 replay。所有预注册 arm 都必须出具 per-target、per-seed 和三 seed 聚合结果，不能只保留 validation 胜出模型。

完整 test 的统计角色必须区分：

- 冻结的 primary generator 对最强 baseline 是唯一 promotion 主检验，报告 paired target difference、bootstrap 95% CI、effect size 与 solver success；
- 其余 variant/method/seed 结果作为预注册 secondary model inventory 完整记录，用于稳定性、失败模式和后续研究，不得根据 test 排名改写 primary arm；
- 多个 secondary paired comparisons 使用 Holm 校正，同时保留未校正 effect/CI；单 seed、单 morphology 与单路线结果均单列，不并入无边界的“全胞普遍优势”；
- test split 不参与 codec fit、训练、checkpoint 选择、候选预算调整、ranker 选择或任何 loss/阈值修改；任一 test 后修改都必须形成新协议版本并使用新的独立证据，不得回写本轮结果。

生成器只有在以下条件同时满足时晋级：

- 相对最强 baseline 的 paired `replay_mre_bands_2_10_pct` 差值 bootstrap 95% CI 完全低于 0；
- solver success 不下降且所有三 seed 完整；
- surrogate–replay gap、candidate health、运行时间和失败分布均完整报告；
- M9 full-test run 的 manifest、solver provenance、hash 与一次性 WSL solver sentinel 身份一致；
- 不把单路线、单 morphology 或 full-test-1000 结果推广为任意 full-cell/OOD 普遍优势。

[↑ 回顶部](#full-cell-inverse-design-extension-plan)

---

<a id="inverse-artifacts"></a>

## 八、训练、候选与 replay 制品

### 8.1 每个训练 run

45 个训练 run 均遵循 `runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/`，至少保存：

- `run.log`、`run.meta.json`、`config.resolved.yaml`；
- `codec.resolved.json` 与 codec/template/hash；
- `history.csv`：逐 epoch stage、train/val reconstruction、raster、primary/cross surrogate、diversity/collision、learning rate 和 epoch time；
- `figures/training_curve.png` 与 `.pdf`：Stage A/B 分区，至少绘制 train/val total loss、Stage-A codec/raster loss、Stage-B primary/cross-surrogate loss，并标出 best/stop epoch；
- `figures/loss_components.png` 与 `.pdf`：按 stage 绘制 bandgap、reconstruction、continuous/binary diversity、latent response 和 collision 等实际启用分量；未启用分量明确标作 not applicable，不画伪零线；
- `figures/runtime_learning_rate.png` 与 `.pdf`：逐 epoch learning rate、epoch wall time 与累计 train time；
- best Stage-A checkpoint、best Stage-B `model.pt` 及 SHA-256；
- `metrics.json`、`results.json` 和 seed/variant/method 身份。

训练曲线的唯一数值源必须是 `history.csv`，图中指标名与 CSV 列名逐项对应。每个 `(variant, method)` 还必须生成 `figures/training_curve_seed_aggregate.png/.pdf`：三个 seed 用浅色单线展示，并只在共同 epoch 区间计算 mean ± std；不得用 early-stopping 后的末值填充，也不得只保留平滑曲线而隐藏原始 epoch 值。图表生成命令、输入 run id 与源文件 SHA-256 写入 figure manifest。

### 8.2 Candidate 与 replay

核心制品包括：

- `candidate_pool_manifest.json`、`candidate_lineage.csv`、`candidate_health.csv`；
- `strategy_candidates.json`、Top-5 selection 与 geometry-hash cache index；
- smoke/full-validation/test target manifests 及 SHA-256；
- atomic MPB shards、failed-case ledger、solver provenance；
- per-target/per-method/per-seed replay metrics、surrogate–replay gap 和 runtime ledger；
- validation 与 full-test 全模型 leaderboard、paired statistics、promotion decision 和代表性成功/失败图；
- `figures/test_model_leaderboard.png/.pdf`：按路线、变体、方法展示三 seed aggregate 与 95% CI，同时保留 seed-level 点；
- `figures/test_error_distribution.png/.pdf`：per-target replay MRE 的 ECDF/箱线或 violin，不用单一均值掩盖长尾；
- `figures/surrogate_replay_calibration.png/.pdf`：surrogate MRE 对 replay MRE、identity line、分方法校准误差与失败点；
- `figures/test_per_band_k.png/.pdf`：bands 2–10 与 81 个 k 点的误差分解；
- `figures/quality_runtime_pareto.png/.pdf`：candidate sampling、ranking、MPB 与总 wall time 对 replay 质量；
- `tables/test_all_models.csv`：每个 `(route, variant, method, seed)` 的 target 数、solver success、replay/surrogate/gap、runtime 与 CI，作为图和结论的共同数值源。

不要求长期保存 per-batch 日志、逐步 GPU telemetry、profiler traces、中间 activation/gradient、每个 epoch checkpoint、未入 Top-5 的完整预测张量、重复 raster/HDF5 副本或重复 solver stdout。它们可用于临时调试，但不能替代核心制品，也不得进入 Git。

[↑ 回顶部](#full-cell-inverse-design-extension-plan)

---

<a id="inverse-authorization"></a>

## 九、授权、失败与结论边界

本文保持 Draft。以下阶段互不授权：

1. codec/L4 实现与 WSL 单元测试；
2. N40 单卡及 18-job 并发 preflight；
3. 45 个 inverse Stage-A GPU training runs；
4. 102 个 full-cell forward run 的训练、validation 排名与主/辅 surrogate 冻结；
5. inverse Stage B、validation candidate generation 与 candidate-health；
6. M9 每路线 8-target inverse smoke；
7. M9 整个 validation split replay；
8. M9 整个 test split（每路线 1,000、全模型）正式 replay。

任一阶段失败时保留既有成功产物并停止在当前 gate；不得自动重训、扩大 replay、修改 loss 或替换目标 manifest。前向 102-run 完成也不自动授权 inverse，GPU training 完成也不自动授权 M9。

[↑ 回顶部](#full-cell-inverse-design-extension-plan)

---

<a id="inverse-human-record"></a>

## 十、本次调整的人工理解记录

```text
解决了什么问题：明确 full-cell inverse 的五类参数 codec、三种训练方法、两阶段损失、45 个单卡训练 run、逐 epoch/逐分量/三 seed 训练曲线，以及 8-target smoke 到完整 validation 和完整 test-1000 的全模型 replay 路线；同时把 Stage A 与尚未训练的正向 surrogate 解耦。
为什么采用这个方案：解析参数维度和固定字段因变体而异，必须分变体训练；Stage A 只需真实配对标签，Stage B/ranking 才依赖冻结 forward；完整 test 为每个预注册模型保留独立物理结果，但仍由 validation 冻结 primary comparison，避免 test 选模。N40 负责训练/筛选，M9 负责批量 hard-candidate real-MPB，WSL 只保留首次 campaign 前的一次性五变体 solver sentinel。
改变了项目哪一层/哪一模块：主要改变 L3 实验/证据协议，并完成 L4 Stage-A epoch runner、断点续训、训练曲线、单 case preflight、N40 波次 manifest/节流和只读状态入口；Stage B 和 replay 接口仍未实现。
使用什么证据验证：当前 HDF5 schema/参数宽度、真实 10k CPU preflight、小数据 WSL epoch/resume 测试、18-task/9-concurrency 零提交 dry-run、历史 A2 方法与 replay 边界、N40/M9 环境预检，以及仓库的 WSL/制品治理规则。
最大风险、未知或下一 gate：正向 102-run 尚未训练；inverse N40 单卡/18-job 实际 admission、45 个 Stage-A 正式 run、Stage B 和 8-target inverse smoke 均未执行；dry-run 不证明账号 QoS 或实际卡数。full-validation 与 full-test replay 去重前各约 170,000 个 hard candidates，必须分别依据 smoke 实测 runtime/缓存命中重新核算 M9 费用并人工授权。
```

[↑ 回顶部](#full-cell-inverse-design-extension-plan)
