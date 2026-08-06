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
| Status | Deferred Draft；协议已明确，full-cell inverse 实现与执行尚未开始 |
| Primary layer | L3 experiment/evidence；L4 仅实现本文列出的 codec、训练与 replay 接口 |
| Authority | [生成网络调整研究计划](../fno_generative_network_adjustment_plan_20260617.md) |
| Prerequisite | [全胞双路线训练与混合算力计划](full_cell_dual_lane_hybrid_compute_plan.md) |
| Evidence snapshot | 2026-08-06，正式执行前必须刷新环境与 checkpoint 状态 |
| Audience | 逆向设计研究人员、GPU/超算执行人员与项目负责人 |
| Owner | 研究生维护，导师审阅 inverse promotion |
| Last reviewed | Pending human review |
| Reading time | 20 minutes |

本文规定前向双路线完成后的 **解析参数 inverse training → candidate health → 双 surrogate 筛选 → real-MPB replay** 协议。它不属于当前 102 个 Full-Wh 前向训练 run；只有前向 surrogate 按三 seed 冻结、本文实现门通过并取得新的 GPU/M9 授权后，才能执行。

本次明确采用 **每个解析变体 × 每个训练方法 × 每个 seed 一个独立单卡作业**。MPB replay 采用“每路线 8-target smoke → 整个 validation split → 冻结后的 test-100”三级边界，取消原来的 20-target 中间 pilot。

**目录：** [一、当前状态](#inverse-current-status) · [二、准入条件](#inverse-admission) · [三、参数 codec](#inverse-codec) · [四、方法与训练](#inverse-training) · [五、GPU 编排](#inverse-gpu) · [六、候选与筛选](#inverse-candidates) · [七、MPB replay](#inverse-replay) · [八、制品](#inverse-artifacts) · [九、授权与结论](#inverse-authorization) · [十、人工理解记录](#inverse-human-record)

---

<a id="inverse-current-status"></a>

## 一、当前状态与证据边界

| 项目 | 当前状态 | 可得出的结论 |
|---|---|---|
| binary/graded full-cell 10k HDF5 | **ready**；均为 `[10000,81,10]`，split 均为 `8000/1000/1000`，WSL acceptance 已通过 | 数据可以进入 inverse 实现与训练准备 |
| 解析参数 | **data ready**；HDF5 含 `params`、`param_mask`、`family_id` 和 `structure_json` | 可以构建变体专用 codec，但不能直接把 padding 后数组当作统一输出 |
| 通用 inverse runner | **部分可复用**；已有 retrieval、direct、single-mode、mixture、candidate health 和 replay 基础设施 | 当前只接受 `rho_wedge/rho_grid`，不能直接训练 full-cell 解析参数 |
| full-cell codec / 可微解析解码器 | **not implemented** | 在该接口完成前不得提交 45 个正式训练 run |
| N40 GPU | **环境 ready**；三卡分配与单卡四模型预检已通过 | 可做 full-cell inverse 单卡与并发 preflight；尚无 inverse 正式 run |
| N50R5 | **blocked**；`loginCluster` 权限仍被拒绝 | 不计入可用 GPU，也不能与 N40 混合 seed |
| M9 MPB | **环境与队列 ready** | 可在单独授权后执行 replay；当前没有本计划的新 replay 作业 |
| full-cell inverse 8-target smoke | **not run** | 不能把历史 V2 inverse replay 或数据集 WSL acceptance 当作本计划 smoke |

此前完成的“8 个结构”是 graded 10k 数据的 WSL 跨环境验收，用于确认集群生成的原始标签可复现。它没有生成 inverse candidates，也没有比较 `MPB(candidate)` 与目标 `Wh`，因此 **不等于本计划的 8-target inverse smoke**。历史 A2/V2 的 smoke、20-target 或 test625 replay 使用不同数据、参数空间与 checkpoint，同样只能作为方法经验，不能顶替 full-cell gate。

[↑ 回顶部](#full-cell-inverse-design-extension-plan)

---

<a id="inverse-admission"></a>

## 二、执行准入条件

必须同时满足：

1. binary 与 graded continuous 的主、辅 Full-Wh surrogate 已按三 seed validation 冻结，checkpoint、模型配置和 SHA-256 齐全；
2. 主 ranker 是该路线三 seed validation 最优的原生连续/二值 Full-Wh 模型；辅 ranker 选择架构不同且已通过校准门的次优模型；
3. threshold proxy 模型不进入 inverse objective、ranker 或物理结论；
4. 两套数据继续使用各自冻结的 `8000/1000/1000` split，test split 不参与 codec 统计、训练、checkpoint 选择或 validation replay；
5. full-cell codec、hard/soft decoder、geometry hash、周期边界和 `FullCellStructureSpec` round-trip 测试通过；
6. 单卡训练 preflight 与 18 个独立单卡任务的并发 admission/共享 HDF5 探针通过；
7. GPU training、validation candidate generation、M9 smoke、M9 full-validation replay 和 test replay 分阶段单独授权。

主、辅 surrogate 都使用各自三个 seed checkpoint 的预测均值；三个 seed 的预测标准差作为不确定性诊断。任何一个 seed 缺失时，不得把两 seed ensemble 写成正式三 seed ranker。

[↑ 回顶部](#full-cell-inverse-design-extension-plan)

---

<a id="inverse-codec"></a>

## 三、解析参数 codec 与可微解码

### 3.1 五个变体的真实输出合同

HDF5 的 `params` 是按全数据最大宽度 padding 的原始数值表，不是网络应直接学习的统一向量。正式实现按变体拆分：

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
6. 当前 `GenerativeInverseConfig.output_repr` 扩展为明确的 `full_cell_analytic_params`，不得伪装成 `rho_grid`。

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
3. 18 个独立单卡短任务实际进入 `RUNNING`/`COMPLETED`，逐项核对 `AllocTRES`，不以排队数量证明可用卡数；
4. 记录共享 HDF5 的并发吞吐、GPU utilization、峰值显存和启动等待，据此冻结 `max_concurrent_jobs=18|9|3`；
5. 候选生成与 surrogate ranking 继续在 N40 GPU 上执行，可复用相同 18-job 上限，但必须写独立 candidate run；
6. M9 只运行 hard analytic candidates 的 MPB，不参与训练、反向传播或跨站点 DDP。

N40 当前环境为 PyTorch `2.1.2+cu118` / CUDA `11.8`；N50R5 在登录权限和独立 preflight 通过前保持 blocked。环境事实与瞬时队列容量见 [GPU 全链路预检报告](../../results/data-provenance/paracloud_gpu_preflight_20260806.md)。

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

## 七、8-target smoke → full validation → test-100

### 7.1 阶段与最大求解预算

“8 targets”按原计划解释为 **每路线 8 个**：binary 8 个、graded 8 个，共 16 个目标。Smoke 使用 seed 0；完整 validation 使用全部三个训练 seed。

| 阶段 | split | 目标 | 进入的 arm | hard-MPB 最大候选数（去重前） |
|---|---|---:|---|---:|
| Smoke | val | 每路线 8 | 全变体、全部方法、seed 0 | binary 288 + graded 192 = **480** |
| Full validation | val | 每路线完整 1,000 | 全变体、全部方法、seeds 0/1/2 | binary 102,000 + graded 68,000 = **170,000** |
| Formal test | test | 每路线预注册 100 | route-level 胜出 generator + 最强 retrieval/direct baseline，seeds 0/1/2 | 至多约 **3,800** |

预算计算规则：seed-0 smoke 中每变体每目标为 `retrieval 1 + direct 1 + single Top-5 + mixture Top-5 = 12`；full validation 中为 `retrieval 1 + direct 3 seeds + single 3×Top-5 + mixture 3×Top-5 = 34`。实际求解数以全局 geometry-hash 去重后的 manifest 为准，但授权前按上限估算 CPU-hours、存储和 wall time。

原来的 20-target pilot 已取消。Smoke 只验证整条物理链与明显 surrogate exploit；通过后可以直接申请 full-validation replay，但 **170,000 上限的正式 M9 任务必须重新报价和单独授权**，不能由 smoke 授权自动扩张。

### 7.2 8-target smoke

两个 smoke manifest 都从各自 `idx_val` 分层冻结：覆盖全部解析变体，并按材料分数、形态复杂度和 forward-surrogate difficulty 选取。Smoke 必须：

1. 使用 hard analytic decoder；soft raster 不得进入 MPB；
2. 所有候选都有 target/method/variant/seed/latent/rank/geometry hash 谱系；
3. solver success 为 1.0，频率有限且 band ordered，失败 ledger 为空；
4. mean absolute surrogate–replay gap 不超过 5 percentage points；
5. M9 结果中至少抽取每个 `(variant, method)` 一个代表结构，共至少 20 个结构，在本机 WSL `meep_env` 固定 replay 并通过跨环境容差；
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

### 7.4 Test-100 与 promotion

Full validation 冻结每路线一个 generator arm，并冻结最强 retrieval/direct baseline。随后从未使用的 `idx_test` 各预注册 100 个目标，使用三个 seed checkpoint 和相同 Top-5 预算执行正式 replay。

生成器只有在以下条件同时满足时晋级：

- 相对最强 baseline 的 paired `replay_mre_bands_2_10_pct` 差值 bootstrap 95% CI 完全低于 0；
- solver success 不下降且所有三 seed 完整；
- surrogate–replay gap、candidate health、运行时间和失败分布均完整报告；
- M9 结果通过固定 WSL `meep_env` 代表子集重放；
- 不把单路线、单 morphology 或 test-100 结果推广为任意 full-cell/OOD 普遍优势。

[↑ 回顶部](#full-cell-inverse-design-extension-plan)

---

<a id="inverse-artifacts"></a>

## 八、训练、候选与 replay 制品

### 8.1 每个训练 run

45 个训练 run 均遵循 `runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/`，至少保存：

- `run.log`、`run.meta.json`、`config.resolved.yaml`；
- `codec.resolved.json` 与 codec/template/hash；
- `history.csv`：逐 epoch stage、train/val reconstruction、raster、primary/cross surrogate、diversity/collision、learning rate 和 epoch time；
- `figures/training_curve.png` 与 `.pdf`：Stage A/B 分区，标出 best/stop epoch；
- best Stage-A checkpoint、best Stage-B `model.pt` 及 SHA-256；
- `metrics.json`、`results.json` 和 seed/variant/method 身份。

训练曲线的数值源必须是 `history.csv`。三 seed 聚合曲线只在共同 epoch 区间计算 mean ± std，不用 early-stopping 后的末值填充。

### 8.2 Candidate 与 replay

核心制品包括：

- `candidate_pool_manifest.json`、`candidate_lineage.csv`、`candidate_health.csv`；
- `strategy_candidates.json`、Top-5 selection 与 geometry-hash cache index；
- smoke/full-validation/test target manifests 及 SHA-256；
- atomic MPB shards、failed-case ledger、solver provenance；
- per-target/per-method/per-seed replay metrics、surrogate–replay gap 和 runtime ledger；
- validation leaderboard、paired statistics、promotion decision 和代表性成功/失败图。

不要求长期保存 per-batch 日志、逐步 GPU telemetry、profiler traces、中间 activation/gradient、每个 epoch checkpoint、未入 Top-5 的完整预测张量、重复 raster/HDF5 副本或重复 solver stdout。它们可用于临时调试，但不能替代核心制品，也不得进入 Git。

[↑ 回顶部](#full-cell-inverse-design-extension-plan)

---

<a id="inverse-authorization"></a>

## 九、授权、失败与结论边界

本文保持 Deferred Draft。以下阶段互不授权：

1. codec/L4 实现与 WSL 单元测试；
2. N40 单卡及 18-job 并发 preflight；
3. 45 个 inverse GPU training runs；
4. validation candidate generation 与 candidate-health；
5. M9 每路线 8-target smoke；
6. M9 整个 validation split replay；
7. test-100 正式 replay。

任一阶段失败时保留既有成功产物并停止在当前 gate；不得自动重训、扩大 replay、修改 loss 或替换目标 manifest。前向 102-run 完成也不自动授权 inverse，GPU training 完成也不自动授权 M9。

[↑ 回顶部](#full-cell-inverse-design-extension-plan)

---

<a id="inverse-human-record"></a>

## 十、本次调整的人工理解记录

```text
解决了什么问题：明确 full-cell inverse 的五类参数 codec、三种训练方法、两阶段损失、45 个单卡训练 run、GPU 并发方式，以及 8-target smoke 到完整 validation 和 test-100 的 replay 路线。
为什么采用这个方案：解析参数维度和固定字段因变体而异，必须分变体训练；N40 可并发独立单卡任务，M9 只做 hard candidate 的 real-MPB，二者通过冻结 manifest/hash 交接。
改变了项目哪一层/哪一模块：主要改变 L3 实验/证据协议；未来 L4 只实现 codec、可微 soft decoder、full-cell inverse 数据适配与 replay 接口。
使用什么证据验证：当前 HDF5 schema/参数宽度、现有 inverse runner/config、历史 A2 方法与 replay 边界、N40/M9 环境预检，以及仓库的 WSL/制品治理规则。
最大风险、未知或下一 gate：full-cell codec 和 8-target inverse smoke 均未实现/执行；full-validation replay 上限约 170,000 个 hard candidates，必须在 smoke 后重新核算 M9 费用并人工授权。
```

[↑ 回顶部](#full-cell-inverse-design-extension-plan)
