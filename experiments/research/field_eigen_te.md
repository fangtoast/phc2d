<!--
  Frozen B0 research profile for the first TE Bloch-eigenfield task.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="field-eigen-te"></a>

# TE Bloch 本征场研究 Profile

本文定义 B1 的**首个物理任务输入**，不是训练记录或 PINO/MaxwellNet 完成声明。完整字段和 promotion gate 由
[`field_eigen_te_contract.yaml`](../../configs/research/field_eigen_te_contract.yaml) 冻结；阶段证据边界见
[`09_phase1_a0_b0_protocol.md`](../../docs/planning/0718-0801/09_phase1_a0_b0_protocol.md)。

截至 2026-07-20，已在 WSL `Ubuntu` + conda `meep_env` 完成一条 real-MPB record 的预检、[64 条分层 field-data pilot](../../docs/results/field-physics/field_eigen_te_b1_64_20260720.md)、B2 residual gate、[1,000 条 real-MPB field dataset](../../docs/results/field-physics/field_eigen_te_b2_b1_1000_20260720.md)，以及 B3 的 9,216 条 operator-style real-MPB records。B3 数据已通过 schema、normalization 和 solver provenance 审计；data-only v2 与 PINO-adapted 的正式训练证据仍以各自 run 为准。

**目录：** [任务](#任务) · [数据记录](#数据记录) · [验证门槛](#验证门槛)

---

## 一、任务

首轮只研究 square-lattice、TE polarization 的 Bloch 本征场：输入是 `epsilon`、physical k coordinate、band context 和可追溯 solver context；输出是该本征模的 eigenfrequency 与复数 E/H field。k-path 继续使用现有 `square_gxm` 元数据，band 覆盖由当前的 one-based 2–10 区间限定。

本 profile 不引入 source、PML、driven field 或 TM。它们必须以独立 contract 和数据集进入后续阶段，不能与本征任务混写。

---

## 二、数据记录

每条未来样本对应一个 `(structure, k-point, band)`。字段必须同时包含 `rho`、`epsilon`、复数 E/H 场、Bloch phase、`dx/dy`、晶胞信息、normalization 和完整 MPB provenance；场的空间/通道 shape 由保存的 schema 推导。

field label 采用 cell-volume L2 归一化，并将最大幅值分量的全局相位固定为非负实数。这样可避免本征解任意相位导致的监督误差不确定；它不改变 solver 原始数据的独立保存要求。

---

## 三、验证门槛

B1 先以一条 real-MPB record 预检生成器，再生成 64 条 stratified real-MPB records；B2 residual gate 通过后扩容至 1,000 条，并保存 manifest、sample metadata、worker/failed-case ledger。只有下列事实均成立，才允许独立 supervised field baseline：

1. record schema、Bloch boundary、单位、normalization 和 solver provenance 全部可追溯；
2. 解析/构造场、solver reference field 与随机场的 residual tests 给出预期区分；
3. 1,000 条数据集的 split、complex field、normalization 和 provenance 审计通过；
4. 训练模型先在 field relative L2、amplitude/phase 与 residual 上形成独立 forward evidence。

首轮 v1 data-only 训练已完成但双模型门槛未通过：`unet_field` 仅极小幅度超过零预测，`fno_field` 仍低于零预测，详见 [`field_eigen_te_supervised_baseline_20260720.md`](../../docs/results/field-physics/field_eigen_te_supervised_baseline_20260720.md)。B3 v2 已修复该监督基线并通过两个模型三 seed 门槛，因此只解锁 adapted-PINO 的 validation lambda gate；MaxwellNet 与 physical inverse 仍被阻断。

---

## 四、B3 双线接口

B3 将 1,000 条数据保留为消融输入，正式数据集改为 256 个 source structures，每个结构覆盖 4 个 physical-k 与 bands 2–10 全部九条记录。source split 固定为 `192/32/32` 个结构，对应 `6912/1152/1152` 条 records；任何 source structure 不跨 split。真实生成 run 为 `runs/20260720/field_eigen_te_b3_9216_202607201608`，其 1,024 个 `(structure, k)` 多 band MPB jobs 全部完成，最终 HDF5 SHA-256 为 `c51e656cc021502b925d54667e0ad356195fe755d1afa7c91752eed10ed9c85c`。

监督表示仅编码 TE 非零复分量 `Ex/Ey/Hz`，同时保留完整 E/H 重建规则；输入包含 schema 驱动的 epsilon grid、周期坐标和独立 `(kx, ky, band)` conditioner，不包含真实频率。`unet_field_v2` 与 `fno_field_v2` 使用共享的 E-derived 全局相位对齐损失，频率头学习 train-only 标准化的 `log(f)`。

PINO 使用与 data-only `fno_field_v2` 完全相同的架构身份，仅通过 `training.regime: pino_adapted` 区分训练制度。物理项只使用预测 E 与预测频率；validation split 已冻结 `lambda_pde=0.01`，正式三 seed的 Maxwell residual median 相对匹配 data-only FNO v2 降低 `37.48%`，mean E 改善 `0.99%`。该结果只称为 `pino_field_adapted_pilot_pass`。MaxwellNet 仍缺少其原始 driven/source/PML task，不进入本轮主表或正式训练。

## 五、B4 频率保真与 dense-k 准入

B4 已实现 frequency-guarded PINO 接口和 [`frequency-guard 配置`](../../configs/research/field_eigen_te_pino_frequency_guard_9216.yaml)：PDE residual 始终使用预测频率，`detach` 仅隔离其对 frequency head 的梯度；field/frequency 权重、每 40 epoch 的 validation/checkpoint 和梯度审计均由配置冻结。旧 P0 结果只能作为历史比较，不能替代 B4 梯度审计。

[`dense-k 配置`](../../configs/research/field_eigen_te_b4_dense_k_gate.yaml) 固定 V2 的 31 点 Γ–X–M–Γ 路径、30 个唯一坐标及 B3 已见索引 `{0,4,8,12,16,20,24,29}`。64×64 才可在已有 field record k 点报告 E/H relative L2；96×96 与128×128只允许 frequency truth 和 residual 自洽性比较。B4 正式结果必须由对应 run 的 `promotion_decision.json` 给出，未完成前 `field_inverse`、inverse sampling/replay 和 real-solver benefit 仍为 blocked。

[↑ 回顶部](#field-eigen-te)
