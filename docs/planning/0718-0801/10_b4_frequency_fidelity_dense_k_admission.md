<!--
  B4 frequency fidelity, dense-k, and inverse admission plan.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="b4-frequency-fidelity-dense-k-admission"></a>

# B4：PINO 频率保真、密集 k 与 inverse 准入

**定位：** B4 是 B3 adapted PINO physics gate 之后的独立 forward 准入阶段。它修复“通过改变预测频率投机降低 residual”的风险，并建立 dense-k 与跨分辨率边界；它不授权 inverse sampling、rerank、replay 或新 MPB 数据生成。

**目录：** [当前证据](#一当前证据) · [冻结设计](#二冻结设计) · [准入状态](#三准入状态) · [运行边界](#四运行边界)

---

## 一、当前证据

| 项目 | 当前状态 | 证据 |
|---|---|---|
| B3 data-only FNO v2 | `formal_completed` | `runs/20260720/field_eigen_te_v2_baseline_9216_202607201723` |
| B3 adapted PINO | `pilot_pass` | `runs/20260721/field_eigen_te_pino_9216_202607210812`；residual median 降低 37.48%，但 frequency MRE 高于 matched FNO |
| B4 训练/审计实现 | `implemented` | [`field_eigen_v2.py`](../../../src/phc2d/forward/field_eigen_v2.py) |
| B4 validation 消融与冻结三 seed | `formal_completed` | `runs/20260721/field_eigen_te_pino_frequency_guard_9216_202607211039/frozen_candidate_manifest.json` |
| dense-k 与 resolution transfer | `formal_completed` | `runs/20260722/field_eigen_te_b4_dense_k_gate_202607220819` |

`frozen_candidate_manifest.json` 与 `promotion_decision.json` 已完整落盘；三项 promotion status 均为 `blocked`。因此不能把 B3 residual 改善升级为 frequency cross-check、residual rerank、cross-resolution 或 field inverse 证据。

---

## 二、冻结设计

1. `physics_frequency_gradient: full|detach` 的两种模式都用模型预测频率计算 PDE residual；`detach` 只阻断 PDE 到 frequency head 的梯度。
2. 候选 P1/P2/P3 固定 200 epochs，关闭 early stopping，在 40/80/120/160/200 epoch 做 validation、checkpoint 与 gradient audit。P0 仅为历史 comparator，不能代替 B4 审计。
3. validation candidate 必须同时满足 matched FNO 的 frequency MRE 与 E relative L2 均不超过 `1.02×`，且 residual median 至少降低 20%。test split 不参与选择。
4. dense-k 使用冻结 V2 31-index Γ–X–M–Γ path。主统计只用 30 个唯一物理坐标；B3 已见 k 为 `{0,4,8,12,16,20,24,29}`，其余 22 个唯一坐标为 unseen-k。
5. 64×64 在已有 field records 的 k 点才报告 E/H relative L2。96×96、128×128只可报告 frequency truth 对照和 predicted-field residual，禁止声称高分辨率 field accuracy。

---

## 三、准入状态

| 独立状态 | 通过条件 | 当前值 |
|---|---|---|
| `physics_frequency_crosscheck_ready` | unseen-k frequency、seen-k E relative L2 均满足匹配 FNO 阈值 | `blocked`：unseen-k frequency 均值与 seed 2 均超出阈值 |
| `physics_residual_rerank_ready` | unseen-k residual 降低、Spearman CI 下界大于零、四分位校准成立 | `blocked`：seed 1/2 校准未通过 |
| `cross_resolution_zero_shot_pass` | 96/128 的 frequency/residual 退化满足冻结上限且有限 | `blocked`：96/128 residual 退化超限 |

频率 cross-check 通过而 residual rerank 不通过时，后续只允许使用 PINO frequency head 作 cross-check；不允许使用 residual 当 reranking score。跨分辨率失败不改变 64×64 的结论，但禁止跨分辨率 claim。

---

## 四、运行边界

B4 仍不生成 MPB 数据，不启动 MaxwellNet，不运行 inverse sampling/replay。任何 future inverse 比较都必须等到 B4 输出完整 lineage、per-case artifacts 和 `promotion_decision.json` 后，再按固定 target、K、seed 与 solver budget 的协议单独授权。

[返回顶部](#b4-frequency-fidelity-dense-k-admission)
