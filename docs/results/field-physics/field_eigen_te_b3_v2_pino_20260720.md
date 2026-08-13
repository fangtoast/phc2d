---
title: "B3：TE 本征场监督模型 v2 与 adapted PINO"
summary: "日期化实验与验证证据；正文保持原始记录。"
page_type: evidence
status: draft
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 9 minutes
primary_layer: L3
evidence_date: 2026-07-20
evidence_scope: "日期化证据或历史记录；正文未因站点重构而改写。"
---

<!--
  B3 TE Bloch eigenfield operator-data, supervised-v2, and adapted-PINO report.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# B3：TE 本征场监督模型 v2 与 adapted PINO

- 日期：2026-07-20 至 2026-07-21
- 环境：WSL `Ubuntu` + conda `meep_env`
- 数据 run：`runs/20260720/field_eigen_te_b3_9216_202607201608`
- 1k 消融 run：`runs/20260720/field_eigen_te_v2_ablation_1k_202607201608`
- 正式 data-only run：`runs/20260720/field_eigen_te_v2_baseline_9216_202607201723`
- PINO lambda sweep：`runs/20260721/field_eigen_te_pino_lambda_sweep_9216_202607210044`
- 正式 PINO run：`runs/20260721/field_eigen_te_pino_9216_202607210812`

## 结论

B3 的代码、物理接口和 9,216 条真实 MPB 数据已经就绪。1k 消融表明 active-component codec、共享相位对齐、周期坐标、独立 conditioner 与 v2 architecture 能显著改善原始场基线；其中 U-Net v2 与补齐正/负 x modes 的 FNO v2 被选为正式候选。

正式 data-only 六个 GPU run 已全部完成并通过预设门槛：两个模型所有 seed 的 test E/H relative L2 均小于 1，三 seed mean E relative L2 均不超过 0.95，并优于各自 v1。validation-only sweep 冻结 `lambda_pde=0.01` 后，adapted PINO 三 seed 也通过物理门槛：相对完全匹配的 data-only FNO v2，Maxwell residual 三 seed median 降低 `37.48%`，mean E relative L2 不但没有退化，反而改善 `0.99%`。

该结果状态为 `pino_field_adapted_pilot_pass`。它证明当前 source-free TE Bloch eigenfield task 上的 data+residual 训练有效，但不代表 faithful PINO reproduction、跨分辨率结论、MaxwellNet、field inverse 或论文主模型优势。PINO 的 frequency MRE 高于 data-only FNO，且监督 U-Net v2 的场误差和 residual 仍更低；这些边界必须同时保留。

## 数据与谱系

| 项目 | 结果 |
|---|---:|
| source structures | 256 |
| k / structure | 4 |
| bands / `(structure, k)` | 9（one-based 2–10） |
| records | 9,216 |
| train / val / test records | 6,912 / 1,152 / 1,152 |
| `(structure, k)` MPB jobs | 1,024 |
| completed / failed jobs | 1,024 / 0 |
| records / band | 1,024 |
| HDF5 SHA-256 | `c51e656cc021502b925d54667e0ad356195fe755d1afa7c91752eed10ed9c85c` |

最终 field-contract audit 的 normalization failures 与 provenance failures 均为空。一次审计重试修复了对称点等幅分量在 complex64 落盘后重新 `argmax` 可能选择另一并列分量的问题；最终审计使用 canonicalization 阶段写入的确定性 `anchor_flat_index`，L2 与相位阈值没有放宽。

## 1k 固定消融

下表只报告 validation split。原消融 run 中 FNO A2 的关闭谱模参数曾被 registry 默认参数过滤，因此该行不作为 A2 证据；修正后的 A2 来自独立 run `runs/20260720/field_eigen_te_v2_fno_a2_1k_202607201635`。

| 模型与版本 | E relative L2 | H relative L2 | frequency MRE | Maxwell residual | 参数量 |
|---|---:|---:|---:|---:|---:|
| U-Net A1：active codec + phase loss | 0.9799 | 0.9814 | 25.43% | 8.2905 | 466,247 |
| U-Net A3：A1 + conditioner + v2 | **0.8256** | **0.7912** | **4.01%** | **3.6031** | 4,592,263 |
| FNO A1：active codec + phase loss | 0.9843 | 0.9797 | 25.62% | 9.0460 | 594,343 |
| FNO A2：A1 + periodic coords + conditioner | 0.9402 | 0.9219 | 5.61% | **4.9653** | 2,376,103 |
| FNO A3：A2 + 正/负 x modes + residual/norm | **0.9352** | **0.9153** | 6.23% | 7.2101 | 4,735,399 |

A3 按首要 validation E relative L2 优于 A2，因此冻结为正式 FNO v2；A2 residual 更低不改变预先规定的版本选择顺序。test split 未参与选择。

## 9,216 条数据上的正式 data-only v2

下表报告冻结 test split 的三 seed `mean ± sample std`。六个 case 均使用 CUDA FP32、相同数据 hash 与无交叉的 source split，checkpoint 由 validation data loss 选择。

| 模型 | E relative L2 | H relative L2 | frequency MRE | Maxwell residual | 推理时间 / sample | 参数量 |
|---|---:|---:|---:|---:|---:|---:|
| `unet_field_v2` | **0.5015 ± 0.0091** | **0.4765 ± 0.0079** | 1.7591% ± 0.0475% | 2.7183 ± 0.2413 | 2.3000 ± 0.2648 ms | 4,592,263 |
| `fno_field_v2` | 0.6925 ± 0.0048 | 0.6494 ± 0.0023 | 2.8627% ± 0.3955% | 6.9342 ± 1.1039 | **0.7134 ± 0.0443 ms** | 4,735,399 |

U-Net v2 三 seed 的 E relative L2 为 `0.5042/0.5090/0.4914`；FNO v2 为 `0.6871/0.6939/0.6964`。所有预测 residual 均有限，`invalid_predicted_field_count=0`。U-Net 与 FNO 的 architecture fingerprint 分别固定为 `123b4de8...abd` 与 `5993c5a7...380f`；全部 case 的 HDF5 SHA-256 均为 `c51e656c...c85c`。

该结果证明 v2 表示与 operator-style 数据修复了 v1 的学习失败，但不应解释为 faithful FNO reproduction。U-Net 的场精度更好，FNO 的推理更快；PINO 的唯一匹配对照仍是完全相同架构和预算的 data-only FNO v2。

## PINO lambda gate

lambda 只使用 validation split 选择。data-only reference 直接读取正式 FNO v2 seed-0 checkpoint，不重复训练；其 E relative L2 上限按 `1.02×` 固定为 `0.7142`。三个候选均满足 E gate，最终按 Maxwell residual 最低选择 `0.01`。

| validation case | E relative L2 | H relative L2 | frequency MRE | Maxwell residual | 相对 reference residual 降幅 |
|---|---:|---:|---:|---:|---:|
| data-only reference | 0.70016 | 0.65845 | 3.5197% | 5.76965 | — |
| `lambda=1e-4` | 0.70005 | 0.65841 | 3.5268% | 5.73752 | 0.56% |
| `lambda=1e-3` | 0.69915 | 0.65808 | 3.5899% | 5.44622 | 5.61% |
| `lambda=1e-2` | **0.69307** | **0.65520** | 4.2458% | **3.48326** | **39.63%** |

选择文件为 `lambda_selection.json`，SHA-256 为 `da02388da3beb574b4404c602e91863ef8a8fa2b3869c7b525bb2bcd2693e847`。test split 未参与 lambda 选择。

## 正式 adapted PINO 三 seed

PINO 与 data-only FNO v2 使用相同 `fno_field_v2` architecture fingerprint、数据、seed、总 epoch 上限、batch、优化器和调度；唯一差异是 `training.regime` 与已冻结的 physics loss。seed 0 复用 lambda sweep 中已训练的 checkpoint，只在正式 run 中评估 test；seed 1/2 独立训练。

| test 指标 | data-only FNO v2 | adapted PINO |
|---|---:|---:|
| E relative L2, mean ± std | 0.6925 ± 0.0048 | **0.6856 ± 0.0045** |
| H relative L2, mean ± std | 0.6494 ± 0.0023 | **0.6462 ± 0.0016** |
| frequency MRE, mean ± std | **2.8627% ± 0.3955%** | 3.5716% ± 0.3507% |
| Maxwell residual, mean ± std | 6.9342 ± 1.1039 | **4.3301 ± 0.8327** |
| Maxwell residual, median | 7.3492 | **4.5946** |
| boundary residual, mean ± std | 10.7205 ± 1.5295 | **7.5293 ± 1.2069** |

| seed | data-only E | PINO E | E 相对变化 | data-only residual | PINO residual | residual 降幅 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.68712 | 0.68102 | -0.89% | 5.68288 | 3.39718 | 40.22% |
| 1 | 0.69386 | 0.68577 | -1.17% | 7.34925 | 4.59460 | 37.48% |
| 2 | 0.69642 | 0.68997 | -0.93% | 7.77037 | 4.99837 | 35.67% |

三个 seed 均无 NaN/Inf，test E relative L2 均 `<1.0`，`invalid_predicted_field_count=0`。PINO 的 residual median 降幅超过 20%，mean E 未退化，故通过预设 physics gate。frequency MRE 的上升说明当前复合 loss 仍存在多目标权衡，后续不得只展示 residual 而隐藏频率误差。

## 实现边界

- field 输出仅预测 TE active complex components `Ex/Ey/Hz` 的六个实虚通道，评估时重建 `Ez/Hx/Hy=0` 的完整 E/H。
- 模型输入为 epsilon、周期坐标与独立 `(kx, ky, band)` context，真实频率不进入输入。
- 频率头预测 train-only 标准化的 `log(f)`，再还原为正物理频率。
- Torch TE Bloch curl-curl residual 与 NumPy evaluator 共用单位、epsilon lifting、Bloch wrap 与频率换算；物理 loss 只使用预测 E 与预测频率。
- PINO 使用 `fno_field_v2` 架构，通过 `training.regime: pino_adapted` 标识，不创建虚假的独立架构身份。

## 当前门槛

1. data-only v2：已通过，两个模型均升级为 `supervised_field_baseline_v2`，但不是 faithful reproduction 或 inverse evidence。
2. PINO lambda：已在 validation split 冻结 `0.01`，selection lineage 与 hash 完整。
3. PINO 三 seed：已通过，状态为 `pino_field_adapted_pilot_pass`；只允许与匹配 data-only FNO v2 作当前任务内比较。
4. MaxwellNet、field inverse、PML/source/driven-field 仍不启动；进入 inverse 需要独立的接入协议与 solver-backed gate。
