<!--
  A2 seven-recipe post-repair validation and real-MPB pilot report.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="a2-postrepair-validation-v2"></a>

# A2 多 Surrogate Post-Repair 分层验证结果

| 元数据 | 值 |
|---|---|
| 日期 | 2026-08-02 |
| 协议 | [`configs/research/a2_inverse_surrogate_postrepair_validation_v2.yaml`](../../../configs/research/a2_inverse_surrogate_postrepair_validation_v2.yaml) |
| 运行 | `runs/20260802/a2_inverse_surrogate_postrepair_validation_v2_202608021718` |
| 环境 | WSL `Ubuntu` + conda `meep_env`；checkpoint 选择与候选生成使用 `cuda:0`，物理 replay 使用 real MPB |
| 证据状态 | `single_seed_20_target_pilot` |

**目录：** [结论](#一结论) · [问题与边界](#二实验问题与边界) · [协议](#三数据模型与门禁) · [结果](#四结果) · [产物](#五命令与产物) · [限制](#六解释限制与下一证据门)

---

## 一、结论

本轮完成了七种 inverse recipe 的 seed-0、validation-100 checkpoint 选择、固定 2-target real-MPB smoke 和固定四分层 20-target Top-5 real-MPB pilot。七臂全部通过 candidate-health 硬门禁；smoke 的 28 次求解与 pilot 的 700 次求解均成功，未发生 CPU 静默回退、OOM、NaN/Inf 或 solver failure。

20-target pilot 中，`res_mlp_wide` 的原始 replay mean MRE 最低，为 **2.3472%**；`mlp_query` 为 **2.4386%**。两者的单臂 bootstrap 区间重叠，因此按冻结规则继续比较 replay gap、solver success 与总耗时，最终选择 **`mlp_query` 作为最佳 challenger**。`fno2d_band_v2` 的 replay mean MRE 为 **3.2502%**，仍永久保留为参考臂；本结论不删除或替换 FNO，也不授权 100-target 或多种子扩展。

`mlp_query` 相对 FNO 的同目标配对 replay 差值为 **-0.8116 percentage points**，paired bootstrap 95% CI 为 **[-1.2867, -0.3863]**；`res_mlp_wide` 的对应差值为 **-0.9030 pp**，95% CI 为 **[-1.4761, -0.3524]**。这些数值只支持 single-seed 20-target pilot 范围内的 challenger 证据，不能表述为跨 seed 稳定或正式替换结论。

---

## 二、实验问题与边界

本实验回答：利用现有 epoch 10/30/50 checkpoint，在不重新训练的条件下，七种 inverse recipe 是否能通过统一 candidate-health 门禁，并在相同 validation targets、latent seed、候选预算和 MPB protocol 下形成可比较的 seed-0 pilot 证据。

七臂中的六个 stabilized repair 臂都使用了 FNO cross-surrogate 正则，因此本表是**七种 inverse recipe 的系统比较**，不是纯 surrogate 架构消融。

- 第一版 A2 checkpoint 路径与 SHA-256 `58548c...aae` 已由 denylist 拒绝，未作为候选生成源。
- 未读取 test100 或 test625；checkpoint 选择和 pilot targets 均来自固定 validation-100 manifest。
- 未启动新一轮 inverse 训练。
- Field/PINO inverse admission 继续为 `blocked`。
- 旧 screen、旧 smoke 和撤销的第一版 A2 只保留历史诊断身份，未并入当前主表。

---

## 三、数据、模型与门禁

### 3.1 固定输入

| 项目 | 固定值 |
|---|---|
| 数据集 | `data/query_dataset_v2_pilot_10000_mpb.npz` |
| 数据集 SHA-256 | `991cb381...974ad4` |
| checkpoint 选择 targets | validation-100 固定 manifest |
| 候选数 | 每目标 $K=128$ |
| latent seed | `0`，各臂共享 target ID / latent ID 规则 |
| query 协议 | 真实物理 k 坐标，完整恢复 `[candidate, k_points, n_bands]` |
| k-path | 保留首尾重复 Gamma；禁止 index-query fallback |
| smoke | 固定 2 targets，每臂 primary Top-2 |
| pilot | 固定四分层 20 targets，每臂 primary Top-5 |
| MPB | TE、resolution 64、10 bands、tolerance $10^{-7}$ |

### 3.2 Validation-100 checkpoint 选择

所有选中 checkpoint 均满足 validity rate `=1.0`、effective-K mean `≥64`、effective-K p10 `≥32`、continuous candidate std `≥0.04`、threshold-margin fraction `≤0.15`、latent gradient 有限非零、mode coverage `=1.0`，且训练/评估均为 CUDA、无 fallback。

| Recipe | 选中 epoch | Primary MRE | Common-FNO MRE | Effective-K mean / p10 | Continuous std | Threshold margin |
|---|---:|---:|---:|---:|---:|---:|
| `mlp` | 30 | 2.0492% | 3.8568% | 113.79 / 103.0 | 0.0991 | 0.0617 |
| `res_mlp_wide` | 50 | 1.7274% | 2.3968% | 112.79 / 103.9 | 0.1282 | 0.0386 |
| `lifted_cnn` | 50 | 1.9921% | 3.1798% | 106.03 / 97.9 | 0.1676 | 0.0193 |
| `fno2d_band_v2` | 50 | 2.1771% | 2.1771% | 109.59 / 101.9 | 0.1243 | 0.0476 |
| `mlp_query` | 50 | 1.8747% | 3.0416% | 111.85 / 105.9 | 0.1216 | 0.0410 |
| `deep_onet_query_v2` | 30 | 1.7857% | 2.8526% | 103.42 / 93.9 | 0.1202 | 0.0427 |
| `fno2d_query_v2` | 10 | 3.2961% | 3.4838% | 105.05 / 97.0 | 0.0855 | 0.0933 |

选择只使用 validation-100。优先级依次为最小 `max(primary MRE, common-FNO MRE)`、更小 threshold margin、更短采样加排序时间、较早 epoch。

---

## 四、结果

### 4.1 2-target real-MPB smoke

Smoke 完成 **28/28** 次 real-MPB 求解，solver failure 为 0，全部指标有限。七臂 mean absolute surrogate–replay gap 为 **1.3833%**，通过 `≤5%` 门禁。Smoke 仅证明 replay 管线可用且未出现明显 surrogate exploit，不产生模型优劣结论。

### 4.2 20-target Top-5 real-MPB pilot

Pilot 完成 20 targets × 7 recipes × Top-5，即 **700/700** 次 real-MPB 求解。联合结构 SHA-256 去重后仍为 700 个唯一结构，跨臂 Top-5 无缓存复用机会；MPB 总求解时间为 **2756.41 s**，完整 replay wall time 为 **2756.51 s**。

| Recipe | Replay MRE | 单臂 bootstrap 95% CI | 相对 FNO paired Δ | Paired Δ 95% CI | Primary gap | Common-FNO gap | 总耗时 | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mlp` | 4.4918% | [3.7592, 5.2703] | +1.2416 pp | [0.3668, 2.1482] | +2.3362 pp | +1.7501 pp | 188.54 s | 41.50 MB |
| `res_mlp_wide` | **2.3472%** | [2.1397, 2.5651] | **-0.9030 pp** | [-1.4761, -0.3524] | +0.4843 pp | -0.2682 pp | 186.81 s | 41.39 MB |
| `lifted_cnn` | 3.3675% | [3.1515, 3.5700] | +0.1173 pp | [-0.4750, 0.6778] | +1.4281 pp | +0.3256 pp | 226.60 s | 37.42 MB |
| `fno2d_band_v2` | 3.2502% | [2.6990, 3.8468] | 0.0000 pp | [0.0000, 0.0000] | +1.1376 pp | +1.1376 pp | 188.91 s | 37.57 MB |
| `mlp_query` | 2.4386% | [2.1595, 2.7499] | -0.8116 pp | [-1.2867, -0.3863] | **+0.4633 pp** | **-0.1458 pp** | 243.67 s | 38.12 MB |
| `deep_onet_query_v2` | 2.6797% | [2.3051, 3.1298] | -0.5705 pp | [-1.1646, 0.0420] | +0.9016 pp | +0.1584 pp | 224.32 s | 38.14 MB |
| `fno2d_query_v2` | 5.9099% | [4.6829, 7.2622] | +2.6597 pp | [1.4139, 4.0512] | +3.7452 pp | +2.8437 pp | 233.17 s | 39.76 MB |

表中“总耗时”按该臂 validation-100 候选生成/primary 排序、common-FNO 诊断排序及分配的 replay 时间相加；共享 MPB wall time 单独如上报告，不能把七臂总耗时再次相加解释为实际 wall time。峰值显存来自 selected candidate-pool CUDA 运行，MPB 为 CPU 物理求解。

### 4.3 决策解释

`res_mlp_wide` 的 replay mean 最低，但其单臂 95% CI 与 `mlp_query` 重叠。冻结决策规则在 CI 重叠时先比较 replay gap：`mlp_query` 的 primary gap 绝对值为 0.4633 pp、common-FNO gap 绝对值为 0.1458 pp，均小于 `res_mlp_wide` 的 0.4843 pp 和 0.2682 pp，因此 `promotion_decision.json` 将 `mlp_query` 选为最佳 challenger。

该决策表示“当前 pilot 支持继续研究 `mlp_query` challenger”，不表示删除 FNO。`fno2d_band_v2` 继续作为固定参考；本结果也不属于纯 surrogate 架构优劣结论。

---

## 五、命令与产物

各阶段均在 WSL `Ubuntu` + `meep_env` 中通过现有公共入口执行：

```bash
PYTHONPATH=src python -m phc2d.cli inverse benchmark \
  --config configs/research/a2_inverse_surrogate_postrepair_validation_v2.yaml \
  --stage preflight --run-dir runs/20260802/a2_inverse_surrogate_postrepair_validation_v2_202608021718

PYTHONPATH=src python -m phc2d.cli inverse benchmark --config configs/research/a2_inverse_surrogate_postrepair_validation_v2.yaml --stage sample --run-dir runs/20260802/a2_inverse_surrogate_postrepair_validation_v2_202608021718
PYTHONPATH=src python -m phc2d.cli inverse benchmark --config configs/research/a2_inverse_surrogate_postrepair_validation_v2.yaml --stage screen --run-dir runs/20260802/a2_inverse_surrogate_postrepair_validation_v2_202608021718
PYTHONPATH=src python -m phc2d.cli inverse benchmark --config configs/research/a2_inverse_surrogate_postrepair_validation_v2.yaml --stage real-smoke --run-dir runs/20260802/a2_inverse_surrogate_postrepair_validation_v2_202608021718
PYTHONPATH=src python -m phc2d.cli inverse benchmark --config configs/research/a2_inverse_surrogate_postrepair_validation_v2.yaml --stage pilot-replay --run-dir runs/20260802/a2_inverse_surrogate_postrepair_validation_v2_202608021718
PYTHONPATH=src python -m phc2d.cli inverse benchmark --config configs/research/a2_inverse_surrogate_postrepair_validation_v2.yaml --stage promote --run-dir runs/20260802/a2_inverse_surrogate_postrepair_validation_v2_202608021718
```

关键产物均已形成：

- `postrepair_surrogate_manifest.json`
- `checkpoint_selection.csv`
- `candidate_pool_manifest.json`
- `candidate_health.csv`
- `primary_common_diagnostic_metrics.csv`
- `real_mpb_smoke_decision.json`
- `pilot_replay_metrics.csv`
- `selection_overlap.csv`
- `promotion_decision.json`

另有 `candidate_lineage.csv`、`pilot_replay/primary_common_per_target_metrics.csv`、per-target/per-strategy replay CSV 和完整 solver artifacts，用于 target/latent/candidate 追溯与配对复核。

---

## 六、解释限制与下一证据门

1. 当前最高状态只能是 `single_seed_20_target_pilot`；没有 cross-seed 证据。
2. 20 targets 仍是小样本 pilot，不能替代预注册的 100-target 正式比较。
3. 七臂来自不同 inverse objective recipe，且六个 repair 臂带 common-FNO 正则，不能解释为独立 surrogate 架构消融。
4. Pilot 的 700 个 Top-5 结构全部唯一，因此本次没有产生跨模型 MPB cache 节省；缓存实现仍由 smoke/mock 测试覆盖，但本轮真实数据没有命中。
5. Field/PINO inverse admission 未改变，仍为 `blocked`。

**下一证据门：** 另行立项决定是否对 FNO 参考臂、`mlp_query` 与 `res_mlp_wide` challenger 开展 100-target / seeds 1–2 的预注册正式实验。在新立项前，不自动训练、不扩 replay、不升级结论。

[↑ 回顶部](#a2-postrepair-validation-v2)
