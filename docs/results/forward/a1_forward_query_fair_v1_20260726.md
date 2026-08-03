<!--
  Formal A1 10k V2 forward/query fair-baseline result and audit.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="a1-forward-query-fair-v1-20260726"></a>

# A1 10k V2 Forward/Query 公平对照正式结果

- **报告日期：** 2026-07-26
- **实验状态：** `formal_completed`
- **数据：** `data/query_dataset_v2_pilot_10000_mpb.npz`
- **数据 SHA-256：** `991cb3810e881938755495ad0bccfa5b091194c2b74451842a9a50ddce974ad4`
- **环境：** WSL `Ubuntu`、conda `meep_env`、NVIDIA GeForce RTX 4060 Laptop GPU、CUDA FP32
- **证据边界：** neural-surrogate benchmark；本报告不包含 MPB/Meep replay、inverse、PINO、MaxwellNet 或 100k 扩容
- **Forward runs：** 本地 `runs/20260724/a1_forward_fair_v1_202607240020`、`runs/20260726/a1_forward_mlp_ablation_v1_202607261605`
- **Query run：** 本地 `runs/20260724/a1_query_fair_v1_202607240803`

**目录：** [结论](#一结论) · [协议与完整性](#二协议与完整性) · [Forward](#三forward-结果) · [Query](#四query-结果) · [运行成本](#五运行成本) · [身份与边界](#六模型身份与结论边界) · [验收](#七验收结果)

---

## 一、结论

1. **A1 已完成扩展后的公平验收。** Forward 为 `36 validation / 12 selected-LR test / 4 summary`，Query 为 `81 validation / 27 selected-LR test / 9 summary`；所有汇总严格为 validation 选 LR 后的 test `n=3`。
2. **Forward 最佳模型是 `fno2d_band_v2`。** test bands 2–10 MRE 为 `1.0299% ± 0.0545%`；`res_mlp_wide` 为 `1.3309% ± 0.0122%`，但 seed 波动最小。
3. **残差连接具有清晰的同容量收益。** 无残差 `mlp` 与 `res_mlp_wide` 均为 `878,646` 参数，test MRE 分别为 `2.4764%` 和 `1.3309%`。残差连接对应 `1.1455` 个百分点、`46.26%` 的相对误差下降，三个配对 seed 均改善。
4. **Query 中 `fno2d_query_v2` 在三个协议均取得最低 MRE。** all-k、physical interpolation、physical held-out segment 分别为 `1.1004%`、`1.0470%`、`3.6747%`。
5. **held-out segment 是所有 Query 模型的主要困难。** 三个模型的 MRE 均明显高于 all-k/interpolation；`fno2d_query_v2` 虽然 MRE 最低，但 ordering violation 为 `9.22%`，高于 DeepONet-style 和 MLP，因此不能只按单一误差指标宣告全面占优。
6. **当前 A1 不使用 POD。** Forward 模型直接输出完整 `Wh`，Query 模型直接输出查询点 band vector；`pod_res_mlp` 仍是未实现占位符。

---

## 二、协议与完整性

### 2.1 数据与 split

| 项目 | 审计结果 |
|---|---|
| 样本数 | `10,000` |
| `Wh` shape | `10000 × 31 × 10` |
| train / val / test | `6875 / 2500 / 625` |
| train hash | `71d33105f9646085b505e0ffdddc1317ef354af4be707ee09c7d787f4233300d` |
| val hash | `5c0b0a29bd802a09fee582379f4ed341ad2d7654c65d241f1c49baec6a3acd8b` |
| test hash | `ba25abd1dda7ebcd85d5521c06f5e91583522a06720b8896dd4ba38c75ab03bf` |
| sample leakage | train/val/test 两两不相交 |
| seeds | `0 / 1 / 2` |
| learning rates | `1e-3 / 3e-4 / 1e-4` |
| LR 选择 | 仅使用 validation bands 2–10 MRE 的三 seed mean |
| test 使用 | 每个模型/协议只评估 selected LR 的三个 checkpoint |

### 2.2 Physical-k

Query 使用 sidecar `query_dataset_v2_pilot_square_gxm_mpb_reduced_v1`，状态为 `physically_verified_with_documented_numerical_tolerance`。路径为 31 个索引、30 个唯一物理坐标的 Γ–X–M–Γ 路径；末端 Γ 与起点重复。

| 协议 | train k | eval k | k 隔离 |
|---|---:|---:|---|
| `unseen_structure_all_k` | 31 | 31 | 不做 k holdout，只做结构隔离 |
| `unseen_structure_physical_k_interpolation` | 24 | 7 | eval 坐标不出现在 train |
| `unseen_structure_physical_heldout_segment` | 22 | 9 | segment interior 不出现在 train |

三个协议均保持 train/val/test structure ID 隔离。所有 Query validation/test 行均记录 physical coordinate source、`cuda:0`、有限梯度和可读取 checkpoint。

---

## 三、Forward 结果

### 3.1 主表

| 模型 | 输入 | selected LR | test bands 2–10 MRE | 参数量 | train time/seed | 推理 ms/sample | 峰值显存 |
|---|---|---:|---:|---:|---:|---:|---:|
| `fno2d_band_v2` | `epsilon_grid` | `1e-3` | **`1.0299% ± 0.0545%`** | 389,718 | 272.71 s | 0.5267 | 147.61 MB |
| `res_mlp_wide` | `rho_wedge` | `3e-4` | `1.3309% ± 0.0122%` | 878,646 | 80.96 s | 0.0064 | 26.31 MB |
| `lifted_cnn` | `epsilon_grid` | `1e-3` | `2.2234% ± 0.0366%` | 5,458 | 39.97 s | 0.0102 | 31.56 MB |
| `mlp` | `rho_wedge` | `3e-4` | `2.4764% ± 0.0241%` | 878,646 | 65.66 s | 0.0074 | 26.31 MB |

### 3.2 MLP 残差消融

`mlp` 与 `res_mlp_wide` 具有相同输入、输出、hidden width、block 数、每块 Linear 数、激活、dropout 和参数量；前者计算 `GELU(F(x))`，后者计算 `GELU(x + F(x))`。两者不是 alias，架构指纹分别为：

- `mlp`：`6c9fc53f2abe92016897ce6225881237c92dcc80b53a1c7bfc3eaba1895f471e`
- `res_mlp_wide`：`26806893189e30d6b487b847a3d75eb737c87f1a7780fe553337884987747625`

| seed | `res_mlp_wide` MRE | `mlp` MRE | Plain − Residual |
|---:|---:|---:|---:|
| 0 | 1.3211% | 2.5031% | +1.1819 pp |
| 1 | 1.3270% | 2.4700% | +1.1429 pp |
| 2 | 1.3446% | 2.4561% | +1.1115 pp |

残差模型训练时间约高 `23.3%`，但误差下降 `46.26%`。在固定三 seed 证据下，应表述为残差模型稳定更优，不扩展为一般性的统计显著性结论。

### 3.3 Per-band

bands 2–10 的平均 per-band 诊断显示：

| 模型 | 最低误差 band | MRE | 最高误差 band | MRE |
|---|---:|---:|---:|---:|
| `fno2d_band_v2` | 4 | 0.9028% | 10 | 1.2765% |
| `res_mlp_wide` | 4 | 1.2046% | 10 | 1.5814% |
| `lifted_cnn` | 4 | 1.9685% | 10 | 2.3834% |
| `mlp` | 9 | 2.2053% | 2 | 2.8586% |

完整逐 seed、逐 band 数据保存在两个 Forward run 的 `per_band_metrics.csv`。

---

## 四、Query 结果

### 4.1 主表

| 协议 | 模型 | selected LR | test bands 2–10 MRE | ordering violation | 参数量 | train time/seed | ms/query |
|---|---|---:|---:|---:|---:|---:|---:|
| all-k | `fno2d_query_v2` | `1e-3` | **1.1004% ± 0.0218%** | 2.96% | 273,514 | 3782.50 s | 0.0258 |
| all-k | `deep_onet_query_v2` | `3e-4` | 1.7937% ± 0.1339% | 1.35% | 166,666 | 1734.57 s | 0.0097 |
| all-k | `mlp_query` | `3e-4` | 2.1800% ± 0.0341% | 0.83% | 21,386 | 842.39 s | 0.0053 |
| interpolation | `fno2d_query_v2` | `1e-3` | **1.0470% ± 0.0240%** | 2.16% | 273,514 | 2876.61 s | 0.0286 |
| interpolation | `deep_onet_query_v2` | `1e-3` | 1.6308% ± 0.0691% | 0.41% | 166,666 | 1218.45 s | 0.0108 |
| interpolation | `mlp_query` | `3e-4` | 2.2079% ± 0.0719% | 0.42% | 21,386 | 587.99 s | 0.0055 |
| held-out segment | `fno2d_query_v2` | `1e-3` | **3.6747% ± 0.1730%** | 9.22% | 273,514 | 2603.91 s | 0.0286 |
| held-out segment | `deep_onet_query_v2` | `1e-3` | 3.7604% ± 0.1595% | 0.99% | 166,666 | 1057.56 s | 0.0128 |
| held-out segment | `mlp_query` | `3e-4` | 4.2548% ± 0.1493% | 1.00% | 21,386 | 541.13 s | 0.0060 |

### 4.2 Per-band 与 per-k 风险

| 协议 | 模型 | 最差 band（MRE） | 最差 k index（bands 2–10 MRE） |
|---|---|---|---|
| all-k | `mlp_query` | band 6（2.3770%） | 30（2.6170%） |
| all-k | `deep_onet_query_v2` | band 10（2.0590%） | 20（2.1805%） |
| all-k | `fno2d_query_v2` | band 10（1.2913%） | 20（1.6164%） |
| interpolation | `mlp_query` | band 2（2.5398%） | 19（2.3543%） |
| interpolation | `deep_onet_query_v2` | band 2（1.8929%） | 19（1.8216%） |
| interpolation | `fno2d_query_v2` | band 10（1.2742%） | 19（1.3657%） |
| held-out segment | `mlp_query` | band 4（8.2465%） | 26（6.0937%） |
| held-out segment | `deep_onet_query_v2` | band 4（10.6880%） | 26（5.1741%） |
| held-out segment | `fno2d_query_v2` | band 4（8.7716%） | 26（5.0841%） |

held-out segment 的共同最差 k index 为 26，且 band 4 对三个模型均困难。`failure_cases.csv` 只有表头，没有失败 case；所有训练和评估指标均有限。完整明细见 Query run 的 `per_band_metrics.csv`、`per_k_metrics.csv` 和 `protocol_metrics.csv`。

---

## 五、运行成本

| 范围 | 累计 validation train time | runner wall time | 说明 |
|---|---:|---:|---|
| 原 Forward 三模型 | 3571.58 s（0.992 h） | 4067.29 s（1.130 h） | 单次完整 run |
| Plain MLP A1.1 | 596.31 s（0.166 h） | 619.19 s（0.172 h） | 独立 run |
| Query 三协议 | 137254.53 s（38.126 h） | 最后一次恢复 80361.87 s（22.323 h） | 累计 train time 来自 81 个 case；wall time 只是最后一次恢复进程 |
| **A1 合计** | **141422.42 s（39.284 h）** | 不跨恢复强行相加 | selected test 与 validation 的模型评估累计 49.82 s |

Query 的累计 GPU 训练成本按 81 个 case 的持久化 `train_time_s` 重建；没有把最后一次恢复进程的局部 wall time冒充完整 Query 耗时。

---

## 六、模型身份与结论边界

| 模型 | implementation | reproduction grade | A1 evidence |
|---|---|---|---|
| `mlp` | project-native | residual ablation | formal three-seed test |
| `res_mlp_wide` | project-native | baseline | formal three-seed test |
| `lifted_cnn` | project-native | baseline | formal three-seed test |
| `fno2d_band_v2` | project-native implementation | paper-inspired adapted spectral encoder | formal three-seed test |
| `mlp_query` | project-native | baseline | three protocols, formal three-seed test |
| `deep_onet_query_v2` | project-native implementation | DeepONet-style adaptation | three protocols, formal three-seed test |
| `fno2d_query_v2` | project-native implementation | paper-inspired query spectral encoder | three protocols, formal three-seed test |

`fno2d_*` 与 `deep_onet_query_v2` 不能称为 faithful literature reproduction。A1 的 `formal_completed` 仅表示本地冻结 neural-surrogate 协议完成，不表示通过真实 MPB replay，也不表示 PINO/MaxwellNet 或 inverse gate 已通过。2026-07-02 及更早结果只作为历史诊断，不进入本次 validation-selected 主表。

---

## 七、验收结果

### 7.1 产物审计

- Forward：`36 validation / 12 selected-LR test / 4 summary`。
- Query：`81 validation / 27 selected-LR test / 9 summary`。
- 所有 summary 行均 `n_seeds=3`。
- 所有 test 行与 selection manifest 的 selected LR 一致。
- 所有 checkpoint 路径可读，模型 canonical identity 非 alias。
- 数据 hash、split hash、配置、Git commit、设备和 checkpoint lineage 可追溯。
- 所有指标有限；没有 CPU fallback、混合精度或 TF32。

### 7.2 WSL `meep_env` 验收

```text
targeted preflight: 23 passed
full tests: 223 passed
python -m compileall -q src tests: passed
python -m phc2d.cli --help: passed
python -m phc2d.cli plot --help: passed
python -m phc2d.cli forward benchmark --help: passed
```

因此 A1 状态更新为 `formal_completed`。A2 保持未升级；本报告不授权启动 inverse、MPB replay、PINO、MaxwellNet 或 100k 扩容。
