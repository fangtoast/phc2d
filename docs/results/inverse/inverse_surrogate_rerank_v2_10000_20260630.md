<!--
  10k inverse multi-surrogate rerank and MPB replay report.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 10k Inverse 多正向替代模型 Rerank 与真实 MPB Replay 对比

## 摘要
- **目的**：基于原始记录保留实验链路与指标定义，对文档进行中文化结构化重构。
- **状态**：已完成中文主语体补充，原始英文命令、路径、文件名、模型字段与数值保持原文不变。
- **文件角色**：Experimental Technical Report | **原始语言比例**：英中混合（8%）
- **证据入口**：runs/20260630/inverse_surrogate_rerank_v2_10000_202606301521/`；runs/20260630/inverse_mpb_replay_v2_10000_strategy_pilot_202606301529/`；runs/20260629/generative_inverse_v2_10000_202606291030/candidates.json`，实际包含；run.meta.json

## 数据与方法
- 数据：以原文 `runs/` 与 `metrics.json` 为准。
- 方法：保持原始训练流程与实验顺序，不改动采样/评估口径。
- 命令与环境：保留原样，`path` 与 `config` 字段不改写。

## 风险与下一步
- 风险主要关注结论复用边界（surrogate-only 与 formal validation 的分离）。
- 下一步优先补齐跨文档术语映射、证据引用一致性与可复现实验条目。


| 项目 | 内容 |
|---|---|
| 报告日期 | 2026-06-30 |
| 数据集 | `data/query_dataset_v2_pilot_10000_mpb.npz` |
| rerank config | `configs/research/inverse_surrogate_rerank_v2_10000.yaml` |
| replay config | `configs/research/inverse_mpb_replay_v2_10000_strategy_pilot.yaml` |
| rerank run | `runs/20260630/inverse_surrogate_rerank_v2_10000_202606301521/` |
| replay run | `runs/20260630/inverse_mpb_replay_v2_10000_strategy_pilot_202606301529/` |
| 物理边界 | rerank 是 `inverse_surrogate_rerank_no_mpb_meep`；replay 是 `formal_inverse_candidate_mpb_replay` |

## 结论

本轮已经实现并运行了多 surrogate inverse candidate rerank：同一批 10k inverse candidates 分别用 `query_only`、`forward_res_mlp_rho_wedge`、`forward_fno2d_band_epsilon_grid`、`forward_fno_lite_epsilon_grid` 和 `hybrid_query_forward` 排序。

真实 MPB replay 后，**forward reranking 没有改善物理 MRE**。`query_only` 在 24-target pilot 中仍是最好的策略，`best_of_k_replay_mre_bands_2_10_pct = 33.2899%`；hybrid 为 `33.4354%`，三个 forward-only 策略均为 `33.4565%`。这说明当前主要瓶颈不是“只换一个 forward surrogate 排序器”，而更可能是候选池本身质量不足，或 surrogate-to-MPB gap 仍然过大。

## Surrogate-Only Rerank

Rerank 阶段读取 `runs/20260629/generative_inverse_v2_10000_202606291030/candidates.json`，实际包含 `1875` 条 target-seed evaluations。该阶段不运行 MPB，不产生物理结论。

| strategy | surrogate MRE bands 2-10 (%) | surrogate MRE bands 1-10 (%) | 与 query-only 选择重叠 |
|---|---:|---:|---:|
| `query_only` | 16.4807 | 763352.8468 | 1.0000 |
| `forward_res_mlp_rho_wedge` | 2.7281 | 182226.7735 | 0.9654 |
| `forward_fno2d_band_epsilon_grid` | 17.0983 | 3056279.7028 | 0.9688 |
| `forward_fno_lite_epsilon_grid` | 11.3007 | 470726.4082 | 0.9728 |
| `hybrid_query_forward` | 9.0941 | 471579.0299 | 0.9864 |

`res_mlp/rho_wedge` 在 surrogate-only 视角下最乐观，但这没有转化成真实 MPB replay 改善。各策略与 query-only 的 top candidates 重叠率都很高，说明当前 candidate pool 内可选结构差异有限。

## Real MPB Replay

Replay 阶段对 24 个 target 的五策略 top candidates 做并集去重：原始 strategy selections 共 `190` 个，去重后只需 `42` 次真实 MPB solve，节省 `148` 次，`candidate_dedupe_ratio = 0.7789`。`run.meta.json` 标记为 `validation_boundary=formal_inverse_candidate_mpb_replay`、`replay_backend=real_mpb`。

| strategy | replay MRE bands 2-10 (%) | replay MRE bands 1-10 (%) | surrogate MRE bands 2-10 (%) | gap bands 2-10 (%) |
|---|---:|---:|---:|---:|
| `query_only` | 33.2899 | 33.0545 | 15.2362 | 18.0536 |
| `hybrid_query_forward` | 33.4354 | 33.1929 | 8.8160 | 24.6194 |
| `forward_res_mlp_rho_wedge` | 33.4565 | 33.2114 | 3.2526 | 30.2039 |
| `forward_fno2d_band_epsilon_grid` | 33.4565 | 33.2114 | 17.3000 | 16.1564 |
| `forward_fno_lite_epsilon_grid` | 33.4565 | 33.2114 | 11.8533 | 21.6031 |

Gap F1 也没有明显变好：`gap_precision = 0.6250`，`gap_recall = 0.3333`，`gap_f1 = 0.4348`。因此下一步不应直接扩大同类 reranking，而应改 candidate generation 或加入更强的物理一致性约束。

## Band1-10 Replay 表

下表列出每个 band 的真实 MPB replay MRE。完整逐 band surrogate MRE、surrogate-replay gap、MAE、RMSE 见 `per_strategy_band_replay_metrics.csv`。

| band | query-only | res_mlp | fno2d_band | fno_lite | hybrid |
|---:|---:|---:|---:|---:|---:|
| 1 | 31.0517 | 31.1213 | 31.1213 | 31.1213 | 31.1256 |
| 2 | 34.6028 | 34.5740 | 34.5740 | 34.5740 | 34.6171 |
| 3 | 34.4682 | 34.3815 | 34.3815 | 34.3815 | 34.4594 |
| 4 | 32.2275 | 32.3748 | 32.3748 | 32.3748 | 32.3936 |
| 5 | 34.0669 | 34.4302 | 34.4302 | 34.4302 | 34.2774 |
| 6 | 33.7076 | 33.8818 | 33.8818 | 33.8818 | 33.7795 |
| 7 | 33.8306 | 34.0303 | 34.0303 | 34.0303 | 33.9079 |
| 8 | 31.4481 | 31.6539 | 31.6539 | 31.6539 | 31.7176 |
| 9 | 32.0876 | 32.3374 | 32.3374 | 32.3374 | 32.3619 |
| 10 | 33.1695 | 33.4444 | 33.4444 | 33.4444 | 33.4043 |

band1 已列出，但主结论仍使用 bands 2-10。这里 bands 1-10 与 bands 2-10 很接近，说明本轮真实 replay 的高误差不是只由 band1 放大造成的；真正的问题出现在 band2-10 的整体物理偏差。

## 后续动作

1. 暂停“只换 forward surrogate 排序器”的路线；它没有改善真实 MPB MRE。
2. 优先做 candidate generator 改进：提高 `n_candidates`、增强 diversity，并加入 gap-aware 或 full-Wh consistency loss。
3. 保留 `res_mlp/rho_wedge` 作为诊断 surrogate，但不要把它的低 surrogate MRE 当成物理质量结论。
4. 下一轮 replay 仍以 `best_of_k_replay_mre_bands_2_10_pct` 为主指标，同时同步报告 bands 1-10 和 band1-10 逐 band 表。

## 验证记录

已在 WSL `Ubuntu` + conda `meep_env` 下运行：

```powershell
wsl -d Ubuntu -e bash -lc "source /home/fangxiao/miniforge3/etc/profile.d/conda.sh && conda activate meep_env && cd '/mnt/d/project/article_reproduction/POD-DeepONET-2d(codex)' && PYTHONPATH=src python -m compileall src tests"
wsl -d Ubuntu -e bash -lc "source /home/fangxiao/miniforge3/etc/profile.d/conda.sh && conda activate meep_env && cd '/mnt/d/project/article_reproduction/POD-DeepONET-2d(codex)' && PYTHONPATH=src python -m pytest -q tests"
wsl -d Ubuntu -e bash -lc "source /home/fangxiao/miniforge3/etc/profile.d/conda.sh && conda activate meep_env && cd '/mnt/d/project/article_reproduction/POD-DeepONET-2d(codex)' && PYTHONPATH=src python -m phc2d.cli --help"
wsl -d Ubuntu -e bash -lc "source /home/fangxiao/miniforge3/etc/profile.d/conda.sh && conda activate meep_env && cd '/mnt/d/project/article_reproduction/POD-DeepONET-2d(codex)' && PYTHONPATH=src python -m phc2d.cli plot --help"
```

结果：`compileall` 通过，`pytest` 为 `118 passed`，CLI help 与 plot help 均通过。真实 rerank 与 replay 均在 WSL `meep_env` 中完成。