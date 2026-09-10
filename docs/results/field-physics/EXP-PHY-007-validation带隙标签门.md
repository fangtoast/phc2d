---
title: "EXP-PHY-007：validation 带隙标签门结果"
summary: "64 个冻结 validation 结构均完成 real-MPB 自适应加密，但只有 18 个结构同时满足 edge 与 resolution 稳定，标签门失败并停止 loss 训练。"
page_type: evidence
status: draft
audience: "光子晶体物理建模研究者、正向代理研究者与结果审阅者"
owner: "研究生维护，导师审阅标签门与训练准入"
last_reviewed: 2026-09-02
reading_time: 5 minutes
evidence_date: 2026-09-02
primary_layer: L3
evidence_scope: "EXP-PHY-007 K5 Binary/Graded validation-only 32+32 structure adaptive band-edge refinement; no test, training, or N40"
dataset_scope: "K5 Binary/Graded Wh[10000,81,10], all validation samples frozen by the EXP-PHY-005 manifest"
evidence_level: physical_validation
evidence_status: completed
validation_backend: wsl_mpb
source_artifacts:
  - runs/20260902/EXP-PHY-007_202609021705
---

<!--
  EXP-PHY-007 curated result.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-PHY-007：validation 带隙标签门结果

**结论：标签门失败，EXP-FWD-016 不解锁。** 64/64 个冻结 validation 结构均完成 WSL `meep_env` real-MPB 求解，且没有 solver failure；但只有 18/64 个结构的全部九个相邻 band pair 同时满足连续两层 edge 收敛与 resolution 64→96 稳定。终态为 `adaptive_label_gate_fail`，因此本轮不提交 N40 loss 训练，也不调整阈值后重跑。

这不是“自适应加密无效”的结论。末层 edge shift 已在 573/576 个 pair 上低于阈值，主要未通过项是样本级连续两层收敛与 Binary 的 resolution 稳定性。有限四层局部搜索仍不构成连续 BZ 的数学全局证明。

## 标签门结果

| 指标 | Binary | Graded | 合计 | 门控解释 |
|---|---:|---:|---:|---|
| 完成结构 | 32/32 | 32/32 | 64/64 | 求解完整 |
| 连续两层 edge 收敛结构 | 13 | 10 | 23/64 | 未覆盖全部结构 |
| 九 pair 全部同时通过结构 | 8 | 10 | 18/64 | 标签门要求 64/64，失败 |
| 末层 edge 稳定 pair | 285/288 | 288/288 | 573/576 | 三个 Binary pair 仍不稳定 |
| resolution 稳定 pair | 232/288 | 288/288 | 520/576 | 56 个 Binary pair 未过 |
| solver failure | 0 | 0 | 0 | 数值执行完整 |

求解总墙钟时间为 `8733.37 s`。全部 64 个结构都执行至少三层加密；62 个结构达到四层，2 个结构在第三层满足早停合同。

## 粗标签与精标签差异

| 指标 | 结果 | 解释 |
|---|---:|---|
| materially changed pair | 556/576 | $9\times9$ 粗标签普遍高估相邻 gap |
| complete→pseudo 翻转 | 3 | Binary 96 B8–B9、Binary 9980 B9–B10、Graded 3330 B9–B10 |
| coarse / refined complete pair | 8 / 5 | 精标签中的 complete 样本更稀少 |
| 最大粗网格 gap 高估 | 0.04619 | 超过预注册 $5\times10^{-4}\,a/\lambda$ 阈值近两个数量级 |
| 最大 gap center shift | 0.01046 | 粗中心不足以承担精确工作频率监督 |

B1–B2 在 64 个结构中仍为 `0` 个 refined complete gap。若只看 EXP-FWD-016 使用的 B2–B10 候选，refined 最大 gap 的 complete 样本为 Binary 4/32、Graded 1/32；粗到精的最大 pair 选择分别改变 3/32 与 7/32。候选排序与 complete-recall 目标因此同时受到标签漂移和类别稀少限制。

## 证据完整性与边界

| 项目 | 证据 |
|---|---|
| 正式 run | `runs/20260902/EXP-PHY-007_202609021705/` |
| 启动代码身份 | `10e71dd733c07b66ce39528f665694ce0ab8a94f` |
| resolved config SHA-256 | `f3ad79d458555ae11e243fd9efea6019b77375f159ad0504c1d0db99f9520d2b` |
| sample manifest SHA-256 | `6edb684906147fd64628ae69f9882c3e62673fb28ac005aa10507d4eab38a4c1` |
| pair CSV SHA-256 | `d268e3b2d97b59fc59876bb0951e5f64e65452314653042987142077a099f0da` |
| refined NPZ SHA-256 | `c759949cbf85af599ef8ed1161d73da454b528a5b5382a521d9984e13759c3a3` |
| 禁止项 | `training_started=false`、`test_targets_loaded=false`、`n40_used=false` |

长任务运行期间仓库 HEAD 被其他提交推进，通用元数据在结束时记录了结束 commit `c059938`。`run.provenance.correction.json` 依据 reflog 与首条 run 日志记录启动 commit `10e71dd`；从启动到结束，正式配置、加密器、物理 primitive 与 solver 路径无差异。该 correction 不修改 solver 数组或指标。

## 决策

- 按预注册停止条件，不提交 EXP-FWD-016，不扩 seed，不读取 held-out test。
- 不把 18 个通过结构外推为完整 validation 标签可用，也不降低阈值追求训练准入。
- 若要继续，应另立物理标签实验，先决定是增加自适应层数，还是提高 Binary 主分辨率；二者成本与科学问题不同，不能在本 run 上事后混合。

原始逐层历史见 `history.csv`，逐 pair 指标见 `gap_refinement_samples.csv`，完整 k 点与能带见 `refined_bands.npz`，终态见 `metrics.json` 与 `results.json`。
