---
title: "EXP-FWD-012：工作频率先验覆盖审计"
summary: "在公平 4k+4k K5 train 与 validation 上审计结构条件 Bragg 频率窗口，决定是否准入 loss-only 训练。"
page_type: experiment
plan_state: completed
status: draft
audience: "正向代理研究者、实验执行者与结果审阅者"
owner: "研究生维护，导师审阅工作频率假设"
last_reviewed: 2026-08-27
primary_layer: L3
---

<!--
  EXP-FWD-012 experiment contract.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-FWD-012：工作频率先验覆盖审计

**研究问题：**只由输入材料场计算的 $X$ 方向一阶 Bragg 尺度，是否比同宽的 train-only 固定中心窗口更能覆盖 K5 Binary/Graded 的真实频率坐标与最大 gap center？

| 合同 | 固定值 |
|---|---|
| 数据 | EXP-FWD-010 同一稳定哈希子集：Binary 4,000 + Graded 4,000；另读各 lane validation 1,000 |
| 测试集 | 只核对 split 身份与 hash；禁止加载 test `Wh` 或材料场 |
| 主先验 | $\bar n=\langle\sqrt\varepsilon\rangle$，$\tilde f_0^{(X)}=1/(2\bar n)$ |
| 敏感性 | $\tilde f_0^{(M)}=\sqrt2/(2\bar n)$；不参与主 gate |
| $\epsilon_f$ | 两 lane train pooled 正相邻 band 间隔的 25% 分位数 |
| $\Delta_f$ | 两 lane train pooled 最近频率坐标距离的 90% 分位数 |
| 固定中心对照 | 两 lane train pooled 最大 gap center 中位数，使用相同 $\Delta_f$ |
| gap 范围 | lower band 2–9；complete/pseudo 状态分开保留 |
| 证据边界 | `surrogate_train_validation_prior_audit_no_test`；无训练、无 test、无 MPB/Meep |

## 通过与失败门

Binary、Graded validation 必须分别同时满足：

1. $X$ 窗口的坐标覆盖率不低于 `0.85`；
2. $X$ 窗口的最大 gap-center 命中率比固定中心同宽窗口至少高 `0.05`。

任一 lane 失败即记录 `prior_audit_fail`，停止 EXP-FWD-013 的自动准入；不得改用 test 调整中心、窗口或阈值。通过只允许提出 loss-only 训练授权，不构成代理性能、模型 promotion 或物理结论。

## 预期证据

标准 run 产物外，必须保存 `work_frequency_prior.json` 与 `work_frequency_metrics.json`，并在 `run.meta.json` 明确 `test_targets_loaded=false`。原始数组和逐样本记录留在 run，不复制进结果文档。

相关计划：[工作频率与带隙损失探索计划](../../docs/planning/工作频率与带隙损失探索计划.md)。

## 执行结论

2026-08-27 的 train/validation-only audit 得到 `prior_audit_fail`：Binary/Graded 坐标覆盖门通过，但最大 gap-center 相对固定中心改善门均失败，EXP-FWD-013 未准入。详见[结果页](../../docs/results/forward/EXP-FWD-012-工作频率先验覆盖审计.md)。
