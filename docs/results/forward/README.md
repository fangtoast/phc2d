---
title: 正向代理结果
summary: 正向频带代理的公平对照、全胞双路线训练与升级模型评估索引。
page_type: landing
status: draft
audience: 模型研究者、复现执行者与结果审阅者
owner: 研究生维护
last_reviewed: 2026-08-14
reading_time: 3 minutes
primary_layer: L3
---

<!--
  Forward surrogate result landing page.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 正向代理结果

建议从公平协议进入，再阅读数据规模升级和模型升级结果。训练集重构或 final-fit monitor 不能替代冻结的 held-out test。

| 结果页面 | 主要内容与边界 |
|---|---|
| [A1 Forward/Query 公平对照](a1_forward_query_fair_v1_20260726.md) | 10k V2、三 seed 的公平协议比较 |
| [全胞双路线 10k](full_cell_dual_lane_forward_training_20260806.md) | 4 models × 3 seeds × 7 表征、交叉评估和分辨率实验；threshold real-MPB 尚未执行 |
| [全胞双路线 50k](full_cell_dual_lane_50000_forward_training_20260808.md) | 50k 波次、held-out test、跨路线和分辨率辨别；final-fit 仅作 monitor |
| [Full-Wh 升级模型 50k](full_cell_forward_model_upgrade_50k_evaluation_20260810.md) | 五个升级模型的单 seed held-out test 与基线比较；保留 proxy 边界 |

[返回研究结果与报告](../README.md)
