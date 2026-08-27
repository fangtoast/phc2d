---
title: 正向代理结果
summary: 正向频带代理的公平对照、全胞双路线训练与升级模型评估索引。
page_type: landing
status: draft
audience: 模型研究者、复现执行者与结果审阅者
owner: 研究生维护
last_reviewed: 2026-08-27
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
| [Full-Wh 升级模型 50k](full_cell_forward_model_upgrade_50k_evaluation_20260810.md) | 五个升级模型的三 seed held-out test 与基线比较；保留 proxy 边界 |
| [Full-Wh 50k 单 seed R²补充](全胞前向50K模型R方补充评估_20260819.md) | seed 0 的 18 个 native 单元；逐坐标中心化 R²，不与 POST 数值直接排名 |
| [K20 与 50K 跨数据域正向验证](全胞K20与50K跨数据域正向验证_20260820.md) | N40 四个 Full-cell 域、68 个冻结 checkpoint、272 个 held-out surrogate cells；Draft |
| [EXP-FWD-007：K5训练与六域验证](EXP-FWD-007-K5训练与六域验证.md) | K5 17-task 与 F50=50,000 的六域 510-cell held-out 重推已完成；旧 272-cell 回归通过，bootstrap/resource 汇总仍缺，Draft 不作 promotion |
| [EXP-FWD-008：K5 固定 200 轮训练与六域验证](EXP-FWD-008-K5固定200轮训练与六域验证.md) | 17/17 完成 200 epoch，但仅 5/17 满足原主收敛门；未按原合同直接进入六域正式排名，后续由 EXP-FWD-009 接续 |
| [EXP-FWD-009/010/011：K5 权重热启动、joint 公平数据量与六域验证](EXP-FWD-009-K5权重热启动与六域验证.md) | 009 完成 510-cell；010/011 的 joint 各 4k 完成 200/300 轮与两组 6-cell 评估。300 轮 Graded/等权未收敛，六域相对 200 轮五升一降，只在 K5 Binary 击败最佳单路线，Draft |
| [EXP-FWD-012：工作频率先验覆盖审计](EXP-FWD-012-工作频率先验覆盖审计.md) | 公平 4k+4k train 与 validation 的无训练审计；X/M Bragg 先验最大 gap-center 命中均为 0，停止 loss-only 训练路线，Draft |
| [超参数灵敏度分析（历史单 seed）](灵敏度分析.md) | 局部搜索的解释性记录；残差 MLP 加深是待多 seed 确认的假设，不作正式排名或 promotion |

[返回研究结果与报告](../README.md)
