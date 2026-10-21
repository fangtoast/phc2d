---
title: 正向代理结果
summary: 正向频带代理的公平对照、全胞双路线训练与升级模型评估索引。
page_type: landing
status: draft
audience: 模型研究者、复现执行者与结果审阅者
owner: 研究生维护
last_reviewed: 2026-09-20
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
| [EXP-FWD-022：分支依赖与独立能力](EXP-FWD-022-分支依赖与独立能力.md) | 1,080 冻结干预与 30 项独立训练完成；均值排序两同两反，按 seed 为一同向、两反向、一不稳定；固定预算 Full 领先但仅 11/30 通过收敛门，Draft |
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
| [EXP-FWD-014：gap 决策损失短预算筛选](EXP-FWD-014-gap决策损失短预算筛选.md) | N40 seed0 配对训练；k 点与多数 gap 指标改善，但全谱 MRE 安全门失败，停止自动三 seed 扩展，Draft |
| [EXP-FWD-015：gap 决策近简并审计](EXP-FWD-015-gap决策近简并审计.md) | 近简并只带来最多 3.0 个百分点容差收益；低命中主要是模型排序与 complete-gap 召回问题，Draft |
| [EXP-FWD-019：组件归因三种子确认](EXP-FWD-019-组件归因三种子确认.md) | 42项F50 validation-only确认完成；ConvNeXt大核信号最强，FNO/Joint频谱删除受容量混杂，TFNO channel未通过两路线一致性门，Draft |
| [EXP-FWD-020：全局—局部带隙算子六域交叉评估](EXP-FWD-020-全局局部带隙算子六域交叉评估.md) | 36 个冻结 checkpoint 的 216-cell held-out surrogate 评估已完成；跨域 MRE 明显高于 native，7x7 对 3x3 没有全局一致优势；G2 未通过，不作 promotion，Draft |
| [EXP-FWD-023：固定模型十二域交叉测试](EXP-FWD-023-十二域交叉测试.md) | 63 个冻结 checkpoint 的 K5/K8/K10/K15/K20/F50 双路线 756-cell 交叉测试；S3-BG/TFNO 十二域等权最低，限 seed0 surrogate，Draft |
| [EXP-FWD-023：七种训练制度十二域完整结果](EXP-FWD-023-七种训练制度十二域完整结果.md) | 按 T01–T07 分组，以 M01–M09 为行、K5/K8/K10/K15/K20/F50 为列，完整呈现 Binary/Graded 十二域的 756/756 个结果格，Draft |
| [EXP-FWD-023：多域统一模型与历史单域模型对比](EXP-FWD-023-多域与历史单域模型对比.md) | T01–T07 七种混合 8k 制度与 K5/K20/F50 六类历史单域模型的同测试样本描述性对比；列出 M01–M09 全部旧六域结果及七种制度的新 K 域结果，保留预算/轮数不一致边界，Draft |
| [超参数灵敏度分析（历史单 seed）](灵敏度分析.md) | 局部搜索的解释性记录；残差 MLP 加深是待多 seed 确认的假设，不作正式排名或 promotion |

[返回研究结果与报告](../README.md)
