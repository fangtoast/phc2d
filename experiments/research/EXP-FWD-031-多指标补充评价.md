---
title: EXP-FWD-031 多指标补充评价
summary: 对四域完整报告的冻结预测补算相对 L²、绝对误差与尾部误差。
page_type: research_experiment
status: draft
audience: 实验执行者与论文作者
owner: fangxiao
last_reviewed: 2026-10-29
primary_layer: L3
---

<!--
  Frozen multi-metric supplement to the four-domain forward campaign.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-FWD-031 多指标补充评价

本补充评价检验模型排序是否依赖误差定义，复用[四域完整报告](../../docs/reports/EXP-FWD-031四域完整测试结果_20261029.md)的 checkpoint、数据和测试范围。其状态独立于原 MRE 测试完整性；本页为 Draft。

| 项目 | 冻结合同 |
|---|---|
| 身份 | EXP-FWD-031 的跨历史批次补充评价；历史模型仍保留 EXP-FWD-026/027/028/029/030 来源 |
| 四域主矩阵 | 13 配置 × 4 来源 × 4 目标 × 3 seed = 624 单元，原域只计对角一次 |
| 新几何 | 13 配置 × 4 来源 × 2 目标 × 3 seed = 312 单元 |
| 混合训练消融 | 3×3/7×7 × 2 lane × 5 分支 × 4 独立目标 × 3 seed = 240 单元；K5+K20 混合训练身份不变 |
| 总清单 | 1,176 个唯一单元，392 组三 seed；不补造 mixed 自身缺失的 test |
| 核心指标 | Band 1–10 MRE、逐结构相对 L²、MAE、MSE、全局 RMSE、逐结构 L² 的 P95 |
| 辅助范围 | Band 2–10、Band 1、逐 band；R² 与能带顺序违反率 |
| 来源门 | 文件 SHA、checkpoint SHA、数据 SHA、split/索引顺序、k 点顺序与原指标范围匹配 |
| 数值门 | MRE 绝对差不超过 1e-5 个百分点；RMSE² = MSE；零真值范数暂停验收 |
| 汇总 | seed 0/1/2，ddof=1；先每 seed 算分位数；四域等权且完整才报告宏平均 |
| 推理门 | 031 先完成全部 36 个原域单元及 MRE 回归，再扩展到全部 216 个单元；任何失败停止，不自动重试 |
| 禁止变更 | 不重训、不重选 checkpoint、不覆盖原结果、不据新排名修改原测试合同 |

指标公式、输入 schema 与命令见[频谱多指标合同](../../docs/参考/频谱多指标合同.md)。使用已有 NPY、NPZ 或 `predictions.h5`；无完整预测时才安排冻结 checkpoint 重推。历史单元和 031 的模型规模、训练策略、收敛状态继续分开记录。

实现分别由纯数组指标、预测文件审计、冻结推理和绘图模块承担。`jobs/paracloud_spectral_metrics.sh` 只负责已授权 allocation 内的薄启动，不自动提交任务。远端资源范围须按 `AGENTS.md` 在只读预检后取得明确授权；本协议本身不代表作业已提交。

产物写入独立 `runs/` 目录，包含冻结输入清单、逐单元审计、逐样本压缩 CSV、逐 band 表、三 seed 表、等权比较表、源码与 manifest 哈希以及图件。未通过或未处理单元保留 blocked 状态。只有 1,176 个单元和 392 组三 seed 全部通过才可称本补充评价完整。

工程验收使用 WSL Ubuntu / `meep_env`；真实数组复算与公式测试分别报告。证据保持 surrogate held-out/transfer，不升级为独立 MPB/Meep 物理验收。当前结果见[补充评价结果](../../docs/results/forward/EXP-FWD-031-多指标补充评价.md)。
