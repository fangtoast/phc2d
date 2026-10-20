---
title: 单孔形状迁移测试协议
summary: 以六种文献采用孔形、等面积扫描和冻结模型检验几何迁移，先过求解准入门。
page_type: plan
status: draft
audience: 正向代理研究者与论文审阅者
owner: fangtoast
last_reviewed: 2026-10-20
reading_time: 5 minutes
primary_layer: L3
plan_state: active
---

<!--
  Frozen supplemental experiment contract.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="single-hole-transfer"></a>
# 单孔形状迁移测试协议

**选取圆孔、椭圆孔、等边三角孔、方孔、矩形孔、正六边形孔六族，每族200个确定参数点，共1200个。** 目前只冻结几何及求解准入协议；解析结构已接入现有接口不表示MPB标签已生成或已物理验收。正式CPU生成保持未提交，先完成32结构数值pilot。

## 一、孔形依据与适用范围

| 形状 | 原始文献 | 本实验借用范围 |
|---|---|---|
| 圆、方、正六边形 | [Dyogtyev等，JAP 2010](https://eprints.gla.ac.uk/41394/)，DOI 10.1063/1.3247544 | 文中明确研究这些孔形；不是商业使用频次调查 |
| 等边三角形 | [Andreani与Gerace，PRB 2006](https://arxiv.org/abs/0706.0395)，DOI 10.1103/PhysRevB.73.235114 | 有限厚度高折射率膜中的孔形研究，不迁移其slab带隙结论 |
| 椭圆、矩形、正六边形 | [Kassa-Baghdouche，APPA 2020原文](https://przyrbwn.icm.edu.pl/APP/PDF/138/app138z3p12.pdf)，DOI 10.12693/APhysPolA.138.421 | 旋转和轴比的参数化依据；原文为三角晶格slab，本实验仍为二维方晶格 |

三篇DOI与标题已通过Crossref核验，孔形根据作者/机构原始来源核查。它们支持“文献采用的代表性孔形”，不证明市场占有率或所有孔形同样常见。圆角方孔可作为制造圆角敏感性补充；槽孔、十字孔、环孔涉及额外尺度或拓扑，当前不扩展。

## 二、面积控制与取样

主集孔面积比例统一为0.10–0.20，晶格常数1、ε_air=1、ε_material=8.9、TE。固定中心，孔与周期像之间保留正间距；不通过平移或对称旋转重复增加独立样本。

| 形状 | 参数数目 | 数量 |
|---|---|---:|
| 圆 | 200个面积比例 | 200 |
| 椭圆、矩形 | 各10个面积比例 × 4个轴比[1.25,1.5,2,3] × 5个角度[0,45°] | 各200 |
| 三角、方、六边形 | 各20个面积比例 × 10个角度 | 各200 |

圆与椭圆半轴按 `sqrt(f*q/pi)`、`sqrt(f/(pi*q))`；矩形半边长按 `sqrt(f*q)/2`、`sqrt(f/q)/2`；正n边形外接圆半径按 `sqrt(2*f/(n*sin(2*pi/n)))`。角度按方晶格D4与孔本身旋转对称性取基本区间：方孔0–45°，三角/六边形0–15°。孔形类型和面积比例、轴比、角度全部写入解析参数，不用随机极坐标近似多边形。

该集是确定性参数扫描，不是制造分布的iid抽样。主结果同时报告逐族指标与等族宏平均；参数点bootstrap不得解释为制造总体或跨训练seed不确定性。

## 三、求解与冻结迁移门

输入沿用256主网格面积平均至64的Binary表示；MPB直接读取同一解析几何。目标严格保持原81点顺序、10条TE能带、4条guard、mesh_size=3、tolerance=1e-7。先比较32个覆盖六族与参数边界的结构在r64/r96/r128下的差异，另用原生circle接口检查解析接口。r128只是数值参考；随机结构的历史分辨率门不自动覆盖尖角孔。

CPU生产复用星仪一节点64个单线程worker。正式标签须待pilot、求解器合同、WSL Ubuntu + meep_env小样本复算过门后再提交；不静默换分辨率或重采样。原81点结果不作完整带隙证明。[MPB官方带边说明](https://mpb.readthedocs.io/en/stable/Python_Tutorial/)。

冻结checkpoint与normalizer后比较三个核心模型，先做无微调直接迁移。报告Band1、Bands2–10、Bands1–10 MRE、逐结构P95、MAE、逐band与参数误差分布；代表样本按预定面积/轴比/角度层与最差例规则选择。角点误差要与标签数值误差分别记录。不得根据该测试重新挑模型或拟合normalizer。

配置：[EXP-DAT-007](../../configs/research/full_cell_single_hole_exp_dat_007_v1.yaml)。几何接口为 `phc2d.data.single_hole.build_single_hole_test_plan` 与 `analytic_single_hole_v1`。数据输出遵守标准run制品、解析参数、样本计划、配置和solver版本/耗时/hash，不将配置就绪标为标签完成。

数值pilot预定失败条件：任一solver失败、非有限值、目标shape错误或256 raster面积误差超过0.01均停止；相对r128的Bands2–10 MRE要求总体median≤0.10%、P95≤0.50%、每族P95≤0.75%。这些沿用历史数值门作为pilot准入，32结构不能证明所有参数点已收敛；若未通过，保留失败证据并重新审阅分辨率合同，不自动升档。
