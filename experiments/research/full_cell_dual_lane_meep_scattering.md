<!--
  Finite-array Meep near/far-field response experiment profile.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 双路线全胞有限阵列 Meep 近远场响应

| Metadata | Value |
|---|---|
| Status | Draft experiment profile |
| Primary layer | L3 experiment/evidence |
| Audience | 光子晶体研究者、实验复核者与图件维护者 |
| Owner | 研究生维护，导师确认散射合同是否用于论文主张 |
| Last reviewed | 2026-08-06 |
| Reading time | 5 minutes |

本文回答“冻结样本在一套明确的有限阵列、入射源和开放边界条件下会产生怎样的受激近场与角向远场”。它不回答无限周期晶体的唯一散射响应，也不把一次有限尺寸计算升级为数据集验收。

## 任务合同

| 字段 | 合同 |
|---|---|
| Goal | 为既有 `S01–S16` 样本计算真正的受激 Meep 近场和远场 |
| Evidence inputs | 已冻结的样本索引、解析介质函数和 MPB 带边频率 |
| Allowed changes | 新增 Meep 求解器、响应画廊、配置、测试与 run 产物 |
| Non-goals | 不改变原始 HDF5、MPB 频带、数据集验收状态或项目级结论 |
| Forbidden promotions | smoke 不能写成正式 32 样本结果；有限阵列响应不能写成无限晶体固有远场 |
| Required validation | WSL `meep_env`、有限数组检查、近远场有限性、图件渲染与完整测试 |
| Human acceptance | 确认 5×5 阵列、正入射和带边频率是否符合论文要回答的物理问题 |

## 冻结散射问题

- 结构：把每个解析全胞沿 (x/y) 各复制 5 个周期，阵列外为空气。
- 激励：从左向右（`+x`）正入射的二维 TE 脉冲，Meep 分量为 `Hz`。
- 频率：同一样本最大完整带隙或最大伪带隙的上边缘频率。
- 边界：四周 PML；源位于闭合 near-to-far 监测框之外。
- 近场：DFT 稳态电能密度 (arepsilon(|E_x|^2+|E_y|^2)/4)，逐样本归一化用于显示。
- 箭头：(langle\mathbf S\rangle=\tfrac12\operatorname{Re}(\mathbf E\times\mathbf H^*)) 的时间平均 Poynting 矢量；青色箭头表示入射方向。
- 远场：闭合空气监测框经 Meep near-to-far 变换得到的径向 Poynting 通量，画为完整 $360^\circ$ 极坐标方向图。
- 收敛：DFT 场必须在最大运行时间之前达到衰减阈值，否则 fail closed，不生成可接受响应。

配置入口为 [`full_cell_dual_lane_meep_scattering_v1.yaml`](../../configs/research/full_cell_dual_lane_meep_scattering_v1.yaml)。正式执行必须位于 WSL Ubuntu 的 `meep_env`：

```bash
PYTHONPATH=src python -m phc2d.cli plot full-cell-scattering-gallery \
  --config configs/research/full_cell_dual_lane_meep_scattering_v1.yaml
```

双路线各一个样本的 smoke 使用：

```bash
PYTHONPATH=src python -m phc2d.cli plot full-cell-scattering-gallery \
  --config configs/research/full_cell_dual_lane_meep_scattering_v1.yaml \
  --max-samples-per-dataset 1
```

## 证据边界

这套响应依赖有限阵列尺寸、入射方向、脉冲带宽、PML、分辨率与监测框。修改任何一项都可能改变近场热点、能流线与远场瓣形。图中的逐样本归一化颜色和极径不能比较不同样本的绝对散射功率；若论文需要绝对反射、透射或散射截面，还必须增加空场基准和功率归一化。

## 正式结果

32 个样本的正式 WSL `meep_env` 运行、数值审计和图件说明见 [2026-08-06 Meep 近远场结果报告](../../docs/results/field-physics/full_cell_dual_lane_meep_scattering_20260806.md)。
