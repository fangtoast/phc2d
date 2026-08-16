<!--
  Author: Ming Lyu <caref.lm@gmail.com>
  Copyright (c) 2026 Ming Lyu. All rights reserved.
-->

# 全胞数据生成方法与分布审查底稿

| Metadata | Value |
|---|---|
| Status | Draft factual reference / evaluation pending |
| Scope | 50k binary 与 50k graded 全胞数据 |
| Purpose | 客观整理当前生成方法，为后续分布、覆盖和训练有效性分析提供依据 |

相关依据：

- [B 50k 配置](../../../configs/research/full_cell_binary_bz9_50000_xingyi_v1.yaml)
- [G 50k 配置](../../../configs/research/full_cell_graded_bz9_50000_xingyi_v1.yaml)
- [随机结构生成](../../../src/phc2d/generators/full_cell.py)
- [材料场计算](../../../src/phc2d/data/full_cell.py)
- [样本调度与 split](../../../src/phc2d/data/full_cell_generation.py)
- [碰撞与介电常数内容哈希](full_cell_dielectric_hash_collision_design.md)

## 一、本文边界

本文不判断当前数据集是否有利于训练，也不提出 generator 改造方案。它只整理当前方法，
以支持后续集中分析以下问题：

1. 拒绝采样后的实际数据密度如何分布；
2. 固定 family 数量与有效独立样本数 $N_{\mathrm{eff}}$ 是否一致；
3. train、val、test 来自同一生成方法时，三个 split 的关系是什么；
4. 介电常数输入的分布与能带标签分布是否同时得到覆盖。

配置给出的是候选参数分布，最终 HDF5 保存的是通过验收门后的条件分布。本文分别称为
“提议分布”和“接受后分布”，不预设二者是否接近。

## 二、完整生成链路

当前每条路线的生成过程为：

```text
固定配置与 master seed
  -> 按样本索引确定 generator family / variant
  -> 由 experiment ID 和索引构造 sample ID
  -> 由 master seed、sample ID、family 派生样本 RNG
  -> 依次生成候选并计算 256×256 审计统计
  -> 第一个通过全部验收门的候选成为正式结构
  -> 按 stratum 和 sample ID 确定 split
  -> 对同一解析结构生成 32/64/128 栅格并运行 MPB
```

样本 RNG 的当前派生形式为：

$$
s_i=\operatorname{uint32}\left(
\operatorname{SHA256}(s_{\mathrm{master}},\mathrm{sample\_id}_i,\mathrm{family}_i)
\right).
$$

每个样本只有一个 RNG。若候选被拒绝，下一次 attempt 继续使用同一 RNG 的后续随机数，
而不是重新以 attempt 编号播种。不同 worker 数不改变样本的随机序列。

## 三、数据集组成与固定调度

两条路线各有 50000 个结构，均使用 seed `20260806`，split 均为
`40000/5000/5000`，材料常数均为 $\epsilon_{\mathrm{air}}=1.0$、
$\epsilon_{\mathrm{material}}=8.9$。

| 路线 | stratum | 调度规则 | 总数 |
|---|---|---|---:|
| B | `polar_hole_1` | 索引模 4 为 0 | 12,500 |
| B | `polar_hole_2` | 索引模 4 为 1 | 12,500 |
| B | `spectral_threshold` | 索引模 4 为 2 或 3 | 25,000 |
| G | `graded_core_1` | 偶数索引 | 25,000 |
| G | `graded_core_2` | 奇数索引 | 25,000 |

这些数量由确定性调度产生，不是由 categorical RNG 抽样得到。HDF5 的原始样本顺序也
保留上述周期模式。

## 四、B 路线的提议与接受规则

### 4.1 单孔与双孔极坐标结构

每个孔的边界为：

$$
r(\theta)=r_0\exp\left[
\sum_{m=2}^{5}(a_m\cos m\theta+b_m\sin m\theta)
\right].
$$

随机变量如下：

| 变量 | 单孔 | 双孔 |
|---|---|---|
| 基准半径 $r_0$ | $U(0.14,0.25)$ | 每孔 $U(0.10,0.18)$ |
| $a_m,b_m$ | 每阶独立 $N(0,0.12/m)$ | 相同 |
| 中心 | 在由该孔最大半径确定的内部区域均匀抽样 | 每孔相同 |
| 边界余量 | 最大半径外增加 $0.03$ | 相同 |

中心抽样区间依赖该次抽样得到的最大半径。双孔还要求两中心距离 $d$ 满足：

$$
0.60(R_1+R_2)\le d\le1.50(R_1+R_2),
$$

其中 $R_1,R_2$ 为两孔的最大径向范围。正式接受条件为：

- 空气占比在 $[0.10,0.50]$；
- 周期四邻域空气连通块数在 1 到 `hole_count` 之间；
- 普通栅格标记得到的最小空气块等效直径不小于 $0.04a$。

每个样本最多尝试 512 次。最终参数分布是上述随机变量在距离和验收条件下的接受后分布。

### 4.2 频谱阈值结构

配置使用 Fourier cutoff 8。对每个非零半平面频率 $(m_x,m_y)$，独立抽样正弦和余弦
系数，其标准差为：

$$
\sigma_{\mathbf m}=\exp\left[-\left(
\frac{\lVert\mathbf m\rVert}{k_c}
\right)^4\right],\qquad k_c\sim U(2.5,4.5).
$$

随机场在 256×256 审计栅格上归一化到标准差 1。随后抽样：

$$
f_{\mathrm{air}}^{\mathrm{target}}\sim U(0.10,0.50),
$$

$$
q=\operatorname{clip}\left(
f_{\mathrm{air}}^{\mathrm{target}}+U(-0.03,0.03),
0.07,0.53
\right).
$$

阈值取随机场的 $q$ 分位数；场值低于阈值的位置为空气。正式接受条件为：

- 实际空气占比在 $[0.07,0.53]$；
- 周期四邻域空气连通块数在 1 到 3 之间；
- 普通栅格标记得到的最小空气块等效直径不小于 $0.04a$。

每个样本最多尝试 512 次。保存参数同时包含 $k_c$、目标空气占比、实际阈值分位数、
阈值和全部 Fourier 系数。

## 五、G 路线的提议与接受规则

每个样本固定使用一个或两个周期椭圆空气井。每个井独立抽样：

| 变量 | 分布或变换 |
|---|---|
| 基准尺度 $b$ | $U(0.08,0.22)$ |
| 长宽参数 $a$ | $U(0.75,1.33)$ |
| 横向尺度 | $\sigma_x=b\sqrt{a}$ |
| 纵向尺度 | $\sigma_y=b/\sqrt{a}$ |
| 中心 | $x,y\sim U(-0.5,0.5)$，使用周期距离 |
| 角度 | $U(0,2\pi)$ |
| 空气井幅值 | $-U(0.90,1.40)$ |

背景 Fourier 场使用 cutoff 3。每个实模态的正弦和余弦系数独立服从零均值高斯分布，
标准差为：

$$
\sigma_{\mathbf m}=(1+\lVert\mathbf m\rVert^2)^{-1.25}.
$$

其余样本级随机变量为：

| 变量 | 分布 |
|---|---|
| Fourier 总强度 | $U(0.03,0.10)$ |
| sigmoid 温度 | $U(0.08,0.16)$ |
| 目标材料权重均值 | $U(0.60,0.78)$ |

生成器在 256×256 审计栅格上通过 80 次二分搜索确定 bias，使材料权重均值匹配随机
目标。正式接受条件为：

- 材料权重均值在配置区间外扩 0.01 的范围内；
- 权重不小于 0.60 的面积比例至少为 0.35；
- 权重不大于 0.10 的面积比例至少为 0.01；
- 最小权重不大于 0.05，最大权重不小于 0.90；
- 权重位于 $(0.05,0.95)$ 的灰度区域比例至少为 0.20；
- 高频能量比例不大于 0.05。

每个样本最多尝试 256 次。保存参数包含全部空气井、Fourier 系数、强度、温度、bias
和目标材料权重均值。

## 六、介电常数场与保存结果

材料权重统一映射为：

$$
\epsilon(x,y)=\epsilon_{\mathrm{air}}+
(\epsilon_{\mathrm{material}}-\epsilon_{\mathrm{air}})w(x,y).
$$

- B 路线的 $w$ 严格取 0 或 1；
- G 路线的 $w$ 是 sigmoid 输出的连续浮点值；
- 所有栅格使用周期单胞内的 cell-centered 坐标；
- 生成验收统计在 256×256 栅格计算；
- HDF5 当前保存 float32 的 `material_weight_grid_32/64/128` 与
  `epsilon_grid_32/64/128`；
- MPB 使用同一结构的解析材料函数，不以保存的栅格图像作为材料输入。

整数灰度阶与容差内容哈希属于单独的拟议数据合同，当前生成数据仍遵循上述浮点保存
方式。

每个已接受结构还计算以下统计量：材料权重均值与分位数、致密材料比例、近空气比例、
空气占比、灰度比例、p4m 误差、高频能量比例、周期与普通连通块数、最小空气块等效
直径及权重极值。

## 七、split 的形成方法

split 在 `polar_hole_1`、`polar_hole_2`、`spectral_threshold`、`graded_core_1` 和
`graded_core_2` 各 stratum 内分别进行。每个 stratum 的样本按
`SHA256(sample_id)` 排序，再依照 80%/10%/10% 分配到 train、val、test。

| stratum | train | val | test |
|---|---:|---:|---:|
| `polar_hole_1` | 10,000 | 1,250 | 1,250 |
| `polar_hole_2` | 10,000 | 1,250 | 1,250 |
| `spectral_threshold` | 20,000 | 2,500 | 2,500 |
| `graded_core_1` | 20,000 | 2,500 | 2,500 |
| `graded_core_2` | 20,000 | 2,500 | 2,500 |

split 哈希排序不读取介电常数场、生成参数或 MPB 标签。三个 split 使用相同的 family、
参数提议范围和验收条件。

## 八、当前制品能够提供的信息

`sample_plan.jsonl` 保存：sample ID、sample index、split、stratum、family、当前
geometry hash、接受后的生成参数和结构统计。

HDF5 保存：sample ID、family、material kind、完整 structure JSON、参数向量、三个
分辨率的材料权重与介电常数栅格、81 个 k 点的 10 条能带，以及三个 split 的索引。

需要注意：从已冻结 sample plan 恢复结构时，metadata 只重建结构统计和恢复标记，不会
恢复原始 attempt。当前制品也不保存每个被拒绝候选的参数和拒绝原因。因此：

- 可以直接统计接受后分布；
- 不能只凭最终 HDF5 还原完整提议分布；
- 不能从恢复后的结构记录稳定获得每个正式样本经历的拒绝次数。

## 九、围绕四个核心问题的待统计量

本节只列出分析输入，不给出有效性判断。

### 9.1 拒绝采样后的真实密度

按 family、stratum 和 split 分别统计：

- 所有保存随机参数的一维分布；
- 尺寸、位置、形状系数与材料占比的联合分布；
- $k_c$、阈值分位数、连通块和最小特征的联合分布；
- G 的尺度、井深、温度、Fourier 强度、目标均值与实际统计量的联合分布；
- 接受后样本在材料占比、频谱能量、连通性和灰度比例上的局部密度。

若需要直接计算接受率，还需额外取得或重新记录 attempt 数、候选参数和拒绝原因。

### 9.2 family 数量与有效样本数

按 family 和分辨率分别统计：

- 与 ID 无关的介电常数内容哈希唯一数；
- 32/64/128 表示层面的重复组；
- 结构特征或栅格空间中的最近邻距离；
- 高密度簇的大小、簇内 split 组成和参数差异；
- 总样本数、唯一内容数和选定相似度定义下的 $N_{\mathrm{eff}}$。

$N_{\mathrm{eff}}$ 的相似度与容差定义应在计算结果前单独冻结。

### 9.3 train、val、test 的关系

对三个 split 分别比较：

- family 与 stratum 数量；
- 接受后生成参数和结构统计分布；
- 内容哈希交集；
- split 内及跨 split 最近邻距离；
- 输入特征分箱占用率；
- 能带标签的边际与联合分布。

这些统计用于描述三个 split 在同一生成协议下的关系，不预先把相似或不相似解释为
优点或缺点。

### 9.4 输入分布与能带标签分布

输入侧至少整理：

- 材料占比、灰度分布、连通性、最小特征和界面长度；
- 空间质心、方向性、p4m 误差；
- 径向与角向频谱、相关长度和高频能量；
- family 与生成参数。

标签侧至少整理：

- 各 k 点和 band 的范围、分位数与局部密度；
- 相邻 band 间距和各类 band gap；
- 能带张量的低维表示与聚集情况；
- 相近输入之间的标签差异，用于描述局部敏感性；
- 输入低密度区与标签低密度区之间的对应关系。

## 十、后续分析的输出边界

基于本文可以形成两类客观制品：

1. 生成方法清单：记录提议变量、固定调度、验收条件、split 和保存字段；
2. 实际分布审计：从正式 sample plan 与 HDF5 计算上述统计量。

只有在实际分布审计完成后，才进入“是否适合训练”“哪些区域覆盖不足”和“如何改善”
的评价与决策。本文本身不作这些结论。
