# 光子能带对介电函数傅里叶分量的敏感度

本文档研究光子晶体第 $n$ 条能带的频率 $\omega_n(\vec k)$ 对介电函数傅里叶分量 $\varepsilon_{\vec G}$ 的敏感度，以及这种敏感度随 $|\vec G|$ 增大而衰减的规律。

本文采用物理学视角。“近似无关”是指在给定结构范围、扰动幅度和误差容限下，高 $|\vec G|$ 分量造成的频率变化已经可以忽略。研究目标是找出这个近似成立的尺度、条件和统计规律。

本文属于旁支研究计划，不进入当前主线项目的完成声明。

## 1. 基本物理图像

低阶光子能带通常对应较长的场变化尺度。介电结构中变化很快的细节与模式能量密度重叠时，其正负贡献容易在单胞积分中相互抵消。因此可以预期：

> 对较低 band，介电函数的低 $|\vec G|$ 分量通常起主要作用；随着 $|\vec G|$ 增大，频率敏感度总体衰减。

band index 增大时，模式通常包含更细的空间结构，因而可能对更大的 $|\vec G|$ 保持敏感。材料反差、结构形态、Bloch 波矢和局部谱结构也会改变实际衰减速度，但不改变上述基本图像。

这里需要区分三个变量：

- $\omega_n$ 是电磁本征频率；
- $\vec k$ 是 Bloch 波矢；
- $\vec G$ 描述介电结构在单胞中的空间频率。

较低的 band index、小 $|\vec k|$ 和小 $|\vec G|$ 并不等价。本研究关心的是在固定 $n$ 和 $\vec k$ 后，$\omega_n(\vec k)$ 对不同 $\vec G$ 的结构分量有多敏感。

“近似无关”应始终带有容差。例如可以问：

> 对 band 1–5，在当前结构分布中保留到多大的 $G_c$，可以使忽略更高空间频率后造成的相对频率变化在绝大多数结构上低于 $0.1\%$？

## 2. 对物理图像的数学化描述

### 2.1 介电结构与频率灵敏度

周期介电函数可以展开为

$$
\varepsilon(\vec r)
=
\sum_{\vec G}
\varepsilon_{\vec G}
e^{i\vec G\cdot\vec r}.
$$

对小介电扰动，Maxwell 本征频率的一阶变化为

$$
\frac{\delta\omega}{\omega}
=
-\frac{1}{2}
\frac{
\int_{\Omega}
\delta\varepsilon(\vec r)
|\vec E(\vec r)|^2\,d\vec r
}{
\int_{\Omega}
\varepsilon(\vec r)
|\vec E(\vec r)|^2\,d\vec r
}.
$$

因此，$\omega_n(\vec k)$ 对 $\varepsilon_{\vec G}$ 的一阶灵敏度正比于模式电能密度

$$
u_{n,\vec k}(\vec r)
=
|\vec E_{n,\vec k}(\vec r)|^2
$$

在 $\vec G$ 处的傅里叶系数：

$$
s_{n,\vec k,\vec G}
\propto
\frac{
\left|
\widehat{u}_{n,\vec k}(\vec G)
\right|
}{
\int_{\Omega}
\varepsilon(\vec r)
u_{n,\vec k}(\vec r)\,d\vec r
}.
$$

这给出了核心数学联系：

> 结构空间频率的物理敏感度谱，可以直接由本征模能量密度的傅里叶谱得到。

### 2.2 高频累计影响与实用 cutoff

单个 $\vec G$ 可能受到晶格方向和模式形态影响，因此更适合研究 cutoff 之外的累计灵敏度：

$$
T_{n,\vec k}(G_c;\varepsilon)
=
\left(
\sum_{|\vec G|>G_c}
s_{n,\vec k,\vec G}^{2}
\right)^{1/2}.
$$

若高频介电扰动的总幅度不超过 $\Delta$，则一阶频率变化可以用

$$
\left|
\delta\log\omega_n(\vec k)
\right|
\lesssim
\Delta
T_{n,\vec k}(G_c;\varepsilon)
$$

控制。

因此，可以把满足

$$
\Delta
T_{n,\vec k}(G_c;\varepsilon)
\le
\tau
$$

的 $G_c$ 视为误差容限 $\tau$ 下的实用空间频率 cutoff。

### 2.3 结构总体上的统计关系

若结构来自概率分布 $P(\varepsilon)$，可以定义概率 cutoff

$$
G_{\tau,q}
\left(
n,\vec k;P,\Delta
\right)
=
\inf
\left\{
G_c:
\Pr_P
\left[
\Delta T_{n,\vec k}(G_c;\varepsilon)
\le\tau
\right]
\ge q
\right\}.
$$

它表示：对分布 $P$ 中至少比例 $q$ 的结构，需要保留到多大的 $G_c$，才能将高频扰动造成的一阶频率变化控制在 $\tau$ 内。

由此可以进一步研究

$$
G_{\tau,q}
=
F
\left(
n,\vec k,
\text{material},
\text{morphology},
P,
\tau,q,\Delta
\right),
$$

这正是本旁支项目希望定量建立的关系。

## 3. 现有的相关理论分析

### 3.1 Maxwell 扰动论

Maxwell 本征频率扰动论已经给出频率对小介电变化的一阶响应。其直接意义是：只要已有本征场，就可以通过一次 FFT 得到全部可解析 $\vec G$ 的局部灵敏度，而不必对每个 Fourier 分量分别重新求解。

### 3.2 Fourier 衰减与场的空间尺度

模式能量密度的 Fourier 系数会随 $|\vec G|$ 增大而衰减，因此高空间频率介电扰动的一阶影响总体趋小。若场更光滑，衰减通常更快；若结构包含高反差和尖锐界面，衰减可能变慢。

在附加正则性条件下，可以得到

$$
s_{n,\vec k,\vec G}
\le
\frac{
C_{n,\vec k,s}
}{
(1+|\vec G|)^s
}.
$$

这为基本物理图像提供了理论依据，但常数和实际衰减区间仍依赖具体结构与模式。

### 3.3 Maxwell 均匀化

均匀化理论说明，当结构尺度远小于观测波长时，快速变化介质常可由有效介质描述。这与本研究的物理图像一致：细尺度结构对低频响应的影响可以被压缩到较少的有效参数中。

均匀化主要研究有限幅度且尺度不断缩小的微结构，本项目则首先研究固定结构附近的 Fourier 灵敏度。两者相关，但对应不同的定量问题。

### 3.4 随机本征问题与 Monte Carlo

随机系数本征问题已有 Monte Carlo、quasi-Monte Carlo 和参数截断理论，可用于估计结构总体上的平均值、分位数和误差。

这些理论提供了统计工具，但尚未直接给出二维光子晶体中

$$
(n,\vec k,\vec G)
\longmapsto
s_{n,\vec k,\vec G}
$$

的定量关系。

## 4. 现有理论尚未给出的部分

### 4.1 缺少可直接使用的 band–cutoff 关系

现有理论能够解释高 $|\vec G|$ 灵敏度为什么衰减，但不能仅根据 band index $n$ 直接给出实用 cutoff。实际尺度还依赖 $\vec k$、材料反差、结构形态、模式空间分布和目标误差。

### 4.2 缺少有限误差下的定量尺度

渐近衰减本身没有回答：

- 从哪个 $|\vec G|$ 开始可以忽略高频贡献；
- 达到给定频率误差需要多大 cutoff；
- cutoff 如何随 band、$\vec k$ 和结构变化；
- 平均趋势是否会被少量高敏感度结构破坏。

### 4.3 缺少从局部灵敏度到有限结构变化的验证

一阶公式描述小扰动，而实际研究还可能涉及低通介电函数、像素变化、边界平滑或重新二值化。需要通过真实 Maxwell 求解器判断一阶灵敏度在多大范围内能够预测有限结构变化。

### 4.4 缺少明确结构分布上的一般统计

“所有光子晶体的平均”没有唯一含义。必须先指定结构分布 $P$，包括材料范围、对称性、占空比、最小特征尺寸和几何先验。

当前项目的 p4m 二值结构分布可以作为第一个明确总体，但其结论不能直接推广到全部二维光子晶体。

## 5. 可进行的研究方向

### 5.1 使用现有数据建立第一版灵敏度谱

当前项目已有：

- 约 10,000 个结构的频率数据，覆盖 $\Gamma-X-M-\Gamma$ 路径和 band 1–10；
- 256 个结构的 9,216 条本征场记录，覆盖路径内 4 个 $\vec k$ 点和 band 2–10；
- 可用于计算 $|\vec E|^2$ Fourier 谱的本征场数据。

第一阶段可以直接：

1. 对每个 structure–$\vec k$–band 计算本征场能量密度；
2. 通过二维 FFT 得到 $s_{n,\vec k,\vec G}$；
3. 按 $|\vec G|$ 壳层和晶格对称性聚合；
4. 计算累计尾部 $T_{n,\vec k}(G_c)$；
5. 比较不同 band、$\vec k$ 和结构形态下的 cutoff。

第一版结论只适用于当前 p4m 二值结构、固定材料反差、TE 模式和高对称路径。若希望推广到整个二维布里渊区，需要增加 off-path $\vec k$ 数据。

### 5.2 使用 MPB 校准物理意义

从低、中、高 $|\vec G|$ 选择代表性扰动，用 MPB 中心有限差分比较本征场公式与实际频率变化：

$$
g_{\vec G}^{\mathrm{FD}}
=
\frac{
\log\omega(\varepsilon+\delta\phi_{\vec G})
-
\log\omega(\varepsilon-\delta\phi_{\vec G})
}{
2\delta
}.
$$

随后对介电函数进行不同 cutoff 的低通重建，并用 real MPB 计算

$$
E_{\omega}(G_c)
=
\operatorname{MRE}
\left(
\omega[\varepsilon_{\le G_c}],
\omega[\varepsilon]
\right).
$$

这一阶段的核心问题是：

> 一阶灵敏度尾部 $T_{n,\vec k}(G_c)$ 能否预测有限低通后的真实频率误差？

### 5.3 建立 Monte Carlo 统计关系

对指定结构分布 $P$，建议报告：

- 灵敏度和累计尾部的均值、RMS 与分位数；
- 不同 band 和 $\vec k$ 的 $G_{\tau,q}$；
- cutoff 超过给定尺度的概率；
- cutoff 与材料反差、结构形态和场局域化的关系。

统计时应以 source structure 为独立抽样单位，并对当前占空比分层数据使用与目标分布一致的权重。

最终希望得到经验关系

$$
G_{\tau,q}
=
F_{\mathrm{empirical}}
\left(
n,\vec k,
\text{material},
\text{morphology},
P,
\tau,q,\Delta
\right).
$$

### 5.4 扩展验证范围

若现有数据中出现稳定规律，再增加：

- 不可约布里渊区内部的 off-path $\vec k$；
- 多个介电反差；
- 多个结构与场分辨率；
- 不同最小特征尺寸和结构族。

这些扩展用于判断经验关系中哪些部分具有一般性，哪些只属于当前数据分布。

### 5.5 与机器学习主线的连接

> **审查建议：** CNN 和 U-Net 以空间局部性和权重共享为基础归纳偏置，优先建模邻近区域关系，再通过层级卷积、降采样和多尺度融合形成大范围理解。光子能带来自带周期边界条件的全局 Maxwell 本征问题，其主要物理关联大概率不完全符合“邻近关系优先”的图像模型假设；U-Net 的多尺度结构可以缓解这种差异，但不能预先保证局部卷积是最合适的归纳偏置。
>
> 可用本研究检验这种潜在失配：若灵敏度主要集中在低 $|\vec G|$ 和大尺度结构上，全局谱表示可能更自然；若较高 $|\vec G|$、界面和局部形态仍有稳定贡献，CNN/U-Net 的局部与多尺度建模可能具有额外价值。

研究结果可以用于：

- 判断结构输入需要保留的空间分辨率；
- 判断 FNO 的 Fourier mode cutoff 是否覆盖物理敏感尺度；
- 分析 CNN 局部卷积是否捕获了有效的界面与局部结构尺度；
- 检查 inverse generator 是否利用了 surrogate 与 real MPB 灵敏度不一致的高频方向。

模型梯度只能用于辅助解释，物理结论应以 Maxwell 本征场和 real-MPB 复核为准。

### 5.6 最小推进门槛

建议先完成：

1. 使用现有本征场计算 FFT 灵敏度谱；
2. 覆盖至少 100 个独立结构、3 个以上 band 和 3 类 $\vec k$ 位置；
3. 对至少 20 个结构完成 MPB 有限差分校准；
4. 检验灵敏度尾部能否预测低通后的 real-MPB 频率误差。

若 cutoff 在结构之间没有稳定统计规律，或不能预测有限结构变化，则将该方法保留为模式诊断工具；若通过上述验证，再扩展数据范围。

## 6. 参考文献

1. S. G. Johnson et al., “Perturbation theory for Maxwell's equations with shifting material boundaries,” Physical Review E 65, 066611 (2002), [论文 PDF](https://math.mit.edu/~stevenj/papers/JohnsonIb02.pdf)，[DOI](https://doi.org/10.1103/PhysRevE.65.066611)。
2. P. D. Lamberti and M. Lanza de Cristoforis, “A few results on permittivity variations in electromagnetic cavities,” Journal of Differential Equations 330, 65–101 (2022), [论文页面](https://www.sciencedirect.com/science/article/pii/S0022039622003916)。
3. G. B. Folland, Fourier Analysis and Its Applications, Wadsworth & Brooks/Cole (1992)。
4. M. H. Taibleson, “Fourier coefficients of functions of bounded variation,” Proceedings of the American Mathematical Society 18 (1967), 766，[DOI](https://doi.org/10.1090/S0002-9939-1967-0212477-6)。
5. M. Costabel, M. Dauge and S. Nicaise, “Singularities of Maxwell interface problems,” ESAIM: M2AN 33(3), 627–649 (1999), [论文与书目信息](https://www.numdam.org/item/M2AN_1999__33_3_627_0/)。
6. G. Kristensson and N. Wellander, “Homogenization of the Maxwell Equations at Fixed Frequency,” SIAM Journal on Applied Mathematics 64(1), 170–195 (2003), [DOI](https://doi.org/10.1137/S0036139902403366)。
7. K. Cherednichenko and S. D'Onofrio, “Operator-norm homogenisation estimates for the system of Maxwell equations on periodic singular structures,” Calculus of Variations and Partial Differential Equations 61, 67 (2022), [开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8827329/)。
8. A. D. Gilbert et al., “Analysis of quasi-Monte Carlo methods for elliptic eigenvalue problems with stochastic coefficients,” Numerische Mathematik 142, 863–915 (2019), [预印本](https://arxiv.org/abs/1808.02639)。
9. [MPB Developer Information](https://mpb.readthedocs.io/en/latest/Developer_Information/)。

## 7. 当前判断

该方向值得继续。已有理论解释了高 $|\vec G|$ 灵敏度为何总体衰减，并给出了可由本征场直接计算的表达式；尚缺的是对实际光子晶体结构族的有限 cutoff、band 依赖和统计置信度进行定量测量。

合适的研究定位是：

> 利用大样本 Maxwell 本征场和真实求解器校准，建立带容差、带结构分布、带 band 和 $\vec k$ 条件的介电空间频率敏感度尺度。
