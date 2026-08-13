---
title: 结构与 MPB
summary: 定义材料场、Bloch 本征问题、频带与带隙的稳定数学约定。
page_type: reference
status: draft
audience: 物理建模者、数据生产者与模型审阅者
owner: 研究生维护
last_reviewed: 2026-08-13
reading_time: 6 minutes
primary_layer: L1
---

<!--
  Structural and MPB formula reference for phc2d.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 结构与 MPB

二维周期结构可由材料参数 $\boldsymbol\rho$ 映射为晶胞内介电常数场

$$
\varepsilon(\mathbf r)=\mathcal G(\boldsymbol\rho;\,\text{lattice},\text{materials}).
$$

几何生成器 $\mathcal G$、对称约束、像素化分辨率和材料值都属于数据身份。

## Bloch 本征问题

对周期部分 $\mathbf u_{n\mathbf k}$，频域 Maxwell 问题可写为抽象形式

$$
\mathcal L_{\varepsilon,\mathbf k}\,\mathbf u_{n\mathbf k}
=\lambda_{n\mathbf k}\,\mathbf u_{n\mathbf k},
\qquad
\lambda_{n\mathbf k}=\left(\frac{\omega_{n\mathbf k}}{c}\right)^2.
$$

实际 TE/TM 表达式、离散网格和求解容差以 MPB 配置为准。网络预测频率或场时，应明确目标是 $\omega$、$f$、$fa/c$ 还是其他归一化量。

## 带隙

相邻第 $n$ 与 $n+1$ 条带之间的全局间隙可由

$$
f_\mathrm{lower}=\max_{\mathbf k} f_n(\mathbf k),
\qquad
f_\mathrm{upper}=\min_{\mathbf k} f_{n+1}(\mathbf k)
$$

确定。当 $f_\mathrm{upper}>f_\mathrm{lower}$ 时存在 gap。相对带隙常写为

$$
g=\frac{f_\mathrm{upper}-f_\mathrm{lower}}
{\tfrac12(f_\mathrm{upper}+f_\mathrm{lower})}.
$$

比较带隙前必须对齐 k 路径、band 索引、极化、分辨率和频率单位。代理预测的 gap 只能称为 surrogate gap，真实 MPB 重算后才是 replay gap。

