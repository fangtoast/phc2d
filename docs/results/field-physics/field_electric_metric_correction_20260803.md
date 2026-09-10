---
title: "电场度规勘误（2026-08-03）"
summary: "日期化实验与验证证据；正文保持原始记录。"
page_type: archive
status: historical
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 4 minutes
primary_layer: L3
evidence_date: 2026-08-03
evidence_scope: "日期化证据或历史记录；正文未因站点重构而改写。"
---

<!--
  Correction note for the electric-field metric used by field/PINO workflows.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="field-electric-metric-correction-20260803"></a>

# 电场度规勘误（2026-08-03）

!!! warning "历史证据"
    本页保留旧度规的修正记录；其中受影响的历史数值不能直接作为当前正式比较。

本文修正 TE Bloch eigenfield、`fno_field_v2` 与 adapted PINO 工作流中的电场内积、归一化、相位对齐及 relative $L_2$ 定义。修正前的 B3/B4 数值来自未加介电常数权重的电场度规，只能保留为历史诊断；在完成数据迁移、重训和三 seed 重评前，不得称为修正后结果。

**目录：** [正确度规](#正确度规) · [分母是否为一](#分母是否为一) · [实现范围](#实现范围) · [证据边界](#证据边界)

---

<a id="正确度规"></a>

## 正确度规

对广义本征问题

$$
\nabla\times\nabla\times E=\omega^2\varepsilon E,
$$

电场空间使用介电加权内积

$$
\langle E_1,E_2\rangle_\varepsilon
=\int_\Omega E_1^*(\mathbf r)\cdot\varepsilon(\mathbf r)E_2(\mathbf r)\,dV.
$$

预测场 $\widehat E$ 与真值 $E$ 的共享全局相位由

$$
\alpha=
\frac{\langle \widehat E,E\rangle_\varepsilon}
{|\langle \widehat E,E\rangle_\varepsilon|}
$$

确定，并将同一个 $\alpha$ 同时作用于预测的 $E$ 与 $H$。修正后的电场 relative $L_2$ 为

$$
e_E=
\frac{\|\alpha\widehat E-E\|_\varepsilon}
{\|E\|_\varepsilon}.
$$

磁场仍使用 $\mu=1$ 下的普通 $L_2$ 度规，但沿用由电场确定的同一全局相位，以保持 $E/H$ 的 Maxwell 关系。

---

<a id="分母是否为一"></a>

## 分母是否为一

若数据生成严格满足

$$
\|E\|_\varepsilon^2
=\sum_{x,y}\varepsilon(x,y)|E(x,y)|^2\,\Delta x\Delta y
=1,
$$

则电场 relative $L_2$ 的分母在该离散积分约定下恒为 1，此时 relative $L_2$ 与介电加权 absolute $L_2$ 数值相同。

实现仍显式计算真值分母，不把它硬编码为 1，原因有三点：旧 HDF5 使用未加权归一化；`complex64` 落盘会产生浮点偏差；指标函数也需要拒绝或显式暴露不满足新契约的数据。注意，若代码中的求和省略公共的 $\Delta x\Delta y$，原始离散和本身不等于 1，但该因子在 relative $L_2$ 的分子和分母中严格约去。

$H$ 没有被独立归一化；生成链路对 $E/H$ 使用同一个由 $E$ 得到的缩放因子。因此 $H$ relative $L_2$ 的分母一般不恒为 1。

---

<a id="实现范围"></a>

## 实现范围

- [`field_eigen.py`](https://github.com/fangtoast/phc2d/blob/master/src/phc2d/data/field_eigen.py)：新数据按 $\int E^*\varepsilon E\,dV=1$ 归一化，并将 schema 升为 `phc2d.field_eigen.v2`。
- [`field_eigen_v2.py`](https://github.com/fangtoast/phc2d/blob/master/src/phc2d/forward/field_eigen_v2.py)：训练相位、E loss 与评估 E relative $L_2$ 改用介电加权度规。
- [`field_eigen_b4.py`](https://github.com/fangtoast/phc2d/blob/master/src/phc2d/forward/field_eigen_b4.py)：seen-$k$ 场对照使用相同度规。
- [`torch_eigen_residual.py`](https://github.com/fangtoast/phc2d/blob/master/src/phc2d/physics/torch_eigen_residual.py)：PINO 场归一化项改为 $\|E\|_\varepsilon$；Rayleigh 商改为 $\langle E,\nabla\times\nabla\times E\rangle/\langle E,\varepsilon E\rangle$。
- [`field_eigen_te_contract.yaml`](https://github.com/fangtoast/phc2d/blob/master/configs/research/field_eigen_te_contract.yaml)：冻结新的归一化和内积契约。

---

<a id="证据边界"></a>

## 证据边界

本次完成的是指标、损失、数据契约和单元测试修正，没有生成新 MPB 数据，也没有重训模型。此前报告中的 U-Net/FNO/PINO E relative $L_2$、lambda selection、checkpoint selection 和基于旧 Rayleigh consistency 的结论均未被新实现重新验证。

下一证据门槛是：迁移或重新生成 v2 field dataset，使用修正损失重训匹配的 data-only FNO 与 adapted PINO，再按相同 split、seed、epoch 和 validation-only selection 协议重算场误差、频率误差、residual 与准入状态。
