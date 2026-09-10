---
title: 本征场与 adapted PINO
summary: 从复相位不唯一性、TE curl-curl residual、Rayleigh 项、实际 tensor 和源码解释监督 Field 与 adapted PINO。
page_type: explanation
status: draft
audience: 物理建模研究者、模型开发者与评审者
owner: 研究生维护，导师审阅物理主张
last_reviewed: 2026-08-14
reading_time: 18 minutes
primary_layer: L2
---

<!--
  Stable explanation of eigenfield models and adapted PINO.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 本征场与 adapted PINO

本征场任务从结构、Bloch 波矢和 band 条件预测 TE 本征对。难点不只是输出像素更多：本征模存在全局复相位不唯一性，场与频率必须一致，离散 residual 还依赖网格、边界和 normalization。

## 1. 问题与输入输出

模型学习

$$
G_\theta:(\varepsilon,\mathbf k,n)
\mapsto(\widehat{\mathbf E},\widehat{\mathbf H},\widehat f).
$$

项目的历史 TE 合同以 5 通道结构/坐标网格和 3 维 context 为输入，输出 6 个实通道表示 3 个复场分量，并由独立 frequency head 输出标准化 log-frequency。

`unet_field_v2` 与 `fno_field_v2` 都在此合同下训练。adapted PINO 复用 `fno_field_v2` 主干并增加物理损失；它不是独立 faithful PINO 架构。

## 2. 复相位不变监督

若 $\mathbf u$ 是本征模，则 $e^{i\phi}\mathbf u$ 表示同一物理解。直接逐点比较会把任意相位当作误差。项目以电场的介电加权内积确定公共相位，并对 E/H 同时对齐：

$$
\langle E_1,E_2\rangle_\varepsilon
=\int \overline{E_1(\mathbf r)}\,\varepsilon(\mathbf r)\,E_2(\mathbf r)\,d\mathbf r,
\qquad
q=\frac{\langle \widehat E,E\rangle_\varepsilon}
{|\langle \widehat E,E\rangle_\varepsilon|}.
$$

对齐后 $\widetilde E=q\widehat E$、$\widetilde H=q\widehat H$，监督项为

$$
\mathcal L_{\mathrm{field}}
=\lambda_E\frac{\|\widetilde E-E\|_\varepsilon^2}{\|E\|_\varepsilon^2}
+\lambda_H\frac{\|\widetilde H-H\|_2^2}{\|H\|_2^2}.
$$

频率头预测标准化 log-frequency：

$$
y_f=\frac{\log f-\mu_{\log f}}{\sigma_{\log f}},
\qquad \mathcal L_f=(\widehat y_f-y_f)^2.
$$

## 3. TE Bloch residual 与 Rayleigh 项

对 active electric components，离散 residual 对应

$$
\mathbf r=
\nabla_{\mathbf k}\times\nabla_{\mathbf k}\times\widehat{\mathbf E}
-(2\pi\widehat f)^2\varepsilon\widehat{\mathbf E}.
$$

实现使用带 Bloch 相位的有限差分。相对 residual 为

$$
R^2=\frac{\|\mathbf r\|_2^2}
{\|(2\pi\widehat f)^2\varepsilon\widehat{\mathbf E}\|_2^2+\epsilon}.
$$

Rayleigh 频率从同一 curl-curl 算子计算：

$$
f_R=\frac{1}{2\pi}
\sqrt{\frac{\langle E,\nabla_{\mathbf k}\times\nabla_{\mathbf k}\times E\rangle}
{\langle E,\varepsilon E\rangle}}.
$$

组合目标按配置启用：

$$
\mathcal L=
\mathcal L_{\mathrm{field}}+\lambda_f\mathcal L_f
+\lambda_{\mathrm{PDE}}\mathcal L_{\mathrm{PDE}}
+\lambda_R\mathcal L_R.
$$

## 4. 实际 tensor shape

| 对象 | shape | 语义 |
|---|---|---|
| `grid` | `[B,5,H,W]` | 结构与周期坐标/条件通道 |
| `context` | `[B,3]` | 归一化的 $k_x,k_y$ 与 band 条件 |
| `field` | `[B,6,H_f,W_f]` | 3 个复场分量的实/虚通道 |
| complex active field | `[B,H_f,W_f,3]` | 通道恢复后的复数 E/H |
| `frequency_z` | `[B]` 或 `[B,1]` | 标准化 log-frequency |
| `epsilon` | `[B,H_\varepsilon,W_\varepsilon]` | 在 loss 内派生到 field grid |

`H,W,H_f` 来自 HDF5 schema 与配置；64/96/128 只是在历史 gate 中使用的具体分辨率，不应写进通用模型。

## 5. 核心源码

以下相位对齐与介电加权损失是实际训练代码节选，核对版本为 `c347845`。完整源码见 [`forward/field_eigen_v2.py`](https://github.com/fangtoast/phc2d/blob/c347845/src/phc2d/forward/field_eigen_v2.py)。

```python
def phase_invariant_field_loss(
    torch, prediction, target, *, epsilon, e_weight=0.5, h_weight=0.5
):
    pred = _torch_active_complex(torch, prediction)
    truth = _torch_active_complex(torch, target)
    eps = derive_epsilon_on_field_grid_torch(
        epsilon.to(dtype=pred.real.dtype, device=pred.device),
        (int(pred.shape[1]), int(pred.shape[2])),
    )
    inner = (pred[..., :2].conj() * eps[..., None] * truth[..., :2]).flatten(1).sum(1)
    phase = inner / inner.abs().clamp_min(1.0e-12)
    aligned = pred * phase[:, None, None, None]
    e_loss = (
        (eps[..., None] * (aligned[..., :2] - truth[..., :2]).abs().square())
        .flatten(1).sum(1)
        / (eps[..., None] * truth[..., :2].abs().square())
        .flatten(1).sum(1).clamp_min(1.0e-12)
    ).mean()
    h_loss = (
        (aligned[..., 2:] - truth[..., 2:]).abs().square().flatten(1).sum(1)
        / truth[..., 2:].abs().square().flatten(1).sum(1).clamp_min(1.0e-12)
    ).mean()
    return (e_weight * e_loss + h_weight * h_loss) / (e_weight + h_weight), e_loss, h_loss, aligned
```

TE residual 与 resolution-calibrated PINO loss 见 [`physics/torch_eigen_residual.py`](https://github.com/fangtoast/phc2d/blob/c347845/src/phc2d/physics/torch_eigen_residual.py)。

## 6. 配置与最小命令

数据合同先冻结 source structure split、k/band 选择和 solver provenance；训练配置再选择 data-only 或 PINO regime。示意字段来自当前研究配置：

```yaml
selection:
  operator_learning: true
  source_structure_split_counts: {train: 192, val: 32, test: 32}
  k_per_structure: 4
promotion_gate:
  required_status: b2_residual_gate_pass
```

```powershell
python -m phc2d.cli forward train --config configs/research/field_eigen_te_v2_baseline_9216.yaml
python -m phc2d.cli forward benchmark --config configs/research/field_eigen_te_pino_9216.yaml
```

这些历史任务可能调用 real-MPB，正式执行与验收必须使用 WSL `meep_env`。

## 7. 产物与证据边界

Field/PINO run 应分开报告 E/H relative error、frequency error、residual、normalization、Rayleigh consistency、分辨率、source-structure split 与 runtime；checkpoint 必须记录 regime 和物理损失权重。

较低 residual 只能说明在声明的离散算子下更自洽，不能单独证明频率更准、场更准、跨分辨率稳定或 inverse reranking 有益。当前 Field/PINO 记录已归入[历史证据](../results/field-physics/README.md)；inverse admission 仍以冻结 gate 为准。
