---
title: 本征场与 PINO
summary: 说明本征场监督、物理残差和 adapted PINO 的指标边界。
page_type: reference
status: draft
audience: 场模型开发者、物理损失维护者与审阅者
owner: 研究生维护
last_reviewed: 2026-08-13
reading_time: 6 minutes
primary_layer: L2
---

<!--
  Eigenfield and PINO formula reference for phc2d.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 本征场与 PINO

场学习任务同时涉及监督误差、频率一致性和离散物理残差。三者测量不同对象，不能用一个下降替代另一个成功。

## 监督项

对预测场 $\hat{\mathbf u}$ 与参考场 $\mathbf u$，典型相对误差为

$$
\mathcal L_\mathrm{field}
=\frac{\|\hat{\mathbf u}-\mathbf u\|_2}
{\|\mathbf u\|_2+\epsilon}.
$$

复场需要先声明实虚部、幅相或其他表示，并处理本征模的相位或符号不唯一性。

## 物理残差

给定预测本征值 $\hat\lambda$，残差可抽象为

$$
\mathbf r=\mathcal L_{\varepsilon,\mathbf k}\hat{\mathbf u}
-\hat\lambda\hat{\mathbf u},
\qquad
\mathcal L_\mathrm{phys}=\frac{\|\mathbf r\|_2}
{\|\hat\lambda\hat{\mathbf u}\|_2+\epsilon}.
$$

离散算子、边界处理、网格间距和 normalization 必须与数据协议一致。否则 residual 数值没有可比较性。

## 组合目标与证据边界

adapted PINO 可使用

$$
\mathcal L=\alpha\mathcal L_\mathrm{field}
+\beta\mathcal L_\mathrm{freq}
+\gamma\mathcal L_\mathrm{phys}.
$$

权重和启用阶段属于实验配置。报告应分别展示 field、frequency 与 residual 指标，并说明是否经过真实求解器复核；较低 residual 本身不是频率更准或物理验收通过。

