<!--
  Supervised TE eigenfield baseline pilot evidence.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 1k TE Bloch 本征场监督基线，2026-07-20

在 WSL `Ubuntu` + conda `meep_env` 的 RTX 4060 GPU 上，使用冻结的 1k TE field dataset 完成 data-only `unet_field` 与 `fno_field` 各三个 seed 的训练。输入为 epsilon、physical k 与 band；输出为复数 E/H 与本征频率。训练中 `lambda_pde=0`，没有 PINO、MaxwellNet、MPB replay 或 inverse 集成。

| 模型 | test E relative L2 (mean ± std) | test H relative L2 | 结论 |
|---|---:|---:|---|
| `unet_field` | 0.99769 ± 0.00032 | 0.99935 ± 0.00026 | 仅以极小幅度优于零预测；`partial` |
| `fno_field` | 1.00123 ± 0.00031 | 1.00117 ± 0.00023 | 未达到零预测基线；`blocked` |

有效 benchmark 为本地 `runs/20260720/field_eigen_te_supervised_baseline_202607201328`。六个 case 均有 checkpoint、per-case 指标、模型身份和固定 750/125/125 source split lineage。

因此本阶段**不**构成稳定的双模型 supervised field baseline，也不能推进 data+residual、PINO、MaxwellNet 或 physics-aware inverse。下一道门槛是诊断并改善 data-only field 学习（条件编码、输出参数化、模型容量或训练方案），随后以相同三 seed 协议重跑两模型。
