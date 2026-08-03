<!--
  Near-field surrogate research profile.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# Near-Field Surrogate

目标：把已有 near-field 产物纳入 `phc2d` 任务族，评估 `epsilon_grid + frequency/k/polarization -> TE/TM field` 是否适合作为 FNO、U-Net、NeurOLight 或 Maxwell residual pilot 的训练目标。

配置入口：[`../../configs/research/near_field_surrogate.yaml`](../../configs/research/near_field_surrogate.yaml)

## Known Data

- `data/field_tiny_mpb.h5` 已存在，包含 `epsilon`、`rho`、`k_points`、`freqs`、`efield`、`hfield`。
- `runs/20260608/plot_dataset_near_fields_202606081551` 已存在，记录了 100 个样本、TE/TM、bands 2-4、resolution 64 的可视化 replay。

## Gate

1. 先完成 HDF5 schema inspection 与 dummy loader。
2. 有足够字段时跑 tiny train/eval；字段不足时只产出数据需求和 MPB/Meep 生成计划。
3. TE/TM 指标分开报告，不能只看图片是否好看。
4. Maxwell residual 只在 source、frequency、boundary、grid spacing 和 field components 足够时进入完整实现。
