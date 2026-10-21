---
title: "Full-cell inverse 解析参数 codec 预检"
summary: "日期化实验与验证证据；正文保持原始记录。"
page_type: archive
status: historical
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 6 minutes
primary_layer: L3
evidence_date: 2026-08-06
evidence_scope: "日期化证据或历史记录；正文未因站点重构而改写。"
---

<!--
  Evidence report for the full-cell analytic inverse codec preflight.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="full-cell-inverse-codec-preflight"></a>

# Full-cell inverse 解析参数 codec 预检

!!! warning "历史证据"
    本页是解析参数 codec 的接口预检，不属于正式 inverse 训练、筛选或 real-MPB replay 结果。

| Metadata | Value |
|---|---|
| Status | Draft evidence report；codec checks passed，runner admission pending |
| Date | 2026-08-06 |
| Primary layer | L4 implementation |
| Implementation commits | `def517f`、`d7b489e` |
| Environment | Windows local Python；WSL Ubuntu + conda `meep_env` |
| Data | binary/graded full-cell 10k HDF5 的冻结 `idx_train` |
| Formal MPB/Meep | Not run；本报告不构成 replay 证据 |
| Owner | 研究生维护，待人工审阅 |
| Reading time | 5 minutes |

## 一、结论

五个 full-cell 解析变体的 train-split-only codec、hard analytic decode 和 Torch soft differentiable decode 已实现，并通过真实 10k HDF5 schema/宽度检查。该结果解除的是 **参数表示与可微解码的 L4 前置门**，不表示 inverse runner、GPU 训练、候选健康或 MPB replay 已完成。

旧 inverse runner 仍以 `rho_wedge/rho_grid` 为输出，并且现有网络末层、validation/early-stopping 与 checkpoint 元数据尚未适配 analytic tensor。因此 `full_cell_analytic_params` 暂未加入正式训练配置，避免出现“配置解析成功但训练语义错误”的假准入。

## 二、实现范围

实现位于 [`src/phc2d/inverse/full_cell_codec.py`](https://github.com/fangtoast/phc2d/blob/main/src/phc2d/inverse/full_cell_codec.py)，包括：

- 五变体身份识别、固定字段/辅助字段/学习字段分离；
- train-only robust/log transform，周期中心与 angle 的 sin/cos 编码；
- codec manifest 序列化与恢复；
- `FullCellStructureSpec` hard decode 与精确 round-trip；
- graded RBF/Fourier、binary polar holes 和 binary spectral threshold 的 Torch soft raster；
- 从 HDF5 `idx_train` 拟合 codec，读取时不加载大体积 raster 数据集；
- production physical-width fail-closed 检查。
- train-only 冻结物理范围与 hard decode 越界拒绝。

对应测试位于 [`tests/test_phc2d_full_cell_inverse_codec.py`](https://github.com/fangtoast/phc2d/blob/main/tests/test_phc2d_full_cell_inverse_codec.py)。

## 三、真实数据合同结果

| 数据 | 变体 | train 样本 | 物理宽度 | 网络编码宽度 | 结果 |
|---|---|---:|---:|---:|---|
| binary 10k | `binary_polar_holes_1` | 2,000 | 11 | 13 | pass |
| binary 10k | `binary_polar_holes_2` | 2,000 | 22 | 26 | pass |
| binary 10k | `binary_spectral_threshold` | 4,000 | 289 | 289 | pass |
| graded 10k | `graded_rbf_fourier_1` | 4,000 | 57 | 60 | pass |
| graded 10k | `graded_rbf_fourier_2` | 4,000 | 63 | 69 | pass |

编码宽度大于物理宽度的部分来自周期中心和 angle 的 sin/cos 对，不是新增物理自由度。HDF5 数据本身及跨环境 acceptance 证据仍以 [GPU/超算预检报告](../data-provenance/paracloud_gpu_preflight_20260806.md) 和对应数据生成记录为准。

每个变体进一步从冻结 train/validation 索引抽取 32 个结构，共 160 个样本。五类结果均为 `geometry_hash=32/32`、64 raster exact `32/32`、128 raster exact `32/32`；soft decoder gradient 全部有限且非零，梯度范数范围为 `0.0370–0.2685`。

## 四、执行命令与结果

Windows 编辑期检查：

```powershell
python -m pytest -q tests/test_phc2d_full_cell_inverse_codec.py
python -m ruff check src/phc2d/inverse/full_cell_codec.py tests/test_phc2d_full_cell_inverse_codec.py
```

结果为 `10 passed`，Ruff 为 `All checks passed`。

WSL 正式 Python 合同检查：

```powershell
wsl -d Ubuntu -e bash -lc "source /home/fangxiao/miniforge3/etc/profile.d/conda.sh && conda activate meep_env && cd '/mnt/d/project/article_reproduction/POD-DeepONET-2d(codex)' && PYTHONPATH=src python -m pytest -q tests/test_phc2d_full_cell_inverse_codec.py tests/test_phc2d_full_cell.py tests/test_phc2d_inverse_candidate_protocol.py"
```

结果为 `50 passed`。同一环境下，binary/graded 两个真实 HDF5 均通过 train-only codec 拟合和 production width 检查。未调用 MPB/Meep，因此不能把这些结果写成物理 replay 验收。

## 五、仍未通过的 gate

在提交任何 full-cell inverse GPU 训练前，还必须完成：

1. 新 runner 数据适配器按变体加载 `Wh`、encoded parameters 和冻结 split；
2. analytic 输出取消旧 `Sigmoid` 假设，并为周期 sin/cos 对增加单位圆约束；
3. Stage A 加入 validation loss、best checkpoint、patience 与 decoded-raster loss；
4. checkpoint 保存 condition normalization、codec manifest、variant 和数据 hash；
5. Stage B 把 analytic tensor 经过 soft decoder 接入冻结 Full-Wh surrogate；
6. hard candidate、geometry lineage、candidate-health 与 M9 replay manifest 接通；
7. N40 单卡训练 smoke 通过后，再验证 18-job 瞬时并发能力。

8-target inverse MPB smoke 截至本报告日期仍未执行。

## 六、人工理解记录

```text
解决了什么问题：把 padding 后 params 改造成五个变体各自可训练、可逆且可微的解析参数表示。
为什么这样实现：周期坐标和 angle 不能按普通欧氏标量学习，固定 mode/order 也不能让网络重复预测。
改变了哪一层：L4 inverse codec 与测试；没有改变 L3 promotion 或 MPB 结论。
用什么验证：生成结构 round-trip、hard/soft raster 对齐、Torch 梯度、manifest 恢复、真实 10k idx_train 宽度检查和 WSL 回归测试。
剩余风险：runner 仍是 rho 输出合同，正式 GPU 训练和 MPB replay 均未准入。
```

[↑ 回顶部](#full-cell-inverse-codec-preflight)
