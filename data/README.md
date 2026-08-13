<!--
  Local dataset catalog and directory policy for the phc2d workspace.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 数据目录

`data/` 保存本机数据集及其本地辅助文件。二进制数据继续由 Git 忽略；仅本说明和 [`catalog.yaml`](catalog.yaml) 纳入版本控制。

## 目录约定

| 位置 | 用途 |
|---|---|
| `data/` | 仍被配置、脚本或测试直接引用的兼容性入口 |
| `data/metadata/` | manifest、k-path sidecar 和 solver provenance |
| `data/archive/legacy_source/` | 已退出活动入口的早期全量/小样本数据 |
| `data/archive/legacy_random/` | 旧随机结构数据及配套生成产物 |
| `data/archive/paper_like_previews/` | 仅结构或预览阶段的 paper-like 数据与产物 |
| `data/archive/validation_probes/` | 不再被活动入口使用的历史验证小样本 |
| `data/artifacts/paper_like/` | 活动 paper-like 数据集的生成辅助产物 |
| `data/diagnostics/full_ibz/` | 历史 full-IBZ 调试与真实求解探针 |

根目录保留为兼容性命名空间，避免移动活动数据后破坏既有配置、脚本、测试和历史结果引用。新增数据应先登记到 [`catalog.yaml`](catalog.yaml)，再决定是否需要稳定的根目录入口。

## 当前主要数据

| 数据集 | 角色 | 主要形状 | 当前证据边界 |
|---|---|---|---|
| `full_cell_binary_bz9_10000_xingyi_v1.h5` | 二值全 BZ 正式数据 | `Wh=[10000,81,10]` | 星逸 real-MPB 生成完成，WSL `meep_env` 跨环境验收通过 |
| `full_cell_graded_bz9_10000_paracloud_m9_v1.h5` | 渐进介质全 BZ 正式数据 | `Wh=[10000,81,10]` | M9 real-MPB 生成完成，WSL `meep_env` 跨环境验收通过 |
| `query_dataset_v2_pilot_10000_mpb.npz` | V2 query/forward/inverse 主数据 | `rho=[10000,36]`, `Wh=[10000,31,10]` | manifest 记录为已完成物理复核 |
| `paper_like_50k_mpb.npz` | legacy paper-like 主 benchmark | `rho=[50000,36]`, `Wh=[50000,31,10]` | provenance 候选，正式物理复核仍受限 |
| `dataset_100k_mpb_train70000_val25000_test5000.npz` | legacy 100k benchmark | `rho=[100000,36]`, `Wh=[100000,31,10]` | 仅供明确标注为 legacy 的配置使用 |

较小的 `tiny`、64/256/1000 query 数据和两条 full-cell pilot64 数据仍保留在根目录，因为 smoke、收敛测试或数据缩放配置正在直接引用它们。

## 本次归档

以下文件没有活动配置、脚本、测试或实验入口引用，已移动但未删除：

| 原路径 | 新路径 |
|---|---|
| `data/dataset_small.npz` | `data/archive/legacy_source/dataset_small.npz` |
| `data/dataset_full.npz` | `data/archive/legacy_source/dataset_full.npz` |
| `data/dataset_full_mpb.npz` | `data/archive/legacy_source/dataset_full_mpb.npz` |
| `data/dataset_fem_check.npz` | `data/archive/validation_probes/dataset_fem_check.npz` |
| `data/legacy_random_64_mpb.npz` | `data/archive/legacy_random/legacy_random_64_mpb.npz` |
| `data/paper_like_100_structure_only.npz` | `data/archive/paper_like_previews/paper_like_100_structure_only.npz` |
| `data/paper_like_64_preview.npz` | `data/archive/paper_like_previews/paper_like_64_preview.npz` |
| `data/paper_like_64_structure_only.npz` | `data/archive/paper_like_previews/paper_like_64_structure_only.npz` |

相应的预览/生成产物和 full-IBZ 诊断目录也已移动到上表所述分类目录。归档文件内部的历史路径与 provenance 不做改写。

## 维护规则

- 不把 `.npz`、`.h5`、原始分片或大规模生成产物提交到 Git。
- 不依据文件名推断正式性；以 manifest、run artifacts 和验收状态为准。
- 移动活动数据前，必须同时审计 `configs/`、`scripts/`、`src/`、`tests/` 和 `experiments/` 的路径引用。
- 正式 MPB/Meep 验证继续遵守仓库规定的 WSL `Ubuntu` + conda `meep_env` 边界。
