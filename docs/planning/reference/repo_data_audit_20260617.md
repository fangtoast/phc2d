---
title: "Repo 与数据审计记录"
summary: "历史研究记录；正文保持原始时间语境，不代表当前状态。"
page_type: archive
status: historical
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 5 minutes
primary_layer: L3
evidence_date: 2026-06-17
evidence_scope: "日期化证据或历史记录；正文未因站点重构而改写。"
---

<!--
  Repository and data audit for the phc2d research plan.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# Repo 与数据审计记录

## 摘要
- **目的**：基于原始记录保留实验链路与指标定义，对文档进行中文化结构化重构。
- **状态**：已完成中文主语体补充，原始英文命令、路径、文件名、模型字段与数值保持原文不变。
- **文件角色**：Project Docs | **原始语言比例**：英中混合（8%）
- **证据入口**：runs/20260608/plot_dataset_near_fields_202606081551`；results.json

## 任务边界
- 目标边界：文档结构重写与中文化表达，不修改代码、实验脚本与模型参数。
- 不覆盖运行产物，仅整理文档链路和表述。

## 进展状态
- 已完成：中文化主框架与摘要落位。
- 未完成：全文逐段逐句人工校对（保留英文原文仅作证据对照）。

## 当前风险
- 部分文件存在大量英文原文，术语一致性依赖后续逐条复核。
- 命令/路径/文件名中英文杂糅时需保持原写。

## 下一步
- 统一 `runs/`、`run_dir`、`surrogate_eval`、`mpb_replay_eval` 术语解释。
- 对新增章节与原文索引做一次 `rg` 链接可读性扫描。


审计日期：2026-06-17
仓库：`D:\project\article_reproduction\POD-DeepONET-2d(codex)`

## 当前架构

| 区域 | 状态 |
|---|---|
| `src/phc2d` | 活跃公共包，已有 data、generators、models、training、inverse、surrogates、replay、plot、runs、cli 骨架 |
| `configs` | 已按 `reproduction`、`research`、`smoke`、`formal` 分组 |
| `experiments/research` | 研究 profile 入口 |
| 旧 `legacy/reproduction` / `research/ai4phc` | 已退出活跃目录，只能从 Git history 恢复算法逻辑 |
| `phc2d cli` | `plot` 已有可执行后端；`data/forward/inverse/replay` 是预留入口，需要补 YAML + native executor |

## 现有 forward 数据

| 路径 | schema 摘要 | 备注 |
|---|---|---|
| `data/tiny.npz` | `rho=(12,36)`、`Wh=(12,31,10)`、split、`config_yaml` | tiny smoke dataset |
| `data/legacy_random_64_mpb.npz` | `rho=(64,36)`、`Wh=(64,31,10)`、split、gap test fields | MPB small |
| `data/paper_like_64_mpb.npz` | `rho=(64,36)`、`Wh=(64,31,10)`、split、gap test fields | paper-like smoke |
| `data/paper_like_50k_mpb.npz` | `rho=(50000,36)`、`Wh=(50000,31,10)`、split、gap test fields | larger benchmark |
| `data/dataset_100k_mpb_train70000_val25000_test5000.npz` | near-field replay run 使用的数据源 | large MPB dataset |

结论：现有 forward 数据仍以 `rho_wedge -> Wh` 为主，但 shape 应由 schema 推断，不应写死进模型和指标。`Wh=(31,10)` 是当前数据实例，不是平台协议。

## near-field 数据与 run

### `data/field_tiny_mpb.h5`

字段：

| field | shape | dtype | 含义 |
|---|---:|---|---|
| `rho` | `(1,36)` | float32 | p4m wedge |
| `epsilon` | `(1,16,16)` | float32 | dielectric grid |
| `k_points` | `(1,2)` | float32 | k point |
| `freqs` | `(1,1)` | float32 | mode frequency |
| `band_index` | `(1,)` | int64 | band id |
| `efield` | `(1,16,16,1,3,2)` | float32 | electric field components, likely real/imag in last dim |
| `hfield` | `(1,16,16,1,3,2)` | float32 | magnetic field components, likely real/imag in last dim |

attrs: `metadata_json={"backend":"mpb","resolution":16,"polarization":"TE"}`。

### `runs/20260608/plot_dataset_near_fields_202606081551`

该 run 存在，`results.json` 显示：

- 数据源：`data/dataset_100k_mpb_train70000_val25000_test5000.npz`
- `limit=100`，`k_index=0`
- bands：2、3、4
- polarizations：TE、TM
- resolution：64
- WSL probe：`ok=true`，`meep_version=1.33.0-beta`，`has_mpb=true`
- near-field：`n_samples=100`、`n_ok=100`、`n_failed=0`
- TE dominant label：`|Hz|^2`，vector label：`Re(E)`
- TM dominant label：`|Ez|^2`，vector label：`Re(H)`
- TE frequency check：`n_checked=300`，mean relative error 约 `2.40e-08`

结论：near-field 不是空想方向，已经有正式 WSL `meep_env` 可视化/replay 证据。但当前 `field_tiny_mpb.h5` 规模极小，且 source、boundary condition、grid spacing、field normalization 等字段还不足以直接支撑完整 MaxwellNet / NeurOLight benchmark。

## 当前缺口

1. 缺统一 `DataRepresentationSpec` / `TargetSpec` 驱动的数据派生协议。
2. 缺 `phc2d data inspect` / `data convert` CLI 后端。
3. 缺 `ModelSpec` 声明模型支持的 input/output target。
4. 缺 native forward train/eval executor。
5. 缺 field dataset loader、field metrics、Maxwell residual feasibility report。
6. 缺 candidate protocol、morphology/diversity metrics 与 replay selection 的统一输出。

## 审计决策

1. Gate 0 通过：仓库边界和数据形态已确认。
2. Gate 1 应先实现 representation/schema 协议。
3. Gate 2 才恢复 baseline 模型和 smoke train。
4. Gate 4 的 near-field/operator 分支可以进入 feasibility 和 tiny loader smoke。
5. MaxwellNet 不做完整实现，直到补齐 source、boundary、grid spacing 与 residual 定义。