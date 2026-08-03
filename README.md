<!--
  Unified phc2d workspace README for 2D photonic-crystal reproduction and research.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<div align="center">

<h1>phc2d · POD-DeepONet / AI4PhC</h1>

<p><b>二维光子晶体能带代理、论文复现与逆向设计统一工作区</b><br />公共能力集中在 <code>src/phc2d</code>，论文复现和后续研究通过 experiment profile 区分。</p>

<p>
  <a href="#readme-zh-cn">中文</a>
  &nbsp;·&nbsp;
  <a href="#english-version">English</a>
  &nbsp;·&nbsp;
  <a href="docs/README.md">文档索引</a>
  &nbsp;·&nbsp;
  <a href="docs/architecture.md">架构</a>
  &nbsp;·&nbsp;
  <a href="AGENTS.md">协作规则</a>
</p>

</div>

---

<span id="readme-zh-cn"></span>

## 什么是 phc2d

本仓库现在按一个统一项目维护：公共代码放在 [`src/phc2d`](src/phc2d)，论文复现和后续研究都作为 [`experiments`](experiments) 下的 profile 存在。旧 `legacy/reproduction` 和 `research/ai4phc` 分裂式目录不再作为活跃工作目录保留。

| 层级 | 目录 | 作用 |
|---|---|---|
| 公共能力 | [`src/phc2d`](src/phc2d) | 数据协议、结构生成、模型 registry、训练/逆向配置、MPB replay、plot、run 输出 |
| 论文复现 profile | [`experiments/reproduction/wang2026`](experiments/reproduction/wang2026) | Wang 等 2026 POD-DeepONet 对齐配置与说明 |
| 研究 profile | [`experiments/research`](experiments/research) | forward benchmark、dispersion inverse、near-field、MPB replay 等后续实验 |
| 项目文档 | [`docs`](docs) | 架构、约定、规划、结果分析和报告 |
| 本地产物 | `data/`、`runs/`、`checkpoints/` | 大数据、运行输出、模型权重；Git ignored |

原则：**目录按能力分层，实验按配置区分，入口按任务复用**。以后新增网络、数据源、逆向目标或 plot 报告，优先注册到 `src/phc2d`，再用 experiment profile 组合。

当前 field 主线已在 9,216 条 real-MPB TE Bloch eigenfield records 上完成 `unet_field_v2`、`fno_field_v2` 与 adapted PINO 的三 seed 验证。PINO 相对匹配 data-only FNO v2 的 Maxwell residual median 降低 `37.48%`，mean E relative L2 未退化；证据与 claim 边界见 [`B3 结果报告`](docs/results/field-physics/field_eigen_te_b3_v2_pino_20260720.md)。该结果不是 faithful PINO/MaxwellNet reproduction，也不构成 field inverse 收益。

## 快速上手

### 环境约定

| 环境 | 用途 | 边界 |
|---|---|---|
| Windows `.venv` | 编辑期临时调试、包安装检查 | 不再作为后续测试验收环境 |
| WSL `Ubuntu` + conda `meep_env` | Python 单测、CLI help、schema/plot smoke、MPB 数据生成、MPB/FEM 对比、near-field、inverse MPB replay | 后续测试与正式物理验证统一环境 |
| Git ignored outputs | `data/`、`runs/`、`checkpoints/` | 本地保留，仅同步清单与入口说明 |

### 安装

PowerShell：

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

WSL / bash：

```bash
python -m pip install -e .
```

### WSL `meep_env` 检查

```powershell
wsl -d Ubuntu -e bash -lc "
source /home/fangxiao/miniforge3/etc/profile.d/conda.sh &&
conda activate meep_env &&
cd '/mnt/d/project/article_reproduction/POD-DeepONET-2d(codex)' &&
PYTHONPATH=src python -m compileall src tests &&
PYTHONPATH=src python -m pytest -q tests &&
PYTHONPATH=src python -m phc2d.cli --help &&
PYTHONPATH=src python -m phc2d.cli plot --help
"
```

### WSL `meep_env` MPB/Meep 正式验证

```powershell
wsl -d Ubuntu -e bash -lc "
source /home/fangxiao/miniforge3/etc/profile.d/conda.sh &&
conda activate meep_env &&
cd '/mnt/d/project/article_reproduction/POD-DeepONET-2d(codex)' &&
PYTHONPATH=src python -m phc2d.cli --help
"
```

汇报时必须明确写出 **WSL `meep_env` tests** 与 **WSL `meep_env` MPB validation**。Windows `.venv` 输出不再作为后续验收测试结果。

## 常用命令

```powershell
phc2d data generate --help
phc2d forward train --help
phc2d forward eval --help
phc2d inverse train --help
phc2d inverse eval --help
phc2d replay mpb --help
phc2d plot --help
phc2d report
```

目前 `phc2d plot` 已直接使用 `phc2d.plot` 实现；训练、数据生成、逆向和 replay 的可执行后端需要继续迁入 `src/phc2d`。

## 任务表

| 你想做什么 | 推荐入口 |
|---|---|
| 读架构和依赖边界 | [docs/architecture.md](docs/architecture.md) |
| 写/改项目说明类 Markdown | 使用 `md-write` skill，并同步 [docs/README.md](docs/README.md) |
| 定义复现实验 | [experiments/reproduction/wang2026](experiments/reproduction/wang2026) |
| 定义后续研究 | [experiments/research](experiments/research) |
| 画 medium/band/metrics/field 图 | `phc2d plot` |
| 查结果分析报告 | [docs/results/README.md](docs/results/README.md) |
| 查当前项目进展和证据边界 | [docs/reports/phc2d_project_full_introduction_20260724.md](docs/reports/phc2d_project_full_introduction_20260724.md) |

## Architecture / 仓库树摘要

```text
src/phc2d/                         # 统一公共包
configs/                           # reproduction/research/smoke/formal 配置入口
experiments/reproduction/wang2026/ # 论文复现 profile
experiments/research/              # 后续研究 profile
docs/                              # 架构、约定、规划、结果报告
data/ runs/ checkpoints/           # 本地大数据、运行产物、权重，Git 忽略
```

## Documentation / 规范与 AI

| 条目 | 链接 |
|---|---|
| 文档索引 | [docs/README.md](docs/README.md) |
| 统一架构 | [docs/architecture.md](docs/architecture.md) |
| 工程约定 | [docs/conventions.md](docs/conventions.md) |
| 协作与验证规则 | [AGENTS.md](AGENTS.md) |
| 规划文档 | [docs/planning](docs/planning) |
| 结果分析 | [docs/results](docs/results) |

## Logging / 输出

所有脚本输出统一写入：

```text
runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/
```

标准文件：`run.log`、`run.meta.json`、`config.resolved.yaml`、`results.json`、`metrics.json`、`history.csv`、`figures/`。[`docs/results`](docs/results) 只放人工整理后的分析报告，不放原始 run 产物。

---

<span id="english-version"></span>

## English Summary

`phc2d` is a unified workspace for 2D photonic-crystal band surrogates, Wang et al. 2026 POD-DeepONet reproduction, inverse design, and MPB replay. Shared code lives in [`src/phc2d`](src/phc2d); reproduction and research work are selected through experiment profiles under [`experiments`](experiments). The old split `legacy/reproduction` and `research/ai4phc` directories are no longer active workspace directories.

Install the root package:

```bash
python -m pip install -e .
```

Use `phc2d --help` for the unified CLI. Future tests and MPB/Meep validation should run in WSL `Ubuntu` with conda environment `meep_env`; Windows `.venv` output is not an acceptance-test result.
