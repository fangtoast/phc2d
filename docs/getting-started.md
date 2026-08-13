<!--
  Clean-clone installation and non-physical smoke guide for phc2d.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="getting-started-home"></a>

# 快速上手：从干净克隆到最小训练

| Metadata | Value |
|---|---|
| Status | Canonical onboarding guide |
| Audience | 第一次使用仓库的研究者、学生和开发者 |
| Owner | 研究生维护 |
| Last reviewed | 2026-08-06 |
| Reading time | 8 minutes |

本文只回答如何安装、检查 CLI 并运行非物理 smoke。它不解释研究结论，也不把 smoke 当作论文复现或物理验证；正式复现见 [`reproducibility.md`](reproducibility.md)。

## 1. 环境层级

| 环境 | 用途 | 可以声明什么 |
|---|---|---|
| 普通 Python 3.10+ | 安装、CLI、模型清单、文档审计 | 接口可用 |
| 安装 `train` extra | 合成数据上的一轮最小训练 | R1 smoke 通过 |
| WSL `Ubuntu` + conda `meep_env` | 正式测试、MPB/Meep、replay | 由具体 gate 决定 |
| Windows `.venv` | 编辑和临时调试 | 不构成项目验收 |

`torch`、`h5py`、MPB 和 Meep 均不是基础 CLI 的强制依赖。

## 2. Windows PowerShell：R0 接口检查

```powershell
git clone https://github.com/fangtoast/phc2d.git
Set-Location phc2d
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m phc2d.cli --version
.\.venv\Scripts\python.exe -m phc2d.cli --help
.\.venv\Scripts\python.exe -m phc2d.cli models list
```

R0 成功只表示包、CLI 和轻量 registry 可以加载。

## 3. Linux / WSL：R0 接口检查

```bash
git clone https://github.com/fangtoast/phc2d.git
cd phc2d
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m phc2d.cli --version
python -m phc2d.cli --help
python -m phc2d.cli models list
```

项目正式 WSL 验收使用现有 conda `meep_env`，不使用这里新建的普通 `.venv` 代替。

## 4. R1：生成非物理合成数据

先安装训练依赖：

```bash
python -m pip install -e ".[train]"
```

`train` extra 包含 PyTorch。全新环境第一次下载和安装可能明显慢于 R0；这属于训练依赖准备，不是模型训练本身。项目协作者可直接使用已配置的 WSL `meep_env`，但仍应在干净克隆中执行下面的数据生成与训练命令。

再生成配置驱动的数据并检查 schema：

```bash
python -m phc2d.cli data make-smoke --config configs/smoke/synthetic_dataset.yaml
python -m phc2d.cli data inspect --path data/generated/phc2d_synthetic_smoke.npz
```

生成器从配置读取样本数、`rho_wedge` 长度、k 点数、band 数、split 和 seed，不在模型中硬编码数据形状。输出包含：

- `data/generated/phc2d_synthetic_smoke.npz`；
- 相邻的 `phc2d_synthetic_smoke.manifest.yaml`；
- 显式 `idx_train`、`idx_val`、`idx_test`；
- 每个样本的 ID、split 和 `synthetic_nonphysical_smoke` 状态。

默认拒绝覆盖已有数据。只有明确要按同一配置重新生成时才使用 `--overwrite`。

## 5. R1：运行最小训练

```bash
python -m phc2d.cli forward train --config configs/smoke/forward_synthetic_res_mlp.yaml
```

成功运行会在标准目录下生成 `run.log`、`run.meta.json`、`config.resolved.yaml`、`results.json`、`metrics.json`、`history.csv` 和 `figures/`。这些产物只证明训练链路可执行，不能进入论文性能表、模型优越性结论或 MPB 验收。

## 6. 文档和接口自检

```bash
python -m phc2d.cli report docs-audit --root README.md --strict
python -m phc2d.cli plot --help
```

文档审计只使用 Git 已跟踪 Markdown，因此本地未跟踪草稿不会被误判为正式项目文档。

## 7. 常见问题

| 现象 | 含义与处理 |
|---|---|
| `torch` 未安装 | R0 不需要；运行 R1 前安装 `.[train]` |
| smoke 数据已存在 | 默认拒绝覆盖；确认是合成数据后再显式使用 `--overwrite` |
| 没有 `data/query_dataset...` | 公共仓库不提交正式数据；转到 [复现指南](reproducibility.md) |
| Windows 测试通过 | 只能作为编辑期检查，不能写成 WSL 或物理验收 |
| MPB/Meep 导入失败 | 使用项目规定的 WSL `meep_env`，并先完成对应实验 preflight |

## 下一步

- 要理解项目：回到 [文档目录](README.md)。
- 要复现实验：阅读 [分级复现指南](reproducibility.md)。
- 要修改项目：先读 [协作治理](collaboration.md) 和 [`AGENTS.md`](../AGENTS.md)。
