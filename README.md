<!--
  Stable project entrypoint for the unified phc2d research workspace.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->
# phc2d

**二维光子晶体数据生成、神经代理、逆向设计与 MPB 复核工具链。**

*A research toolkit for 2D photonic-crystal surrogate modeling and solver-verified inverse design.*

`phc2d` 面向二维光子晶体的计算与设计研究，将结构和材料表示、MPB 数据生成、神经代理训练、条件逆向设计以及真实求解器复核组织在同一个 Python 项目中。

仓库提供统一的命令行入口、可追溯的实验配置和标准化运行产物。当前研究进展、已知结论和待解决问题集中维护在[项目地图](PROJECT_MAP.md)，不会写入这个稳定入口。

## 主要能力

- 生成和检查带有结构、材料、k 点与求解器 provenance 的 MPB 数据。
- 训练和评估 Full-Wh、Continuous Query 与 TE Bloch eigenfield 代理模型。
- 根据目标能带生成多个候选结构，并执行合法性、去重和代理排序。
- 使用 real-MPB replay 复核逆向设计候选，区分代理指标与物理求解结果。
- 通过统一 CLI、研究配置和实验 profiles 组织数据、训练、评估、绘图与报告。



## 环境要求


| 用途                   | 要求                              |
| -------------------- | ------------------------------- |
| 基础接口与轻量工具            | Python 3.10+                    |
| 数据处理与模型训练            | `train` 可选依赖，包括 PyTorch、h5py 等  |
| MPB / Meep 数据生成与物理复核 | WSL `Ubuntu` + conda `meep_env` |


基础命令不会强制导入 PyTorch、MPB 或 Meep。Windows、Linux/WSL 和完整求解器环境说明见[快速上手](docs/getting-started.md)。

## 安装

```bash
git clone https://github.com/fangtoast/phc2d.git
cd phc2d
python -m pip install -e .
```

需要训练模型时安装可选依赖：

```bash
python -m pip install -e ".[train]"
```



## Quick Start

检查 CLI 和已注册模型：

```bash
python -m phc2d.cli --version
python -m phc2d.cli --help
python -m phc2d.cli models list
```



### Smoke example

下面的示例生成确定性的合成数据，并运行一次最小训练，用于检查数据、模型和运行产物链路：

```bash
python -m phc2d.cli data make-smoke --config configs/smoke/synthetic_dataset.yaml
python -m phc2d.cli forward train --config configs/smoke/forward_synthetic_res_mlp.yaml
```

> 该示例使用非物理合成数据，不代表论文复现、代理模型有效性或 MPB 物理验证。正式复现所需的数据、环境和证据等级见[复现指南](docs/reproducibility.md)。



## 项目组成

- **Data**：数据 schema、结构表示、k 路径、MPB 适配器和数据集检查。
- **Forward / Query / Field**：完整能带、连续查询和本征场代理模型。
- **Inverse / Replay**：条件候选生成、筛选、排序和真实求解器回放。
- **CLI / Configs / Experiments**：命令行接口、可复现实验配置和协议入口。

```text
src/phc2d/    公共 Python 包与 CLI
configs/      smoke 和 research 配置
experiments/  复现与研究实验 profiles
docs/         方法、复现、结果、规划和报告
tests/        单元测试与合约测试
```

完整数据集、模型权重和原始运行目录不随 Git 仓库发布。

## 文档

- [快速上手](docs/getting-started.md)：安装、环境和最小训练。
- [项目地图](PROJECT_MAP.md)：当前主线、结论、阻塞项与下一步。
- [完整文档目录](docs/README.md)：理论、架构、公式、实验和结果索引。
- [复现指南](docs/reproducibility.md)：从接口检查到 real-MPB 验证的分级流程。
- [实验与配置](experiments/README.md)：实验 profiles；配置入口见 [configs/](configs/README.md)。



## 验证与协作

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m phc2d.cli report docs-audit --root README.md --strict
```

提交代码、实验或文档改动前，请阅读[工程约定](docs/conventions.md)和[协作治理](docs/collaboration.md)。正式测试与 MPB/Meep 验收以 WSL `Ubuntu` + conda `meep_env` 的结果为准。
