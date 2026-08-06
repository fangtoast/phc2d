<!--
  Stable project entrypoint for the unified phc2d research workspace.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<div align="center">

<h1>phc2d</h1>

<p><b>二维光子晶体神经代理、生成式逆向设计与 real-MPB 物理回放</b></p>

<p>
  <a href="#readme-zh-cn">中文</a>
  &nbsp;·&nbsp;
  <a href="#english-version">English</a>
  &nbsp;·&nbsp;
  <a href="PROJECT_MAP.md">项目地图</a>
  &nbsp;·&nbsp;
  <a href="docs/README.md">文档目录</a>
  &nbsp;·&nbsp;
  <a href="docs/getting-started.md">快速上手</a>
</p>

</div>

---

<span id="readme-zh-cn"></span>

| Metadata | Value |
|---|---|
| Status | Canonical entrypoint |
| Audience | 导师、研究者、新用户、开发者与 AI 协作者 |
| Owner | 研究生维护，导师审阅科学边界 |
| Last reviewed | 2026-08-06 |
| Reading time | 5 minutes |

## 这个项目要做什么

`phc2d` 研究如何用神经网络和神经算子加速二维光子晶体的 Maxwell/MPB 求解，并从目标能带反向生成可由真实求解器复核的结构。项目围绕一条可追溯闭环组织：

```text
结构与材料表示
  -> real-MPB 数据
  -> Full-Wh / Query / Field 代理
  -> 多候选条件生成
  -> 合法性、多样性与交叉代理门控
  -> real-MPB replay 最终物理裁判
```

本仓库不是单一模型示例，也不把 surrogate 误差、field residual 或 smoke test 当作物理成功。当前进度、阻塞项与最近 gate 只在 [`PROJECT_MAP.md`](PROJECT_MAP.md) 维护。

## 按你的任务进入

| 你是谁 / 想做什么 | 第一入口 | 下一步 |
|---|---|---|
| 导师或项目负责人：十分钟掌握主线 | [项目地图](PROJECT_MAP.md) | [完整项目报告](docs/reports/phc2d_project_full_introduction_20260724.md) · [当前证据](docs/results/README.md) |
| 新用户：安装并确认仓库可用 | [快速上手](docs/getting-started.md) | [分级复现指南](docs/reproducibility.md) |
| 研究者：重现实验或物理验证 | [分级复现指南](docs/reproducibility.md) | [实验 profiles](experiments/README.md) · [研究配置](configs/README.md) |
| 开发者或 AI：修改代码、实验或文档 | [架构](docs/architecture.md) | [协作治理](docs/collaboration.md) · [固定规则](AGENTS.md) |

完整的理论、公式、代码、实验和证据阅读顺序见 [`docs/README.md`](docs/README.md)。

## 五分钟接口检查（R0）

要求 Python 3.10 或更高版本。下面的检查不需要数据、PyTorch、MPB 或 Meep：

```bash
git clone https://github.com/fangtoast/phc2d.git
cd phc2d
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m phc2d.cli --version
python -m phc2d.cli --help
python -m phc2d.cli models list
```

虚拟环境激活方式和 Windows / WSL 分支命令见 [快速上手](docs/getting-started.md)。

## 可运行的最小训练（R1）

R1 会生成一个确定性的**非物理合成数据集**，只检查数据、模型、训练和 run 产物链路：

```bash
python -m pip install -e ".[train]"
python -m phc2d.cli data make-smoke --config configs/smoke/synthetic_dataset.yaml
python -m phc2d.cli forward train --config configs/smoke/forward_synthetic_res_mlp.yaml
```

R1 不是论文复现、MPB 数据生成或物理验收。正式代理结果与 real-MPB 验证分别属于 R2、R3，所需输入和命令见 [复现指南](docs/reproducibility.md)。

## 主要研究模块

| 模块 | 回答的问题 | 深入阅读 |
|---|---|---|
| 数据与 MPB | 结构、材料、k 点和标签怎样产生并保留 provenance？ | [公式手册：结构与数据](docs/reports/phc2d_physics_data_formula_guide_20260803.md#formula-structure-data) |
| Full-Wh | 如何一次预测完整能带矩阵？ | [完整报告：前向模型](docs/reports/phc2d_project_full_introduction_20260724.md#report-forward-models) |
| Continuous Query | 如何查询任意 band / physical-k 响应？ | [A1 实验 profile](experiments/research/a1_fair_forward_query.md) |
| Field / adapted PINO | 如何预测 TE Bloch 本征场并审计物理残差？ | [Field profile](experiments/research/field_eigen_te.md) |
| Inverse / replay | 如何生成多个候选并由 real-MPB 最终复核？ | [完整报告：逆向闭环](docs/reports/phc2d_project_full_introduction_20260724.md#report-inverse-models) |

## 仓库边界

```text
src/phc2d/       公共数据、模型、训练、inverse、replay、plot 和 reporting 能力
configs/         smoke 与 research 配置
experiments/     复现和研究实验 profiles
docs/            书籍式知识目录、规划、结果、报告和协作规则
data/            本地数据与 tracked metadata；大数组不进入 Git
runs/            标准运行产物；完整 runs 不进入 Git
checkpoints/     本地模型权重；不进入 Git
```

公共仓库不包含完整数据集、checkpoint 或原始 runs。任何结果声明都必须回链到相应配置、数据 provenance 和运行制品。未来正式测试与 MPB/Meep 验收统一使用 WSL `Ubuntu` + conda `meep_env`；Windows `.venv` 仅用于编辑和临时调试。

## 文档与审计

```bash
python -m phc2d.cli report docs-audit --root README.md --strict
python -m phc2d.cli report model-audit \
  --inventory docs/reports/phc2d_model_inventory_v1.yaml \
  --report docs/reports/phc2d_project_full_introduction_20260724.md
```

- [项目地图](PROJECT_MAP.md)：当前状态唯一短入口。
- [文档目录](docs/README.md)：理论、公式、代码、实验与复现的书籍式目录。
- [协作治理](docs/collaboration.md)：知识层级、文档角色、人工审阅与 AI 边界。
- [固定规则](AGENTS.md)：代码、证据、测试和 Git 的不可违反约束。

---

<span id="english-version"></span>

## English Summary

| Metadata | Value |
|---|---|
| Status | Compact mirror of the Chinese entrypoint |
| Audience | New users and external collaborators |
| Owner | Student maintainer; scientific boundaries reviewed with the supervisor |
| Last reviewed | 2026-08-06 |
| Reading time | 2 minutes |

`phc2d` is a research workspace for real-MPB photonic-crystal data, full-band and continuous-query surrogates, TE Bloch eigenfield models, multi-candidate inverse design, and real-MPB replay. The solver replay—not a smoke test, surrogate metric, or physics residual—is the final physical judge.

Start with the [project map](PROJECT_MAP.md), the [documentation book](docs/README.md), or the [clean-clone guide](docs/getting-started.md). The public repository excludes full datasets, checkpoints, and raw runs; the [reproducibility guide](docs/reproducibility.md) states the required inputs and evidence level for each workflow.
