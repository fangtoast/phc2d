<!--
  Book-style knowledge map and documentation index for phc2d.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

> 超算物理数据生成的统一 plan/shard/finalize/status/audit 合同见 [`hpc_physics_campaigns.md`](hpc_physics_campaigns.md)。该文档明确区分已实现的 full-cell 参考适配器与尚待迁移的任务族。

<a id="docs-book-home"></a>

# phc2d 知识地图与阅读目录

| Metadata | Value |
|---|---|
| Status | Canonical documentation index |
| Audience | 导师、研究生、研究者、开发者与 AI 协作者 |
| Owner | 研究生维护，导师审阅研究叙事 |
| Last reviewed | 2026-08-08 |
| Reading time | 10 minutes |

这里是项目的“书籍目录”，负责把理论、公式、方法、代码、实验和证据连接起来；它不重复各章正文。今天的状态先看 [`PROJECT_MAP.md`](../PROJECT_MAP.md)，第一次运行先看 [`getting-started.md`](getting-started.md)。

## 按读者选择路线

### 导师路线：问题、主线、证据

1. [项目地图](../PROJECT_MAP.md)：当前主线、已知结论、阻塞项和下一 gate。
2. [完整项目报告](reports/phc2d_project_full_introduction_20260724.md)：带日期的完整技术叙述。
3. [当前实验进展](results/README.md)：逐项证据与边界。
4. [研究规划](planning/README.md)：计划、TODO 与尚未完成的 gate。

### 新用户路线：安装、理解、复现

1. [快速上手](getting-started.md)：干净克隆、R0 和 R1。
2. [架构说明](architecture.md)：公共代码、数据流和验证边界。
3. [分级复现](reproducibility.md)：R2 surrogate 与 R3 real-MPB。
4. [实验 profiles](../experiments/README.md)：具体实验入口。

### 开发者 / AI 路线：边界、实现、验收

1. [协作治理](collaboration.md)：知识层级、文档角色与任务契约。
2. [固定规则](../AGENTS.md)：代码、数据、测试、证据与 Git 约束。
3. [工程约定](conventions.md)：目录、产物和 Markdown 规则。
4. [配置索引](../configs/README.md) · [数据索引](../data/README.md) · [运行产物](../runs/README.md)。
5. [Full-Wh 升级实现记录](planning/full_cell_forward_inverse_training/full_cell_forward_model_upgrade_implementation_plan.md) · [10k/50k 训练验证方案](planning/full_cell_forward_inverse_training/full_cell_forward_model_upgrade_10k_50k_training_validation_plan.md) · [50K 实验报告](results/forward/full_cell_forward_model_upgrade_50k_evaluation_20260810.md)：五个升级模型的实现、实验合同与正式主评估入口；threshold proxy 尚待补充。

## 全书目录

| 章节 | 回答的问题 | 权威入口 |
|---|---|---|
| 封面 | 项目是什么、从哪里进入？ | [`README.md`](../README.md) |
| 当前状态 | 现在做到哪里、卡在哪里？ | [`PROJECT_MAP.md`](../PROJECT_MAP.md) |
| 第一章：研究全貌 | 为什么做、实现了什么、证据多强？ | [综合报告索引](reports/README.md) · [完整项目报告](reports/phc2d_project_full_introduction_20260724.md) · [带隙—近远场—逆向设计科学动机](reports/phc2d_bandgap_field_inverse_design_rationale_20260806.md) |
| 第二章：物理与公式 | 结构怎样变成数据，损失和指标如何定义？ | [公式手册](reports/phc2d_physics_data_formula_guide_20260803.md) |
| 第三章：代码架构 | 模块、数据流、CLI 和可选依赖怎样组织？ | [架构说明](architecture.md) |
| 第四章：实验设计 | 每项实验的输入、非目标、命令和 gate 是什么？ | [实验 profiles](../experiments/README.md) |
| 第五章：证据 | 实际运行得到什么、不能证明什么？ | [结果索引](results/README.md) |
| 第六章：复现 | 从接口检查到 real-MPB 验收怎样执行？ | [分级复现](reproducibility.md) |
| 第七章：研究路线 | 下一步为何这样安排？ | [规划索引](planning/README.md) |
| 附录：协作规则 | 人与 AI 怎样在同一抽象层工作？ | [协作治理](collaboration.md) |

## 模块交叉索引

| 模块 | 方法叙述 | 公式 | 代码/配置 | 实验与证据 |
|---|---|---|---|---|
| 数据与 MPB | [报告：数据表征](reports/phc2d_project_full_introduction_20260724.md#report-data-representation) | [结构与数据](reports/phc2d_physics_data_formula_guide_20260803.md#formula-structure-data) | [`phc2d.data`](../src/phc2d/data) · [数据目录](../data/README.md) | [Data provenance](results/README.md#data-provenance-results) |
| Full-Wh | [前向模型与升级实现](reports/phc2d_project_full_introduction_20260724.md#report-forward-models) | [损失与指标](reports/phc2d_physics_data_formula_guide_20260803.md#formula-forward-metrics) | [`phc2d.forward`](../src/phc2d/forward) · [A1 配置](../configs/research/a1_forward_fair_v1.yaml) · [升级实现记录](planning/full_cell_forward_inverse_training/full_cell_forward_model_upgrade_implementation_plan.md) | [A1](results/forward/a1_forward_query_fair_v1_20260726.md) · [升级 50K 主结果](results/forward/full_cell_forward_model_upgrade_50k_evaluation_20260810.md) · [实验合同](planning/full_cell_forward_inverse_training/full_cell_forward_model_upgrade_10k_50k_training_validation_plan.md) |
| Query | [Query 模型](reports/phc2d_project_full_introduction_20260724.md#report-forward-models) | [数据表示](reports/phc2d_physics_data_formula_guide_20260803.md#formula-representation) | [`phc2d.query`](../src/phc2d/query) · [A1 配置](../configs/research/a1_query_fair_v1.yaml) | [Query 证据](results/README.md#query-results) |
| Field / PINO | [Field 模型](reports/phc2d_project_full_introduction_20260724.md#report-forward-models) | [本征场与 PINO](reports/phc2d_physics_data_formula_guide_20260803.md#formula-field-pino) | [`phc2d.physics`](../src/phc2d/physics) · [Field profile](../experiments/research/field_eigen_te.md) | [Field 证据](results/README.md#field-physics-results) |
| Inverse | [逆向模型](reports/phc2d_project_full_introduction_20260724.md#report-inverse-models) | [逆向目标](reports/phc2d_physics_data_formula_guide_20260803.md#formula-inverse) | [`phc2d.inverse`](../src/phc2d/inverse) · [A2 profile](../experiments/research/a2_inverse_single_seed.md) | [Inverse 证据](results/README.md#inverse-results) |
| Replay | [物理闭环](reports/phc2d_project_full_introduction_20260724.md#report-mpb-replay) | [Replay 指标](reports/phc2d_physics_data_formula_guide_20260803.md#formula-replay) | [`phc2d.replay`](../src/phc2d/replay) | [复现 R3](reproducibility.md#repro-r3) |

## 知识层级

| 层级 | 问题 | 主要文档 |
|---|---|---|
| L0 | 研究目标是什么？ | 根 README、PROJECT_MAP |
| L1 | 为什么选择这条路线？ | 完整报告、研究计划 |
| L2 | 系统由哪些模块组成？ | 完整报告、架构、公式手册 |
| L3 | 如何验证，实际证据是什么？ | experiment profiles、results、runs |
| L4 | 具体怎样实现和运行？ | `src/phc2d`、configs、测试、CLI |

详细任务约束见 [协作治理](collaboration.md)。

## 权威与证据边界

当不同文档冲突时，按以下顺序处理：

```text
运行制品 / solver 输出 / metrics / checkpoint
  > 当前代码、配置、测试和 CLI
  > 经筛选的结果报告
  > Research TODO
  > 研究计划
  > 综合报告与摘要
```

计划不能证明实验完成；smoke 不能证明模型有效；surrogate 不能替代 replay；较低 residual 不能单独证明 inverse 收益。综合报告与公式手册是重要深层资料，但均保留其证据日期和适用边界。

## 文档完整性检查

```bash
python -m phc2d.cli report docs-audit --root README.md --strict
```

新增或重命名正式文档后，必须从本目录或相邻子索引建立可达链接。未跟踪草稿不参与正式文档审计。
