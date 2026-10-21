---
title: "工程与文档约定"
summary: "定义仓库边界、运行产物、正式验证、文档页面契约和 Git 纪律。"
page_type: reference
status: draft
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 7 minutes
primary_layer: L4
---

<!--
  Repository conventions for the unified phc2d workspace.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 工程与文档约定

本文记录本仓库的目录、代码、文档、运行产物和验证约定，面向维护者和 AI 协作者。详细架构见[系统架构](参考/系统架构.md)，知识分层、文档角色与人工审阅规则见[文档与人机协作治理](collaboration.md)。

## Code Boundaries

- 公共接口放在 [`../src/phc2d`](https://github.com/fangtoast/phc2d/tree/main/src/phc2d)：data、generators、models、training、inverse、replay、plotting、runs、cli。
- 论文复现作为 [`../experiments/reproduction/wang2026`](https://github.com/fangtoast/phc2d/tree/main/experiments/reproduction/wang2026) profile 维护。
- 后续研究作为 [`../experiments/research`](https://github.com/fangtoast/phc2d/tree/main/experiments/research) profile 维护。
- 旧 `legacy/reproduction` 与 `research/ai4phc` 已退出活跃目录；需要旧逻辑时从 Git history 恢复并迁入 `src/phc2d`。
- 新模型通过 `phc2d.models` registry 接入，新数据源通过 `phc2d.data` / `phc2d.generators` / `phc2d.solvers` 接入，新图表通过 `phc2d.plot` 接入。

## Experiment Profiles

每次实验应能被一个 `ExperimentSpec` 描述：

| 字段 | 说明 |
|---|---|
| `experiment_id` | 稳定实验名，用于 run 目录 |
| `family` | `reproduction` 或 `research` |
| `stage` | `data`、`forward_train`、`forward_eval`、`inverse_train`、`surrogate_eval`、`mpb_replay`、`report` |
| `dataset` | 数据路径、split、schema、metadata |
| `model` | 模型名、checkpoint、adapter |
| `solver` | FEM/MPB/replay backend |
| `output` | runs、figures、report 输出规则 |

## Run Outputs

脚本输出写入：

```text
runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/
```

标准文件：

| 文件 | 说明 |
|---|---|
| `run.log` | 日志 |
| `run.meta.json` | 脚本、环境、git commit、时间、关键路径 |
| `config.resolved.yaml` | 解析后的配置快照 |
| `results.json` | 结构化结果 |
| `metrics.json` | 指标 |
| `history.csv` | 训练历史 |
| `figures/` | 图表 |

[`results/`](results/README.md) 只放人工整理后的报告，不放原始运行产物。

## MPB / Meep Validation

后续测试与 MPB/Meep 正式验证统一使用：

```text
WSL Ubuntu + conda environment: meep_env
```

Windows `.venv` 只作为编辑期临时调试便利，不再作为测试验收环境。Python 合约、CLI、plot、schema、非 MPB smoke、MPB 数据生成、MPB/FEM 对比、near-field、inverse replay 和正式 physics benchmark 都应通过 WSL `meep_env` 执行。报告中必须明确写出 WSL `meep_env` tests 与 WSL `meep_env` MPB validation 状态。

## Markdown 与站点

- 本站的信息架构、页面职责、研究证据语气和迁移验收使用独立 `research-docs-site` skill；通用 Markdown 排版任务仍可使用 `md-write`。
- 稳定页分为 tutorial、how-to、explanation、reference；日期记录分为 evidence、plan、archive。索引页使用 landing，图件索引使用 asset-index。
- 中文为主，命令、路径、API、配置键保持英文原文。
- `docs/` 下新建的面向人工阅读文档，文件名默认优先使用简洁、可辨识的中文；固定入口名、代码/API 标识、实验 ID、机器消费产物及兼容性敏感路径继续使用英文。不要仅为统一语言批量重命名已有稳定文档。
- `docs/` 页面以 YAML front matter 开头，并包含 `title`、`summary`、`page_type`、`status`、`audience`、`owner`、`last_reviewed`、`reading_time` 和 `primary_layer`；evidence 与 archive 另加日期和范围。
- 新建或大改 Markdown 应在 front matter 后包含 HTML 注释文件头。
- 站内页面与资产使用相对链接；站外仓库文件使用可在发布站访问的仓库 URL。
- 根 README 是面向外部读者的项目首页，只保留稳定定义、主要能力、最小 quick start 和文档入口；易变化状态进入 [`../PROJECT_MAP.md`](https://github.com/fangtoast/phc2d/blob/main/PROJECT_MAP.md)。
- 一个文档只承担一种主要角色；长细节通过链接下沉，不在多个权威文件中复制。
- 日期化报告、计划与历史记录的原正文只增补元数据、状态、导航和回链，不因重构而改写成当前结论。
- 新建或重命名正式文档后，更新 `mkdocs.yml` 或 `not_in_nav`，并运行文档审计和 MkDocs strict build。

## Files and Headers

新增或本次大改的 Python 与 Markdown 文件应包含用途简述、作者、邮箱和版权。已有历史文件不要求一次性补齐，但修改时可顺手补。

## Git Hygiene

不要提交 datasets、checkpoints、run outputs、`.venv`、cache、密钥、本地临时文件或生成二进制产物。大规模迁移后应保持活跃目录短而清晰，避免重新引入顶层 legacy/research 分裂。
