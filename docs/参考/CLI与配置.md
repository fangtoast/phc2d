---
title: CLI 与配置
summary: 提供 phc2d 命令入口、配置分层、运行前检查和帮助命令约定。
page_type: reference
status: draft
audience: 复现执行者、实验维护者与自动化作者
owner: 研究生维护
last_reviewed: 2026-08-13
reading_time: 7 minutes
primary_layer: L4
---

<!--
  CLI and configuration reference for phc2d.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# CLI 与配置

从 `python -m phc2d.cli --help` 发现当前命令树，再进入子命令帮助。文档中的示例用于定位入口；具体参数、默认值和可选依赖以当前 CLI 帮助与配置 schema 为准。

## 基础入口

```bash
python -m phc2d.cli --help
python -m phc2d.cli models list
python -m phc2d.cli plot --help
python -m phc2d.cli report --help
```

训练、评估、数据生成或回放任务应继续查询对应子命令的 `--help`，不要从旧报告复制整条命令后直接运行。

## 配置分层

| 来源 | 作用 | 记录要求 |
| --- | --- | --- |
| 代码默认值 | 提供稳定、保守的基础行为 | 由测试覆盖 |
| `configs/research` | 冻结研究实验参数 | 保留配置路径与版本 |
| 命令行覆盖 | 处理单次运行差异 | 写入 resolved config |
| 数据 metadata | 提供 shape、表示与目标契约 | 禁止被硬编码值覆盖 |

每次正式运行应保存 `config.resolved.yaml`，并在 `run.meta.json` 中记录代码、数据、模型和环境身份。若恢复 checkpoint，先核对 architecture fingerprint、数据 schema 和 normalization。

## 报告维护命令

```bash
python -m phc2d.cli report docs-audit --root README.md --strict
python -m phc2d.cli report model-audit \
  --inventory docs/reports/phc2d_model_inventory_v1.yaml \
  --report <report-path>
```

证据归属机器上可为模型审计增加 `--require-source-artifacts`。文档站构建命令见[分级复现](../指南/分级复现.md)。

## 失败时先确认

1. 当前 shell 是否为 WSL `Ubuntu` 与 `meep_env`。
2. `PYTHONPATH=src` 或 editable install 是否指向当前 checkout。
3. config、数据 metadata 和 checkpoint 是否来自同一实验身份。
4. 重依赖是否只在对应子命令执行时导入。
5. 失败属于命令解析、数据契约、模型身份还是物理求解器。

