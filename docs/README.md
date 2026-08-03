<!--
  Documentation index for the synchronized phc2d workspace.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# phc2d 文档入口

本目录只同步当前仍有效的架构约定、研究权威、最新进展和项目级报告。旧计划、过程快照和被替代的报告保留在本机备份中，不进入协作仓库；原始数据、模型和完整运行产物仍位于 Git 之外的 `data/`、`checkpoints/` 与 `runs/`。

## 当前入口

| 目的 | 文档 |
|---|---|
| 了解公共代码、数据流和验证边界 | [`architecture.md`](architecture.md) |
| 了解目录、运行产物和 Markdown 约定 | [`conventions.md`](conventions.md) |
| 查看研究权威、TODO 与当前阶段计划 | [`planning/README.md`](planning/README.md) |
| 查看当前可引用的实验进展 | [`results/README.md`](results/README.md) |
| 阅读项目级完整报告与模型清单 | [`reports/README.md`](reports/README.md) |

文档中的计划、诊断结果和正式验证必须保持分离。任何结果声明都应回链到代码、配置和可核验的源运行目录；Windows 本地输出不能替代 WSL `Ubuntu` + `meep_env` 的正式验收。
