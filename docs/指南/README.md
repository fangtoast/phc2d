---
title: 操作指南
summary: 按研究任务提供前置条件、执行步骤、产物和停止条件。
page_type: landing
status: draft
audience: 需要运行或复现实验的研究使用者
owner: 研究生维护
last_reviewed: 2026-08-13
reading_time: 3 minutes
primary_layer: L4
---

<!--
  Task-oriented guide index for phc2d.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 操作指南

指南回答“怎样完成一个具体任务”，不会重复完整的 API 或实验结果。

- [分级复现](分级复现.md)：先选择 R0–R3 和允许表述。
- [数据生成](数据生成.md)：从配置、preflight 到 schema/provenance 检查。
- [训练与评估](训练与评估.md)：冻结数据、模型、checkpoint 选择和 test。
- [逆向候选与回放](逆向候选与回放.md)：生成、健康检查、排序与 real-MPB replay。
- [超算物理任务](超算物理任务.md)：plan、shard、finalize、status 和 audit。

若前置数据、环境、checkpoint、授权或证据 gate 不满足，应停止在对应 preflight，而不是用更弱的检查替代。
