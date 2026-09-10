---
title: 逆向设计结果
summary: 候选生成、代理排序、Objective-Bridge、真实 MPB 回放和全胞 Stage-A 预检索引。
page_type: landing
status: draft
audience: 逆向设计研究者、复现执行者与证据审阅者
owner: 研究生维护
last_reviewed: 2026-08-14
reading_time: 5 minutes
primary_layer: L3
---

<!--
  Inverse design result landing page.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 逆向设计结果

## 已形成训练或回放证据

| 结果页面 | 状态与边界 |
|---|---|
| [多代理 surrogate rerank](inverse_surrogate_rerank_v2_10000_20260630.md) | 历史代理排序证据 |
| [10k inverse 重训](inverse_retrain_v2_10000_20260706.md) | 历史 inverse retrain 证据 |
| [Test625 real-MPB replay](inverse_mpb_replay_mixture_v2_10000_test625_20260706.md) | 625-target 真实求解器回放；保留协议边界 |
| [A2 单 seed 正式协议](a2_inverse_single_seed_formal_20260727.md) | 单 seed 结果，不能外推为跨 seed 结论 |
| [Objective-Bridge seed0](a2_inverse_objective_bridge_seed0_v1_20260727.md) | 目标桥接的 seed0 证据 |
| [多代理修复后验证](a2_inverse_surrogate_postrepair_validation_v2_20260802.md) | 候选健康度与 pilot 验证 |

## 全胞 Stage-A 准备证据

| 结果页面 | 证明了什么 |
|---|---|
| [Full-cell codec 预检](full_cell_inverse_codec_preflight_20260806.md) | codec 和 decoder 合同；不是训练或 replay 证据 |
| [Stage-A CPU 预检](full_cell_inverse_stage_a_cpu_preflight_20260806.md) | 真实 10k 数据上的 15-case 合同预检 |
| [Stage-A runner smoke](full_cell_inverse_stage_a_runner_smoke_20260806.md) | 可断点续训 runner 的小数据 smoke |
| [Stage-A 波次 dry-run](full_cell_inverse_stage_a_wave_dryrun_20260806.md) | manifest、状态和并发编排；不证明实际 GPU 容量 |

逆向报告必须区分 surrogate MRE、候选排序和 real-MPB replay MRE；smoke、dry-run 与 preflight 不能写成正式训练结果。

[返回研究结果与报告](../README.md)
