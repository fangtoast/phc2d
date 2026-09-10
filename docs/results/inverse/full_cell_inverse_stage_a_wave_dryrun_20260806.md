---
title: "Full-cell inverse Stage-A N40 波次 dry-run（2026-08-06）"
summary: "日期化实验与验证证据；正文保持原始记录。"
page_type: archive
status: historical
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 4 minutes
primary_layer: L3
evidence_date: 2026-08-06
evidence_scope: "日期化证据或历史记录；正文未因站点重构而改写。"
---

<!--
  Full-cell inverse Stage-A N40 wave dry-run evidence.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# Full-cell inverse Stage-A N40 波次 dry-run（2026-08-06）

!!! warning "历史证据"
    本页是波次 dry-run，只验证调度与参数展开；没有产生可用于科学结论的正式训练结果。

| Metadata | Value |
|---|---|
| Status | orchestration dry-run passed；未调用 Slurm、未分配 GPU |
| Primary layer | L4 implementation；仅验证调度合同，不产生 L3 模型结果 |
| Code revision | `678b6da` (`feat(cluster): orchestrate inverse Stage-A GPU waves`) |
| Environment | WSL `Ubuntu` + conda `meep_env`；N40 登录节点部署与远端只读数据哈希/manifest 演练 |
| Evidence boundary | 证明任务矩阵、节流依赖、配置/数据 hash 和状态解析入口；不证明 N40 QoS、9/18 卡可分配、显存、吞吐或训练收敛 |

## 演练输入

- mode：`probe`；
- variants：`binary_polar_holes_1`、`graded_rbf_fourier_1`；
- methods：`direct_inverse_mlp`、`single_mode_generator`、`mixture_conditional_generator`；
- seeds：`0/1/2`；
- job count：`2 × 3 × 3 = 18`；
- concurrency：`9`；
- config：[`full_cell_inverse_stage_a_n40_preflight_v1.yaml`](https://github.com/fangtoast/phc2d/blob/master/configs/research/full_cell_inverse_stage_a_n40_preflight_v1.yaml)；
- dry-run manifest：`runs/20260806/inverse_stage_a_r1_dryrun_v2_probe_20260806173951/submission_manifest.csv`（ignored run artifact，不进入 Git）。

演练前重新计算并核对了 binary/graded 两份 10k HDF5 SHA-256。配置 loader 同时验证 N40 的 batch 64、`64×64` soft raster、FP32、AMP/TF32 off 和三类网络宽度合同。

## 验证结果

| 检查 | 结果 |
|---|---|
| manifest 行数 | 18 |
| 唯一 `(variant, method, seed)` 身份 | 18/18 |
| 第一组 9 个任务 | 无 dependency |
| 第二组 9 个任务 | 全部为 `afterany:DRYRUN-001:...:DRYRUN-009` |
| submit state | 18/18 为 `dry_run` |
| `sbatch` 调用 | 0 |
| active-wave pointer | 未创建 |
| 只读 status 汇总 | 18/18 可解析，状态为 `DRY_RUN` |
| Windows targeted pytest | 9 passed |
| WSL codec + Stage-A targeted pytest | 19 passed |
| Ruff / compile / three shell scripts `bash -n` | passed |
| N40 deployed commit | `b93cc2d`；部署前备份已保留 |
| N40 remote dry-run | 18/18 身份、9+9 dependency、无 `sbatch`、无 active pointer |

单 case preflight 现在可通过 `--variant --method --seed` 选择一个任务；默认不传筛选参数时仍保持原 15-case 单卡合同。正式波次只允许每个任务申请一张 `gpu_4090`，allocation 内使用 `cuda:0`，并支持 `3|9|18` 三档节流。

N40 远端 dry-run manifest 为 `runs/20260806/inverse_stage_a_r1_remote_dryrun_probe_20260806174540/submission_manifest.csv`。17:45 的只读资源快照显示平台公共 `gpu_4090` 当时报告 45 张可用、144 张总卡；这是瞬时公共池信息，不是本账号可同时分配 18 张的承诺。N40 与 M9 的用户队列快照均无本任务作业。

## 当前门禁

1. 本报告没有提交任何 N40 作业，不能把 18 行 manifest 写成“18 张卡可用”；
2. 下一步仍是单独授权 15-case N40 单卡 preflight；
3. 单卡通过后，需再次授权实际 18-task concurrency probe，并以 `sacct`/`AllocTRES` 和逐任务 artifacts 判定 18/9/3 档位；
4. 只有 admission 通过后才可提交 Stage-A 正式训练；正向 surrogate 未训练仍阻塞 Stage B、candidate ranking 和 inverse MPB replay。

## 人工理解记录

```text
解决了什么问题：补齐 inverse Stage-A 从单任务到 9/18 任务波次、3/9/18 节流、manifest 和只读状态审计的执行入口。
为什么采用这个方案：每个 variant/method/seed 保持独立单卡和独立 run dir，既能利用 N40 多卡，又能在容量不足时不改变科学协议地降并发。
改变了项目哪一层/哪一模块：L4 preflight CLI、N40 wrapper、波次提交与状态脚本；没有启动正式实验。
使用什么证据验证：双 10k HDF5 hash、18 个唯一身份、9+9 dependency dry-run、Windows/WSL targeted tests 和 shell syntax checks。
最大风险、未知或下一 gate：实际 N40 QoS、卡数、共享 HDF5 吞吐和 batch-64 显存仍未知，必须由经授权的真实 preflight 证明。
```
