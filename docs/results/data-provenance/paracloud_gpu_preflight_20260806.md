---
title: "Paracloud GPU 全链路准备与单卡预检"
summary: "日期化实验与验证证据；正文保持原始记录。"
page_type: archive
status: historical
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 7 minutes
primary_layer: L3
evidence_date: 2026-08-06
evidence_scope: "日期化证据或历史记录；正文未因站点重构而改写。"
---

<!--
  Paracloud GPU and MPB handoff preflight evidence for phc2d.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# Paracloud GPU 全链路准备与单卡预检

!!! warning "历史证据"
    本页是部署与单卡预检记录；环境可用不等于模型训练、评估或物理复核已经完成。

| Metadata | Value |
|---|---|
| Status | Draft evidence report |
| Date | 2026-08-06 |
| Primary layer | L4 implementation/runtime |
| Current N40 deployed commit | `b93cc2d`；原三卡/单卡 Slurm 证据产生于更早部署，job id 与制品边界不变 |
| GPU environment | N40 `gpu_4090`, PyTorch `2.1.2+cu118`, CUDA `11.8` |
| Evidence boundary | 资源、环境、数据和单卡训练预检；不是正式训练结果 |
| Reading time | 6 minutes |

## 执行结论

N40 已打通登录、代码部署、数据导入、Slurm 分配、CUDA/PyTorch 环境和训练预检。Slurm 作业 `5390073` 实际同时获得 3 张 RTX4090；单卡作业 `5390162` 完成四个计划内模型的 batch-64 前反向、FNO 10 轮训练、HDF5 多 worker 读取和 checkpoint 精确恢复。因此 N40 可作为三 seed 正式训练的同型号三卡路线。

两套 10k full-cell HDF5 已在 N40 就位并通过 SHA-256 核对。渐变数据另在本机 WSL `Ubuntu` + `meep_env` 完成固定 real-MPB 跨环境验收，训练数据门已经解除。

N50R5 尚未打通：网关返回 `loginCluster` 权限拒绝，因此未在该平台安装环境或提交 GPU 作业。它是可选的 RTX5090 路线，不影响按计划使用 N40 三卡，但在平台同步 SSH 公钥或登录权限前不能称为可用。

正式 102-run 训练未开始，M9 threshold real-MPB preflight/pilot 也未提交；两者仍分别等待人工授权。

2026-08-06 17:44–17:45（Asia/Shanghai）又完成一次 **非作业部署检查**：N40 部署树从 `b86ed7e` 更新到 `b93cc2d`，同步 inverse Stage-A runner、单 case preflight、正式配置、波次调度和只读 status 脚本。覆盖前备份保存在 `/HOME/scw6g7a/run/deployments/phc2d_before_b93cc2d_20260806174434.tar.gz`；部署归档 SHA-256 为 `ac291ddd46ca909e097d6bc8cc9dc23d85ea67a78e35879f3734f818a403df34`。远端 compile、CLI help 和 shell syntax 通过后才原子更新 `DEPLOYED_GIT_COMMIT`。

随后在 N40 登录节点运行 `PHC2D_SUBMIT_DRY_RUN=1`：远端重新核对两份 10k 数据哈希，生成 18 个唯一任务的 9+9 dependency manifest，未调用 `sbatch`、未创建 inverse active-wave pointer。登录节点直接导入 Torch 会因未加载 CUDA module 报动态库符号错误；这不是计算节点环境回归证据，正式 wrapper 仍必须在 Slurm allocation 内由 `paracloud_gpu_common.sh` 加载 `cuda/11.8` 后验证 CUDA。

## 资源与职责

| 资源 | 已验证状态 | 本计划中的职责 | 明确非目标 |
|---|---|---|---|
| N40 `gpu_4090` | 三卡分配与单卡训练预检通过 | seed 0/1/2 神经网络训练、推理、指标汇总 | 不运行 real-MPB，不做跨站点 DDP |
| N50R5 `gpu` | 登录权限被平台拒绝 | 权限恢复后的 RTX5090 备选/加速路线 | 当前不纳入可用资源计数 |
| M9 CPU 超算 | SSH 与既有数据 hash 复核通过 | 经单独授权的 real-MPB 计算 | 不训练神经网络，不因 GPU 就绪自动提交 |
| 本机 WSL | `Ubuntu` + `meep_env` 验收通过 | 固定 replay、checkpoint 回收复核 | Windows 本地结果不作为正式验收 |

资源参数依据 Paratera 的 [N40 使用文档](https://kbs.paratera.com/info/1503) 和 [N50R5 使用文档](https://kbs.paratera.com/info/1661)。N40 每卡由队列分配 6 CPU 与 60,000 MB 内存；三 seed 采用三张同型号卡各跑一个独立进程，不使用 DDP。

## 数据与 provenance

| 数据 | 大小 | SHA-256 | N40 状态 |
|---|---:|---|---|
| `full_cell_binary_bz9_10000_xingyi_v1.h5` | 250,275,716 bytes | `1f131f2bda7ccb1f1736809bdf6bafd534096cb275e22ce3e56fc5d910530c2f` | 从本机重新导入，远端 hash 一致 |
| `full_cell_graded_bz9_10000_paracloud_m9_v1.h5` | 1,210,491,259 bytes | `d16d838e098215c05dc299bf7c4bbc8c6bc00f6ef35b9772546aa4e7c2b8b728` | 从本机导入，N40/M9/本机 hash 一致 |

渐变数据的固定 WSL 验收 run 为 `runs/20260806/full_cell_graded_paracloud_m9_wsl_acceptance_v1_202608061417/`：8 个分层结构、每个 9 个代表 k 点、bands 2–10 全部完成，失败数为 0；median MRE 为 0，p95 MRE 为 `1.0455171661760687e-07`，分别低于 `0.005` 和 `0.01` 门限。该结果只证明集群数据能在固定 WSL real-MPB 环境复现，不是模型精度。

## 三卡分配证据

| 项目 | 观测值 |
|---|---|
| Slurm job | `5390073` |
| Partition / node | `gpu_4090` / `g0361` |
| State / exit | `COMPLETED` / `0:0` |
| Allocation | `gres/gpu=3`, `cpu=18`, `mem=180000M` |
| Visible devices | `CUDA_VISIBLE_DEVICES=0,1,2`，共 3 张 |
| GPU | 3×NVIDIA GeForce RTX4090，每张报告 24,564 MiB |

这是一次 2 秒硬件探针，只证明三个设备能够在同一 Slurm 作业中被真实分配和枚举；它不是多卡训练，也不证明 102 个正式 run 已完成。

## 单卡训练预检

标准 run：`runs/20260806/full_cell_gpu_single_card_preflight_202608061531/`；Slurm job `5390162`，节点 `g0902`，状态 `COMPLETED`，用时 29 秒，分配 `gpu=1,cpu=6,mem=60000M`。

| 检查 | 结果 |
|---|---:|
| CUDA 设备 | NVIDIA GeForce RTX 4090，24,210.3 MiB |
| PyTorch / CUDA / NCCL | `2.1.2+cu118` / `11.8` / `2.18.6` |
| HDF5 读取 | 1,024 samples，4 workers，2,061.46 samples/s |
| Batch size | 64 |
| MLP / ResMLP / CNN / FNO | 全部 `forward_backward_pass`，输出均为 `[64,81,10]` |
| FNO 训练 | 10 epochs，1.61 s |
| FNO 峰值显存 | 743.62 MiB |
| Checkpoint 恢复 | 输出逐元素精确一致 |
| Checkpoint SHA-256 | `13704159c79ed7ca32bd7964e686b7a85e0c016fa7232f09c65546d2c0500765` |
| Run status | `gpu_single_card_preflight_pass` |

完整 checkpoint 保留在 N40 标准 run 目录；本地已回收并逐项核对 `config.resolved.yaml`、`history.csv`、`metrics.json`、`results.json`、`run.log`、`run.meta.json` 和 Slurm 日志。该 run 明确记录 `formal_training_started=false` 与 `mpb_meep_used=false`。

## 已处理的失败与修正

1. N40 公网下载 PyTorch 速度不稳定，改用平台本地 `torch-2.1.2+cu118` wheel；依赖固定为 CentOS 7 可用的 manylinux2014 版本，并禁止在登录节点源码编译。
2. `--mem=60G` 会换算为 61,440 MB，超过每卡 60,000 MB 上限；最终由 `gpu_4090` 队列按每卡默认资源分配。
3. 作业 `5390157` 因 prefix Conda 环境不存在 `bin/activate` 而在 1 秒内失败；公共脚本改为显式设置该环境的 `PATH`，随后 `5390162` 通过。
4. N50R5 的 `loginCluster` 权限仍是外部阻塞，尚无该平台 CUDA/PyTorch 或 GPU 分配证据。

## 下一 gate

1. 人工确认是否启动 full-cell inverse N40 15-case 单卡 preflight；通过后再单独授权 18-task concurrency probe，不能把 dry-run 当作真实分配。
2. full-cell forward 102-run 和 inverse Stage-A 正式训练仍需按各自冻结计划单独授权；正向 checkpoint 未冻结前不得进入 inverse Stage B/ranking/replay。
3. N50R5 若要作为备选，需要平台恢复登录权限后重新执行同一硬件探针与单卡预检；不能沿用 N40 结果代替。
4. 任何 M9 threshold real-MPB preflight、pilot 或扩展任务都需单独授权，并与 surrogate 指标分表报告。

## 人工理解记录

```text
解决了什么问题：打通 N40 GPU 登录、环境、数据、三卡分配和单卡训练预检，并完成渐变 10k 数据的 WSL 验收。
为什么采用这个方案：三 seed 固定到三张同型号 GPU，GPU 只训练；M9 只承担经授权的 real-MPB，使用 hash 和冻结制品异步交接。
改变了项目哪一层/哪一模块：主要改变 L4 集群运行时、预检脚本和证据状态；没有改变 L0/L1 科学主张。
使用什么证据验证：Slurm jobs 5390073/5390162、标准 run 产物、两份 HDF5 SHA-256，以及 WSL meep_env real-MPB acceptance run。
最大风险、未知或下一 gate：N50 登录权限仍阻塞；正式三 seed 训练与 M9 threshold 物理验证均尚未获本轮启动授权。
```
