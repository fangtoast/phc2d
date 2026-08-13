<!--
  Parallel Computing Cloud M9 deployment contract for graded full-cell generation.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 并行超算云 M9 连续渐变数据生成

## 证据边界

本任务在 `amd_m9_768` 的单个 Rocky Linux 8.10 计算节点上，从冻结的 10000 行结构计划重新计算全部 real-MPB 标签。现有 WSL 渐变分片不上传、不恢复、不合并。集群完成状态只能是 `cluster_generated_pending_wsl_acceptance`；必须再由 WSL Ubuntu + conda `meep_env` 交叉复算才能晋级。

冻结计划 SHA-256：

```text
5a046251e18bd0964ea8c70f4d52fa3bc48bff1b7e8997d4a3f17855be267ec2
```

项目与环境归档固定部署到：

```text
/publicfs10/fs10-m9/home/m9s005801/phc2d
/publicfs10/fs10-m9/home/m9s005801/envs/meep_env_paracloud_m9.tar.gz
```

公共文件系统不展开 Conda 的数万个小文件。每个 Slurm 作业先校验环境归档 SHA-256，再解包到 `${SLURM_TMPDIR:-/tmp}/phc2d_${USER}_${SLURM_JOB_ID}/meep_env`，运行 `conda-unpack`，作业退出时仅清理该精确节点本地目录。

## 提交顺序

环境包上传和解压后，先提交兼容性检查：

```bash
sbatch -p amd_m9_768 -N 1 -n 1 -c 4 --mem=16G \
  -o runs/20260805/paracloud_m9_environment_%j.log \
  jobs/paracloud_m9_environment_preflight.sh
```

随后提交 solver oracle 和 4 样本渐变预检：

```bash
PHC2D_SAMPLE_PLAN=/path/to/frozen/sample_plan.jsonl \
PHC2D_PREFLIGHT_ROOT=runs/20260805 \
sbatch --export=ALL -p amd_m9_768 -N 1 -n 1 -c 4 --mem=32G \
  -o runs/20260805/paracloud_m9_solver_preflight_%j.log \
  jobs/paracloud_m9_solver_preflight.sh
```

两项均通过后，使用单个 256 CPU allocation 顺序执行自适应基准：

```bash
PHC2D_SAMPLE_PLAN=/path/to/frozen/sample_plan.jsonl \
PHC2D_BENCHMARK_ROOT=runs/20260805/paracloud_m9_graded_benchmark \
sbatch --export=ALL -p amd_m9_768 -N 1 -n 1 -c 256 --mem=690G \
  -o runs/20260805/paracloud_m9_benchmark_%j.log \
  jobs/paracloud_m9_graded_benchmark.sh
```

M9 分区限制 `MaxMemPerCPU=2764 MB`；256 CPU 搭配 700 GiB 会被 Slurm
拒绝，因而基准预留 690 GiB。正式任务仍使用 `worker_selection.json` 给出的
实测内存申请。

`worker_selection.json` 给出正式任务的 CPU、内存与时限。正式任务必须创建标准运行目录，只复制原始冻结计划，保持 `worker_shards/` 为空。`PHC2D_AUTHORIZE_GRADED=YES` 只能在用户另行授权后设置。

## 2026-08-05 现场证据

- 环境预检作业 `4190605`：`COMPLETED/0:0`；Rocky/RHEL 8.10、glibc
  2.28、Meep 1.33.0-beta、MPB 1.12.0。
- solver oracle 与 4 样本预检作业 `4190628`：`COMPLETED/0:0`；
  `Wh=[4,81,10]`，有限值、频带升序、失败数 0，`graded_core_1/2`
  各 2 个。状态仍为 `cluster_real_mpb_preflight_pass_pending_wsl_acceptance`。
- 自适应基准作业 `4190638`：`COMPLETED/0:0`；六档均零失败且无 OOM。
  32/64/96/128/192/256 workers 的吞吐分别为
  1.817/3.274/3.258/4.862/5.870/6.694 samples/s，峰值 RSS 分别为
  9.35/18.39/27.40/36.43/54.54/72.55 GiB。
- 自动选择 256 workers；正式资源建议为 256 CPU、96 GiB、24 小时。
  该结果只授权准备正式命令，不授权提交 10000 样本作业。

## 恢复与验收

失败恢复必须使用同一 Git commit、配置、计划和运行目录，并在审计现有分片后显式设置 `PHC2D_ALLOW_RESUME=1`。不得重新生成结构或把本地148个分片并入集群结果。

完成后运行：

```bash
PYTHONPATH=src python -m phc2d.cli data accept-cluster \
  --dataset data/full_cell_graded_bz9_10000_paracloud_m9_v1.h5 \
  --config configs/research/full_cell_graded_paracloud_m9_wsl_acceptance_v1.yaml
```

通过门为8个固定结构、9个代表性 k 点、bands 2–10 的 median MRE 不高于 `0.5%` 且 P95 不高于 `1%`。
