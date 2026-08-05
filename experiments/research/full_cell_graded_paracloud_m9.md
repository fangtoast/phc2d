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

项目与环境固定部署到：

```text
/publicfs10/fs10-m9/home/m9s005801/phc2d
/publicfs10/fs10-m9/home/m9s005801/envs/meep_env_paracloud_m9
```

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
sbatch --export=ALL -p amd_m9_768 -N 1 -n 1 -c 256 --mem=700G \
  -o runs/20260805/paracloud_m9_benchmark_%j.log \
  jobs/paracloud_m9_graded_benchmark.sh
```

`worker_selection.json` 给出正式任务的 CPU、内存与时限。正式任务必须创建标准运行目录，只复制原始冻结计划，保持 `worker_shards/` 为空。`PHC2D_AUTHORIZE_GRADED=YES` 只能在用户另行授权后设置。

## 恢复与验收

失败恢复必须使用同一 Git commit、配置、计划和运行目录，并在审计现有分片后显式设置 `PHC2D_ALLOW_RESUME=1`。不得重新生成结构或把本地148个分片并入集群结果。

完成后运行：

```bash
PYTHONPATH=src python -m phc2d.cli data accept-cluster \
  --dataset data/full_cell_graded_bz9_10000_paracloud_m9_v1.h5 \
  --config configs/research/full_cell_graded_paracloud_m9_wsl_acceptance_v1.yaml
```

通过门为8个固定结构、9个代表性 k 点、bands 2–10 的 median MRE 不高于 `0.5%` 且 P95 不高于 `1%`。
