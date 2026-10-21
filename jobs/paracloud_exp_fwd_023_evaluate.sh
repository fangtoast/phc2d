#!/usr/bin/env bash
# Evaluate one frozen new-domain cell; no training or raw-test loading.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
set -euo pipefail
: "${PHC2D_ROOT:?Set the frozen deployment}"
export PHC2D_GPU_PLATFORM=n56
export PHC2D_CUDA_MODULE=cuda/12.8
export PHC2D_ENV="${PHC2D_ENV:-/data/run01/scxl892/envs/phc2d_gpu_cu128_torch271_r1}"
export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_TRANSFER_CAMPAIGN:?Set the frozen transfer campaign}"
: "${SLURM_ARRAY_TASK_ID:?Run as a Slurm array}"
python -m phc2d.forward.full_cell_multidomain_campaign run-evaluation-task \
  --campaign "$PHC2D_TRANSFER_CAMPAIGN" --cell-index "$SLURM_ARRAY_TASK_ID"
