#!/usr/bin/env bash
# Run one frozen EXP-FWD-021 checkpoint-to-target evaluation cell on N56.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d-exp-fwd-021}"
export PHC2D_GPU_PLATFORM=n56
export PHC2D_CUDA_MODULE=cuda/12.8
export PHC2D_ENV="${PHC2D_ENV:-/data/run01/scxl892/envs/phc2d_gpu_cu128_torch271_r1}"
export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_FWD021_CAMPAIGN:?Set the frozen campaign directory.}"
: "${SLURM_ARRAY_TASK_ID:?Run as a Slurm array task.}"

python -m phc2d.forward.full_cell_multidomain_campaign run-evaluation-task \
  --campaign "$PHC2D_FWD021_CAMPAIGN" --cell-index "$SLURM_ARRAY_TASK_ID"
