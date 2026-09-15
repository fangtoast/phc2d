#!/usr/bin/env bash
# Run one frozen EXP-FWD-021 training or admission task on N56 RTX 5090.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

#SBATCH --partition=hp_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60000
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

arguments=(--campaign "$PHC2D_FWD021_CAMPAIGN" --task-index "$SLURM_ARRAY_TASK_ID")
if [[ "${PHC2D_FWD021_ADMISSION:-0}" == 1 ]]; then
  arguments+=(--admission-probe --max-train-batches 1 --max-validation-samples 64)
fi
python -m phc2d.forward.full_cell_multidomain_campaign run-training-task "${arguments[@]}"
