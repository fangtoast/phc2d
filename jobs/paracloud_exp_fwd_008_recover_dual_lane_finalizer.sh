#!/usr/bin/env bash
# Recover missing EXP-FWD-008 dual-lane finalizer artifacts without retraining.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
: "${PHC2D_RECOVERY_RUN:?PHC2D_RECOVERY_RUN is required}"
: "${PHC2D_SOURCE_SLURM_JOB_ID:?PHC2D_SOURCE_SLURM_JOB_ID is required}"
export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
python -m phc2d.forward.full_cell_training_campaign recover-dual-lane-finalizer \
  --run-dir "$PHC2D_RECOVERY_RUN" \
  --source-slurm-job-id "$PHC2D_SOURCE_SLURM_JOB_ID"
