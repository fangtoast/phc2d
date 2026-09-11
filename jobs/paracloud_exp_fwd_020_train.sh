#!/usr/bin/env bash
# Run one frozen EXP-FWD-020 G1 or G2 task on N56 RTX 5090.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d-exp-fwd-020}"
export PHC2D_GPU_PLATFORM=n56
export PHC2D_CUDA_MODULE=cuda/12.8
export PHC2D_ENV="${PHC2D_ENV:-/data/run01/scxl892/envs/phc2d_gpu_cu128_torch271_r1}"
export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_FWD020_MANIFEST:?Set frozen task manifest.}"
: "${SLURM_ARRAY_TASK_ID:?Run as a Slurm array task.}"
python -m phc2d.forward.global_local_bridge_campaign run-training-task \
  --manifest "$PHC2D_FWD020_MANIFEST" --task-index "$SLURM_ARRAY_TASK_ID" \
  --repository-root "$PHC2D_ROOT"
