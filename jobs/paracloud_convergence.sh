#!/bin/bash -l
# Execute dynamically claimed tasks while bounding each platform to eight GPUs.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail
: "${PHC2D_ROOT:?Set isolated deployment}"
: "${PHC2D_SUITE:?Set frozen convergence suite}"
: "${PHC2D_STAGE:?Set admission or train}"
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 10 > "$PHC2D_SUITE/gpu_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
python -u -m phc2d.forward.convergence_campaign "$PHC2D_STAGE" --suite "$PHC2D_SUITE"
