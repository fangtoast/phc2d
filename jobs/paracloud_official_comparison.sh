#!/bin/bash -l
# Immutable node-local deployment for EXP-FWD-031 single-GPU workers.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail
: "${PHC2D_ROOT:?Set shared deployment}"
: "${PHC2D_SUITE:?Set campaign}"
: "${PHC2D_STAGE:?Set admission or train}"
: "${PHC2D_ARCHIVE_SHA256:?Set archive SHA}"
shared_root="$PHC2D_ROOT"
printf '%s  %s\n' "$PHC2D_ARCHIVE_SHA256" "$shared_root/deployment.tar.gz" | sha256sum -c -
task_root=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/phc2d-official-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "$task_root"' EXIT
tar -xzf "$shared_root/deployment.tar.gz" -C "$task_root"
cp "$shared_root/DEPLOYED_GIT_COMMIT" "$task_root/DEPLOYED_GIT_COMMIT"
export PHC2D_ROOT="$task_root"
case "${PHC2D_GPU_PLATFORM:?Set n40 or n56}" in
  n40)
    export PHC2D_ENV=/HOME/scw6g7a/run/envs/phc2d_gpu_cu118
    official_overlay=/data/run01/scw6g7a/envs/phc2d_exp_fwd_031/official_overlay
    ;;
  n56)
    export PHC2D_ENV=/data/run01/scxl892/envs/phc2d_gpu_cu128_torch271_r1
    official_overlay=/data/run01/scxl892/envs/phc2d_exp_fwd_031/official_overlay
    ;;
  *) echo "Unsupported platform" >&2; exit 2 ;;
esac
[[ -d "$official_overlay/neuralop" ]] || { echo "Missing isolated official package overlay" >&2; exit 2; }
(cd "$official_overlay" && sha256sum -c "$task_root/jobs/official_neuraloperator_2_0_0.sha256")
source "$task_root/jobs/paracloud_gpu_common.sh"
export PYTHONPATH="$official_overlay:$PYTHONPATH"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
python -c 'import importlib.metadata as m, torch, neuralop; assert m.version("neuraloperator") == "2.0.0"; assert m.version("tensorly") == "0.9.0"; print("runtime", torch.__version__, neuralop.__version__, m.version("tensorly"))'
cp "$task_root/jobs/official_neuraloperator_2_0_0.sha256" "$PHC2D_SUITE/official_source_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.sha256"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 10 > "$PHC2D_SUITE/gpu_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true; rm -rf -- "$task_root"' EXIT
python -u -m phc2d.forward.official_comparison_campaign "$PHC2D_STAGE" --suite "$PHC2D_SUITE"
