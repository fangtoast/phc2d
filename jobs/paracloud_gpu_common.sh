#!/usr/bin/env bash
# Shared runtime setup for phc2d jobs on Parallel Computing Cloud GPU partitions.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

: "${SLURM_JOB_ID:?This script must run inside a Slurm allocation.}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required.}"

PHC2D_GPU_PLATFORM="${PHC2D_GPU_PLATFORM:-n40}"
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
case "$PHC2D_GPU_PLATFORM" in
  n40)
    PHC2D_ENV="${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu118}"
    PHC2D_CUDA_MODULE="${PHC2D_CUDA_MODULE:-cuda/11.8}"
    ;;
  n50)
    PHC2D_ENV="${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu128}"
    PHC2D_CUDA_MODULE="${PHC2D_CUDA_MODULE:-cuda/12.8}"
    ;;
  *)
    echo "Unsupported PHC2D_GPU_PLATFORM: $PHC2D_GPU_PLATFORM" >&2
    exit 2
    ;;
esac

if [[ ! -x "$PHC2D_ENV/bin/python" ]]; then
  echo "Missing GPU environment: $PHC2D_ENV" >&2
  exit 2
fi
if [[ ! -d "$PHC2D_ROOT/src/phc2d" ]]; then
  echo "Missing phc2d deployment: $PHC2D_ROOT" >&2
  exit 2
fi
if [[ ! -f "$PHC2D_ROOT/DEPLOYED_GIT_COMMIT" ]]; then
  echo "Missing deployed Git commit provenance: $PHC2D_ROOT/DEPLOYED_GIT_COMMIT" >&2
  exit 2
fi

module purge >/dev/null 2>&1 || true
module load "$PHC2D_CUDA_MODULE"
export PATH="$PHC2D_ENV/bin:$PATH"
export PYTHONPATH="$PHC2D_ROOT/src"
export PHC2D_CLUSTER_ID="paracloud_${PHC2D_GPU_PLATFORM}"
export PHC2D_DEPLOYED_GIT_COMMIT="$(tr -d '[:space:]' < "$PHC2D_ROOT/DEPLOYED_GIT_COMMIT")"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "$PHC2D_ROOT"
if [[ ! "$PHC2D_DEPLOYED_GIT_COMMIT" =~ ^[0-9a-f]{7,40}$ ]]; then
  echo "Invalid deployed Git commit provenance: $PHC2D_DEPLOYED_GIT_COMMIT" >&2
  exit 2
fi
if [[ "${PHC2D_DEFER_CUDA_CHECK_TO_RUNNER:-0}" != "1" ]]; then
  python -c "import torch; assert torch.cuda.is_available(); print('CUDA:', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
fi
