#!/usr/bin/env bash
# Verify that a Parallel Computing Cloud Slurm allocation exposes every requested GPU.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

: "${SLURM_JOB_ID:?This script must run inside a Slurm allocation.}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required.}"

expected_gpus="${PHC2D_EXPECTED_GPUS:-3}"
if ! [[ "$expected_gpus" =~ ^[1-8]$ ]]; then
  echo "PHC2D_EXPECTED_GPUS must be an integer in [1,8]." >&2
  exit 2
fi

visible_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d '[:space:]')"
if [[ "$visible_gpus" != "$expected_gpus" ]]; then
  echo "Expected $expected_gpus visible GPUs, got $visible_gpus." >&2
  exit 3
fi

echo "status=three_gpu_hardware_probe_pass"
echo "slurm_job_id=$SLURM_JOB_ID"
echo "slurm_partition=${SLURM_JOB_PARTITION:-unknown}"
echo "slurm_node=$SLURM_JOB_NODELIST"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "visible_gpu_count=$visible_gpus"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pci.bus_id --format=csv,noheader
