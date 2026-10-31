#!/bin/bash -l
# Execute one EXP-FWD-032 admission or training worker on node-local code.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail
: "${PHC2D_ROOT:?Set immutable shared deployment}"
: "${PHC2D_SUITE:?Set frozen campaign}"
: "${PHC2D_STAGE:?Set admission or train}"
: "${PHC2D_ARCHIVE_SHA256:?Set deployment archive hash}"
shared_root="$PHC2D_ROOT"
printf '%s  %s\n' "$PHC2D_ARCHIVE_SHA256" "$shared_root/deployment.tar.gz" | sha256sum -c -
task_root=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/phc2d-032-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "$task_root"' EXIT
tar -xzf "$shared_root/deployment.tar.gz" -C "$task_root"
cp "$shared_root/DEPLOYED_GIT_COMMIT" "$task_root/DEPLOYED_GIT_COMMIT"
export PHC2D_ROOT="$task_root"
source "$task_root/jobs/paracloud_gpu_common.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 10 > "$PHC2D_SUITE/gpu_${PHC2D_STAGE}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true; rm -rf -- "$task_root"' EXIT
python -u -m phc2d.forward.single_domain_branch_campaign "$PHC2D_STAGE" --suite "$PHC2D_SUITE"
