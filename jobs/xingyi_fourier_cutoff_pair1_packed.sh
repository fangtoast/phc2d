#!/usr/bin/env bash
# Run all 32 paired Fourier-cutoff members inside one Xingyi allocation.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/HDD_POOL/phc2d}"
: "${PHC2D_CUTOFF_ROOT:?Set the fixed campaign root.}"
: "${PHC2D_AUTHORIZE_FOURIER_CUTOFF:?Explicit authorization is required.}"
[[ "$PHC2D_AUTHORIZE_FOURIER_CUTOFF" == YES ]] || exit 3
(( ${SLURM_CPUS_PER_TASK:-0} >= 32 )) || { echo "32 CPUs are required." >&2; exit 2; }

pids=()
for task in $(seq 0 31); do
  member_log="$PHC2D_CUTOFF_ROOT/logs/member_$(printf '%02d' "$task").log"
  (
    export SLURM_ARRAY_TASK_ID="$task"
    bash "$PHC2D_ROOT/jobs/xingyi_fourier_cutoff_pair1_array.sh"
  ) >"$member_log" 2>&1 &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failures=$((failures + 1)); fi
done
if (( failures > 0 )); then
  echo "$failures cutoff members failed." >&2
  exit 1
fi
echo "All 32 paired Fourier-cutoff members completed."
