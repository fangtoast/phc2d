#!/usr/bin/env bash
# Benchmark binary full-cell MPB generation at fixed Xingyi worker counts.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/HDD_POOL/phc2d}"
source "$PHC2D_ROOT/jobs/xingyi_common.sh"

: "${PHC2D_BENCHMARK_ROOT:?Set PHC2D_BENCHMARK_ROOT to a runs/YYYYMMDD directory.}"
: "${PHC2D_SAMPLE_PLAN:?Set PHC2D_SAMPLE_PLAN to the frozen 10,000-row sample plan.}"
if [[ "$(wc -l < "$PHC2D_SAMPLE_PLAN")" -ne 10000 ]]; then
  echo "PHC2D_SAMPLE_PLAN must contain exactly 10000 rows." >&2
  exit 2
fi
mkdir -p "$PHC2D_BENCHMARK_ROOT"

for workers in 16 32 48 64; do
  run_dir="$PHC2D_BENCHMARK_ROOT/full_cell_binary_xingyi_benchmark64_w${workers}_${SLURM_JOB_ID}"
  mkdir -p "$run_dir"
  python -m phc2d.data.xingyi_deployment \
    --prepare-plan-source "$PHC2D_SAMPLE_PLAN" \
    --prepare-plan-output "$run_dir/sample_plan.jsonl" \
    --config configs/research/full_cell_binary_bz9_10000_xingyi_v1.yaml \
    --count 64
  python -m phc2d.cli data generate \
    --config configs/research/full_cell_binary_bz9_10000_xingyi_v1.yaml \
    --run-dir "$run_dir" \
    --max-samples 64 \
    --workers "$workers" &
  solver_pid=$!
  python jobs/monitor_process_tree.py \
    --pid "$solver_pid" \
    --output "$run_dir/peak_process_tree_rss_bytes.txt" &
  monitor_pid=$!
  wait "$solver_pid"
  wait "$monitor_pid"
done

python -m phc2d.data.xingyi_deployment \
  --benchmark-root "$PHC2D_BENCHMARK_ROOT" \
  --output "$PHC2D_BENCHMARK_ROOT/worker_selection.json"
