#!/usr/bin/env bash
# Run adaptive graded-medium worker benchmarks on Parallel Computing Cloud M9.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paracloud_m9_common.sh"

: "${PHC2D_BENCHMARK_ROOT:?Set PHC2D_BENCHMARK_ROOT to a runs/YYYYMMDD directory.}"
: "${PHC2D_SAMPLE_PLAN:?Set PHC2D_SAMPLE_PLAN to the frozen 10,000-row plan.}"
EXPECTED_PLAN_SHA256="5a046251e18bd0964ea8c70f4d52fa3bc48bff1b7e8997d4a3f17855be267ec2"
ACTUAL_PLAN_SHA256="$(sha256sum "$PHC2D_SAMPLE_PLAN" | awk '{print $1}')"
if [[ "$ACTUAL_PLAN_SHA256" != "$EXPECTED_PLAN_SHA256" ]]; then
  echo "Frozen sample plan SHA-256 mismatch: $ACTUAL_PLAN_SHA256" >&2
  exit 2
fi
if (( ${SLURM_CPUS_PER_TASK:-0} < 256 )); then
  echo "Adaptive benchmark requires --cpus-per-task=256 for comparable sequential cases." >&2
  exit 2
fi
mkdir -p "$PHC2D_BENCHMARK_ROOT"

run_case() {
  local samples="$1"
  local workers="$2"
  local run_dir="$PHC2D_BENCHMARK_ROOT/full_cell_graded_paracloud_m9_benchmark${samples}_w${workers}_${SLURM_JOB_ID}"
  mkdir -p "$run_dir"
  python -m phc2d.data.cluster_deployment prepare-plan \
    --source "$PHC2D_SAMPLE_PLAN" \
    --output "$run_dir/sample_plan.jsonl" \
    --config configs/research/full_cell_graded_bz9_10000_paracloud_m9_v1.yaml \
    --count "$samples"
  python -m phc2d.cli data generate \
    --config configs/research/full_cell_graded_bz9_10000_paracloud_m9_v1.yaml \
    --run-dir "$run_dir" \
    --max-samples "$samples" \
    --workers "$workers" &
  local solver_pid=$!
  python jobs/monitor_process_tree.py \
    --pid "$solver_pid" \
    --output "$run_dir/peak_process_tree_rss_bytes.txt" &
  local monitor_pid=$!
  wait "$solver_pid"
  wait "$monitor_pid"
}

for workers in 32 64 96 128; do
  run_case 128 "$workers"
done
if [[ "$(python -m phc2d.data.cluster_deployment extended-gate --benchmark-root "$PHC2D_BENCHMARK_ROOT")" == "YES" ]]; then
  for workers in 192 256; do
    run_case 256 "$workers"
  done
fi
python -m phc2d.data.cluster_deployment select \
  --benchmark-root "$PHC2D_BENCHMARK_ROOT" \
  --output "$PHC2D_BENCHMARK_ROOT/worker_selection.json" \
  --sample-count 10000
