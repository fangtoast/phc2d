#!/usr/bin/env bash
# Run solver-oracle and four-sample graded preflights on Parallel Computing Cloud M9.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-/publicfs10/fs10-m9/home/m9s005801/phc2d}"
source "$PHC2D_ROOT/jobs/paracloud_m9_common.sh"

: "${PHC2D_PREFLIGHT_ROOT:?Set PHC2D_PREFLIGHT_ROOT to a runs/YYYYMMDD directory.}"
: "${PHC2D_SAMPLE_PLAN:?Set PHC2D_SAMPLE_PLAN to the frozen 10,000-row plan.}"
EXPECTED_PLAN_SHA256="5a046251e18bd0964ea8c70f4d52fa3bc48bff1b7e8997d4a3f17855be267ec2"
ACTUAL_PLAN_SHA256="$(sha256sum "$PHC2D_SAMPLE_PLAN" | awk '{print $1}')"
if [[ "$ACTUAL_PLAN_SHA256" != "$EXPECTED_PLAN_SHA256" ]]; then
  echo "Frozen sample plan SHA-256 mismatch: $ACTUAL_PLAN_SHA256" >&2
  exit 2
fi

mkdir -p "$PHC2D_PREFLIGHT_ROOT"
oracle_dir="$PHC2D_PREFLIGHT_ROOT/full_cell_solver_preflight_paracloud_m9_v1_${SLURM_JOB_ID}"
graded_dir="$PHC2D_PREFLIGHT_ROOT/full_cell_graded_paracloud_m9_preflight4_${SLURM_JOB_ID}"
python -m phc2d.cli data generate \
  --config configs/research/full_cell_solver_preflight_paracloud_m9_v1.yaml \
  --run-dir "$oracle_dir"
mkdir -p "$graded_dir"
python -m phc2d.data.cluster_deployment prepare-plan \
  --source "$PHC2D_SAMPLE_PLAN" \
  --output "$graded_dir/sample_plan.jsonl" \
  --config configs/research/full_cell_graded_bz9_10000_paracloud_m9_v1.yaml \
  --count 4
python -m phc2d.cli data generate \
  --config configs/research/full_cell_graded_bz9_10000_paracloud_m9_v1.yaml \
  --run-dir "$graded_dir" \
  --max-samples 4 \
  --workers 4
