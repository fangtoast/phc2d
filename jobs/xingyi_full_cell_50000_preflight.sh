#!/usr/bin/env bash
# Run the Xingyi solver oracle and dual-lane four-sample 50k preflight.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/HDD_POOL/phc2d}"
source "$PHC2D_ROOT/jobs/xingyi_common.sh"

if [[ "${PHC2D_AUTHORIZE_50K_PREFLIGHT:-NO}" != "YES" ]]; then
  echo "50k preflight allocation is not authorized." >&2
  exit 3
fi
: "${PHC2D_PREFLIGHT_ROOT:?Set PHC2D_PREFLIGHT_ROOT to a standard campaign run directory.}"
mkdir -p "$PHC2D_PREFLIGHT_ROOT"

python -m phc2d.cli data generate \
  --config configs/research/full_cell_solver_preflight_xingyi_v1.yaml \
  --run-dir "$PHC2D_PREFLIGHT_ROOT/full_cell_solver_preflight_xingyi_v1_${SLURM_JOB_ID}"

python -m phc2d.cli data generate \
  --config configs/research/full_cell_binary_bz9_50000_xingyi_v1.yaml \
  --run-dir "$PHC2D_PREFLIGHT_ROOT/full_cell_binary_50000_xingyi_preflight4_${SLURM_JOB_ID}" \
  --max-samples 4 \
  --workers 4

python -m phc2d.cli data generate \
  --config configs/research/full_cell_graded_bz9_50000_xingyi_v1.yaml \
  --run-dir "$PHC2D_PREFLIGHT_ROOT/full_cell_graded_50000_xingyi_preflight4_${SLURM_JOB_ID}" \
  --max-samples 4 \
  --workers 4
