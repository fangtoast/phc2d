# Run Xingyi environment, MPB-oracle, and four-sample full-cell preflights.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/xingyi_common.sh"

: "${PHC2D_PREFLIGHT_ROOT:?Set PHC2D_PREFLIGHT_ROOT to a runs/YYYYMMDD directory.}"
mkdir -p "$PHC2D_PREFLIGHT_ROOT"

python -m phc2d.cli data generate \
  --config configs/research/full_cell_solver_preflight_xingyi_v1.yaml \
  --run-dir "$PHC2D_PREFLIGHT_ROOT/full_cell_solver_preflight_xingyi_v1_${SLURM_JOB_ID}"

python -m phc2d.cli data generate \
  --config configs/research/full_cell_binary_bz9_10000_xingyi_v1.yaml \
  --run-dir "$PHC2D_PREFLIGHT_ROOT/full_cell_binary_xingyi_preflight4_${SLURM_JOB_ID}" \
  --max-samples 4 \
  --workers 4
