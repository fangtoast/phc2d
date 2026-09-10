#!/usr/bin/env bash
# Run the authorized 100-structure MPB resolution Stage-B confirmation on Xingyi.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/HDD_POOL/phc2d}"
source "$PHC2D_ROOT/jobs/xingyi_common.sh"

if [[ "${PHC2D_AUTHORIZE_RESOLUTION_STAGE_B:-NO}" != "YES" ]]; then
  echo "MPB resolution Stage-B confirmation is not authorized." >&2
  exit 3
fi
: "${PHC2D_RUN_DIR:?Set PHC2D_RUN_DIR to a fresh standard run directory.}"
if [[ "${SLURM_CPUS_PER_TASK:-0}" -ne 64 ]]; then
  echo "Stage B requires one 64-CPU Xingyi node." >&2
  exit 2
fi
if [[ -e "$PHC2D_RUN_DIR/results.json" || -d "$PHC2D_RUN_DIR/worker_shards" ]]; then
  echo "Stage B requires a fresh run directory; recovery needs separate authorization." >&2
  exit 2
fi

for dataset in \
  data/full_cell_binary_bz9_10000_xingyi_v1.h5 \
  data/full_cell_graded_bz9_10000_paracloud_m9_v1.h5
do
  if [[ ! -f "$dataset" ]]; then
    echo "Missing frozen Stage-B input: $dataset" >&2
    exit 2
  fi
done

mkdir -p "$PHC2D_RUN_DIR"
printf 'resource_usage_source=slurm_sacct\n' > "$PHC2D_RUN_DIR/resource_usage.txt"
python -m phc2d.cli data converge \
  --config configs/research/full_cell_resolution_convergence_xingyi_stage_b_v1.yaml \
  --run-dir "$PHC2D_RUN_DIR"
