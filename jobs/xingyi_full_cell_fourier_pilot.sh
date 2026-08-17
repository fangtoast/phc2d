#!/usr/bin/env bash
# Run one authorized unified-Fourier smoke or 64-sample lane on Xingyi.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/HDD_POOL/phc2d}"
source "$PHC2D_ROOT/jobs/xingyi_common.sh"

: "${PHC2D_LANE:?Set PHC2D_LANE to binary or graded.}"
: "${PHC2D_PHASE:?Set PHC2D_PHASE to smoke or pilot.}"
: "${PHC2D_RUN_DIR:?Set PHC2D_RUN_DIR to a fixed standard run directory.}"

case "$PHC2D_LANE" in
  binary) config="configs/research/full_cell_binary_fourier_bz9_64_xingyi_pilot_v1.yaml" ;;
  graded) config="configs/research/full_cell_graded_fourier_bz9_64_xingyi_pilot_v1.yaml" ;;
  *) echo "PHC2D_LANE must be binary or graded." >&2; exit 2 ;;
esac

case "$PHC2D_PHASE" in
  smoke)
    if [[ "${PHC2D_AUTHORIZE_FOURIER_SMOKE:-NO}" != "YES" ]]; then
      echo "The 8+8 solver smoke is not authorized." >&2
      exit 3
    fi
    sample_count=8
    workers="${PHC2D_WORKERS:-8}"
    ;;
  pilot)
    if [[ "${PHC2D_AUTHORIZE_FOURIER_PILOT:-NO}" != "YES" ]]; then
      echo "The 64+64 pilot is not authorized." >&2
      exit 3
    fi
    sample_count=64
    workers="${PHC2D_WORKERS:-64}"
    if [[ ! -f "$PHC2D_RUN_DIR/sample_plan.jsonl" ]]; then
      echo "The 64-sample pilot requires a frozen sample_plan.jsonl." >&2
      exit 2
    fi
    if [[ "$(wc -l < "$PHC2D_RUN_DIR/sample_plan.jsonl")" -ne 64 ]]; then
      echo "The frozen pilot sample plan must contain exactly 64 rows." >&2
      exit 2
    fi
    ;;
  *) echo "PHC2D_PHASE must be smoke or pilot." >&2; exit 2 ;;
esac

if (( workers < 1 || workers > ${SLURM_CPUS_PER_TASK:-0} )); then
  echo "PHC2D_WORKERS must fit the allocated CPUs." >&2
  exit 2
fi
if [[ -d "$PHC2D_RUN_DIR/worker_shards" ]] && find "$PHC2D_RUN_DIR/worker_shards" -type f -print -quit | grep -q .; then
  if [[ "${PHC2D_ALLOW_RESUME:-0}" != "1" ]]; then
    echo "Existing worker shards require an audited PHC2D_ALLOW_RESUME=1 recovery." >&2
    exit 2
  fi
fi

args=(
  -m phc2d.cli data generate
  --config "$config"
  --run-dir "$PHC2D_RUN_DIR"
  --workers "$workers"
  --plan-workers "${PHC2D_PLAN_WORKERS:-8}"
)
if [[ "$PHC2D_PHASE" == "smoke" ]]; then
  args+=(--max-samples "$sample_count")
fi
python "${args[@]}"
