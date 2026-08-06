#!/usr/bin/env bash
# Run one authorized 50k full-cell lane on a Xingyi compute node.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/HDD_POOL/phc2d}"
source "$PHC2D_ROOT/jobs/xingyi_common.sh"

if [[ "${PHC2D_AUTHORIZE_50K:-NO}" != "YES" ]]; then
  echo "50k generation is not authorized." >&2
  exit 3
fi
: "${PHC2D_LANE:?Set PHC2D_LANE to binary or graded.}"
: "${PHC2D_RUN_DIR:?Set PHC2D_RUN_DIR to the fixed standard run directory.}"
PHC2D_WORKERS="${PHC2D_WORKERS:-64}"
PHC2D_ALLOW_RESUME="${PHC2D_ALLOW_RESUME:-0}"

case "$PHC2D_LANE" in
  binary) config="configs/research/full_cell_binary_bz9_50000_xingyi_v1.yaml" ;;
  graded) config="configs/research/full_cell_graded_bz9_50000_xingyi_v1.yaml" ;;
  *) echo "PHC2D_LANE must be binary or graded." >&2; exit 2 ;;
esac
if (( PHC2D_WORKERS < 1 || PHC2D_WORKERS > 64 )); then
  echo "PHC2D_WORKERS must be in [1,64]." >&2
  exit 2
fi
if [[ ! -f "$PHC2D_RUN_DIR/sample_plan.jsonl" ]]; then
  echo "Frozen sample_plan.jsonl is required in $PHC2D_RUN_DIR." >&2
  exit 2
fi
if [[ "$(wc -l < "$PHC2D_RUN_DIR/sample_plan.jsonl")" -ne 50000 ]]; then
  echo "Frozen sample plan must contain exactly 50000 rows." >&2
  exit 2
fi
if [[ -d "$PHC2D_RUN_DIR/worker_shards" ]] && find "$PHC2D_RUN_DIR/worker_shards" -type f -print -quit | grep -q .; then
  if [[ "$PHC2D_ALLOW_RESUME" != "1" ]]; then
    echo "worker_shards is not empty; audited recovery requires PHC2D_ALLOW_RESUME=1." >&2
    exit 2
  fi
fi

python -m phc2d.cli data generate \
  --config "$config" \
  --run-dir "$PHC2D_RUN_DIR" \
  --workers "$PHC2D_WORKERS"
