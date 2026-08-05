#!/usr/bin/env bash
# Prepared-only Xingyi graded-medium job; explicit authorization is mandatory.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/xingyi_common.sh"

if [[ "${PHC2D_AUTHORIZE_GRADED:-NO}" != "YES" ]]; then
  echo "Graded generation is prepared only and is not authorized for submission." >&2
  exit 3
fi
: "${PHC2D_RUN_DIR:?Set PHC2D_RUN_DIR to the fixed standard run directory.}"
PHC2D_WORKERS="${PHC2D_WORKERS:-64}"

python -m phc2d.cli data generate \
  --config configs/research/full_cell_graded_bz9_10000_xingyi_v1.yaml \
  --run-dir "$PHC2D_RUN_DIR" \
  --workers "$PHC2D_WORKERS"
