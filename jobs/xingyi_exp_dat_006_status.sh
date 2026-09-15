#!/usr/bin/env bash
# Read-only EXP-DAT-006 Xingyi status snapshot; intended for 30-minute polling.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/HDD_POOL/phc2d}"
PHC2D_EXP_DAT_006_CAMPAIGN_ROOT="${PHC2D_EXP_DAT_006_CAMPAIGN_ROOT:-$PHC2D_ROOT/runs/20260915/EXP-DAT-006_xingyi_v1}"
source "$PHC2D_ROOT/jobs/xingyi_common.sh"

squeue -u "$USER" -o '%.18i %.18j %.2t %.10M %.6D %R'
for cutoff in 8 10 15; do
  for lane in binary graded; do
    run_dir="$PHC2D_EXP_DAT_006_CAMPAIGN_ROOT/k${cutoff}/${lane}"
    if [[ -d "$run_dir" ]]; then
      printf '\n[%s K%s]\n' "$lane" "$cutoff"
      python -m phc2d.cli hpc status --run-dir "$run_dir"
    fi
  done
done
