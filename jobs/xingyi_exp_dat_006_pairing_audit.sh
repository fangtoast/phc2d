#!/usr/bin/env bash
# Audit all six frozen EXP-DAT-006 plans on Xingyi without running MPB.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/HDD_POOL/phc2d}"
source "$PHC2D_ROOT/jobs/xingyi_common.sh"
: "${PHC2D_EXP_DAT_006_CAMPAIGN_ROOT:?Set the EXP-DAT-006 campaign root.}"
: "${PHC2D_EXPECTED_COMMIT:?Set the exact deployed commit.}"
[[ "$PHC2D_DEPLOYED_GIT_COMMIT" == "$PHC2D_EXPECTED_COMMIT" ]] || {
  echo "Deployed commit mismatch." >&2
  exit 2
}

args=()
for lane in binary graded; do
  for cutoff in 8 10 15; do
    args+=(--run-dir "$PHC2D_EXP_DAT_006_CAMPAIGN_ROOT/k${cutoff}/${lane}")
  done
done
python -m phc2d.cli data audit-fourier-pairing "${args[@]}" \
  --expected-cutoff 8 --expected-cutoff 10 --expected-cutoff 15 \
  --output "$PHC2D_EXP_DAT_006_CAMPAIGN_ROOT/pairing_audit.json"
