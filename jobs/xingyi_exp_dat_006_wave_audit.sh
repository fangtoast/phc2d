#!/usr/bin/env bash
# Run read-only final artifact audits for one completed EXP-DAT-006 K wave.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/HDD_POOL/phc2d}"
source "$PHC2D_ROOT/jobs/xingyi_common.sh"
: "${PHC2D_EXP_DAT_006_CAMPAIGN_ROOT:?Set the EXP-DAT-006 campaign root.}"
: "${PHC2D_EXP_DAT_006_K:?Set K to 8, 10, or 15.}"
: "${PHC2D_EXPECTED_COMMIT:?Set the exact deployed commit.}"
[[ "$PHC2D_EXP_DAT_006_K" =~ ^(8|10|15)$ ]] || exit 2
[[ "$PHC2D_DEPLOYED_GIT_COMMIT" == "$PHC2D_EXPECTED_COMMIT" ]] || exit 2

wave_dir="$PHC2D_EXP_DAT_006_CAMPAIGN_ROOT/k${PHC2D_EXP_DAT_006_K}"
for lane in binary graded; do
  python -m phc2d.cli hpc audit --run-dir "$wave_dir/$lane" \
    > "$wave_dir/$lane/hpc_audit.json"
done
printf 'status=passed\nk=%s\ndeployed_git_commit=%s\n' \
  "$PHC2D_EXP_DAT_006_K" "$PHC2D_DEPLOYED_GIT_COMMIT" > "$wave_dir/wave_audit.env"
