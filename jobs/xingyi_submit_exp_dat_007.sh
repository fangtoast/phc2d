#!/usr/bin/env bash
# Submit the authorized EXP-DAT-007 single-node, 64-worker MPB production campaign.
# The resolution pilot is user-waived; this launcher does not retry or change resources.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/HDD_POOL/phc2d}"
PHC2D_RUN_ROOT="${PHC2D_RUN_ROOT:-$HOME/HDD_POOL/runs/$(date +%Y%m%d)}"
PHC2D_CAMPAIGN_ID="${PHC2D_CAMPAIGN_ID:-EXP-DAT-007_$(date +%Y%m%d%H%M)}"
PHC2D_SUBMIT_MODE="${PHC2D_SUBMIT_MODE:-dry-run}"
: "${PHC2D_EXPECTED_COMMIT:?Set PHC2D_EXPECTED_COMMIT to the deployed source commit.}"

config="$PHC2D_ROOT/configs/research/full_cell_single_hole_exp_dat_007_v1.yaml"
run_dir="$PHC2D_RUN_ROOT/$PHC2D_CAMPAIGN_ID"

grep -q '^  formal_generation_authorized: true$' "$config"
grep -q '^  solver_resolution_pilot_status: waived_by_user$' "$config"
grep -q '^  automatic_retry_authorized: false$' "$config"
grep -q '^  resource_change_authorized: false$' "$config"

export PHC2D_ROOT PHC2D_SUBMIT_MODE PHC2D_EXPECTED_COMMIT
export PHC2D_ADAPTER=full_cell_band
export PHC2D_CONFIG="$config"
export PHC2D_RUN_DIR="$run_dir"
export PHC2D_PLATFORM_PROFILE="$PHC2D_ROOT/configs/research/hpc_platform_xingyi_v1.yaml"
export PHC2D_CASE_COUNT=1200
export PHC2D_NODE_TIER=1
export PHC2D_AUTHORIZE_FORMAL=YES

exec "$PHC2D_ROOT/jobs/hpc_submit_physics_campaign.sh"
