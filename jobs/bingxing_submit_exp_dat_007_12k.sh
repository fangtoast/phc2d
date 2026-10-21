#!/usr/bin/env bash
# Submit the authorized EXP-DAT-007 12k campaign as four independent 64-CPU shards.
# The launcher does not retry, resume, or change the frozen scientific contract.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_ROOT="${PHC2D_ROOT:-/publicfs10/fs10-m9/home/m9s005801/phc2d}"
PHC2D_RUN_ROOT="${PHC2D_RUN_ROOT:-/publicfs10/fs10-m9/home/m9s005801/runs/$(date +%Y%m%d)}"
PHC2D_CAMPAIGN_ID="${PHC2D_CAMPAIGN_ID:-EXP-DAT-007_12k_bingxing_$(date +%Y%m%d%H%M)}"
PHC2D_SUBMIT_MODE="${PHC2D_SUBMIT_MODE:-dry-run}"
: "${PHC2D_EXPECTED_COMMIT:?Set PHC2D_EXPECTED_COMMIT to the deployed source commit.}"

config="$PHC2D_ROOT/configs/research/full_cell_single_hole_exp_dat_007_bingxing_12k_v1.yaml"
run_dir="$PHC2D_RUN_ROOT/$PHC2D_CAMPAIGN_ID"

grep -q '^  formal_generation_authorized: true$' "$config"
grep -q '^  solver_resolution_pilot_status: waived_by_user$' "$config"
grep -q '^  sample_expansion_scope: six_families_200_to_2000_each$' "$config"
grep -q '^  resource_change_scope: bingxing_m9_four_shards_64_cpu_each$' "$config"
grep -q '^  automatic_retry_authorized: false$' "$config"
grep -q '^  sample_count: 12000$' "$config"
grep -q '^  workers: 64$' "$config"

export PHC2D_ROOT PHC2D_SUBMIT_MODE PHC2D_EXPECTED_COMMIT
export PHC2D_SUBMIT_PYTHON=python3
export PHC2D_ADAPTER=full_cell_band
export PHC2D_CONFIG="$config"
export PHC2D_RUN_DIR="$run_dir"
export PHC2D_PLATFORM_PROFILE="$PHC2D_ROOT/configs/research/hpc_platform_paracloud_m9_v1.yaml"
export PHC2D_CASE_COUNT=12000
export PHC2D_NODE_TIER=4
export PHC2D_CPUS_PER_TASK=64
export PHC2D_MEMORY_GB_PER_TASK=64
export PHC2D_AUTHORIZE_FORMAL=YES

exec bash "$PHC2D_ROOT/jobs/hpc_submit_physics_campaign.sh"
