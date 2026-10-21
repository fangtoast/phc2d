#!/usr/bin/env bash
# Submit the approved 24000-case PCSEL geometry campaign without retries.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail
: "${PHC2D_ROOT:?Set the isolated deployed checkout.}"
: "${PHC2D_RUN_DIR:?Set the frozen run directory.}"
: "${PHC2D_EXPECTED_COMMIT:?Set the deployed source commit.}"
export PHC2D_SUBMIT_MODE="${PHC2D_SUBMIT_MODE:-dry-run}"
export PHC2D_SUBMIT_PYTHON=python3
export PHC2D_ADAPTER=full_cell_band
export PHC2D_CONFIG="$PHC2D_ROOT/configs/research/full_cell_pcsel_exp_dat_008_bingxing_v1.yaml"
export PHC2D_PLATFORM_PROFILE="$PHC2D_ROOT/configs/research/hpc_platform_paracloud_m9_v1.yaml"
export PHC2D_CASE_COUNT=24000
export PHC2D_NODE_TIER=4
export PHC2D_CPUS_PER_TASK=64
export PHC2D_MEMORY_GB_PER_TASK=64
export PHC2D_AUTHORIZE_FORMAL=YES
test ! -e "$PHC2D_RUN_DIR/submission_manifest.json"
python3 - "$PHC2D_RUN_DIR/geometry_audit.json" <<'PY'
import json
import sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text())
assert r["status"] == "passed" and r["case_count"] == 24000
assert r["failed_count"] == 0 and set(r["family_counts"].values()) == {2000}
assert r["duplicate_group_counts"]["parameter_hash"] == 0
PY
exec bash "$PHC2D_ROOT/jobs/hpc_submit_physics_campaign.sh"
