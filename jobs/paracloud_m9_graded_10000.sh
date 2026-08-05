#!/usr/bin/env bash
# Generate the frozen 10,000-sample graded dataset on Parallel Computing Cloud M9.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paracloud_m9_common.sh"

if [[ "${PHC2D_AUTHORIZE_GRADED:-NO}" != "YES" ]]; then
  echo "Set PHC2D_AUTHORIZE_GRADED=YES only after explicit formal-run authorization." >&2
  exit 3
fi
: "${PHC2D_RUN_DIR:?Set PHC2D_RUN_DIR to the fixed standard run directory.}"
: "${PHC2D_EXPECTED_GIT_COMMIT:?Set PHC2D_EXPECTED_GIT_COMMIT to the deployed commit.}"
: "${PHC2D_PREFLIGHT_RUN_DIR:?Set PHC2D_PREFLIGHT_RUN_DIR to the passed solver-oracle run.}"
PHC2D_WORKERS="${PHC2D_WORKERS:?Set PHC2D_WORKERS from worker_selection.json.}"
PHC2D_ALLOW_RESUME="${PHC2D_ALLOW_RESUME:-0}"
EXPECTED_PLAN_SHA256="5a046251e18bd0964ea8c70f4d52fa3bc48bff1b7e8997d4a3f17855be267ec2"

if [[ "$PHC2D_DEPLOYED_GIT_COMMIT" != "$PHC2D_EXPECTED_GIT_COMMIT" ]]; then
  echo "Deployed Git commit does not match PHC2D_EXPECTED_GIT_COMMIT." >&2
  exit 2
fi
if (( PHC2D_WORKERS < 1 || PHC2D_WORKERS > 256 )); then
  echo "PHC2D_WORKERS must be in [1,256]." >&2
  exit 2
fi
if (( ${SLURM_CPUS_PER_TASK:-0} < PHC2D_WORKERS )); then
  echo "SLURM_CPUS_PER_TASK is smaller than PHC2D_WORKERS." >&2
  exit 2
fi
if [[ ! -f "$PHC2D_RUN_DIR/sample_plan.jsonl" ]]; then
  echo "Frozen sample_plan.jsonl is required in $PHC2D_RUN_DIR" >&2
  exit 2
fi
if [[ "$(wc -l < "$PHC2D_RUN_DIR/sample_plan.jsonl")" -ne 10000 ]]; then
  echo "Frozen sample plan must contain exactly 10000 rows." >&2
  exit 2
fi
if [[ "$(sha256sum "$PHC2D_RUN_DIR/sample_plan.jsonl" | awk '{print $1}')" != "$EXPECTED_PLAN_SHA256" ]]; then
  echo "Frozen sample plan SHA-256 mismatch." >&2
  exit 2
fi
python - "$PHC2D_PREFLIGHT_RUN_DIR/results.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "cluster_solver_preflight_pass_pending_wsl_acceptance":
    raise SystemExit(f"solver preflight has not passed: {payload.get('status')}")
PY
if [[ -d "$PHC2D_RUN_DIR/worker_shards" ]] && find "$PHC2D_RUN_DIR/worker_shards" -type f -print -quit | grep -q .; then
  if [[ "$PHC2D_ALLOW_RESUME" != "1" ]]; then
    echo "worker_shards is not empty; audited recovery requires PHC2D_ALLOW_RESUME=1." >&2
    exit 2
  fi
fi

python -m phc2d.cli data generate \
  --config configs/research/full_cell_graded_bz9_10000_paracloud_m9_v1.yaml \
  --run-dir "$PHC2D_RUN_DIR" \
  --workers "$PHC2D_WORKERS"
