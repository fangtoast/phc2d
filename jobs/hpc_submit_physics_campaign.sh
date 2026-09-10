#!/usr/bin/env bash
# Submit a dependency-linked plan, solve array, and finalizer campaign.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_ROOT="${PHC2D_ROOT:-$PWD}"
: "${PHC2D_ADAPTER:?Set PHC2D_ADAPTER.}"
: "${PHC2D_CONFIG:?Set PHC2D_CONFIG.}"
: "${PHC2D_RUN_DIR:?Set PHC2D_RUN_DIR.}"
: "${PHC2D_PLATFORM_PROFILE:?Set PHC2D_PLATFORM_PROFILE.}"
: "${PHC2D_CASE_COUNT:?Set PHC2D_CASE_COUNT.}"

PHC2D_SUBMIT_MODE="${PHC2D_SUBMIT_MODE:-dry-run}"
PHC2D_NODE_TIER="${PHC2D_NODE_TIER:-}"
if [[ -n "$PHC2D_NODE_TIER" ]]; then
  selection=""
else
  : "${PHC2D_BENCHMARK_EVIDENCE:?Set PHC2D_BENCHMARK_EVIDENCE when PHC2D_NODE_TIER is unset.}"
  selection="$(python -m phc2d.cli hpc benchmark \
    --platform-profile "$PHC2D_PLATFORM_PROFILE" \
    --evidence "$PHC2D_BENCHMARK_EVIDENCE" \
    --case-count "$PHC2D_CASE_COUNT")"
fi

readarray -t profile_values < <(python - "$PHC2D_PLATFORM_PROFILE" "$selection" "$PHC2D_NODE_TIER" <<'PY'
import json
import sys
from pathlib import Path
import yaml

profile = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
selection = json.loads(sys.argv[2]) if sys.argv[2] else {}
node_tier = int(sys.argv[3]) if sys.argv[3] else None
print(profile["platform_id"])
print(profile.get("partition", ""))
print(profile["cores_per_node"])
print(profile["memory_gb_per_node"])
print(node_tier if node_tier is not None else selection["selected_nodes"])
print("true" if node_tier is not None else ("true" if selection["target_met"] else "false"))
PY
)
platform_id="${profile_values[0]}"
partition="${profile_values[1]}"
cores="${profile_values[2]}"
memory_gb="${profile_values[3]}"
nodes="${profile_values[4]}"
target_met="${profile_values[5]}"

if [[ -n "$PHC2D_NODE_TIER" ]]; then
  python - "$PHC2D_PLATFORM_PROFILE" "$nodes" <<'PY'
import sys
from pathlib import Path
import yaml

profile = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
nodes = int(sys.argv[2])
allowed = [int(item) for item in profile.get("allowed_node_tiers", [])]
if nodes not in allowed:
    raise SystemExit(f"PHC2D_NODE_TIER={nodes} is not an allowed node tier: {allowed}")
PY
fi

if [[ "$target_met" != "true" ]]; then
  echo "No admitted resource tier meets the target wall time." >&2
  exit 2
fi
if [[ "$PHC2D_SUBMIT_MODE" == "submit" && "${PHC2D_AUTHORIZE_FORMAL:-NO}" != "YES" ]]; then
  echo "Submission requires PHC2D_AUTHORIZE_FORMAL=YES." >&2
  exit 3
fi

partition_args=()
if [[ -n "$partition" ]]; then
  partition_args=(-p "$partition")
fi
common_export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_PLATFORM_ID=$platform_id,PHC2D_ADAPTER=$PHC2D_ADAPTER,PHC2D_CONFIG=$PHC2D_CONFIG,PHC2D_RUN_DIR=$PHC2D_RUN_DIR,PHC2D_SHARD_COUNT=$nodes,PHC2D_AUTHORIZE_FORMAL=YES"
stage_script="$PHC2D_ROOT/jobs/hpc_physics_stage.sh"

if [[ "$PHC2D_SUBMIT_MODE" == "dry-run" ]]; then
  printf 'PLAN: sbatch %q -N 1 -n 1 -c %q --mem=%qG --export=%q %q\n' \
    "${partition_args[*]}" "$cores" "$memory_gb" "$common_export,PHC2D_STAGE=plan" "$stage_script"
  printf 'SOLVE: sbatch --dependency=afterok:PLAN_JOB --array=0-%d %q -N 1 -n 1 -c %q --mem=%qG --export=%q %q\n' \
    "$((nodes - 1))" "${partition_args[*]}" "$cores" "$memory_gb" "$common_export,PHC2D_STAGE=solve-shard" "$stage_script"
  printf 'FINALIZE: sbatch --dependency=afterok:SOLVE_JOB %q -N 1 -n 1 -c %q --mem=%qG --export=%q %q\n' \
    "${partition_args[*]}" "$cores" "$memory_gb" "$common_export,PHC2D_STAGE=finalize" "$stage_script"
  exit 0
fi
if [[ "$PHC2D_SUBMIT_MODE" != "submit" ]]; then
  echo "PHC2D_SUBMIT_MODE must be dry-run or submit." >&2
  exit 2
fi

mkdir -p "$PHC2D_RUN_DIR/slurm"
plan_job="$(sbatch --parsable "${partition_args[@]}" -N 1 -n 1 -c "$cores" --mem="${memory_gb}G" \
  --job-name=phc2d_plan --output="$PHC2D_RUN_DIR/slurm/plan_%j.log" \
  --export="$common_export,PHC2D_STAGE=plan" "$stage_script")"
solve_job="$(sbatch --parsable --dependency="afterok:$plan_job" --array="0-$((nodes - 1))" \
  "${partition_args[@]}" -N 1 -n 1 -c "$cores" --mem="${memory_gb}G" \
  --job-name=phc2d_solve --output="$PHC2D_RUN_DIR/slurm/solve_%A_%a.log" \
  --export="$common_export,PHC2D_STAGE=solve-shard" "$stage_script")"
finalize_job="$(sbatch --parsable --dependency="afterok:$solve_job" "${partition_args[@]}" \
  -N 1 -n 1 -c "$cores" --mem="${memory_gb}G" --job-name=phc2d_finalize \
  --output="$PHC2D_RUN_DIR/slurm/finalize_%j.log" \
  --export="$common_export,PHC2D_STAGE=finalize" "$stage_script")"

python - "$PHC2D_RUN_DIR/submission_manifest.json" "$plan_job" "$solve_job" "$finalize_job" "$nodes" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "plan_job_id": sys.argv[2],
    "solve_array_job_id": sys.argv[3],
    "finalize_job_id": sys.argv[4],
    "solve_nodes": int(sys.argv[5]),
}, indent=2) + "\n", encoding="utf-8")
PY
printf '%s\n' "$PHC2D_RUN_DIR/submission_manifest.json"
