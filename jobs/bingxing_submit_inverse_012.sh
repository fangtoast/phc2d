#!/usr/bin/env bash
# Submit a pre-budgeted set of locked nominal MPB cases on M9.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail
: "${PHC2D_ROOT:?Set isolated deployment root}"
: "${PHC2D_EXPECTED_COMMIT:?Set deployed commit}"
: "${PHC2D_MANIFEST:?Set locked physical manifest}"
: "${PHC2D_PROTOCOL:?Set physical protocol}"
: "${PHC2D_RESOURCES:?Set resource configuration}"
: "${PHC2D_RESOURCE_PLAN:?Set prepared resource plan}"
: "${PHC2D_OUTPUT_ROOT:?Set unique output directory}"
: "${PHC2D_RECEIPT_ROOT:?Set unique submission receipt directory}"
[[ "$(tr -d '[:space:]' < "$PHC2D_ROOT/DEPLOYED_GIT_COMMIT")" == "$PHC2D_EXPECTED_COMMIT" ]] || { echo 'deployment mismatch' >&2; exit 2; }
export PHC2D_MANIFEST_SHA256="$(sha256sum "$PHC2D_MANIFEST" | awk '{print $1}')"
export PHC2D_PROTOCOL_SHA256="$(sha256sum "$PHC2D_PROTOCOL" | awk '{print $1}')"
resource_sha="$(sha256sum "$PHC2D_RESOURCES" | awk '{print $1}')"
details="$(PYTHONPATH="$PHC2D_ROOT/src" python3 - "$PHC2D_MANIFEST" "$PHC2D_PROTOCOL_SHA256" "$PHC2D_RESOURCES" "$resource_sha" "$PHC2D_RESOURCE_PLAN" <<'PY'
import json, sys
from phc2d.inverse.differentiable_resources import plan_physical_resources
from hashlib import sha256
manifest_path, protocol_hash, resources_path, resource_hash, plan_path = sys.argv[1:]
raw = open(manifest_path, 'rb').read()
manifest, resources = json.loads(raw), json.load(open(resources_path))
saved = json.load(open(plan_path))
expected = plan_physical_resources(manifest, resources, manifest_sha256=sha256(raw).hexdigest())
expected['resources_sha256'] = resource_hash
if saved != expected or manifest['protocol_sha256'] != protocol_hash:
    raise SystemExit('resource plan or protocol mismatch')
if not expected['shards']:
    raise SystemExit('no physical cases to submit')
physical = resources['physical_replay']
for item in (expected['case_count'], len(expected['shards']), physical['cases_per_shard'],
             physical['workers_per_shard'], physical['cpus_per_shard'],
             physical['memory_mib_per_shard'], physical['worker_memory_mib'],
             physical['setup_reserve_mib'], physical['wall_seconds_per_shard']):
    print(item)
PY
)"
mapfile -t values <<< "$details"
[[ "${#values[@]}" == 9 ]] || { echo 'invalid resource plan' >&2; exit 2; }
export PHC2D_CASE_COUNT="${values[0]}" PHC2D_CASES_PER_SHARD="${values[2]}"
export PHC2D_WORKERS_PER_SHARD="${values[3]}" PHC2D_WORKER_MEMORY_MIB="${values[6]}"
export PHC2D_SETUP_RESERVE_MIB="${values[7]}" PHC2D_RECORD_WORKER_RESOURCES=1
export PHC2D_ROOT PHC2D_EXPECTED_COMMIT PHC2D_MANIFEST PHC2D_PROTOCOL PHC2D_OUTPUT_ROOT
array="0-$((${values[1]}-1))%${values[1]}"
time_minutes="$(((${values[8]}+59)/60))"
if [[ "${PHC2D_SUBMIT_DRY_RUN:-0}" == 1 ]]; then
  printf 'prepared cases=%s shards=%s cpus=%s memory_mib=%s wall_minutes=%s\n' \
    "$PHC2D_CASE_COUNT" "${values[1]}" "${values[4]}" "${values[5]}" "$time_minutes"
  exit 0
fi
[[ "${PHC2D_SUBMIT_DRY_RUN:-0}" == 0 ]] || { echo 'invalid dry-run flag' >&2; exit 2; }
[[ ! -e "$PHC2D_OUTPUT_ROOT" ]] || { echo 'output already exists' >&2; exit 2; }
mkdir -p "$PHC2D_RECEIPT_ROOT"
pending="$PHC2D_RECEIPT_ROOT/submission.pending"
receipt="$PHC2D_RECEIPT_ROOT/submission.txt"
[[ ! -e "$receipt" ]] || { echo 'submission receipt already exists' >&2; exit 2; }
( set -o noclobber; printf 'manifest_sha256=%s\n' "$PHC2D_MANIFEST_SHA256" > "$pending" ) || { echo 'submission pending; reconcile before retry' >&2; exit 2; }
job_id="$(sbatch --parsable --array="$array" -c "${values[4]}" --mem="${values[5]}M" -t "$time_minutes" \
  -J inv012_mpb -o "$PHC2D_RECEIPT_ROOT/mpb_%A_%a.out" -e "$PHC2D_RECEIPT_ROOT/mpb_%A_%a.err" \
  --export=ALL "$PHC2D_ROOT/jobs/bingxing_inverse_perturb_refine_shard.sh")"
[[ "$job_id" =~ ^[0-9]+([\;][A-Za-z0-9_-]+)?$ ]] || { echo 'unexpected submission result; reconcile before retry' >&2; exit 2; }
printf 'job_id=%s\nmanifest_sha256=%s\nprotocol_sha256=%s\nresources_sha256=%s\ncommit=%s\n' \
  "$job_id" "$PHC2D_MANIFEST_SHA256" "$PHC2D_PROTOCOL_SHA256" "$resource_sha" "$PHC2D_EXPECTED_COMMIT" > "$receipt"
rm "$pending"
printf '%s\n' "$job_id"
