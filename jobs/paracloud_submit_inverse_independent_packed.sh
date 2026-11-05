#!/usr/bin/env bash
# Submit source-packed independent design jobs with immutable receipts.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail
: "${PHC2D_ROOT:?Set frozen scientific code root}"
: "${PHC2D_EXPECTED_COMMIT:?Set frozen scientific commit}"
: "${PHC2D_CONTROLLER_COMMIT:?Set controller commit}"
: "${PHC2D_GPU_PLATFORM:?Set GPU platform}"
: "${PHC2D_DISPATCH:?Set registered dispatch}"
: "${PHC2D_DISPATCH_SHA256:?Set dispatch identity}"
: "${PHC2D_MATRIX_ROOT:?Set independent matrix root}"
: "${PHC2D_SOURCE_DIRECTORY:?Set frozen sources}"
: "${PHC2D_CHECKPOINT:?Set frozen checkpoint}"
: "${PHC2D_OUTPUT_ROOT:?Set unique output root}"
: "${PHC2D_RECEIPT_ROOT:?Set receipt root}"
: "${PHC2D_WORKER:?Set packed worker}"
: "${PHC2D_WORKER_SHA256:?Set packed worker hash}"
: "${PHC2D_VARIANT:?Set two or four}"
[[ "$PHC2D_GPU_PLATFORM" == n56 ]] || { echo 'unsupported GPU platform' >&2; exit 2; }
case "$PHC2D_VARIANT" in two) wall=02:00:00 ;; four|feedback) wall=01:00:00 ;; *) exit 2 ;; esac
[[ "$(tr -d '[:space:]' < "$PHC2D_ROOT/DEPLOYED_GIT_COMMIT")" == "$PHC2D_EXPECTED_COMMIT" ]] || exit 2
printf '%s  %s\n' "$PHC2D_DISPATCH_SHA256" "$PHC2D_DISPATCH" | sha256sum -c - >/dev/null
printf '%s  %s\n' "$PHC2D_WORKER_SHA256" "$PHC2D_WORKER" | sha256sum -c - >/dev/null
count="$(python3 - "$PHC2D_DISPATCH" "$PHC2D_VARIANT" "$PHC2D_CONTROLLER_COMMIT" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding='utf8')); ids=[x['id'] for x in p['sources']]
scope = 'independent_feedback_packed_dispatch' if sys.argv[2]=='feedback' else 'independent_gpu_packed_dispatch'
if p['scope']!=scope or p['variant']!=sys.argv[2] or p['controller_commit']!=sys.argv[3] or len(ids)!=5 or len(set(ids))!=len(ids): raise SystemExit('bad dispatch')
print(len(ids))
PY
)"
mkdir -p "$PHC2D_RECEIPT_ROOT"
receipt="$PHC2D_RECEIPT_ROOT/submission_${PHC2D_VARIANT}.txt"
pending="$PHC2D_RECEIPT_ROOT/submission_${PHC2D_VARIANT}.pending"
[[ ! -e "$receipt" ]] || { echo 'receipt already exists' >&2; exit 2; }
( set -o noclobber; printf 'dispatch=%s\n' "$PHC2D_DISPATCH_SHA256" > "$pending" ) || { echo 'reconcile pending submission' >&2; exit 2; }
export PHC2D_ROOT PHC2D_EXPECTED_COMMIT PHC2D_CONTROLLER_COMMIT PHC2D_GPU_PLATFORM PHC2D_DISPATCH PHC2D_DISPATCH_SHA256
export PHC2D_MATRIX_ROOT PHC2D_SOURCE_DIRECTORY PHC2D_CHECKPOINT PHC2D_OUTPUT_ROOT
job_id="$(sbatch --parsable --array="0-$((count-1))%4" -p gpu_4090 --gres=gpu:1 -c 6 -t "$wall" \
  -J "inv016_${PHC2D_VARIANT}" -o "$PHC2D_RECEIPT_ROOT/${PHC2D_VARIANT}_%A_%a.out" \
  -e "$PHC2D_RECEIPT_ROOT/${PHC2D_VARIANT}_%A_%a.err" --export=ALL "$PHC2D_WORKER")"
[[ "$job_id" =~ ^[0-9]+([\;][A-Za-z0-9_-]+)?$ ]] || { echo 'unexpected job ID; reconcile' >&2; exit 2; }
printf 'job_id=%s\nvariant=%s\ncode_commit=%s\ncontroller_commit=%s\ndispatch_sha256=%s\nworker_sha256=%s\n' \
  "$job_id" "$PHC2D_VARIANT" "$PHC2D_EXPECTED_COMMIT" "$PHC2D_CONTROLLER_COMMIT" "$PHC2D_DISPATCH_SHA256" "$PHC2D_WORKER_SHA256" > "$receipt"
rm "$pending"
printf '%s\n' "$job_id"
