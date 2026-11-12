#!/usr/bin/env bash
# Run immutable inverse refinement shards on M9.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail
: "${PHC2D_ROOT:?Set isolated deployment root}"
: "${PHC2D_MANIFEST:?Set candidate manifest}"
: "${PHC2D_MANIFEST_SHA256:?Set candidate manifest SHA-256}"
: "${PHC2D_REFINEMENT:?Set refinement configuration}"
: "${PHC2D_REFINEMENT_SHA256:?Set refinement configuration SHA-256}"
: "${PHC2D_OUTPUT_ROOT:?Set unique output directory}"
: "${PHC2D_EXPECTED_COMMIT:?Set deployed source commit}"
: "${PHC2D_CASE_COUNT:?Set case count}"
: "${PHC2D_CASES_PER_SHARD:?Set cases per shard}"
: "${SLURM_ARRAY_TASK_ID:?Set shard index}"
source "$PHC2D_ROOT/jobs/paracloud_m9_common.sh"
[[ "$PHC2D_DEPLOYED_GIT_COMMIT" == "$PHC2D_EXPECTED_COMMIT" ]] || { echo 'deployment mismatch' >&2; exit 2; }
start=$((SLURM_ARRAY_TASK_ID * PHC2D_CASES_PER_SHARD))
end=$((start + PHC2D_CASES_PER_SHARD))
for ((index=start; index<end && index<PHC2D_CASE_COUNT; index++)); do
  python -m phc2d.inverse.audit_refine --manifest "$PHC2D_MANIFEST" \
    --manifest-sha256 "$PHC2D_MANIFEST_SHA256" --refinement "$PHC2D_REFINEMENT" \
    --refinement-sha256 "$PHC2D_REFINEMENT_SHA256" --case-index "$index" \
    --output-root "$PHC2D_OUTPUT_ROOT"
done
