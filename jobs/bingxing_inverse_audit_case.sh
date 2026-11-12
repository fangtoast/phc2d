#!/usr/bin/env bash
# Run one frozen inverse audit case inside an M9 allocation.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail
: "${PHC2D_ROOT:?Set isolated deployment root}"
: "${PHC2D_MANIFEST:?Set immutable candidate manifest}"
: "${PHC2D_MANIFEST_SHA256:?Set candidate manifest SHA-256}"
: "${PHC2D_CASE_INDEX:?Set candidate case index}"
: "${PHC2D_OUTPUT_ROOT:?Set unique output directory}"
: "${PHC2D_EXPECTED_COMMIT:?Set deployed source commit}"
source "$PHC2D_ROOT/jobs/paracloud_m9_common.sh"
[[ "$PHC2D_DEPLOYED_GIT_COMMIT" == "$PHC2D_EXPECTED_COMMIT" ]] || { echo 'deployment mismatch' >&2; exit 2; }
python -m phc2d.inverse.audit_mpb --manifest "$PHC2D_MANIFEST" \
  --manifest-sha256 "$PHC2D_MANIFEST_SHA256" --case-index "$PHC2D_CASE_INDEX" \
  --output-root "$PHC2D_OUTPUT_ROOT"
