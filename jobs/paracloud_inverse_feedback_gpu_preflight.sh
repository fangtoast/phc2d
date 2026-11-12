#!/usr/bin/env bash
# One frozen-model robust feedback proposal on N40/N56.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail
: "${PHC2D_ROOT:?Set isolated deployment root}"
: "${PHC2D_MANIFEST:?Set physical candidate manifest}"
: "${PHC2D_MANIFEST_SHA256:?Set candidate manifest digest}"
: "${PHC2D_PROTOCOL:?Set offset protocol}"
: "${PHC2D_PROTOCOL_SHA256:?Set protocol digest}"
: "${PHC2D_ANCHOR:?Set validated anchor solver record}"
: "${PHC2D_ANCHOR_SHA256:?Set anchor record digest}"
: "${PHC2D_CASE_INDEX:?Set anchor case index}"
: "${PHC2D_DESIGN:?Set design contract}"
: "${PHC2D_DESIGN_SHA256:?Set design contract digest}"
: "${PHC2D_CHECKPOINT:?Set frozen checkpoint}"
: "${PHC2D_CHECKPOINT_SHA256:?Set checkpoint digest}"
: "${PHC2D_EXPECTED_COMMIT:?Set deployed commit}"
: "${PHC2D_OUTPUT_ROOT:?Set unique output directory}"
if ! type module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
  set +u
  source /etc/profile.d/modules.sh
  set -u
fi
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
[[ "$PHC2D_DEPLOYED_GIT_COMMIT" == "$PHC2D_EXPECTED_COMMIT" ]] || { echo 'deployment mismatch' >&2; exit 2; }
python -m phc2d.inverse.feedback_gpu_preflight --manifest-path "$PHC2D_MANIFEST" \
  --manifest-sha256 "$PHC2D_MANIFEST_SHA256" --protocol-path "$PHC2D_PROTOCOL" \
  --protocol-sha256 "$PHC2D_PROTOCOL_SHA256" --anchor-path "$PHC2D_ANCHOR" \
  --anchor-sha256 "$PHC2D_ANCHOR_SHA256" --case-index "$PHC2D_CASE_INDEX" \
  --design-path "$PHC2D_DESIGN" --design-sha256 "$PHC2D_DESIGN_SHA256" \
  --checkpoint "$PHC2D_CHECKPOINT" --checkpoint-sha256 "$PHC2D_CHECKPOINT_SHA256" \
  --device cuda:0 --output-root "$PHC2D_OUTPUT_ROOT"
