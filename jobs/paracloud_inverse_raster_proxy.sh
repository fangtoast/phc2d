#!/usr/bin/env bash
# Compare frozen surrogate inputs from two physical raster protocols.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail
: "${PHC2D_ROOT:?Set isolated deployment root}"
: "${PHC2D_MANIFEST:?Set anchor manifest}"
: "${PHC2D_MANIFEST_SHA256:?Set manifest digest}"
: "${PHC2D_PROTOCOL_A:?Set first protocol}"
: "${PHC2D_PROTOCOL_A_SHA256:?Set first protocol digest}"
: "${PHC2D_PROTOCOL_B:?Set second protocol}"
: "${PHC2D_PROTOCOL_B_SHA256:?Set second protocol digest}"
: "${PHC2D_CHECKPOINT:?Set frozen checkpoint}"
: "${PHC2D_CHECKPOINT_SHA256:?Set checkpoint digest}"
: "${PHC2D_EXPECTED_COMMIT:?Set deployed commit}"
: "${PHC2D_OUTPUT:?Set unique output file}"
if ! type module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
  set +u
  source /etc/profile.d/modules.sh
  set -u
fi
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
[[ "$PHC2D_DEPLOYED_GIT_COMMIT" == "$PHC2D_EXPECTED_COMMIT" ]] || { echo 'deployment mismatch' >&2; exit 2; }
python -m phc2d.inverse.audit_raster_proxy --manifest-path "$PHC2D_MANIFEST" \
  --manifest-sha256 "$PHC2D_MANIFEST_SHA256" --protocol-a-path "$PHC2D_PROTOCOL_A" \
  --protocol-a-sha256 "$PHC2D_PROTOCOL_A_SHA256" --protocol-b-path "$PHC2D_PROTOCOL_B" \
  --protocol-b-sha256 "$PHC2D_PROTOCOL_B_SHA256" --checkpoint "$PHC2D_CHECKPOINT" \
  --checkpoint-sha256 "$PHC2D_CHECKPOINT_SHA256" --device cuda:0 --output "$PHC2D_OUTPUT"
