#!/usr/bin/env bash
# Score frozen periodic-offset candidates on one N40/N56 GPU.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail
: "${PHC2D_ROOT:?Set isolated deployment root}"
: "${PHC2D_MANIFEST:?Set candidate manifest}"
: "${PHC2D_MANIFEST_SHA256:?Set manifest digest}"
: "${PHC2D_PROTOCOL:?Set offset protocol}"
: "${PHC2D_PROTOCOL_SHA256:?Set offset protocol digest}"
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
python -m phc2d.inverse.audit_robust_proxy --manifest "$PHC2D_MANIFEST" \
  --manifest-sha256 "$PHC2D_MANIFEST_SHA256" --protocol "$PHC2D_PROTOCOL" \
  --protocol-sha256 "$PHC2D_PROTOCOL_SHA256" --checkpoint "$PHC2D_CHECKPOINT" \
  --checkpoint-sha256 "$PHC2D_CHECKPOINT_SHA256" --device cuda:0 --output "$PHC2D_OUTPUT"
