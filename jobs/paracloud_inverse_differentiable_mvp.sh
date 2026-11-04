#!/usr/bin/env bash
# Execute one immutable differentiable inverse-design shard on N40/N56.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail
: "${PHC2D_ROOT:?Set isolated deployment root}"
: "${PHC2D_CONFIG:?Set configuration path}"
: "${PHC2D_CONFIG_SHA256:?Set configuration SHA-256}"
: "${PHC2D_SOURCE:?Set source path}"
: "${PHC2D_CHECKPOINT:?Set checkpoint path}"
: "${PHC2D_ROLE:?Set shard role}"
: "${PHC2D_OUTPUT:?Set unique output path}"
: "${PHC2D_EXPECTED_COMMIT:?Set deployed commit}"
if [[ "${PHC2D_GPU_PLATFORM:-}" == n56 ]]; then
  export PHC2D_ENV="${PHC2D_ENV:-/data/run01/scxl892/envs/phc2d_gpu_cu128_torch271_r1}"
fi
if ! type module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
  set +u
  source /etc/profile.d/modules.sh
  set -u
fi
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
[[ "$PHC2D_DEPLOYED_GIT_COMMIT" == "$PHC2D_EXPECTED_COMMIT" ]] || { echo 'deployment mismatch' >&2; exit 2; }
printf '%s  %s\n' "$PHC2D_CONFIG_SHA256" "$PHC2D_CONFIG" | sha256sum -c -
if [[ "$PHC2D_ROLE" != preflight ]]; then
  : "${PHC2D_PREFLIGHT_GATE:?Set combined preflight gate}"
  : "${PHC2D_PREFLIGHT_GATE_SHA256:?Set preflight gate SHA-256}"
  printf '%s  %s\n' "$PHC2D_PREFLIGHT_GATE_SHA256" "$PHC2D_PREFLIGHT_GATE" | sha256sum -c -
  python -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["status"] == "passed" and x["config_sha256"] == sys.argv[2] and x["code_commit"] == sys.argv[3]' \
    "$PHC2D_PREFLIGHT_GATE" "$PHC2D_CONFIG_SHA256" "$PHC2D_EXPECTED_COMMIT"
fi
python -m phc2d.inverse.differentiable_mvp --config "$PHC2D_CONFIG" --source "$PHC2D_SOURCE" \
  --checkpoint "$PHC2D_CHECKPOINT" --role "$PHC2D_ROLE" --device cuda:0 --output "$PHC2D_OUTPUT"
