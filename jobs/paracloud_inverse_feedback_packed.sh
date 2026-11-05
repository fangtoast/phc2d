#!/usr/bin/env bash
# Run independent feedback cells for one source under a frozen dispatch.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail
for name in PHC2D_ROOT PHC2D_EXPECTED_COMMIT PHC2D_GPU_PLATFORM PHC2D_DISPATCH PHC2D_DISPATCH_SHA256 PHC2D_MATRIX_ROOT PHC2D_SOURCE_DIRECTORY PHC2D_CHECKPOINT PHC2D_OUTPUT_ROOT PHC2D_FEEDBACK_CONFIG PHC2D_FEEDBACK_CONFIG_SHA256 SLURM_ARRAY_TASK_ID; do
  [[ -n "${!name:-}" ]] || { echo "missing $name" >&2; exit 2; }
done
[[ "$PHC2D_GPU_PLATFORM" == n56 ]] || exit 2
export PHC2D_ENV="${PHC2D_ENV:-/data/run01/scxl892/envs/phc2d_gpu_cu128_torch271_r1}"
if ! type module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
  set +u; source /etc/profile.d/modules.sh; set -u
fi
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
[[ "$PHC2D_DEPLOYED_GIT_COMMIT" == "$PHC2D_EXPECTED_COMMIT" ]] || exit 2
printf '%s  %s\n' "$PHC2D_DISPATCH_SHA256" "$PHC2D_DISPATCH" | sha256sum -c -
printf '%s  %s\n' "$PHC2D_FEEDBACK_CONFIG_SHA256" "$PHC2D_FEEDBACK_CONFIG" | sha256sum -c -
mapfile -t lines < <(python - "$PHC2D_DISPATCH" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding='utf8')); i=int(sys.argv[2]); sources=p['sources']
if p['scope']!='independent_feedback_packed_dispatch' or not 0<=i<len(sources) or p['round'] not in (0,1,2): raise SystemExit('bad dispatch')
print(sources[i]['id'],p['round'])
for cell in p['cells']: print(cell['name'],cell['manifest_sha256'],cell['arms_sha256'])
PY
)
[[ "${#lines[@]}" -gt 1 ]] || exit 2
read -r source_id round <<< "${lines[0]}"
[[ "$source_id" =~ ^[0-9]+$ && "$round" =~ ^[0-2]$ ]] || exit 2
printf -v source_path '%s/source_%04d.json' "$PHC2D_SOURCE_DIRECTORY" "$source_id"
for entry in "${lines[@]:1}"; do
  read -r cell manifest_hash arms_hash <<< "$entry"
  [[ "$cell" =~ ^target_[0-9]+_seed_[0-9]+$ && "$manifest_hash" =~ ^[0-9a-f]{64}$ && "$arms_hash" =~ ^[0-9a-f]{64}$ ]] || exit 2
  manifest="$PHC2D_MATRIX_ROOT/$cell/manifest.json"
  arms="$PHC2D_OUTPUT_ROOT/$cell/round${round}_arms.json"
  printf '%s  %s\n' "$manifest_hash" "$manifest" | sha256sum -c -
  printf '%s  %s\n' "$arms_hash" "$arms" | sha256sum -c -
  python -m phc2d.inverse.differentiable_feedback_gpu --arms "$arms" \
    --base "$PHC2D_MATRIX_ROOT/$cell/base.json" --campaign "$PHC2D_FEEDBACK_CONFIG" \
    --design-manifest "$manifest" --source "$source_path" --source-id "$source_id" \
    --checkpoint "$PHC2D_CHECKPOINT" --round "$round" --device cuda:0 \
    --output "$PHC2D_OUTPUT_ROOT/$cell/round${round}_output"
done
