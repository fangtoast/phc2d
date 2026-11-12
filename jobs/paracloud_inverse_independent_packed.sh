#!/usr/bin/env bash
# Run preregistered independent cells for one source on one GPU.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail
: "${PHC2D_ROOT:?Set frozen scientific code root}"
: "${PHC2D_GPU_PLATFORM:?Set platform}"
: "${PHC2D_EXPECTED_COMMIT:?Set frozen scientific commit}"
: "${PHC2D_CONTROLLER_COMMIT:?Set controller commit}"
: "${PHC2D_DISPATCH:?Set registered work list}"
: "${PHC2D_DISPATCH_SHA256:?Set work-list identity}"
: "${PHC2D_MATRIX_ROOT:?Set independent matrix root}"
: "${PHC2D_SOURCE_DIRECTORY:?Set frozen sources}"
: "${PHC2D_CHECKPOINT:?Set frozen checkpoint}"
: "${PHC2D_OUTPUT_ROOT:?Set unique output root}"
: "${SLURM_ARRAY_TASK_ID:?Set source index}"
[[ "$PHC2D_GPU_PLATFORM" == n56 ]] || { echo 'unsupported GPU platform' >&2; exit 2; }
export PHC2D_ENV="${PHC2D_ENV:-/data/run01/scxl892/envs/phc2d_gpu_cu128_torch271_r1}"
if ! type module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
  set +u
  source /etc/profile.d/modules.sh
  set -u
fi
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
[[ "$PHC2D_DEPLOYED_GIT_COMMIT" == "$PHC2D_EXPECTED_COMMIT" ]] || { echo 'scientific code mismatch' >&2; exit 2; }
printf '%s  %s\n' "$PHC2D_DISPATCH_SHA256" "$PHC2D_DISPATCH" | sha256sum -c -
mapfile -t lines < <(python - "$PHC2D_DISPATCH" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding='utf8')); i=int(sys.argv[2]); sources=p['sources']
if p['scope']!='independent_gpu_packed_dispatch' or not 0<=i<len(sources): raise SystemExit('bad dispatch')
print(sources[i]['id'])
for cell in p['cells']: print(cell['name']+' '+cell['manifest_sha256'])
PY
)
[[ "${#lines[@]}" -gt 1 && "${lines[0]}" =~ ^[0-9]+$ ]] || { echo 'invalid source roster' >&2; exit 2; }
source_id="${lines[0]}"
printf -v source_path '%s/source_%04d.json' "$PHC2D_SOURCE_DIRECTORY" "$source_id"
for entry in "${lines[@]:1}"; do
  read -r cell expected <<< "$entry"
  [[ "$cell" =~ ^target_[0-9]+_seed_[0-9]+$ && "$expected" =~ ^[0-9a-f]{64}$ ]] || { echo 'invalid cell identity' >&2; exit 2; }
  manifest="$PHC2D_MATRIX_ROOT/$cell/manifest.json"
  printf '%s  %s\n' "$expected" "$manifest" | sha256sum -c -
  for method in adam de random; do
    python -m phc2d.inverse.differentiable_campaign_runner --manifest "$manifest" \
      --source "$source_path" --source-id "$source_id" --checkpoint "$PHC2D_CHECKPOINT" \
      --method "$method" --device cuda:0 --output "$PHC2D_OUTPUT_ROOT/$cell"
  done
done
