#!/usr/bin/env bash
# Run the six EXP-DAT-006 real-MPB smoke checks in one exclusive Xingyi allocation.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/HDD_POOL/phc2d}"
source "$PHC2D_ROOT/jobs/xingyi_common.sh"
: "${PHC2D_EXP_DAT_006_CAMPAIGN_ROOT:?Set the EXP-DAT-006 campaign root.}"
: "${PHC2D_EXPECTED_COMMIT:?Set the exact deployed commit.}"
[[ "${PHC2D_AUTHORIZE_EXP_DAT_006_SMOKE:-NO}" == "YES" ]] || {
  echo "EXP-DAT-006 smoke authorization is missing." >&2
  exit 3
}
[[ "$PHC2D_DEPLOYED_GIT_COMMIT" == "$PHC2D_EXPECTED_COMMIT" ]] || {
  echo "Deployed commit mismatch." >&2
  exit 2
}
[[ "${SLURM_CPUS_PER_TASK:-0}" -eq 64 ]] || {
  echo "The packed smoke requires exactly 64 CPUs." >&2
  exit 2
}

available_kb="$(df -Pk "$PHC2D_EXP_DAT_006_CAMPAIGN_ROOT" | awk 'NR==2 {print $4}')"
[[ "${available_kb:-0}" -ge 10485760 ]] || {
  echo "Less than 10 GiB is available for smoke evidence." >&2
  exit 2
}
python - <<'PY'
import multiprocessing
assert multiprocessing.cpu_count() >= 64
print(f"visible_cpu_count={multiprocessing.cpu_count()}")
PY

mkdir -p "$PHC2D_EXP_DAT_006_CAMPAIGN_ROOT/smoke"
pids=()
for lane in binary graded; do
  for cutoff in 8 10 15; do
    config="configs/research/full_cell_${lane}_fourier_k${cutoff}_bz9_10000_xingyi_v1.yaml"
    run_dir="$PHC2D_EXP_DAT_006_CAMPAIGN_ROOT/smoke/${lane}_k${cutoff}"
    python -m phc2d.cli data generate \
      --config "$config" \
      --run-dir "$run_dir" \
      --max-samples 4 \
      --dataset-output "$run_dir/data/smoke.h5" \
      --workers 8 \
      --plan-workers 2 &
    pids+=("$!")
  done
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
[[ "$failed" -eq 0 ]] || {
  echo "At least one EXP-DAT-006 smoke lane failed; no automatic retry is permitted." >&2
  exit 4
}
printf 'status=passed\ndeployed_git_commit=%s\n' "$PHC2D_DEPLOYED_GIT_COMMIT" \
  > "$PHC2D_EXP_DAT_006_CAMPAIGN_ROOT/smoke/preflight.env"
