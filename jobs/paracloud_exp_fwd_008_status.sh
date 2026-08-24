#!/usr/bin/env bash
# Read-only EXP-FWD-008 status check; watch mode samples every 30 minutes.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
campaign="${PHC2D_FWD008_CAMPAIGN_DIR:?Set PHC2D_FWD008_CAMPAIGN_DIR}"
mode="${1:-once}"
root="${PHC2D_ROOT:-$HOME/run/phc2d}"
python_bin="${PHC2D_PYTHON:-$HOME/run/envs/phc2d_gpu_cu118/bin/python}"

check_once() {
  date -Is
  if command -v squeue >/dev/null 2>&1; then squeue -u "${USER:-$(id -un)}" -o '%.18i %.20j %.10T %.10M %.20R' || true; fi
  if command -v sacct >/dev/null 2>&1 && [[ -f "$campaign/submission.env" ]]; then
    job_ids="$(awk -F= '/job_id=|shard_job_id=|finalizer_job_id=/{print $2}' "$campaign/submission.env" | paste -sd, -)"
    [[ -n "$job_ids" ]] && sacct -j "$job_ids" --format=JobID,JobName,State,Elapsed,ExitCode -n -P || true
  fi
  find "$campaign" -name results.json -o -name metrics.json -o -name history.csv -o -name epoch200.pt | wc -l | awk '{print "observed_artifact_files=" $1}'
}

case "$mode" in
  once) check_once ;;
  watch)
    while true; do check_once; sleep 1800; done
    ;;
  *) echo "usage: $0 once|watch" >&2; exit 2 ;;
esac
