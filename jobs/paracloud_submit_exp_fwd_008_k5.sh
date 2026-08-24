#!/usr/bin/env bash
# Freeze, dry-run, or explicitly submit EXP-FWD-008 K5 waves on N40.
# Failed submissions are recorded for human diagnosis; no automatic retry is allowed by the project HPC contract.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
mode="${1:-}"; wave="${2:-}"; authorization_id="${3:-}"
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
PHC2D_ENV="${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu118}"
PHC2D_PYTHON="${PHC2D_PYTHON:-$PHC2D_ENV/bin/python}"
campaign="${PHC2D_FWD008_CAMPAIGN_DIR:-$PHC2D_ROOT/runs/$(date +%Y%m%d)/EXP-FWD-008_K5_$(date +%Y%m%d%H%M%S)}"
profile="${PHC2D_FWD008_PROFILE:-configs/research/exp_fwd_008_k5_campaign_profile_v1.yaml}"
cd "$PHC2D_ROOT"
if [[ "$mode" == "freeze" ]]; then
  PYTHONPATH=src "$PHC2D_PYTHON" -m phc2d.forward.full_cell_training_campaign freeze --profile "$profile" --campaign-dir "$campaign" --repository-root "$PHC2D_ROOT"
  exit 0
fi
[[ "$mode" == "dry-run" || "$mode" == "submit" ]] || { echo "usage: $0 freeze|dry-run|submit ADMISSION_A|ADMISSION_B|W1|W2|W3 AUTHORIZATION_ID" >&2; exit 2; }
[[ "$wave" =~ ^(ADMISSION_A|ADMISSION_B|W1|W2|W3)$ ]] || { echo "wave must be ADMISSION_A, ADMISSION_B, W1, W2, or W3" >&2; exit 2; }
[[ -f "$campaign/freeze.json" ]] || { echo "frozen campaign required" >&2; exit 3; }
[[ "$mode" != "submit" || "${PHC2D_AUTHORIZE_EXP_FWD_008:-}" == "YES" ]] || { echo "submit requires PHC2D_AUTHORIZE_EXP_FWD_008=YES" >&2; exit 4; }
[[ "$mode" != "submit" || -n "$authorization_id" ]] || { echo "explicit authorization id is required" >&2; exit 4; }
if [[ "$wave" == "ADMISSION_A" || "$wave" == "ADMISSION_B" ]]; then
  manifest="$campaign/${wave,,}_manifest.csv"
  [[ -f "$manifest" ]] || { echo "frozen admission submission manifest required: $manifest" >&2; exit 3; }
  if [[ "$wave" == "ADMISSION_B" ]]; then
    "$PHC2D_PYTHON" - "$campaign/admission_a_audit.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.is_file() or json.loads(path.read_text(encoding="utf-8")).get("status") != "pass":
    raise SystemExit("admission B requires a passing admission A audit")
PY
  fi
else
  for audit in "$campaign/admission_a_audit.json" "$campaign/admission_b_audit.json"; do
    "$PHC2D_PYTHON" - "$audit" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.is_file() or json.loads(path.read_text(encoding="utf-8")).get("status") != "pass":
    raise SystemExit("formal training requires passing admission A and B audits")
PY
  done
  source_manifest="$campaign/task_manifest.csv"; manifest="$campaign/${wave,,}_manifest.csv"
  "$PHC2D_PYTHON" - "$source_manifest" "$manifest" "$wave" <<'PY'
import csv, sys
source, target, wave = sys.argv[1:]
rows = list(csv.DictReader(open(source, encoding="utf-8")))
rows = [row for row in rows if row["wave"] == wave]
for index, row in enumerate(rows): row["task_index"] = index
with open(target, "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
PY
fi
count="$(( $(wc -l < "$manifest") - 1 ))"; time_limit="12:00:00"; stage="formal"
if [[ "$wave" == ADMISSION_* ]]; then time_limit="00:30:00"; stage="admission"; fi
if [[ "$mode" == "dry-run" ]]; then echo "campaign_dir=$campaign wave=$wave tasks=$count sbatch --array=0-$((count-1))%8 --gres=gpu:1 -c 6 -t $time_limit"; exit 0; fi
set +e
submit_output="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t "$time_limit" --array="0-$((count-1))%8" -J "fwd008_${wave,,}" -o "$campaign/slurm_${wave,,}_%A_%a.out" -e "$campaign/slurm_${wave,,}_%A_%a.err" --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_FWD008_MANIFEST=$manifest,PHC2D_FWD008_STAGE=$stage" jobs/paracloud_full_cell_k5_exp_fwd_008_train.sh 2>&1)"
submit_status=$?
set -e
if [[ "$submit_status" -ne 0 ]]; then
  printf 'wave=%s\nfailed_at=%s\nsbatch_output=%s\nnext_action=human_diagnosis_and_explicit_resubmission\n' "$wave" "$(date -Is)" "$submit_output" > "$campaign/${wave,,}_submission_failure.txt"
  echo "sbatch failed; reason recorded at $campaign/${wave,,}_submission_failure.txt; no automatic retry" >&2
  exit "$submit_status"
fi
printf 'authorization_id=%s\nwave=%s\njob_id=%s\nmanifest=%s\nsubmitted_at=%s\n' "$authorization_id" "$wave" "$submit_output" "$manifest" "$(date -Is)" >> "$campaign/submission.env"
echo "wave=$wave job_id=$submit_output campaign_dir=$campaign"
