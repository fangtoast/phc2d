#!/usr/bin/env bash
# Freeze, dry-run, or explicitly submit EXP-FWD-007 K5 training waves on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
mode="${1:-}"; wave="${2:-}"; authorization_id="${3:-}"
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
PHC2D_ENV="${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu118}"
PHC2D_PYTHON="${PHC2D_PYTHON:-$PHC2D_ENV/bin/python}"
campaign="${PHC2D_FWD007_CAMPAIGN_DIR:-$PHC2D_ROOT/runs/$(date +%Y%m%d)/EXP-FWD-007_K5_$(date +%Y%m%d%H%M%S)}"
profile="${PHC2D_FWD007_PROFILE:-configs/research/exp_fwd_007_k5_campaign_profile_v1.yaml}"
cd "$PHC2D_ROOT"
if [[ "$mode" == "freeze" ]]; then
  PYTHONPATH=src "$PHC2D_PYTHON" -m phc2d.forward.full_cell_training_campaign freeze --profile "$profile" --campaign-dir "$campaign" --repository-root "$PHC2D_ROOT"; exit 0
fi
[[ "$mode" == "dry-run" || "$mode" == "submit" ]] || { echo "usage: $0 freeze|dry-run|submit ADMISSION|W1|W2|W3 AUTHORIZATION_ID" >&2; exit 2; }
[[ "$wave" =~ ^(ADMISSION|W1|W2|W3)$ ]] || { echo "wave must be ADMISSION, W1, W2, or W3" >&2; exit 2; }
[[ -f "$campaign/freeze.json" ]] || { echo "frozen campaign required" >&2; exit 3; }
[[ "$mode" != "submit" || "${PHC2D_AUTHORIZE_EXP_FWD_007:-}" == "YES" ]] || { echo "submit requires PHC2D_AUTHORIZE_EXP_FWD_007=YES" >&2; exit 4; }
[[ "$mode" != "submit" || -n "$authorization_id" ]] || { echo "explicit authorization id is required" >&2; exit 4; }
source_manifest="$campaign/$([[ "$wave" == "ADMISSION" ]] && echo admission_manifest.csv || echo task_manifest.csv)"
manifest="$campaign/${wave,,}_manifest.csv"
"$PHC2D_PYTHON" - "$source_manifest" "$manifest" "$wave" <<'PY'
import csv, sys
source, target, wave = sys.argv[1:]
rows = list(csv.DictReader(open(source, encoding="utf-8")))
if wave == "W1": rows = [row for row in rows if row["wave"] == "W1"]
elif wave == "W2": rows = [row for row in rows if row["wave"] == "W2"]
elif wave == "W3": rows = [row for row in rows if row["wave"] == "W3"]
for index, row in enumerate(rows): row["task_index"] = index
with open(target, "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
PY
count="$(( $(wc -l < "$manifest") - 1 ))"; time_limit="12:00:00"; stage="formal"
if [[ "$wave" == "ADMISSION" ]]; then time_limit="00:30:00"; stage="admission"; fi
if [[ "$mode" == "dry-run" ]]; then echo "campaign_dir=$campaign wave=$wave tasks=$count sbatch --array=0-$((count-1))%8 --gres=gpu:1 -c 6 -t $time_limit"; exit 0; fi
job_id="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t "$time_limit" --array="0-$((count-1))%8" -J "fwd007_${wave,,}" -o "$campaign/slurm_${wave,,}_%A_%a.out" -e "$campaign/slurm_${wave,,}_%A_%a.err" --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_FWD007_MANIFEST=$manifest,PHC2D_FWD007_STAGE=$stage" jobs/paracloud_full_cell_k5_train.sh)"
printf 'authorization_id=%s\nwave=%s\njob_id=%s\nmanifest=%s\nsubmitted_at=%s\n' "$authorization_id" "$wave" "$job_id" "$manifest" "$(date -Is)" >> "$campaign/submission.env"
echo "wave=$wave job_id=$job_id campaign_dir=$campaign"
