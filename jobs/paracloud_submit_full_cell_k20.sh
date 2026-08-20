#!/usr/bin/env bash
# Freeze or submit the K=20 B/G seed-0 forward campaign on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
mode="${1:-}"
wave="${2:-}"
authorization_id="${3:-}"
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
PHC2D_ENV="${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu118}"
PHC2D_PYTHON="${PHC2D_PYTHON:-$PHC2D_ENV/bin/python}"
PHC2D_CAMPAIGN_DIR="${PHC2D_CAMPAIGN_DIR:-$PHC2D_ROOT/runs/$(date +%Y%m%d)/full_cell_k20_forward_seed0_$(date +%Y%m%d%H%M%S)}"
cd "$PHC2D_ROOT"

if [[ "$mode" == "freeze" ]]; then
  mkdir -p "$(dirname "$PHC2D_CAMPAIGN_DIR")"
  [[ -x "$PHC2D_PYTHON" ]] || { echo "missing Python 3 runtime: $PHC2D_PYTHON" >&2; exit 3; }
  PYTHONPATH=src "$PHC2D_PYTHON" -m phc2d.forward.full_cell_k20_campaign freeze --campaign-dir "$PHC2D_CAMPAIGN_DIR" --repository-root "$PHC2D_ROOT"
  exit 0
fi
if [[ "$mode" != "dry-run" && "$mode" != "submit" ]]; then
  echo "usage: $0 freeze|dry-run|submit ADMISSION|W1|W2|W3 AUTHORIZATION_ID" >&2
  exit 2
fi
if [[ "$wave" != "ADMISSION" && "$wave" != "W1" && "$wave" != "W2" && "$wave" != "W3" ]]; then
  echo "wave must be ADMISSION, W1, W2, or W3" >&2
  exit 2
fi
[[ -f "$PHC2D_CAMPAIGN_DIR/admission_manifest.csv" && -f "$PHC2D_CAMPAIGN_DIR/task_manifest.csv" ]] || { echo "frozen campaign manifests are required" >&2; exit 3; }
if [[ "$mode" == "submit" && "${PHC2D_AUTHORIZE_K20:-}" != "YES" ]]; then
  echo "submit requires PHC2D_AUTHORIZE_K20=YES" >&2
  exit 4
fi
if [[ "$mode" == "submit" ]]; then
  acceptance_lock="${PHC2D_K20_ACCEPTANCE_LOCK:-$PHC2D_CAMPAIGN_DIR/wsl_acceptance.lock.json}"
  [[ -f "$acceptance_lock" ]] || { echo "WSL acceptance lock is required: $acceptance_lock" >&2; exit 5; }
  "$PHC2D_PYTHON" - "$acceptance_lock" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
required = {"binary": "wsl_cross_environment_acceptance_pass", "graded": "wsl_cross_environment_acceptance_pass"}
observed = payload.get("lanes", {})
missing = [lane for lane, status in required.items() if observed.get(lane, {}).get("status") != status]
if missing:
    raise SystemExit("WSL acceptance lock is not passing for: " + ", ".join(missing))
PY
fi

manifest="$PHC2D_CAMPAIGN_DIR/${wave,,}_manifest.csv"
"$PHC2D_PYTHON" - "$PHC2D_CAMPAIGN_DIR/task_manifest.csv" "$PHC2D_CAMPAIGN_DIR/admission_manifest.csv" "$manifest" "$wave" <<'PY'
import csv, sys
source, admission, target, wave = sys.argv[1:]
path = admission if wave == "ADMISSION" else source
rows = list(csv.DictReader(open(path, encoding="utf-8")))
if wave == "W1": rows = rows[:8]
elif wave == "W2": rows = rows[8:16]
elif wave == "W3": rows = rows[16:]
if not rows: raise SystemExit("empty wave manifest")
for index, row in enumerate(rows): row["task_index"] = str(index)
with open(target, "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
PY

task_count="$(($(wc -l < "$manifest") - 1))"
array_limit=8
time_limit="12:00:00"
[[ "$wave" == "ADMISSION" ]] && time_limit="00:30:00"
job_name="fck20_${wave,,}"
if [[ "$mode" == "dry-run" ]]; then
  echo "campaign_dir=$PHC2D_CAMPAIGN_DIR"
  echo "wave=$wave"
  echo "manifest=$manifest"
  echo "tasks=$task_count"
  echo "sbatch -p gpu_4090 --gres=gpu:1 -c 6 -t $time_limit --array=0-$((task_count - 1))%$array_limit -J $job_name"
  exit 0
fi
job_id="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t "$time_limit" --array="0-$((task_count - 1))%$array_limit" -J "$job_name" -o "$PHC2D_CAMPAIGN_DIR/slurm_${wave,,}_%A_%a.out" -e "$PHC2D_CAMPAIGN_DIR/slurm_${wave,,}_%A_%a.err" --export="ALL,PHC2D_K20_MANIFEST=$manifest,PHC2D_K20_ROOT=$PHC2D_ROOT,PHC2D_K20_STAGE=$([[ "$wave" == "ADMISSION" ]] && echo admission || echo formal)" jobs/paracloud_full_cell_k20_train.sh)"
printf 'wave=%s\njob_id=%s\nmanifest=%s\nsubmitted_at=%s\n' "$wave" "$job_id" "$manifest" "$(date -Is)" >> "$PHC2D_CAMPAIGN_DIR/submission.env"
echo "wave=$wave job_id=$job_id campaign_dir=$PHC2D_CAMPAIGN_DIR"
