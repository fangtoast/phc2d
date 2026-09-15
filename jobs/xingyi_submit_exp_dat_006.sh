#!/usr/bin/env bash
# Submit one explicitly selected EXP-DAT-006 stage; dry-run is the default.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_ROOT="${PHC2D_ROOT:-$PWD}"
PHC2D_SUBMIT_MODE="${PHC2D_SUBMIT_MODE:-dry-run}"
PHC2D_EXP_DAT_006_STAGE="${PHC2D_EXP_DAT_006_STAGE:-plan}"
PHC2D_EXP_DAT_006_CAMPAIGN_ROOT="${PHC2D_EXP_DAT_006_CAMPAIGN_ROOT:-$PHC2D_ROOT/runs/20260915/EXP-DAT-006_xingyi_v1}"
: "${PHC2D_EXPECTED_COMMIT:?Set the exact deployed Git commit.}"

[[ "$PHC2D_SUBMIT_MODE" == "dry-run" || "$PHC2D_SUBMIT_MODE" == "submit" ]] || {
  echo "PHC2D_SUBMIT_MODE must be dry-run or submit." >&2
  exit 2
}
if [[ "$PHC2D_SUBMIT_MODE" == "submit" ]]; then
  case "$PHC2D_EXP_DAT_006_STAGE" in
    plan|pairing-audit)
      [[ "${PHC2D_AUTHORIZE_EXP_DAT_006_PLAN:-NO}" == "YES" ]] || {
        echo "Plan/audit submission requires PHC2D_AUTHORIZE_EXP_DAT_006_PLAN=YES." >&2
        exit 3
      }
      ;;
    smoke)
      [[ "${PHC2D_AUTHORIZE_EXP_DAT_006_SMOKE:-NO}" == "YES" ]] || {
        echo "Smoke submission requires PHC2D_AUTHORIZE_EXP_DAT_006_SMOKE=YES." >&2
        exit 3
      }
      ;;
    solve|finalize|wave-audit)
      [[ "${PHC2D_AUTHORIZE_EXP_DAT_006_FORMAL:-NO}" == "YES" && -n "${PHC2D_EXP_DAT_006_AUTHORIZATION_ID:-}" ]] || {
        echo "Formal submission requires a fresh authorization flag and ID." >&2
        exit 3
      }
      ;;
  esac
fi

stage_script="$PHC2D_ROOT/jobs/hpc_physics_stage.sh"
common="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_PLATFORM_ID=xingyi,PHC2D_ADAPTER=full_cell_band,PHC2D_SHARD_COUNT=8,PHC2D_EXPECTED_COMMIT=$PHC2D_EXPECTED_COMMIT"
sbatch_base=(-p xyfree -N 1 -n 1 -c 64 --mem=128G --exclusive)
formal_worker_auth=NO
smoke_worker_auth=NO
if [[ "$PHC2D_SUBMIT_MODE" == "submit" ]]; then
  mkdir -p "$PHC2D_EXP_DAT_006_CAMPAIGN_ROOT/slurm"
  formal_worker_auth=YES
  smoke_worker_auth=YES
fi

emit_or_submit() {
  if [[ "$PHC2D_SUBMIT_MODE" == "dry-run" ]]; then
    printf 'DRY-RUN:'
    printf ' %q' sbatch "$@"
    printf '\n'
    LAST_JOB_ID=dry-run
  else
    LAST_JOB_ID="$(sbatch --parsable "$@")"
    printf '%s\n' "$LAST_JOB_ID"
  fi
}

record_submission() {
  [[ "$PHC2D_SUBMIT_MODE" == "submit" ]] || return 0
  ledger="$PHC2D_EXP_DAT_006_CAMPAIGN_ROOT/submission_ledger.tsv"
  if [[ ! -f "$ledger" ]]; then
    printf 'submitted_utc\tstage\tlane\tk\tjob_id\tdeployed_git_commit\tauthorization_id\n' > "$ledger"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$3" "$LAST_JOB_ID" \
    "$PHC2D_EXPECTED_COMMIT" "${PHC2D_EXP_DAT_006_AUTHORIZATION_ID:-not-applicable}" >> "$ledger"
}

config_for() {
  printf '%s/configs/research/full_cell_%s_fourier_k%s_bz9_10000_xingyi_v1.yaml' \
    "$PHC2D_ROOT" "$1" "$2"
}
run_for() {
  printf '%s/k%s/%s' "$PHC2D_EXP_DAT_006_CAMPAIGN_ROOT" "$2" "$1"
}

case "$PHC2D_EXP_DAT_006_STAGE" in
  plan)
    for lane in binary graded; do
      for cutoff in 8 10 15; do
        config="$(config_for "$lane" "$cutoff")"
        run_dir="$(run_for "$lane" "$cutoff")"
        [[ "$PHC2D_SUBMIT_MODE" != "submit" ]] || mkdir -p "$run_dir/slurm"
        export_spec="$common,PHC2D_CONFIG=$config,PHC2D_RUN_DIR=$run_dir,PHC2D_STAGE=plan"
        emit_or_submit "${sbatch_base[@]}" -t 01:00:00 -J "dat006_p_${lane:0:1}${cutoff}" \
          -o "$run_dir/slurm/plan_%j.log" --export="$export_spec" "$stage_script"
        record_submission plan "$lane" "$cutoff"
      done
    done
    ;;
  pairing-audit)
    emit_or_submit "${sbatch_base[@]}" -t 01:00:00 -J dat006_pair_audit \
      -o "$PHC2D_EXP_DAT_006_CAMPAIGN_ROOT/slurm/pairing_audit_%j.log" \
      --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_EXP_DAT_006_CAMPAIGN_ROOT=$PHC2D_EXP_DAT_006_CAMPAIGN_ROOT,PHC2D_EXPECTED_COMMIT=$PHC2D_EXPECTED_COMMIT" \
      "$PHC2D_ROOT/jobs/xingyi_exp_dat_006_pairing_audit.sh"
    record_submission pairing-audit both all
    ;;
  smoke)
    emit_or_submit "${sbatch_base[@]}" -t 02:00:00 -J dat006_smoke \
      -o "$PHC2D_EXP_DAT_006_CAMPAIGN_ROOT/slurm/smoke_%j.log" \
      --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_EXP_DAT_006_CAMPAIGN_ROOT=$PHC2D_EXP_DAT_006_CAMPAIGN_ROOT,PHC2D_EXPECTED_COMMIT=$PHC2D_EXPECTED_COMMIT,PHC2D_AUTHORIZE_EXP_DAT_006_SMOKE=$smoke_worker_auth" \
      "$PHC2D_ROOT/jobs/xingyi_exp_dat_006_smoke.sh"
    record_submission smoke both all
    ;;
  solve)
    : "${PHC2D_EXP_DAT_006_K:?Set K to 8, 10, or 15 for a solve wave.}"
    [[ "$PHC2D_EXP_DAT_006_K" =~ ^(8|10|15)$ ]] || exit 2
    if [[ "$PHC2D_SUBMIT_MODE" == "submit" ]]; then
      python - "$PHC2D_EXP_DAT_006_CAMPAIGN_ROOT/pairing_audit.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file() or json.loads(path.read_text(encoding="utf-8")).get("status") != "passed":
    raise SystemExit("Passing pairing audit evidence is required before solve submission.")
PY
      grep -qx 'status=passed' "$PHC2D_EXP_DAT_006_CAMPAIGN_ROOT/smoke/preflight.env" || {
        echo "Passing six-config real-MPB smoke evidence is required." >&2
        exit 4
      }
      case "$PHC2D_EXP_DAT_006_K" in
        10) prior_k=8 ;;
        15) prior_k=10 ;;
        *) prior_k= ;;
      esac
      if [[ -n "$prior_k" ]]; then
        grep -qx 'status=passed' "$PHC2D_EXP_DAT_006_CAMPAIGN_ROOT/k${prior_k}/wave_audit.env" || {
          echo "K${prior_k} wave audit must pass before K${PHC2D_EXP_DAT_006_K}." >&2
          exit 4
        }
      fi
    fi
    authorization_id="${PHC2D_EXP_DAT_006_AUTHORIZATION_ID:-UNAUTHORIZED_DRY_RUN}"
    for lane in binary graded; do
      config="$(config_for "$lane" "$PHC2D_EXP_DAT_006_K")"
      run_dir="$(run_for "$lane" "$PHC2D_EXP_DAT_006_K")"
      [[ "$PHC2D_SUBMIT_MODE" != "submit" ]] || mkdir -p "$run_dir/slurm"
      export_spec="$common,PHC2D_CONFIG=$config,PHC2D_RUN_DIR=$run_dir,PHC2D_STAGE=solve-shard,PHC2D_AUTHORIZE_FORMAL=$formal_worker_auth,PHC2D_EXP_DAT_006_AUTHORIZATION_ID=$authorization_id"
      emit_or_submit --array=0-7 "${sbatch_base[@]}" -t 06:00:00 \
        -J "dat006_s_${lane:0:1}${PHC2D_EXP_DAT_006_K}" \
        -o "$run_dir/slurm/solve_%A_%a.log" --export="$export_spec" "$stage_script"
      record_submission solve "$lane" "$PHC2D_EXP_DAT_006_K"
    done
    ;;
  finalize)
    : "${PHC2D_EXP_DAT_006_K:?Set K to 8, 10, or 15 for finalizers.}"
    [[ "$PHC2D_EXP_DAT_006_K" =~ ^(8|10|15)$ ]] || exit 2
    authorization_id="${PHC2D_EXP_DAT_006_AUTHORIZATION_ID:-UNAUTHORIZED_DRY_RUN}"
    if [[ "$PHC2D_SUBMIT_MODE" == "submit" ]]; then
      python - "$PHC2D_EXP_DAT_006_CAMPAIGN_ROOT" "$PHC2D_EXP_DAT_006_K" <<'PY'
import sys
from pathlib import Path

from phc2d.hpc import collect_campaign_status

root = Path(sys.argv[1]) / f"k{sys.argv[2]}"
for lane in ("binary", "graded"):
    status = collect_campaign_status(root / lane)
    if status.get("state") != "solver_shards_complete":
        raise SystemExit(f"{lane} shards are not complete: {status.get('state')}")
PY
    fi
    for lane in binary graded; do
      config="$(config_for "$lane" "$PHC2D_EXP_DAT_006_K")"
      run_dir="$(run_for "$lane" "$PHC2D_EXP_DAT_006_K")"
      [[ "$PHC2D_SUBMIT_MODE" != "submit" ]] || mkdir -p "$run_dir/slurm"
      export_spec="$common,PHC2D_CONFIG=$config,PHC2D_RUN_DIR=$run_dir,PHC2D_STAGE=finalize,PHC2D_AUTHORIZE_FORMAL=$formal_worker_auth,PHC2D_EXP_DAT_006_AUTHORIZATION_ID=$authorization_id"
      emit_or_submit "${sbatch_base[@]}" -t 02:00:00 \
        -J "dat006_f_${lane:0:1}${PHC2D_EXP_DAT_006_K}" \
        -o "$run_dir/slurm/finalize_%j.log" --export="$export_spec" "$stage_script"
      record_submission finalize "$lane" "$PHC2D_EXP_DAT_006_K"
    done
    ;;
  wave-audit)
    : "${PHC2D_EXP_DAT_006_K:?Set K to 8, 10, or 15 for the wave audit.}"
    [[ "$PHC2D_EXP_DAT_006_K" =~ ^(8|10|15)$ ]] || exit 2
    emit_or_submit "${sbatch_base[@]}" -t 01:00:00 -J "dat006_a_k${PHC2D_EXP_DAT_006_K}" \
      -o "$PHC2D_EXP_DAT_006_CAMPAIGN_ROOT/k${PHC2D_EXP_DAT_006_K}/wave_audit_%j.log" \
      --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_EXP_DAT_006_CAMPAIGN_ROOT=$PHC2D_EXP_DAT_006_CAMPAIGN_ROOT,PHC2D_EXP_DAT_006_K=$PHC2D_EXP_DAT_006_K,PHC2D_EXPECTED_COMMIT=$PHC2D_EXPECTED_COMMIT" \
      "$PHC2D_ROOT/jobs/xingyi_exp_dat_006_wave_audit.sh"
    record_submission wave-audit both "$PHC2D_EXP_DAT_006_K"
    ;;
  *)
    echo "Stage must be plan, pairing-audit, smoke, solve, finalize, or wave-audit." >&2
    exit 2
    ;;
esac
