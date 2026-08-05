#!/usr/bin/env bash
# Shared runtime setup for phc2d jobs on Parallel Computing Cloud M9.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

: "${SLURM_JOB_ID:?This script must run inside a Slurm allocation.}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required.}"

export PHC2D_CLUSTER_ID="${PHC2D_CLUSTER_ID:-paracloud_bscc_m9}"
PHC2D_ROOT="${PHC2D_ROOT:-/publicfs10/fs10-m9/home/m9s005801/phc2d}"
PHC2D_ENV_ARCHIVE="${PHC2D_ENV_ARCHIVE:-/publicfs10/fs10-m9/home/m9s005801/envs/meep_env_paracloud_m9.tar.gz}"
PHC2D_ENV_ARCHIVE_SHA256="b2fd8f0cf039d27d7f203bab16ae1428b9f8cd470777475c92169e6833611117"
PHC2D_JOB_TMP="${SLURM_TMPDIR:-/tmp}/phc2d_${USER}_${SLURM_JOB_ID}"
PHC2D_ENV="$PHC2D_JOB_TMP/meep_env"

cleanup_paracloud_m9_job_environment() {
  case "$PHC2D_JOB_TMP" in
    "${SLURM_TMPDIR:-/tmp}"/phc2d_"${USER}"_"${SLURM_JOB_ID}")
      rm -rf -- "$PHC2D_JOB_TMP"
      ;;
    *)
      echo "Refusing to clean unexpected job-local path: $PHC2D_JOB_TMP" >&2
      ;;
  esac
}
trap cleanup_paracloud_m9_job_environment EXIT

if [[ ! -f "$PHC2D_ENV_ARCHIVE" ]]; then
  echo "Missing packed environment archive: $PHC2D_ENV_ARCHIVE" >&2
  exit 2
fi
if [[ ! -d "$PHC2D_ROOT/src/phc2d" ]]; then
  echo "Missing phc2d checkout: $PHC2D_ROOT" >&2
  exit 2
fi
if [[ "$(sha256sum "$PHC2D_ENV_ARCHIVE" | awk '{print $1}')" != "$PHC2D_ENV_ARCHIVE_SHA256" ]]; then
  echo "Packed environment archive SHA-256 mismatch." >&2
  exit 2
fi
if [[ -f "$PHC2D_ROOT/DEPLOYED_GIT_COMMIT" ]]; then
  export PHC2D_DEPLOYED_GIT_COMMIT="$(tr -d '[:space:]' < "$PHC2D_ROOT/DEPLOYED_GIT_COMMIT")"
elif git -C "$PHC2D_ROOT" rev-parse HEAD >/dev/null 2>&1; then
  export PHC2D_DEPLOYED_GIT_COMMIT="$(git -C "$PHC2D_ROOT" rev-parse HEAD)"
else
  echo "Missing deployed Git commit provenance." >&2
  exit 2
fi

module purge >/dev/null 2>&1 || true
mkdir -p "$PHC2D_ENV"
tar -xzf "$PHC2D_ENV_ARCHIVE" -C "$PHC2D_ENV"
export PATH="$PHC2D_ENV/bin:$PATH"
"$PHC2D_ENV/bin/conda-unpack"
source "$PHC2D_ENV/bin/activate"
export PYTHONPATH="$PHC2D_ROOT/src"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$PHC2D_ROOT"
python -c "import meep; from meep import mpb; assert meep.__version__ == '1.33.0-beta'; assert mpb.__version__ == '1.12.0'; print('Meep/MPB:', meep.__version__, mpb.__version__)"
