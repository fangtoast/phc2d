# Unpack and verify the offline phc2d environment on a Parallel Computing Cloud M9 node.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

: "${SLURM_JOB_ID:?This script must run inside a Slurm allocation.}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required.}"
PHC2D_ENV="${PHC2D_ENV:-/publicfs10/fs10-m9/home/m9s005801/envs/meep_env_paracloud_m9}"
if [[ ! -x "$PHC2D_ENV/bin/conda-unpack" ]]; then
  echo "Missing conda-unpack in $PHC2D_ENV" >&2
  exit 2
fi

source "$PHC2D_ENV/bin/activate"
"$PHC2D_ENV/bin/conda-unpack"
echo "host=$(hostname)"
cat /etc/os-release
ldd --version | head -n 1
python -c "import platform; import meep; from meep import mpb; print(platform.libc_ver()); print(meep.__version__, mpb.__version__)"
touch "$PHC2D_ENV/.paracloud_m9_unpacked"
