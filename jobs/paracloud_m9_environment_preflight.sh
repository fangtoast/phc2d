#!/usr/bin/env bash
# Unpack and verify the offline phc2d environment on a Parallel Computing Cloud M9 node.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-/publicfs10/fs10-m9/home/m9s005801/phc2d}"
source "$PHC2D_ROOT/jobs/paracloud_m9_common.sh"

echo "host=$(hostname)"
cat /etc/os-release
ldd --version | head -n 1
python -c "import platform; import meep; from meep import mpb; print(platform.libc_ver()); print(meep.__version__, mpb.__version__)"
