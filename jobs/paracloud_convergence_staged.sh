#!/bin/bash -l
# Stage an immutable source archive on node-local storage to avoid shared metadata I/O.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail
: "${PHC2D_ROOT:?Set shared deployment}"
: "${PHC2D_ARCHIVE_SHA256:?Set frozen archive hash}"
archive="$PHC2D_ROOT/deployment.tar.gz"
printf '%s  %s\n' "$PHC2D_ARCHIVE_SHA256" "$archive" | sha256sum -c -
task_root=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/phc2d-convergence-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "$task_root"' EXIT
tar -xzf "$archive" -C "$task_root"
cp "$PHC2D_ROOT/DEPLOYED_GIT_COMMIT" "$task_root/DEPLOYED_GIT_COMMIT"
export PHC2D_ROOT="$task_root"
bash -l "$task_root/jobs/paracloud_convergence.sh"
