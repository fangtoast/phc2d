#!/usr/bin/env bash
# Stage EXP-FWD-020 datasets from the N40 evidence source into an isolated N56 pool.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

: "${PHC2D_N40_HOST:=paracloud-n40}"
: "${PHC2D_N40_DATA_ROOT:=/HOME/scw6g7a/run/phc2d/data}"
: "${PHC2D_N56_DATA_ROOT:=/data/run01/scxl892/data/exp_fwd_020}"

declare -A expected=(
  [full_cell_binary_fourier_k5_bz9_10000_xingyi_v1.h5]=7d2c6bef128c7ee61e0c5ca0eb9561b95992ccf8e53f325274c4889752ea86b6
  [full_cell_graded_fourier_k5_bz9_10000_xingyi_v1.h5]=eb140534d941cbdb53addca279f00337810124a34fade20a19bfb969e0e6af20
  [full_cell_binary_fourier_bz9_10000_xingyi_v1.h5]=2661eb1e9e3b398a7b69c4083daefa761ea9c26a1081c478b968fd6a951d6b56
  [full_cell_graded_fourier_bz9_10000_xingyi_v1.h5]=96b127b43f2c31fb7591e9634317a7ed35332ceae1077b2277de15058902fd72
  [full_cell_binary_bz9_50000_xingyi_v1.h5]=e2d99e0c59c3bf5dfebc507e756dbcd18c8623f331ad6193832d774df5213f00
  [full_cell_graded_bz9_50000_xingyi_v1.h5]=172786bc4b689d13f5938480fe82c25a4b8d9bad9a7d98bcd7eac778dad38a46
)

mkdir -p "$PHC2D_N56_DATA_ROOT"
for name in "${!expected[@]}"; do
  target="$PHC2D_N56_DATA_ROOT/$name"
  partial="$target.partial"
  source="$PHC2D_N40_DATA_ROOT/$name"
  want="${expected[$name]}"
  got_source="$(ssh "$PHC2D_N40_HOST" "sha256sum '$source'" | awk '{print $1}')"
  [[ "$got_source" == "$want" ]] || { echo "N40 source hash mismatch: $name" >&2; exit 3; }
  if [[ -e "$target" ]]; then
    got_target="$(sha256sum "$target" | awk '{print $1}')"
    [[ "$got_target" == "$want" ]] || { echo "refusing to overwrite mismatched N56 target: $target" >&2; exit 4; }
    continue
  fi
  rsync --partial --append-verify "$PHC2D_N40_HOST:$source" "$partial"
  got_partial="$(sha256sum "$partial" | awk '{print $1}')"
  [[ "$got_partial" == "$want" ]] || { echo "N56 transfer hash mismatch: $name" >&2; exit 5; }
  mv -n "$partial" "$target"
done
