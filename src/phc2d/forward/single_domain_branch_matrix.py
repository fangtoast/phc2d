"""Build the provisional single-domain branch-ablation identity matrix.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def build_matrix(profile: dict) -> dict:
    """Enumerate unique runs and test cells without granting execution rights."""
    if profile.get("experiment_id") != "EXP-FWD-032" or profile.get("status") != "planned":
        raise ValueError("expected the planned EXP-FWD-032 profile")
    domains, seeds = profile["domains"], profile["seeds"]
    variants, targets, platforms = profile["variants"], profile["test"]["targets"], profile["platforms"]
    expected_variants = {"full_k3", "full_k7", "g_only", "g_only_cap", "l_only_k3", "l_only_k7", "l_only_cap_k3", "l_only_cap_k7"}
    if (len(domains) != 4 or len(set(domains)) != 4 or seeds != [0, 1, 2]
            or set(variants) != expected_variants or len(targets) != 6 or len(set(targets)) != 6):
        raise ValueError("experiment matrix is incomplete or contains duplicates")
    owner = {}
    for platform, data in platforms.items():
        for domain in data["domains"]:
            if domain in owner:
                raise ValueError("domain assigned to multiple platforms")
            owner[domain] = platform
    if set(owner) != set(domains):
        raise ValueError("domain partition is incomplete")
    full_counts = [variants[name]["parameters"] for name in ("full_k3", "full_k7")]
    for index, name in enumerate(("l_only_cap_k3", "l_only_cap_k7")):
        if abs(variants[name]["parameters"] / full_counts[index] - 1) > profile["parameter_tolerance"]:
            raise ValueError("local capacity control outside tolerance")
    if any(abs(variants["g_only_cap"]["parameters"] / full - 1) > profile["parameter_tolerance"] for full in full_counts):
        raise ValueError("shared global capacity control outside tolerance")
    runs, cells = [], []
    for domain in domains:
        for variant in variants:
            for seed in seeds:
                case_id = f"{domain}_{variant}_s{seed}"
                baseline = variant.startswith("full_")
                runs.append(dict(case_id=case_id, domain=domain, variant=variant, seed=seed,
                                 platform=owner[domain], disposition="candidate_reuse" if baseline else "new_training"))
                if not baseline:
                    cells.extend(dict(case_id=case_id, target=target, platform=owner[domain]) for target in targets)
    return dict(status="provisional_not_authorized", experiment_id=profile["experiment_id"],
                runs=runs, new_test_cells=cells)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    matrix = build_matrix(yaml.safe_load(args.profile.read_text(encoding="utf-8")))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(matrix, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
