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


def build_equal_mixed_matrix(profile: dict, variant_profile: dict) -> dict:
    """Enumerate 48 new mixed-source runs and their 288 planned test cells."""
    if profile.get("experiment_id") != "EXP-FWD-033" or profile.get("status") != "planned":
        raise ValueError("expected the planned EXP-FWD-033 profile")
    if variant_profile.get("experiment_id") != "EXP-FWD-032":
        raise ValueError("expected the EXP-FWD-032 architecture contract")
    variants = variant_profile["variants"]
    names = {"full_k3", "full_k7", "g_only", "g_only_cap", "l_only_k3", "l_only_k7", "l_only_cap_k3", "l_only_cap_k7"}
    lanes, seeds, targets = profile["lanes"], profile["seeds"], profile["test"]["targets"]
    expected_targets = {"k5_binary", "k5_graded", "f50_binary", "f50_graded", "exp_dat_007_12k", "exp_dat_008_24k"}
    if (set(variants) != names or lanes != ["binary", "graded"] or seeds != [0, 1, 2]
            or len(targets) != 6 or set(targets) != expected_targets):
        raise ValueError("mixed matrix identities are incomplete or duplicated")
    data = profile["data"]
    if (data["domains"] != ["k5", "k20"] or data["parent_train_per_domain"] != 8000
            or data["train_per_domain"] != 4000 or data["validation_per_domain"] != 1000
            or data["subset_salt"] != "exp-fwd-021-domain-ranking-v1"):
        raise ValueError("mixed train or validation count changed")
    training = profile["training"]
    expected_training = dict(initialization="from_scratch", train_size=8000, micro_batch_size=64,
                             effective_batch_size=64, optimizer="adamw", learning_rate=0.0005,
                             weight_decay=0.00001, loss="normalized_full_matrix_mse",
                             checkpoint_selection="validation_bands_2_10_mre", max_epochs=1000)
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise ValueError("mixed training contract changed")
    expected_platforms = {"n40": dict(lanes=["binary"], partition="gpu_4090", cpus_per_gpu=6, max_gpus=8),
                          "n56": dict(lanes=["graded"], partition="gpu_5090", cpus_per_gpu=8, max_gpus=8)}
    if profile["platforms"] != expected_platforms:
        raise ValueError("mixed platform allocation changed")
    owner = {}
    for platform, spec in profile["platforms"].items():
        for lane in spec["lanes"]:
            if lane in owner:
                raise ValueError("lane assigned to multiple platforms")
            owner[lane] = platform
    if set(owner) != set(lanes):
        raise ValueError("lane assignment incomplete")
    full_counts = [variants[name]["parameters"] for name in ("full_k3", "full_k7")]
    tolerance = variant_profile["parameter_tolerance"]
    if (any(abs(variants["g_only_cap"]["parameters"] / n - 1) > tolerance for n in full_counts)
            or any(abs(variants[name]["parameters"] / n - 1) > tolerance
                   for name, n in zip(("l_only_cap_k3", "l_only_cap_k7"), full_counts, strict=True))):
        raise ValueError("parameter control outside Full tolerance")
    runs, cells = [], []
    for lane in lanes:
        for variant in variants:
            for seed in seeds:
                case_id = f"mixed_{lane}_{variant}_s{seed}"
                runs.append(dict(case_id=case_id, lane=lane, variant=variant, seed=seed, platform=owner[lane]))
                cells.extend(dict(case_id=case_id, target=target, platform=owner[lane]) for target in targets)
    if len(runs) != 48 or len(cells) != 288:
        raise ValueError("unexpected mixed matrix size")
    return dict(status="provisional_not_authorized", experiment_id=profile["experiment_id"],
                runs=runs, test_cells=cells)


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
