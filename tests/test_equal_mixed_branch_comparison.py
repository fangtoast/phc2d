"""Complete synthetic matrix and provenance checks for EXP-FWD-033 comparison.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import csv
import json

import pytest

from phc2d.forward.budget_confirmation import file_digest
from phc2d.forward.equal_mixed_branch_comparison import TARGETS, VARIANTS, compare


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture_history(root):
    root.mkdir()
    single, old = [], []
    for domain in ("k5_binary", "k5_graded", "f50_binary", "f50_graded"):
        for target in TARGETS:
            for seed in range(3):
                for variant in ("g_only", "g_only_cap", "l_only_k3", "l_only_k7", "l_only_cap_k3", "l_only_cap_k7"):
                    single.append(dict(source_domain=domain, variant=variant, target=target, seed=seed, mre1=2.0,
                                       target_sha256="sha-"+target, split_sha256="split-"+target, sample_count=10))
                for kernel in (3, 7):
                    old.append(dict(kind="single_full", source=domain, variant="full", kernel=kernel, target=target,
                                    seed=seed, mre1=1.0, target_sha256="sha-"+target,
                                    split_status="verified" if kernel==7 else "historical_unverified"))
    for lane in ("binary", "graded"):
        for target in TARGETS:
            for seed in range(3):
                for kernel in (3, 7):
                    for variant in ("full", "g_only", "g_only_cap", "l_only", "l_only_cap"):
                        old.append(dict(kind="mixed", source=lane, variant=variant, kernel=kernel, target=target,
                                        seed=seed, mre1=1.5, target_sha256="sha-"+target,
                                        split_status="verified" if kernel==7 else "historical_unverified"))
    write_csv(root/"new_test_cells.csv", single)
    write_csv(root/"historical_reference_cells.csv", old)
    (root/"comparison_audit.json").write_text(json.dumps(dict(new_cells=432, outputs={
        name:file_digest(root/name) for name in ("new_test_cells.csv", "historical_reference_cells.csv")})))
    (root/"verification.json").write_text(json.dumps(dict(status="pass")))


def fixture_suite(root, lane):
    root.mkdir()
    rows, cells = [], []
    for variant in VARIANTS:
        for seed in range(3):
            for target in TARGETS:
                case = f"mixed_{lane}_{variant}_s{seed}"
                rows.append(dict(case_id=case, domain="mixed_"+lane, variant=variant, seed=seed, target=target,
                                 checkpoint_sha256=case, dataset_sha256="sha-"+target,
                                 split_sha256="split-"+target, sample_count=10))
                cells.append(dict(case_id=case, target=target, checkpoint_sha256=case, target_sha256="sha-"+target,
                                  sample_count=10, source_status="formal_training_success",
                                  bands_1_10_mre_pct=1.2 if variant.startswith("full_") else 2.2,
                                  bands_2_10_mre_pct=2.0))
    write_csv(root/"test_manifest.csv", rows)
    (root/"test_freeze.json").write_text(json.dumps(dict(authorization_id="approved", manifest_sha256=file_digest(root/"test_manifest.csv"))))
    (root/"suite.json").write_text(json.dumps(dict(experiment_id="EXP-FWD-033")))
    (root/"test_audit.json").write_text(json.dumps(dict(status="complete_pending_independent_review", cell_count=144,
                                                       three_seed_group_count=48, cells=cells)))


def test_complete_mixed_comparison_and_target_split_gate(tmp_path):
    historical, binary, graded = tmp_path/"history", tmp_path/"binary", tmp_path/"graded"
    fixture_history(historical)
    fixture_suite(binary, "binary")
    fixture_suite(graded, "graded")
    output = tmp_path/"comparison"
    result = compare([binary, graded], historical, output)
    assert (result["mixed_cells"], result["seed_groups"], result["paired_branch_cells"], result["reference_rows"]) == (288, 96, 288, 720)
    assert result["reference_groups"] == 240
    with (output/"paired_branch_cells.csv").open(newline="") as f:
        assert all(abs(float(row["branch_minus_full_pct_points"])-1.0)<1e-12 for row in csv.DictReader(f))
    with pytest.raises(FileExistsError):
        compare([binary, graded], historical, output)
    with (binary/"test_manifest.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["split_sha256"] = "changed"
    write_csv(binary/"test_manifest.csv", rows)
    (binary/"test_freeze.json").write_text(json.dumps(dict(authorization_id="approved", manifest_sha256=file_digest(binary/"test_manifest.csv"))))
    with pytest.raises(ValueError, match="target identities"):
        compare([binary, graded], historical, tmp_path/"invalid")
    assert not (tmp_path/"invalid").exists()
