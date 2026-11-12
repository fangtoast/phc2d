"""Compare completed equal-count mixed runs with audited historical evidence.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import csv
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from phc2d.forward.budget_confirmation import file_digest, write_json, write_table

TARGETS = ("k5_binary", "k5_graded", "f50_binary", "f50_graded", "dat007_12000", "dat008_24000")
TARGET_MAP = {"exp_dat_007_12k": "dat007_12000", "exp_dat_008_24k": "dat008_24000"}
VARIANTS = ("full_k3", "full_k7", "g_only", "g_only_cap", "l_only_k3", "l_only_k7", "l_only_cap_k3", "l_only_cap_k7")


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _metric(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite comparison metric")
    return number


def compare(suites: list[Path], historical: Path, output: Path) -> dict:
    """Write a complete 288-cell comparison without changing source evidence."""
    if output.exists():
        raise FileExistsError(output)
    if len(suites) != 2 or len(set(suites)) != 2:
        raise ValueError("two distinct completed suites are required")
    prior = json.loads((historical / "comparison_audit.json").read_text())
    verification = json.loads((historical / "verification.json").read_text())
    if verification.get("status") != "pass" or prior.get("new_cells") != 432:
        raise ValueError("historical EXP-FWD-032 evidence is not audited")
    for name in ("new_test_cells.csv", "historical_reference_cells.csv"):
        if file_digest(historical / name) != prior["outputs"][name]:
            raise ValueError("historical comparison input hash changed")
    single = {(r["source_domain"], r["variant"], r["target"], int(r["seed"])): r
              for r in _rows(historical / "new_test_cells.csv")}
    old = {(r["kind"], r["source"], r["variant"], int(r["kernel"]), r["target"], int(r["seed"])): r
           for r in _rows(historical / "historical_reference_cells.csv")}
    if len(single) != 432 or len(old) != 504:
        raise ValueError("historical reference cells are incomplete or duplicated")
    cells, lanes = {}, set()
    for suite in suites:
        info = json.loads((suite / "suite.json").read_text())
        audit = json.loads((suite / "test_audit.json").read_text())
        freeze = json.loads((suite / "test_freeze.json").read_text())
        if (info.get("experiment_id") != "EXP-FWD-033" or audit.get("status") != "complete_pending_independent_review"
                or audit.get("cell_count") != 144 or audit.get("three_seed_group_count") != 48
                or not freeze.get("authorization_id") or file_digest(suite / "test_manifest.csv") != freeze["manifest_sha256"]):
            raise ValueError("mixed test suite is incomplete or not frozen")
        manifest = _rows(suite / "test_manifest.csv")
        indexed = {(r["case_id"], TARGET_MAP.get(r["target"], r["target"])): r for r in manifest}
        if len(manifest) != 144 or len(indexed) != 144:
            raise ValueError("mixed test identities are incomplete or duplicated")
        for result in audit["cells"]:
            target = TARGET_MAP.get(result["target"], result["target"])
            row = indexed[(result["case_id"], target)]
            lane, variant, seed = row["domain"].removeprefix("mixed_"), row["variant"], int(row["seed"])
            if lane not in {"binary", "graded"} or variant not in VARIANTS or seed not in {0, 1, 2} or target not in TARGETS:
                raise ValueError("unexpected mixed test identity")
            if (result["checkpoint_sha256"] != row["checkpoint_sha256"] or result["target_sha256"] != row["dataset_sha256"]
                    or int(result["sample_count"]) != int(row["sample_count"])
                    or result["source_status"] not in {"formal_training_success", "training_complete_convergence_not_met"}):
                raise ValueError("mixed test result differs from frozen manifest")
            reference = single[(f"k5_{lane}", "g_only", target, seed)]
            if (row["dataset_sha256"] != reference["target_sha256"] or row["split_sha256"] != reference["split_sha256"]
                    or int(row["sample_count"]) != int(reference["sample_count"])):
                raise ValueError("mixed and single-domain target identities differ")
            key = (lane, variant, target, seed)
            if key in cells:
                raise ValueError("duplicate mixed test cell")
            lanes.add(lane)
            cells[key] = dict(lane=lane, variant=variant, target=target, seed=seed,
                              mre1=_metric(result["bands_1_10_mre_pct"]), mre2=_metric(result["bands_2_10_mre_pct"]),
                              checkpoint_sha256=result["checkpoint_sha256"], target_sha256=row["dataset_sha256"],
                              split_sha256=row["split_sha256"], sample_count=int(row["sample_count"]),
                              source_status=result["source_status"], source_suite=str(suite))
    if len(cells) != 288 or lanes != {"binary", "graded"}:
        raise ValueError("288 unique mixed test cells and both lanes are required")

    summaries, paired, references = [], [], []
    for lane in ("binary", "graded"):
        for variant in VARIANTS:
            for target in TARGETS:
                values = [cells[(lane, variant, target, seed)]["mre1"] for seed in range(3)]
                summaries.append(dict(lane=lane, variant=variant, target=target, mean=statistics.mean(values),
                                      sample_sd=statistics.stdev(values), seed_count=3))
                for seed in range(3):
                    value = cells[(lane, variant, target, seed)]["mre1"]
                    kernels = (3, 7) if variant in {"g_only", "g_only_cap"} else (int(variant[-1]),)
                    for kernel in kernels:
                        base_variant = "full" if variant.startswith("full_") else variant
                        if base_variant.startswith("l_only"):
                            base_variant = base_variant.rsplit("_k", 1)[0]
                        mixed16 = old[("mixed", lane, base_variant, kernel, target, seed)]
                        if mixed16["target_sha256"] != cells[(lane, variant, target, seed)]["target_sha256"]:
                            raise ValueError("mixed 16k reference target changed")
                        for domain in (f"k5_{lane}", f"f50_{lane}"):
                            reference = (old[("single_full", domain, "full", kernel, target, seed)]
                                         if variant.startswith("full_") else single[(domain, variant, target, seed)])
                            if reference["target_sha256"] != cells[(lane, variant, target, seed)]["target_sha256"]:
                                raise ValueError("single-domain reference target changed")
                            references.append(dict(lane=lane, variant=variant, kernel=kernel, target=target, seed=seed,
                                reference_source=domain, mixed8k_mre=value, single8k_mre=_metric(reference["mre1"]),
                                mixed16k_mre=_metric(mixed16["mre1"]),
                                single_split_status=reference.get("split_status", "manifest_split_sha256_verified"),
                                mixed16_split_status=mixed16["split_status"],
                                note="descriptive; training sources and validation contracts differ"))
                    if not variant.startswith("full_"):
                        for kernel in kernels:
                            full = cells[(lane, f"full_k{kernel}", target, seed)]["mre1"]
                            paired.append(dict(lane=lane, variant=variant, kernel=kernel, target=target, seed=seed,
                                               full_mre=full, branch_mre=value, branch_minus_full_pct_points=value-full))
    if len(summaries) != 96 or len(paired) != 288 or len(references) != 720:
        raise ValueError("comparison matrix size changed")
    grouped = defaultdict(list)
    for row in paired:
        grouped[(row["lane"], row["variant"], row["kernel"], row["target"])].append(row["branch_minus_full_pct_points"])
    paired_summary = [dict(lane=k[0], variant=k[1], kernel=k[2], target=k[3],
                           mean=statistics.mean(v), sample_sd=statistics.stdev(v), seed_count=len(v))
                      for k, v in sorted(grouped.items())]
    if len(paired_summary) != 96 or any(r["seed_count"] != 3 for r in paired_summary):
        raise ValueError("incomplete paired three-seed groups")
    ref_groups = defaultdict(list)
    for row in references:
        ref_groups[(row["lane"], row["variant"], row["kernel"], row["target"], row["reference_source"])].append(row)
    reference_summary = []
    for key, rows in sorted(ref_groups.items()):
        if {row["seed"] for row in rows} != {0, 1, 2}:
            raise ValueError("incomplete historical reference seed group")
        summary = dict(lane=key[0], variant=key[1], kernel=key[2], target=key[3], reference_source=key[4],
                       single_split_status=rows[0]["single_split_status"], mixed16_split_status=rows[0]["mixed16_split_status"])
        for field in ("mixed8k_mre", "single8k_mre", "mixed16k_mre"):
            values = [row[field] for row in rows]
            summary[field + "_mean"] = statistics.mean(values)
            summary[field + "_sample_sd"] = statistics.stdev(values)
        summary["seed_count"] = len(rows)
        reference_summary.append(summary)
    if len(reference_summary) != 240:
        raise ValueError("unexpected historical reference summary count")
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("mixed8k_cells.csv", list(cells.values())), ("mixed8k_summary.csv", summaries),
                       ("paired_branch_cells.csv", paired), ("paired_branch_summary.csv", paired_summary),
                       ("historical_reference_comparison.csv", references),
                       ("historical_reference_summary.csv", reference_summary)):
        write_table(output / name, rows)
    record = dict(status="complete_pending_independent_review", mixed_cells=288, seed_groups=96,
                  paired_branch_cells=len(paired), reference_rows=len(references), reference_groups=len(reference_summary),
                  historical_audit_sha256=file_digest(historical / "comparison_audit.json"),
                  suite_audit_sha256={str(s): file_digest(s / "test_audit.json") for s in suites},
                  output_sha256={p.name: file_digest(p) for p in output.glob("*.csv")},
                  historical_3x3_split="per-cell historical split hash not verified", physical_validation=False)
    write_json(output / "comparison_audit.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--suite", type=Path, action="append", required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.suite, args.historical, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
