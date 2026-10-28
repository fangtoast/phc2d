"""Recompute EXP-FWD-031 test cells and complete three-seed summaries.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
import yaml

EXTERNAL_DIGESTS = {
    "dat007_12000": "badd2081180406d6eaac451d758c8d17259a23e13a2c00198b2c82059c9b85a8",
    "dat008_24000": "8c3721e03603a01dd5a8e5c8f6b183e8c70259a259e18ac2f53483585e04aa5c",
}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def audit_cell(out: Path, row: dict, target: str, expected_sha: str,
               expected_count: int) -> dict:
    """Verify provenance and recompute aggregate metrics from raw rows."""
    result = json.loads((out / "results.json").read_text(encoding="utf-8"))
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    samples = _csv(out / "per_sample_metrics.csv")
    bands = _csv(out / "per_band_metrics.csv")
    source = Path(row["run_dir"])
    source_result = json.loads((source / "results.json").read_text(encoding="utf-8"))
    if (result["status"] != "test_complete" or result["model"] != row["model"] or
            Path(result["source_run"]) != source or result["target_sha256"] != expected_sha or
            result["checkpoint_sha256"] != source_result["checkpoint_sha256"] or
            file_digest(source / "model.pt") != result["checkpoint_sha256"]):
        raise ValueError(f"test/source identity mismatch: {out}")
    if len(samples) != expected_count or len({int(s["dataset_index"]) for s in samples}) != expected_count:
        raise ValueError(f"missing or duplicate test samples: {out}")
    if result["sample_count"] != expected_count or metrics["samples"] != expected_count:
        raise ValueError(f"sample count mismatch: {out}")
    full = np.asarray([float(s["full_matrix_mre_pct"]) for s in samples], dtype=np.float64)
    b2 = np.asarray([float(s["bands_2_10_mre_pct"]) for s in samples], dtype=np.float64)
    per_band = np.asarray([float(b["mre_pct"]) for b in bands], dtype=np.float64)
    if not (np.isfinite(full).all() and np.isfinite(b2).all() and np.isfinite(per_band).all()):
        raise ValueError(f"nonfinite test metrics: {out}")
    if [int(b["band_one_based"]) for b in bands] != list(range(1, len(bands) + 1)):
        raise ValueError(f"per-band order mismatch: {out}")
    checks = ((full.mean(), metrics["bands_1_10_mre_pct"]),
              (b2.mean(), metrics["bands_2_10_mre_pct"]),
              (np.percentile(full, 95), metrics["sample_p95_mre_pct"]),
              (per_band.mean(), metrics["bands_1_10_mre_pct"]),
              (per_band[1:].mean(), metrics["bands_2_10_mre_pct"]))
    if any(not math.isclose(float(actual), float(saved), rel_tol=1e-8, abs_tol=1e-8)
           for actual, saved in checks):
        raise ValueError(f"aggregate metrics disagree with raw rows: {out}")
    by_family = {}
    if target.startswith("dat"):
        for sample in samples:
            by_family.setdefault(sample.get("family"), []).append(float(sample["full_matrix_mre_pct"]))
        expected_families = 6 if target == "dat007_12000" else 12
        if len(by_family) != expected_families or any(len(values) != 2000 for values in by_family.values()):
            raise ValueError(f"new-geometry family evidence incomplete: {out}")
    return dict(case_id=row["case_id"], model=row["model"], seed=int(row["seed"]),
                target=target, sample_count=expected_count,
                source_status=source_result["status"],
                bands_1_10_mre_pct=float(full.mean()), bands_2_10_mre_pct=float(b2.mean()),
                sample_p95_mre_pct=float(np.percentile(full, 95)),
                checkpoint_sha256=result["checkpoint_sha256"],
                target_sha256=result["target_sha256"],
                family_mre_pct={str(name): statistics.fmean(values) for name, values in by_family.items()},
                equal_family_mre_pct=(statistics.fmean(statistics.fmean(values)
                                      for values in by_family.values()) if by_family else None),
                evaluation_source_sha256=result.get("evaluation_source_sha256", "legacy_first_batch"))


def audit_suite(suite: Path, *, require_complete: bool = False) -> dict:
    profile = yaml.safe_load((suite / "config.resolved.yaml").read_text(encoding="utf-8"))
    import h5py

    rows = _csv(suite / "task_manifest.csv")
    if len(rows) != 18 or len({r["case_id"] for r in rows}) != 18:
        raise ValueError("expected 18 unique source checkpoints per server")
    targets = {}
    for name, spec in profile["datasets"].items():
        path = Path(profile["dataset_root"]) / Path(spec["path"]).name
        with h5py.File(path, "r") as handle:
            targets[name] = (spec["sha256"], int(len(handle["idx_test"])))
    targets.update({name: (digest, 12000 if name == "dat007_12000" else 24000)
                    for name, digest in EXTERNAL_DIGESTS.items()})
    cells, missing, failed = [], [], []
    for row in rows:
        for target, (digest, count) in targets.items():
            out = suite / "test" / row["case_id"] / target
            if (out / "results.json").is_file():
                cells.append(audit_cell(out, row, target, digest, count))
            else:
                missing.append(f"{row['case_id']}:{target}")
            claim = suite / "test_claims" / row["case_id"] / target
            if (claim / "failure.json").is_file():
                failed.append(f"{row['case_id']}:{target}")
    by_group = {}
    for cell in cells:
        domain = next(name for name in profile["domains"] if cell["case_id"].startswith(name + "_"))
        by_group.setdefault((domain, cell["model"], cell["target"]), []).append(cell)
    summaries = []
    for (domain, model, target), group in sorted(by_group.items()):
        if {cell["seed"] for cell in group} != {0, 1, 2} or len(group) != 3:
            continue
        values = [cell["bands_1_10_mre_pct"] for cell in group]
        summaries.append(dict(source_domain=domain, model=model, target=target,
                              role="native_test" if domain == target else
                                   "new_geometry_transfer" if target.startswith("dat") else "cross_domain_test",
                              seeds=[0, 1, 2], bands_1_10_mre_mean_pct=statistics.fmean(values),
                              bands_1_10_mre_sample_std_pct=statistics.stdev(values)))
    report = dict(status="complete" if len(cells) == 108 and len(summaries) == 36 and not failed else "partial",
                  expected_cells=108, verified_cells=len(cells), missing_cells=missing,
                  failed_cells=failed, expected_three_seed_groups=36,
                  verified_three_seed_groups=len(summaries), cells=cells, three_seed_groups=summaries)
    if require_complete and report["status"] != "complete":
        raise ValueError(f"local test suite incomplete: {len(cells)}/108 cells, {len(summaries)}/36 groups")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = audit_suite(args.suite, require_complete=args.require_complete)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "verified_cells", "verified_three_seed_groups")}), flush=True)


if __name__ == "__main__":
    main()
