"""Merge disjoint spectral metric sidecars against a full frozen inventory.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from phc2d.forward.budget_confirmation import file_digest
from phc2d.forward.spectral_metric_evaluation import (
    METRICS, summarize_comparisons, summarize_seeds, write_csv,
)


def merge_metric_evaluations(manifest_path: Path, runs: list[Path], output: Path) -> dict:
    """Reject overlapping or changed inputs; preserve every uncomputed cell."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cells = manifest["cells"]
    expected = {c["cell_id"]: c for c in cells}
    identities = {(c["model"], c["source_domain"], c["target_domain"], int(c["seed"])) for c in cells}
    if len(expected) != len(cells) or len(identities) != len(cells) or len(cells) != int(manifest.get("expected_cells", len(cells))):
        raise ValueError("duplicate IDs or truncated full inventory")
    selected, metrics, bands, origins = {}, [], [], []

    def read(path):
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    for run in runs:
        audit = json.loads((run / "aggregation_audit.json").read_text(encoding="utf-8"))
        if audit["schema"] != "phc2d.spectral_regression.v1":
            raise ValueError("incompatible metric schema")
        for name in ("cell_audit.csv", "cell_metrics.csv", "per_band_metrics.csv"):
            if file_digest(run / name) != audit["output_files_sha256"][name]:
                raise ValueError(f"changed source table: {run / name}")
        accepted = set()
        for row in read(run / "cell_audit.csv"):
            if row["status"] != "computed":
                continue
            key = row["cell_id"]
            if key in selected:
                raise ValueError(f"overlapping source runs: {key}")
            cell = expected[key]
            for field in ("model", "source_domain", "target_domain", "comparison_role"):
                if row[field] != cell[field]:
                    raise ValueError(f"cell identity mismatch: {key}/{field}")
            if int(row["seed"]) != int(cell["seed"]) or float(row["mre_delta_pct"]) > 1e-5:
                raise ValueError("seed or MRE regression mismatch")
            for field in ("checkpoint_sha256", "dataset_sha256", "split_sha256", "k_points_sha256"):
                if cell.get(field) and row[field] != cell[field]:
                    raise ValueError(f"cell provenance mismatch: {key}/{field}")
            cell_dir = run / "cells" / key
            for name, field in (("metrics.json", "metrics_file_sha256"), ("per_sample_metrics.csv.gz", "sample_file_sha256")):
                if file_digest(cell_dir / name) != row[field]:
                    raise ValueError(f"changed cell artifact: {key}/{name}")
            selected[key] = row, cell_dir
            accepted.add(key)
        for row in read(run / "cell_metrics.csv"):
            if row["cell_id"] not in accepted:
                raise ValueError("metric row has no accepted cell")
            for name in METRICS:
                row[name] = float(row[name]) if row[name] else None
            metrics.append(row)
        bands.extend(read(run / "per_band_metrics.csv"))
        origins.append(dict(run=str(run), audit_sha256=file_digest(run / "aggregation_audit.json")))
    for key, (row, cell_dir) in selected.items():
        groups = [r for r in metrics if r["cell_id"] == key]
        headline = [r for r in groups if r["band_group"] == "all_bands"]
        if len(headline) != 1 or abs(headline[0]["mre_pct"] - float(expected[key]["expected_mre_pct"])) > 1e-5:
            raise ValueError("missing or changed headline metric")
    if any(row["cell_id"] not in selected for row in bands):
        raise ValueError("per-band row has no accepted cell")

    output.mkdir(parents=True, exist_ok=False)
    (output / "cells").mkdir()
    records = []
    for cell in cells:
        key = cell["cell_id"]
        if key in selected:
            row, source = selected[key]
            shutil.copytree(source, output / "cells" / key)
            records.append(row)
        else:
            records.append(dict(**{k: cell[k] for k in ("cell_id", "model", "source_domain", "target_domain", "seed", "comparison_role")},
                                status="blocked", reason=cell.get("blocked_reason", "source metric run unavailable")))
    summary = summarize_seeds(metrics)
    write_csv(output / "cell_audit.csv", records)
    write_csv(output / "cell_metrics.csv", metrics)
    write_csv(output / "per_band_metrics.csv", bands)
    write_csv(output / "three_seed_summary.csv", summary)
    write_csv(output / "comparison_summary.csv", summarize_comparisons(metrics, cells))
    complete_groups = sum(r["status"] == "complete" and r["band_group"] == "all_bands" for r in summary)
    expected_groups = len({(c["model"], c["source_domain"], c["target_domain"]) for c in cells})
    audit = dict(status="complete" if len(selected) == len(cells) and complete_groups == expected_groups else "partial",
                 expected_cells=len(cells), computed_cells=len(selected), blocked_cells=len(cells)-len(selected),
                 expected_seed_groups=expected_groups, complete_seed_groups=complete_groups,
                 manifest_sha256=file_digest(manifest_path), source_runs=origins,
                 schema="phc2d.spectral_regression.v1", evidence_level="surrogate_held_out",
                 aggregator_sha256=file_digest(Path(__file__)),
                 output_files_sha256={name: file_digest(output / name) for name in
                     ("cell_audit.csv", "cell_metrics.csv", "per_band_metrics.csv", "three_seed_summary.csv", "comparison_summary.csv")})
    (output / "config.resolved.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name in ("results.json", "aggregation_audit.json"):
        (output / name).write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(merge_metric_evaluations(args.manifest, args.run, args.output)))


if __name__ == "__main__":
    main()
