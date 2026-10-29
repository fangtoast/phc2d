"""Audited sidecar evaluation of existing frozen spectral predictions.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from phc2d.forward.budget_confirmation import file_digest
from phc2d.forward.full_cell_metrics import compute_spectral_regression_metrics


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write heterogeneous records, including blocked cell records."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_cell_arrays(cell: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load NPY, NPZ or audited HDF5 exports after checking file hashes."""
    for name, digest in cell["files_sha256"].items():
        if file_digest(Path(name)) != digest:
            raise ValueError(f"file hash mismatch: {name}")
    if cell.get("h5_path"):
        import h5py
        with h5py.File(cell["h5_path"], "r") as handle:
            prediction, target = np.asarray(handle["prediction"]), np.asarray(handle["label"])
            indices = np.asarray(handle["dataset_index"])
            for attribute, key in (("checkpoint_sha256", "checkpoint_sha256"),
                                   ("target_sha256", "dataset_sha256"),
                                   ("test_idx_sha256", "split_sha256")):
                if handle.attrs[attribute] != cell[key]:
                    raise ValueError(f"HDF5 provenance mismatch: {attribute}")
        required = [cell["h5_path"]]
    elif cell.get("archive_path"):
        with np.load(cell["archive_path"], allow_pickle=False) as archive:
            prediction, target = archive["prediction"], archive["target"]
        indices = np.load(cell["indices_path"], allow_pickle=False)
        required = [cell["archive_path"], cell["indices_path"]]
    else:
        prediction = np.load(cell["prediction_path"], mmap_mode="r", allow_pickle=False)
        target = np.load(cell["target_path"], mmap_mode="r", allow_pickle=False)
        indices = np.load(cell["indices_path"], allow_pickle=False)
        required = [cell["prediction_path"], cell["target_path"], cell["indices_path"]]
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("sample indices must be one-dimensional integers")
    if len(indices) != len(prediction) or len(np.unique(indices)) != len(indices):
        raise ValueError("sample index count or uniqueness mismatch")
    if np.any(indices < 0) or len(indices) != int(cell["sample_count"]):
        raise ValueError("sample count or nonnegative index contract mismatch")
    if hashlib.sha256(indices.tobytes()).hexdigest() != cell["split_sha256"]:
        raise ValueError("sample order/split hash mismatch")
    if any(path not in cell["files_sha256"] for path in required):
        raise ValueError("missing array file hash")
    for key in ("checkpoint_sha256", "dataset_sha256", "split_sha256", "k_points_sha256"):
        value = cell[key]
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"missing or invalid provenance: {key}")
    return prediction, target, indices


METRICS = ("mre_pct", "relative_l2_mean_pct", "relative_l2_median_pct",
           "relative_l2_p95_pct", "relative_l2_global_pct", "mae", "mse",
           "rmse_global", "r2_variance_weighted", "band_order_violation_rate")


def summarize_seeds(rows: list[dict]) -> list[dict]:
    """Only complete seed 0/1/2 groups qualify; P95 is not pooled."""
    grouped = defaultdict(list)
    keys = ("model", "source_domain", "target_domain", "comparison_role", "band_group")
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    output = []
    for identity, values in sorted(grouped.items()):
        record = dict(zip(keys, identity))
        complete = len(values) == 3 and {int(v["seed"]) for v in values} == {0, 1, 2}
        record.update(seed_count=len(values), status="complete" if complete else "blocked_seed_set")
        if complete:
            for metric in METRICS:
                data = [v[metric] for v in values]
                record[metric + "_mean"] = float(np.mean(data)) if None not in data else None
                record[metric + "_seed_sd"] = float(np.std(data, ddof=1)) if None not in data else None
        output.append(record)
    return output


def summarize_comparisons(rows: list[dict], cells: list[dict]) -> list[dict]:
    """Equal comparison-pair means per seed; incomplete scopes stay blocked."""
    def key(cell):
        target = cell["target_domain"]
        return cell["model"], cell["comparison_role"], target if target.startswith("dat") else "manifest_domains"
    expected = defaultdict(set)
    for cell in cells:
        expected[key(cell)].add((cell["source_domain"], cell["target_domain"], int(cell["seed"])))
    observed = defaultdict(list)
    for row in rows:
        if row["band_group"] == "all_bands":
            observed[key(row)].append(row)
    output = []
    for identity, identities in sorted(expected.items()):
        values = observed[identity]
        seen = {(v["source_domain"], v["target_domain"], int(v["seed"])) for v in values}
        pairs = {(i[0], i[1]) for i in identities}
        balanced = all({i[2] for i in identities if i[:2] == pair} == {0, 1, 2} for pair in pairs)
        complete = seen == identities and balanced
        record = dict(zip(("model", "comparison_role", "target_scope"), identity))
        record.update(expected_cells=len(identities), computed_cells=len(values),
                      status="complete" if complete else "blocked_incomplete_scope",
                      weighting="equal_comparison_pairs_within_each_seed")
        if complete:
            for metric in METRICS:
                data = [v[metric] for v in values]
                means = [np.mean([v[metric] for v in values if int(v["seed"]) == s]) for s in (0, 1, 2)] if None not in data else None
                record[metric + "_mean"] = float(np.mean(means)) if means is not None else None
                record[metric + "_seed_sd"] = float(np.std(means, ddof=1)) if means is not None else None
        output.append(record)
    return output


def evaluate_manifest(manifest_path: Path, output: Path) -> dict:
    """Create a new evidence directory; never modify inputs or existing outputs."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cells = manifest["cells"]
    ids = [c["cell_id"] for c in cells]
    identities = [(c["model"], c["source_domain"], c["target_domain"], int(c["seed"])) for c in cells]
    if not cells or len(ids) != len(set(ids)) or len(identities) != len(set(identities)):
        raise ValueError("empty manifest or duplicate cell identity")
    if len(cells) != int(manifest.get("expected_cells", len(cells))):
        raise ValueError("manifest does not cover declared expected_cells")
    group_count = len({identity[:3] for identity in identities})
    if group_count != int(manifest.get("expected_seed_groups", group_count)):
        raise ValueError("manifest does not cover declared expected_seed_groups")
    if any(not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value) for value in ids):
        raise ValueError("cell IDs must be safe path components")
    output.mkdir(parents=True, exist_ok=False)
    (output / "cells").mkdir()
    (output / "run.meta.json").write_text(json.dumps(dict(
        start_utc=datetime.now(timezone.utc).isoformat(), python_executable=sys.executable,
        platform=platform.platform(), evaluator="phc2d.forward.spectral_metric_evaluation"), indent=2), encoding="utf-8")
    (output / "config.resolved.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    records, metrics_rows, band_rows = [], [], []
    for cell in cells:
        record = {k: cell[k] for k in ("cell_id", "model", "source_domain", "target_domain", "seed", "comparison_role")}
        record.update({k: cell[k] for k in ("source_status", "scope") if k in cell})
        try:
            if cell.get("blocked_reason"):
                raise FileNotFoundError(cell["blocked_reason"])
            pred, truth, indices = read_cell_arrays(cell)
            result, sample, per_band = compute_spectral_regression_metrics(pred, truth)
            headline = result["groups"]["all_bands"]
            delta = abs(headline["mre_pct"] - float(cell["expected_mre_pct"]))
            if delta > 1.0e-5:
                raise ValueError(f"MRE regression exceeds 1e-5 percentage points: {delta}")
            if any(v["status"] != "computed" for v in result["groups"].values()):
                raise ValueError("undefined relative L2; acceptance paused")
            for values in result["groups"].values():
                if not np.isclose(values["rmse_global"] ** 2, values["mse"], rtol=1e-12, atol=1e-20):
                    raise ValueError("RMSE/MSE consistency failure")
            cell_out = output / "cells" / cell["cell_id"]
            cell_out.mkdir()
            (cell_out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
            with gzip.open(cell_out / "per_sample_metrics.csv.gz", "wt", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["dataset_index", "band_group", *next(iter(sample.values()))])
                writer.writeheader()
                for group, values in sample.items():
                    for i, index in enumerate(indices):
                        writer.writerow(dict(dataset_index=int(index), band_group=group,
                                             **{k: float(v[i]) for k, v in values.items()}))
            for group, values in result["groups"].items():
                metrics_rows.append(dict(**record, band_group=group, **values))
            band_rows.extend(dict(**record, **band) for band in per_band)
            record.update(status="computed", mre_delta_pct=delta,
                          checkpoint_sha256=cell["checkpoint_sha256"],
                          dataset_sha256=cell["dataset_sha256"], split_sha256=cell["split_sha256"],
                          k_points_sha256=cell["k_points_sha256"],
                          metrics_file_sha256=file_digest(cell_out / "metrics.json"),
                          sample_file_sha256=file_digest(cell_out / "per_sample_metrics.csv.gz"))
        except (OSError, ValueError, KeyError, ImportError) as error:
            record.update(status="blocked", reason=str(error))
        records.append(record)
    summary = summarize_seeds(metrics_rows)
    write_csv(output / "cell_audit.csv", records)
    write_csv(output / "cell_metrics.csv", metrics_rows)
    write_csv(output / "per_band_metrics.csv", band_rows)
    write_csv(output / "three_seed_summary.csv", summary)
    write_csv(output / "comparison_summary.csv", summarize_comparisons(metrics_rows, cells))
    expected_groups = {(c["model"], c["source_domain"], c["target_domain"]) for c in cells}
    completed_groups = {(r["model"], r["source_domain"], r["target_domain"]) for r in summary if r["status"] == "complete" and r["band_group"] == "all_bands"}
    audit = dict(status="complete" if all(r["status"] == "computed" for r in records) and completed_groups == expected_groups else "partial",
                 expected_cells=len(cells), computed_cells=sum(r["status"] == "computed" for r in records),
                 blocked_cells=sum(r["status"] == "blocked" for r in records),
                 expected_seed_groups=len(expected_groups), complete_seed_groups=len(completed_groups),
                 manifest_sha256=file_digest(manifest_path),
                 evaluator_sha256=file_digest(Path(__file__)),
                 metric_source_sha256=file_digest(Path(__file__).with_name("full_cell_metrics.py")),
                 environment=dict(platform=platform.platform(), python=platform.python_version()),
                 output_files_sha256={name: file_digest(output / name) for name in
                     ("cell_audit.csv", "cell_metrics.csv", "per_band_metrics.csv", "three_seed_summary.csv", "comparison_summary.csv")},
                 evidence_level="surrogate_held_out", schema="phc2d.spectral_regression.v1")
    for filename in ("results.json", "aggregation_audit.json"):
        (output / filename).write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output / "run.log").write_text(f"Computed {audit['computed_cells']}/{len(cells)} cells.\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_manifest(args.manifest, args.output)))


if __name__ == "__main__":
    main()
