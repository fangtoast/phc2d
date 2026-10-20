"""Project all-band MRE tables while retaining complete native test evidence.

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
import time
from pathlib import Path


PRIMARY_METRIC = "bands_1_10_mre_pct"


def export_primary_reports(campaign: Path) -> dict[str, object]:
    """Export presentation tables without altering raw metrics or selection."""
    result = json.loads((campaign / "results.json").read_text(encoding="utf-8"))
    if result.get("status") != "fixed300_native_surrogate_evaluation_complete":
        raise ValueError("all native tests must complete before primary reporting")
    hashes = {}
    outputs = {}
    for source, destination, columns in (
        ("native_test_cells.csv", "native_test_primary_cells.csv",
         ["source_domain", "model", "seed", PRIMARY_METRIC]),
        ("native_test_summary.csv", "native_test_primary_summary.csv",
         ["source_domain", "model", "seed_count", PRIMARY_METRIC + "_mean", PRIMARY_METRIC + "_seed_sd"]),
    ):
        path = campaign / source
        hashes[source] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not set(columns).issubset(reader.fieldnames or []):
                raise ValueError("all-band MRE is missing; subset metrics cannot substitute")
            rows = [{column: row[column] for column in columns} for row in reader]
        if not rows or any(not math.isfinite(float(row[column])) for row in rows
                           for column in columns if column.startswith(PRIMARY_METRIC)):
            raise ValueError("primary MRE requires finite nonempty results")
        if source == "native_test_cells.csv" and len(rows) != int(result["test_cell_count"]):
            raise ValueError("primary cell count does not match completed test evidence")
        output = campaign / destination
        temporary = output.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output)
        outputs[destination] = len(rows)
        if hashes[source] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise RuntimeError("raw evidence changed while exporting primary tables")
    record = {"status": "primary_all_band_mre_report_complete", "primary_metric": PRIMARY_METRIC,
              "default_summary": "native_test_primary_summary.csv", "outputs": outputs,
              "raw_source_sha256": hashes, "auxiliary_bands_2_10_retained": True,
              "checkpoint_selection_changed": False, "training_changed": False}
    (campaign / "primary_reporting.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=96)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout_hours * 3600
    while args.wait:
        try:
            ready = json.loads((args.campaign / "results.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            ready = {}
        if ready.get("status") == "fixed300_native_surrogate_evaluation_complete":
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("test results are not complete; no training or test retry was attempted")
        time.sleep(60)
    print(json.dumps(export_primary_reports(args.campaign)))


if __name__ == "__main__":
    main()
