"""Audit approved PCSEL plans without solving or changing parameter points.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from phc2d.data.full_cell_hashing import hash_raster_content
from phc2d.data.pcsel_geometry import build_pcsel_test_plan, certify_pcsel_clearance, evaluate_pcsel


def pcsel_symmetry_group(parameters: Mapping[str, Any]) -> str:
    """Canonicalize analytic primitive sets under translations, D4, and relabeling."""
    variants = []
    gs = parameters["primitives"]
    for flip in [1, -1]:
        for turns in range(4):
            a = turns*np.pi/2
            transform = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.diag([1, flip])
            if gs[0]["kind"] == "polygon":
                vertices = np.asarray(gs[0]["vertices"]) @ transform.T
                signature = sorted(tuple(float(x) for x in np.round(v, 12)) for v in vertices)
                variants.append(json.dumps(signature))
                continue
            centers = np.asarray([g["center"] for g in gs]) @ transform.T
            covariances = []
            for g in gs:
                basis = np.array([[g["cos"], -g["sin"]], [g["sin"], g["cos"]]])
                covariance = transform @ basis @ np.diag([g["a"]**2, g["b"]**2]) @ basis.T @ transform.T
                covariances.append([covariance[0, 0], covariance[0, 1], covariance[1, 1]])
            for origin in centers:
                shifted = np.round((centers-origin) % 1, 12) % 1
                signature = sorted(tuple(float(x) for x in np.round([*c, *v], 12)) for c, v in zip(shifted, covariances))
                variants.append(json.dumps(signature))
    return hashlib.sha256(min(variants).encode()).hexdigest()


def _audit_case(row: dict[str, Any]) -> dict[str, Any]:
    p = row["parameters"]
    grid_axis = (np.arange(256)+.5)/256-.5
    x, y = np.meshgrid(grid_axis, grid_axis)
    material = evaluate_pcsel(x, y, p)
    grid64 = material.reshape(64, 4, 64, 4).mean(axis=(1, 3))
    gap = certify_pcsel_clearance(p)
    area_error = abs(float(np.mean(1-material))-p["air_fraction"])
    return dict(sample_id=row["sample_id"], family=p["shape"],
                area_error_r256=area_error, separation_lower_bound_a=gap,
                parameter_hash=hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest(),
                symmetry_group=pcsel_symmetry_group(p),
                content_hash_r256=hash_raster_content(material, material_kind="binary"),
                content_hash_r64=hash_raster_content(grid64, material_kind="area_fraction"),
                passed=bool(gap > .008 and area_error <= .01 and np.isfinite(material).all()))


def audit_pcsel_plan(output: str | Path, workers: int = 8) -> dict[str, Any]:
    """Write complete area, clearance, input collision and analytic group evidence."""
    start = time.perf_counter()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    cases = build_pcsel_test_plan()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_audit_case, cases, chunksize=25))
    else:
        rows = [_audit_case(r) for r in cases]
    with (out/"geometry_audit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    collisions = {}
    for field in ["parameter_hash", "symmetry_group", "content_hash_r256", "content_hash_r64"]:
        groups: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row["sample_id"])
        collisions[field] = [dict(hash=k, count=len(v), sample_ids=v) for k, v in sorted(groups.items()) if len(v) > 1]
    (out/"geometry_collision_groups.json").write_text(json.dumps(collisions, indent=2), encoding="utf-8")
    summary = dict(status="passed" if all(r["passed"] for r in rows) and not collisions["parameter_hash"] else "failed",
                   case_count=len(rows), family_counts=dict(Counter(r["family"] for r in rows)),
                   failed_count=sum(not r["passed"] for r in rows),
                   max_area_error_r256=max(r["area_error_r256"] for r in rows),
                   minimum_separation_lower_bound_a=min(r["separation_lower_bound_a"] for r in rows),
                   duplicate_group_counts={k: len(v) for k, v in collisions.items()},
                   input_collision_policy="retain_with_distinct_analytic_identity_and_group_evidence",
                   elapsed_s=time.perf_counter()-start, workers=workers,
                   solver_executed=False)
    (out/"geometry_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    report = audit_pcsel_plan(args.output, args.workers)
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report["status"] == "passed" else 1)
