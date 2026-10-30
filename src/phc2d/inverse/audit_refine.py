"""Immutable adaptive refinement of frozen inverse-audit candidates.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.feedback_evaluator import evaluate_feedback_candidate
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig


def read_checked_json(path: str | Path, expected: str) -> tuple[dict[str, Any], str]:
    """Load an immutable JSON object after verifying its declared digest."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected:
        raise ValueError("input SHA-256 mismatch")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("input must be a JSON object")
    return value, digest


def run_case(manifest_path: str | Path, *, expected_sha256: str,
             refinement_path: str | Path, expected_refinement_sha256: str,
             case_index: int, output_root: str | Path, solver=None) -> Path:
    """Evaluate one candidate, retaining finite-grid numerical gates and provenance."""
    manifest, digest = read_checked_json(manifest_path, expected_sha256)
    options, options_digest = read_checked_json(refinement_path, expected_refinement_sha256)
    rows = manifest["rows"]
    if not isinstance(case_index, int) or isinstance(case_index, bool) or not 0 <= case_index < len(rows):
        raise ValueError("case index out of bounds")
    record = rows[case_index]
    structure = FullCellStructureSpec.from_mapping(record["structure"])
    if structure.parameter_hash != record["parameter_hash"]:
        raise ValueError("candidate parameter identity mismatch")
    config = FullCellMpbBandSolverConfig(**manifest["solver_config"])
    points = np.asarray(manifest["k_points"], dtype=np.float64)
    output = Path(output_root) / f"case_{case_index:04d}"
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        evaluated = evaluate_feedback_candidate(structure, k_points=points, config=config,
            lower_band=manifest["lower_band"], refinement=options, solver=solver)
        trace = evaluated['traces'][0]
        results, arrays, history, provenance = [trace['row']], trace['arrays'], trace['history'], trace['provenance']
        payload = {"case_index": case_index, "source_index": record["source_index"],
                   "label": record["label"], "parameter_hash": structure.parameter_hash,
                   "manifest_sha256": digest, "refinement_sha256": options_digest,
                   "elapsed_s": time.monotonic() - start, "rows": results,
                   "solver_work": evaluated['solver_work'],
                   "arrays": {key: value.tolist() for key, value in arrays.items()},
                   "history": history, "provenance": provenance,
                   "evidence_scope": "exploratory_finite_local_refinement"}
        temp = output / "results.json.tmp"
        temp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf8")
        temp.replace(output / "results.json")
    except Exception as exc:
        (output / "failure.json").write_text(json.dumps({"case_index": case_index,
            "manifest_sha256": digest, "refinement_sha256": options_digest,
            "error_type": type(exc).__name__, "message": str(exc),
            "elapsed_s": time.monotonic() - start}, indent=2), encoding="utf8")
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine one frozen inverse-audit candidate")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--refinement", required=True)
    parser.add_argument("--refinement-sha256", required=True)
    parser.add_argument("--case-index", required=True, type=int)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    print(run_case(args.manifest, expected_sha256=args.manifest_sha256,
                   refinement_path=args.refinement, expected_refinement_sha256=args.refinement_sha256,
                   case_index=args.case_index, output_root=args.output_root))


if __name__ == "__main__":
    main()
