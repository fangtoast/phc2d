"""Isolated solver evaluation for frozen inverse-audit candidate manifests.

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
from typing import Any, Callable

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.physical_screen import evaluate_sampled_candidate
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig, solve_bands_for_full_cell_mpb


def run_case(manifest_path: str | Path, *, expected_sha256: str, case_index: int,
             output_root: str | Path, solver: Callable = solve_bands_for_full_cell_mpb) -> Path:
    """Run one immutable case; a preexisting case directory prohibits reruns."""
    path = Path(manifest_path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError('candidate manifest SHA-256 mismatch')
    manifest = json.loads(raw)
    rows = manifest['rows']
    if not isinstance(case_index, int) or isinstance(case_index, bool) or not 0 <= case_index < len(rows):
        raise ValueError('case index out of bounds')
    record = rows[case_index]
    structure = FullCellStructureSpec.from_mapping(record['structure'])
    if structure.parameter_hash != record['parameter_hash']:
        raise ValueError('candidate parameter identity mismatch')
    points = np.asarray(manifest['k_points'], dtype=np.float64)
    config = FullCellMpbBandSolverConfig(**manifest['solver_config'])
    output = Path(output_root) / f'case_{case_index:04d}'
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        result = evaluate_sampled_candidate(structure, k_points=points, config=config,
                                            lower_band=manifest['lower_band'],
                                            target_frequency=manifest['target_frequency'], solver=solver)
        result.update(case_index=case_index, source_index=record['source_index'], label=record['label'],
                      manifest_sha256=digest, elapsed_s=time.monotonic()-start,
                      evidence_scope='sampled_solver_no_refinement')
        (output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    except Exception as exc:
        (output / 'failure.json').write_text(json.dumps({'case_index': case_index, 'manifest_sha256': digest,
            'error_type': type(exc).__name__, 'message': str(exc), 'elapsed_s': time.monotonic()-start},
            indent=2), encoding='utf8')
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate one frozen inverse candidate with MPB')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--case-index', required=True, type=int)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    print(run_case(args.manifest, expected_sha256=args.manifest_sha256,
                   case_index=args.case_index, output_root=args.output_root))


if __name__ == '__main__':
    main()
