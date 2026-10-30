"""Refine a frozen inverse-audit candidate under periodic boundary offsets.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.audit_refine import read_checked_json
from phc2d.inverse.feedback_evaluator import evaluate_feedback_candidate
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig


def run_case(manifest_path: str | Path, *, expected_sha256: str, protocol_path: str | Path,
             expected_protocol_sha256: str, case_index: int, output_root: str | Path, solver=None) -> Path:
    """Evaluate identical rasterized geometry representations across offsets."""
    manifest, digest = read_checked_json(manifest_path, expected_sha256)
    protocol, protocol_digest = read_checked_json(protocol_path, expected_protocol_sha256)
    rows = manifest['rows']
    if not isinstance(case_index, int) or isinstance(case_index, bool) or not 0 <= case_index < len(rows):
        raise ValueError('case index out of bounds')
    source = rows[case_index]
    structure = FullCellStructureSpec.from_mapping(source['structure'])
    if structure.parameter_hash != source['parameter_hash']:
        raise ValueError('candidate parameter identity mismatch')
    offsets = tuple(protocol['offsets'])
    options = protocol['refinement']
    config = FullCellMpbBandSolverConfig(**manifest['solver_config'])
    points = np.asarray(manifest['k_points'], dtype=np.float64)
    output = Path(output_root) / f'case_{case_index:04d}'
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        evaluated = evaluate_feedback_candidate(structure, k_points=points, config=config,
            lower_band=manifest['lower_band'], refinement=options, offsets=offsets,
            raster_resolution=protocol['raster_resolution'], solver=solver)
        observations = [{**trace, 'arrays': {key: value.tolist() for key, value in trace['arrays'].items()}}
                        for trace in evaluated['traces']]
        aggregate = {**evaluated['observation'], 'status': evaluated['status']}
        payload = {'case_index': case_index, 'source_index': source['source_index'], 'label': source['label'],
                   'parameter_hash': structure.parameter_hash, 'manifest_sha256': digest,
                   'protocol_sha256': protocol_digest, 'elapsed_s': time.monotonic() - started,
                   'observations': observations, 'aggregate': aggregate,
                   'evidence_scope': 'exploratory_finite_offsets_and_local_refinement'}
        temp = output / 'results.json.tmp'
        temp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf8')
        temp.replace(output / 'results.json')
    except Exception as exc:
        (output / 'failure.json').write_text(json.dumps({'case_index': case_index, 'manifest_sha256': digest,
            'protocol_sha256': protocol_digest, 'error_type': type(exc).__name__, 'message': str(exc),
            'elapsed_s': time.monotonic() - started}, indent=2), encoding='utf8')
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description='Refine one candidate under periodic offsets')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--case-index', required=True, type=int)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    print(run_case(args.manifest, expected_sha256=args.manifest_sha256, protocol_path=args.protocol,
                   expected_protocol_sha256=args.protocol_sha256, case_index=args.case_index,
                   output_root=args.output_root))


if __name__ == '__main__':
    main()
