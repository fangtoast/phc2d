"""Reconcile frozen proxy and sampled solver inverse-audit evidence.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.audit_ranking import paired_candidate_ranking
from phc2d.inverse.robust_gap import common_gap


def _validate_sampled_row(expected: Mapping[str, Any], row: Mapping[str, Any], *,
                          manifest: Mapping[str, Any], manifest_sha: str,
                          points: np.ndarray) -> np.ndarray:
    """Check one solver result against the frozen structure and k-grid contract."""
    structure = FullCellStructureSpec.from_mapping(expected['structure'])
    if structure.parameter_hash != expected['parameter_hash'] or row['parameter_hash'] != expected['parameter_hash']:
        raise ValueError('candidate parameter identity mismatch')
    if row['source_index'] != expected['source_index'] or row['label'] != expected['label']:
        raise ValueError('candidate source or label mismatch')
    if row.get('manifest_sha256') != manifest_sha or row.get('geometry_hash') != structure.geometry_hash:
        raise ValueError('manifest or geometry identity mismatch')
    if not np.array_equal(np.asarray(row['k_points'], dtype=np.float64), points):
        raise ValueError('solver k-point identity mismatch')
    metadata, config = row['solver_metadata'], manifest['solver_config']
    for key in ('resolution', 'polarization', 'num_bands', 'mesh_size', 'band_guard_count'):
        if metadata.get(key) != config.get(key):
            raise ValueError(f'solver {key} mismatch')
    bands = np.asarray(row['bands'], dtype=np.float64)
    if bands.shape != (len(points), int(config['num_bands'])):
        raise ValueError('solver band shape mismatch')
    gap = common_gap(bands[None], lower_band=manifest['lower_band'], target_frequency=manifest['target_frequency'])
    if not np.isclose(gap.target_margin, float(row['gap']['target_margin']), rtol=1e-10, atol=1e-10):
        raise ValueError('solver gap does not reproduce from bands')
    return bands


def reconcile_sampled_solver(manifest_path: str | Path, solver: Sequence[Mapping[str, Any]], *,
                             expected_manifest_sha256: str,
                             expected_solver_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate an entire solver-only matrix and return ordered bands for stratification.

    This verifies a finite k grid only. It does not establish band-edge
    convergence, perturbation robustness, or an independent test population.
    """
    raw = Path(manifest_path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_manifest_sha256:
        raise ValueError('manifest SHA-256 mismatch')
    manifest = json.loads(raw)
    points = np.asarray(manifest['k_points'], dtype=np.float64)
    expected = manifest['rows']
    indexed = {row['case_index']: row for row in solver}
    if len(solver) != len(expected) or len(indexed) != len(expected) or set(indexed) != set(range(len(expected))):
        raise ValueError('incomplete or duplicate sampled solver matrix')
    if not expected:
        raise ValueError('sampled solver matrix must not be empty')
    spectra = []
    for i, source in enumerate(expected):
        row = indexed[i]
        if row.get('status') != 'sampled_solver_only':
            raise ValueError('invalid sampled solver status')
        if expected_solver_identity is not None:
            metadata = row['solver_metadata']
            if any(metadata.get(key) != value for key, value in expected_solver_identity.items()):
                raise ValueError('sampled solver identity mismatch')
        spectra.append(_validate_sampled_row(source, row, manifest=manifest, manifest_sha=digest, points=points))
    return {'status': 'sampled_solver_only', 'manifest_sha256': digest,
            'source_indices': np.asarray([row['source_index'] for row in expected], dtype=np.int64),
            'spectra': np.stack(spectra)}


def reconcile_audit(manifest_path: str | Path, proxy: Mapping[str, Any],
                    solver: Sequence[Mapping[str, Any]], *,
                    expected_checkpoint_sha256: str, checkpoint_k_sha256: str) -> dict[str, Any]:
    """Fail closed on missing or inconsistent identities before computing rankings.

    This contract concerns the fixed sampled k grid, not refined or converged
    gaps. The checkpoint k hash must come from its independently verified file.
    """
    raw = Path(manifest_path).read_bytes()
    manifest_sha = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    points = np.asarray(manifest['k_points'], dtype=np.float64)
    k_hash = hashlib.sha256(np.ascontiguousarray(points, dtype=np.float32).tobytes()).hexdigest()
    if k_hash != checkpoint_k_sha256:
        raise ValueError('checkpoint and manifest k-point identities differ')
    if proxy.get('checkpoint_sha256') != expected_checkpoint_sha256 or proxy.get('dataset_sha256') != manifest['dataset_sha256']:
        raise ValueError('proxy checkpoint or dataset identity mismatch')
    if proxy.get('candidate_manifest_sha256') != manifest.get('source_manifest_sha256'):
        raise ValueError('proxy candidate manifest identity mismatch')
    if proxy.get('target_frequency') != manifest['target_frequency'] or proxy.get('lower_band') != manifest['lower_band'] or proxy.get('k_count') != len(points):
        raise ValueError('proxy target or k count differs')
    expected_rows = manifest['rows']
    if len(solver) != len(expected_rows) or len(proxy['rows']) != len(expected_rows):
        raise ValueError('incomplete candidate matrix')
    proxy_keys = {(row['source_index'], row['label']): row for row in proxy['rows']}
    if len(proxy_keys) != len(proxy['rows']):
        raise ValueError('duplicate proxy candidate identity')
    solved = {int(row['case_index']): row for row in solver}
    if len(solved) != len(solver) or set(solved) != set(range(len(expected_rows))):
        raise ValueError('missing, duplicate or unexpected case indices')
    for i, expected in enumerate(expected_rows):
        row = solved[i]
        p = proxy_keys.get((expected['source_index'], expected['label']))
        if p is None or p['parameter_hash'] != expected['parameter_hash'] or p['grid_sha256'] != expected['master_grid_sha256']:
            raise ValueError('proxy structure or raster identity mismatch')
        _validate_sampled_row(expected, row, manifest=manifest, manifest_sha=manifest_sha, points=points)
    ranking = paired_candidate_ranking(proxy['rows'], solver)
    return {'status': 'exploratory_sampled_solver_only', 'manifest_sha256': manifest_sha,
            'checkpoint_sha256': expected_checkpoint_sha256, 'k_points_sha256': k_hash,
            'candidate_count': len(expected_rows), 'source_count': len(ranking), 'ranking': ranking}
