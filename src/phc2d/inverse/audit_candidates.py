"""Deterministic local Fourier candidates for validation-only inverse audits.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.fourier_design import update_fourier_binary


def local_fourier_audit_candidates(
    structure: FullCellStructureSpec, *, coefficient_delta: float,
    threshold_delta: float, max_mode_radius: float, raster_resolution: int,
) -> tuple[dict[str, Any], ...]:
    """Return anchor and four signed perturbations around one Fourier structure.

    The first radial-lowest Fourier mode and its cosine coefficient are used;
    ties follow the source mode order. All candidates retain original analytic
    source provenance. Duplicate raster identities are explicitly marked.
    """
    if structure.generator_family != 'unified_fourier_binary' or structure.material_kind != 'binary':
        raise ValueError('expected unified Fourier binary structure')
    if not np.isfinite([coefficient_delta, threshold_delta, max_mode_radius]).all() or min(coefficient_delta, threshold_delta, max_mode_radius) <= 0 or raster_resolution <= 0:
        raise ValueError('invalid candidate perturbation contract')
    modes = structure.parameters['modes']
    eligible = [(float(np.hypot(mode['mx'], mode['my'])), index) for index, mode in enumerate(modes)
                if 0 < np.hypot(mode['mx'], mode['my']) <= max_mode_radius]
    if not eligible:
        raise ValueError('no Fourier mode within configured radius')
    _, index = min(eligible)
    original = float(modes[index]['cos'])
    threshold = float(structure.parameters['threshold'])
    perturbations = [('anchor', 0., 0.), ('coefficient_minus', -coefficient_delta, 0.),
                     ('coefficient_plus', coefficient_delta, 0.), ('threshold_minus', 0., -threshold_delta),
                     ('threshold_plus', 0., threshold_delta)]
    rows = []
    seen: dict[str, str] = {}
    for label, dc, dt in perturbations:
        candidate = update_fourier_binary(structure, sample_id=f'{structure.sample_id}-{label}',
            coefficients={(index, 'cos'): original + dc}, threshold=threshold + dt,
            max_mode_radius=max_mode_radius)
        grid = np.ascontiguousarray(candidate.rasterize((raster_resolution, raster_resolution)), dtype=np.uint8)
        digest = hashlib.sha256(grid.tobytes()).hexdigest()
        first = seen.setdefault(digest, label)
        rows.append({'label': label, 'mode_index': index, 'coefficient_delta': float(dc),
                     'threshold_delta': float(dt), 'parameter_hash': candidate.parameter_hash,
                     'master_grid_sha256': digest, 'duplicate_of': None if first == label else first,
                     'structure': candidate.to_mapping()})
    return tuple(rows)
