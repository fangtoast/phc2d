"""Evaluate Fourier feedback candidates with adaptive physical band edges.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.perturbation_screen import periodic_perturbation_grids
from phc2d.inverse.refinement_feedback import aggregate_refined_perturbations
from phc2d.replay.gap_refinement import refine_inverse_candidate
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig


def evaluate_feedback_candidate(structure: FullCellStructureSpec, *, k_points: np.ndarray,
                                config: FullCellMpbBandSolverConfig, lower_band: int,
                                refinement: Mapping[str, Any],
                                offsets: tuple[float, ...] | None = None,
                                raster_resolution: int | None = None, solver=None) -> dict[str, Any]:
    """Return a gateable observation plus raw finite-grid refinement records.

    For offset evaluations, all cases including zero use the same pixel-grid
    representation. Callers persist raw records before consuming a budgeted
    feedback observation. This function does not establish a formal gate.
    """
    started = time.monotonic()
    if offsets is None:
        rows, arrays, history, provenance = refine_inverse_candidate(
            structure, k_points=k_points, config=config, lower_band=lower_band,
            solver=solver, **refinement)
        if len(rows) != 1:
            raise ValueError('expected one target band pair')
        observation = {'row': rows[0], 'provenance': provenance}
        traces = [{'offset': None, 'row': rows[0], 'provenance': provenance,
                   'arrays': arrays, 'history': history}]
    else:
        if raster_resolution is None:
            raise ValueError('offset evaluation requires a raster resolution')
        grids = periodic_perturbation_grids(structure, resolution=raster_resolution, offsets=offsets)
        traces = []
        for offset, grid in zip(offsets, grids):
            rows, arrays, history, provenance = refine_inverse_candidate(
                structure, k_points=k_points, config=config, lower_band=lower_band,
                solver=solver, material_grid=grid, **refinement)
            if len(rows) != 1:
                raise ValueError('expected one target band pair')
            traces.append({'offset': offset, 'row': rows[0], 'provenance': provenance,
                           'arrays': arrays, 'history': history})
        aggregate = aggregate_refined_perturbations(traces, expected_offsets=offsets)
        provenance = dict(aggregate['provenance'])
        identities = [(trace['provenance']['initial_solver'].get('solver_backend'),
                       trace['provenance']['initial_solver'].get('solver_version')) for trace in traces]
        if identities and all(backend and version for backend, version in identities):
            if any(identity != identities[0] for identity in identities):
                raise ValueError('offset solver identities differ')
            provenance['solver_identity'] = {'solver_backend': identities[0][0], 'solver_version': identities[0][1]}
        observation = {'row': aggregate['row'], 'provenance': provenance}
    row = observation['row']
    stable = all(row.get(key) is True for key in
                 ('last_refinement_stable', 'resolution_stable', 'adaptive_refinement_converged'))
    work = []
    for trace in traces:
        arrays = trace['arrays']
        primary_count = len(arrays['k_points'])
        verification_count = len(arrays['verification_k_points'])
        work.append({'offset': trace['offset'],
                     'primary_resolution': int(trace['provenance']['primary_resolution']),
                     'primary_k_points': primary_count,
                     'verification_resolution': int(trace['provenance']['verification_resolution']),
                     'verification_k_points': verification_count,
                     'solver_calls': 2 + sum(int(level['new_k_points'] > 0) for level in trace['history'])})
    return {'status': 'numerically_stable' if stable else 'numerical_gate_failed',
            'observation': observation, 'traces': traces, 'elapsed_s': time.monotonic() - started,
            'solver_work': work,
            'evidence_scope': 'finite_local_refinement_only'}
