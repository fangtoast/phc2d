"""Contract checks for static surrogate candidate selection.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import pytest

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.fourier_design import update_fourier_binary
from phc2d.inverse.trace_selection import lock_static_nominal_candidates


def test_static_selection_deduplicates_grids_and_breaks_ties_by_trace_order():
    source = generate_unified_fourier_binary('anchor', master_seed=1, material_fraction_bin=1,
        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    def make(delta, name):
        return update_fourier_binary(source, sample_id=name, coefficients={},
            threshold=source.parameters['threshold'] + delta, max_mode_radius=1)
    first, second = make(-.2, 'first'), make(.2, 'second')
    def row(index, item, score):
        return {'candidate_index': index, 'status': 'surrogate_only', 'structure': item.to_mapping(),
                'parameter_hash': item.parameter_hash, 'selection_score': score}
    traces = [[row(0, source, 100), row(1, first, .4)], [row(0, first, .9), row(1, second, .4)]]
    selected = lock_static_nominal_candidates(source, traces, raster_resolution=32,
                                               candidates=2, evaluations_per_trace=2)
    assert selected['status'] == 'locked_surrogate_only'
    assert [(r['trace_index'], r['candidate_index']) for r in selected['selected']] == [(0, 1), (1, 1)]
    assert selected['excluded_duplicates'] == [{'trace_index': 0, 'candidate_index': 0},
                                                {'trace_index': 1, 'candidate_index': 0}]
    assert selected['physical_validation'] == 'not_run'
    assert lock_static_nominal_candidates(source, traces, raster_resolution=32, candidates=3,
        evaluations_per_trace=2)['selected'] == []
    traces[1][1]['perturbation_offsets'] = [-.1, 0., .1]
    with pytest.raises(ValueError, match='not nominal'):
        lock_static_nominal_candidates(source, traces, raster_resolution=32, candidates=2, evaluations_per_trace=2)
    del traces[1][1]['perturbation_offsets']
    traces[1][1]['parameter_hash'] = 'tampered'
    with pytest.raises(ValueError, match='parameter identity'):
        lock_static_nominal_candidates(source, traces, raster_resolution=32, candidates=2, evaluations_per_trace=2)
