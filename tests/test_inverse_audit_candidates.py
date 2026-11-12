"""Tests for deterministic Fourier neighborhood audit candidates.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import pytest

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.audit_candidates import local_fourier_audit_candidates


def test_candidates_are_reproducible_and_preserve_analytic_identity():
    s = generate_unified_fourier_binary('source', master_seed=2, material_fraction_bin=1,
                                        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    kw = dict(coefficient_delta=.1, threshold_delta=.05, max_mode_radius=1, raster_resolution=16)
    rows = local_fourier_audit_candidates(s, **kw)
    assert rows == local_fourier_audit_candidates(s, **kw)
    assert len(rows) == 5 and rows[0]['label'] == 'anchor'
    assert all(r['structure']['metadata']['inverse_design']['source_parameter_hash'] == s.parameter_hash for r in rows)
    assert len({r['parameter_hash'] for r in rows}) == 5


def test_candidate_contract_rejects_missing_mode():
    s = generate_unified_fourier_binary('source', master_seed=2, material_fraction_bin=1,
                                        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    with pytest.raises(ValueError, match='no Fourier mode'):
        local_fourier_audit_candidates(s, coefficient_delta=.1, threshold_delta=.05,
                                       max_mode_radius=.1, raster_resolution=16)
