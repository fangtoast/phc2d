"""Contract tests for non-generative Fourier design edits.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import numpy as np
import pytest

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.fourier_design import update_fourier_binary
from phc2d.data.full_cell_spec import FullCellStructureSpec


def structure():
    return generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
                                           config={'fourier_cutoff': 3, 'audit_resolution': 32})


def test_zero_edit_preserves_geometry_and_source():
    s = structure()
    edited = update_fourier_binary(s, sample_id='candidate', coefficients={},
                                   threshold=s.parameters['threshold'], max_mode_radius=1)
    np.testing.assert_array_equal(s.rasterize((32, 32)), edited.rasterize((32, 32)))
    assert FullCellStructureSpec.from_mapping(edited.to_mapping()).parameter_hash == edited.parameter_hash
    assert edited.metadata['inverse_design']['source_parameter_hash'] == s.parameter_hash


def test_edit_updates_serialized_parameter_identity():
    s = structure()
    mode = s.parameters['modes'][0]
    value = mode['cos'] + 0.3
    radius = np.hypot(mode['mx'], mode['my'])
    edited = update_fourier_binary(s, sample_id='candidate', coefficients={(0, 'cos'): value},
                                   threshold=0.1, max_mode_radius=radius)
    assert edited.parameters['modes'][0]['cos'] == value
    assert edited.parameter_vector[edited.parameter_names.index('modes[0].cos')] == value
    assert edited.parameter_hash != s.parameter_hash
    assert s.parameters['modes'][0]['cos'] != value


def test_rejects_out_of_radius():
    s = structure()
    with pytest.raises(ValueError, match='outside'):
        update_fourier_binary(s, sample_id='c', coefficients={(0, 'cos'): 0},
                              threshold=0, max_mode_radius=0.1)
