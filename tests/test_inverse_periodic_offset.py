"""Periodic fabrication perturbation geometry tests.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import numpy as np
import pytest

from phc2d.inverse.morphology import periodic_binary_offset


def test_dilation_wraps_across_both_periodic_boundaries():
    grid = np.zeros((8, 8), dtype=np.uint8)
    grid[0, 0] = 1
    result = periodic_binary_offset(grid, offset=1 / 8)
    assert result[0, 7] and result[7, 0] and result[0, 1] and result[1, 0]
    assert not result[7, 7]


def test_translation_equivariance_and_erosion():
    grid = np.ones((8, 8), dtype=np.uint8)
    grid[0, 0] = 0
    result = periodic_binary_offset(grid, offset=-1 / 8)
    assert not result[0, 7] and not result[7, 0]
    shifted = periodic_binary_offset(np.roll(grid, (2, 3), axis=(0, 1)), offset=-1 / 8)
    np.testing.assert_array_equal(shifted, np.roll(result, (2, 3), axis=(0, 1)))


def test_identity_and_single_phase_are_stable():
    grid = np.eye(5, dtype=np.uint8)
    np.testing.assert_array_equal(periodic_binary_offset(grid, offset=0), grid)
    for value in (0, 1):
        g = np.full((5, 5), value)
        for offset in (-1, 1):
            np.testing.assert_array_equal(periodic_binary_offset(g, offset=offset), g)


def test_subpixel_offsets_are_noops_not_resolved_perturbations():
    grid = np.zeros((8, 8), dtype=np.uint8)
    grid[2:6, 2:6] = 1
    for offset in (-0.99 / 8, 0.99 / 8):
        np.testing.assert_array_equal(periodic_binary_offset(grid, offset=offset), grid)
    assert not np.array_equal(periodic_binary_offset(grid, offset=1 / 8), grid)
    assert not np.array_equal(periodic_binary_offset(grid, offset=-1 / 8), grid)


@pytest.mark.parametrize('grid,offset', [(np.ones((2, 3)), 0), (np.full((2, 2), .5), 0),
                                        (np.zeros((2, 2)), float('nan'))])
def test_invalid_inputs_fail_closed(grid, offset):
    with pytest.raises(ValueError):
        periodic_binary_offset(grid, offset=offset)
