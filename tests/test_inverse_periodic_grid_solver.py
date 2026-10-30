"""Contract tests for periodic raster MPB material evaluation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from phc2d.solvers.full_cell import build_periodic_binary_material_function


class FakeMeep:
    @staticmethod
    def Medium(*, epsilon):
        return epsilon


def test_mapping_orientation_and_periodic_wrapping():
    grid = np.array([[1, 0], [0, 0]])
    callback = build_periodic_binary_material_function(FakeMeep, grid, epsilon_air=1, epsilon_material=9)
    assert callback(SimpleNamespace(x=-.25, y=-.25)) == 9
    assert callback(SimpleNamespace(x=.25, y=-.25)) == 1
    assert callback(SimpleNamespace(x=.75, y=-.25)) == 9
    assert callback(SimpleNamespace(x=-.25, y=.75)) == 9


def test_callback_copies_original_grid_and_rejects_gray_values():
    grid = np.ones((2, 2))
    callback = build_periodic_binary_material_function(FakeMeep, grid, epsilon_air=1, epsilon_material=9)
    grid[:] = 0
    assert callback(SimpleNamespace(x=0, y=0)) == 9
    with pytest.raises(ValueError, match='binary'):
        build_periodic_binary_material_function(FakeMeep, np.full((2, 2), .5), epsilon_air=1, epsilon_material=9)
