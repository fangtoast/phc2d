"""Periodic phase topology and raster feature diagnostics.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import numpy as np
import pytest

from phc2d.data.periodic_geometry import periodic_component_sizes
from phc2d.generators.full_cell_legacy import _periodic_component_count
from phc2d.inverse.morphology import periodic_binary_geometry_metrics, periodic_geometry_gate


def test_components_join_across_opposing_cell_edges():
    grid = np.zeros((12, 12), dtype=np.uint8)
    grid[4:7, :2] = 1
    grid[4:7, -2:] = 1
    grid[9, 6] = 1
    assert periodic_component_sizes(grid) == (1, 12)
    assert _periodic_component_count(grid) == 2
    metrics = periodic_binary_geometry_metrics(grid, opening_radius_pixels=1)
    assert metrics['material']['component_count'] == 2
    assert metrics['material']['component_areas_over_lattice_squared'] == pytest.approx([1/144, 12/144])
    assert metrics['material']['opening_removed_fraction_of_phase'] > 0
    assert metrics['void']['component_count'] == 1
    gate = periodic_geometry_gate(grid, opening_radius_pixels=1, min_component_area=.01,
                                  max_opening_removed_fraction=1.)
    assert gate['status'] == 'rejected' and 'material_small_component' in gate['failures']


def test_periodic_translation_preserves_all_geometry_metrics():
    grid = np.zeros((16, 16), dtype=np.uint8)
    grid[1:6, 2:8] = 1
    grid[9, 12] = 1
    expected = periodic_binary_geometry_metrics(grid, opening_radius_pixels=2)
    actual = periodic_binary_geometry_metrics(np.roll(grid, (5, 8), axis=(0, 1)), opening_radius_pixels=2)
    assert expected == actual


def test_void_pore_crossing_periodic_boundary_is_one_component():
    grid = np.ones((16, 16), dtype=np.uint8)
    grid[:2, 6:10] = 0
    grid[-2:, 6:10] = 0
    metrics = periodic_binary_geometry_metrics(grid, opening_radius_pixels=1)
    assert metrics['void']['component_count'] == 1
    assert metrics['void']['component_areas_over_lattice_squared'] == pytest.approx([16/256])


def test_empty_phase_and_invalid_raster_are_explicit():
    metrics = periodic_binary_geometry_metrics(np.ones((12, 12)), opening_radius_pixels=1)
    assert metrics['void']['component_count'] == 0
    assert metrics['void']['opening_removed_fraction_of_phase'] is None
    with pytest.raises(ValueError, match='binary'):
        periodic_binary_geometry_metrics(np.full((12, 12), .5), opening_radius_pixels=1)
    with pytest.raises(ValueError, match='radius'):
        periodic_binary_geometry_metrics(np.ones((12, 12)), opening_radius_pixels=3)
