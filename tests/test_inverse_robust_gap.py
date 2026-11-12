"""Contract tests for sampled robust common band gaps.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import numpy as np
import pytest

from phc2d.inverse.robust_gap import common_gap


def test_common_interval_uses_extrema_across_k_and_perturbations():
    spectra = np.array([[[0.2, 0.8], [0.4, 0.9]], [[0.3, 0.7], [0.35, 0.85]]])
    result = common_gap(spectra, lower_band=1, target_frequency=0.55)
    assert result.lower_edge == pytest.approx(0.4)
    assert result.upper_edge == pytest.approx(0.7)
    assert result.target_margin == pytest.approx(2 * 0.15 / 0.55)
    assert result.contains_target


def test_individual_gaps_can_have_no_common_interval():
    spectra = np.array([[[0.1, 0.3]], [[0.5, 0.7]]])
    result = common_gap(spectra, lower_band=1, target_frequency=0.4)
    assert result.signed_width == pytest.approx(-0.2)
    assert result.target_margin < 0
    assert not result.contains_target


def test_target_outside_existing_gap_and_boundary():
    spectra = np.array([[[0.2, 0.4], [0.25, 0.5]]])
    assert common_gap(spectra, lower_band=1, target_frequency=0.6).target_margin < 0
    assert not common_gap(spectra, lower_band=1, target_frequency=0.25).contains_target


@pytest.mark.parametrize("spectra,band,target", [
    (np.empty((0, 1, 2)), 1, 0.5),
    (np.array([[[0.2, np.nan]]]), 1, 0.5),
    (np.array([[[0.5, 0.2]]]), 1, 0.5),
    (np.array([[[0.2, 0.5]]]), 2, 0.5),
    (np.array([[[0.2, 0.5]]]), 1, 0.0),
])
def test_rejects_invalid_contract(spectra, band, target):
    with pytest.raises(ValueError):
        common_gap(spectra, lower_band=band, target_frequency=target)
