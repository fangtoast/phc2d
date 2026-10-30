"""Tests for validation-only target-margin audit sampling.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import numpy as np
import pytest

from phc2d.inverse.audit_sampling import stratify_target_margin


def test_stratification_is_seeded_and_order_independent():
    spectra = np.array([[[.2, .8]], [[.49, .51]], [[.6, .7]], [[.1, .9]]])
    ids = np.array([10, 20, 30, 40])
    kwargs = dict(lower_band=1, target_frequency=.5, margin_threshold=.05, per_stratum=1, seed=7)
    a = stratify_target_margin(spectra, ids, **kwargs)
    b = stratify_target_margin(spectra[::-1], ids[::-1], **kwargs)
    assert a == b and a['status'] == 'ready'
    assert a['counts'] == {'positive': 2, 'near_zero': 1, 'negative': 1}


def test_missing_strata_stays_insufficient_and_rejects_duplicate_ids():
    spectra = np.array([[[.2, .8]], [[.2, .8]]])
    kwargs = dict(lower_band=1, target_frequency=.5, margin_threshold=.05, per_stratum=1, seed=7)
    assert stratify_target_margin(spectra, np.array([1, 2]), **kwargs)['status'] == 'insufficient_strata'
    with pytest.raises(ValueError, match='unique'):
        stratify_target_margin(spectra, np.array([1, 1]), **kwargs)
