"""Reproducible validation-only stratification for inverse-design audits.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import numpy as np


def stratify_target_margin(
    spectra: np.ndarray, indices: np.ndarray, *, lower_band: int,
    target_frequency: float, margin_threshold: float, per_stratum: int, seed: int,
) -> dict:
    """Sample positive/near-zero/negative coarse target margins without replacement.

    Inputs must contain only the explicitly permitted development split. No
    physical conclusion follows from coarse sampled spectral labels.
    """
    values = np.asarray(spectra, dtype=np.float64)
    ids = np.asarray(indices)
    if values.ndim != 3 or not values.size or len(values) != len(ids) or ids.ndim != 1:
        raise ValueError('expected aligned nonempty [sample,k,band] spectra and indices')
    if not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids) or np.any(ids < 0):
        raise ValueError('sample indices must be unique nonnegative integers')
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(np.diff(values, axis=2) < 0):
        raise ValueError('spectra must be finite, nonnegative and ordered')
    if isinstance(lower_band, bool) or not isinstance(lower_band, (int, np.integer)) or not 1 <= lower_band < values.shape[2]:
        raise ValueError('invalid adjacent band pair')
    if not np.isfinite([target_frequency, margin_threshold]).all() or target_frequency <= 0 or margin_threshold <= 0 or per_stratum <= 0:
        raise ValueError('invalid target or sampling parameters')
    lower = values[:, :, lower_band-1].max(axis=1)
    upper = values[:, :, lower_band].min(axis=1)
    margin = 2 * np.minimum(target_frequency - lower, upper - target_frequency) / target_frequency
    masks = {'positive': margin > margin_threshold, 'near_zero': np.abs(margin) <= margin_threshold,
             'negative': margin < -margin_threshold}
    rng = np.random.default_rng(seed)
    selected = []
    counts = {}
    for name, mask in masks.items():
        locations = np.flatnonzero(mask)
        locations = locations[np.argsort(ids[locations], kind='stable')]
        counts[name] = int(len(locations))
        chosen = rng.choice(locations, size=min(per_stratum, len(locations)), replace=False)
        for j in chosen:
            selected.append({'index': int(ids[j]), 'stratum': name, 'coarse_lower_edge': float(lower[j]),
                             'coarse_upper_edge': float(upper[j]), 'coarse_margin': float(margin[j])})
    return {'status': 'ready' if all(n >= per_stratum for n in counts.values()) else 'insufficient_strata',
            'counts': counts, 'selected': selected, 'seed': int(seed), 'lower_band': int(lower_band),
            'target_frequency': float(target_frequency), 'margin_threshold': float(margin_threshold),
            'per_stratum': int(per_stratum)}
