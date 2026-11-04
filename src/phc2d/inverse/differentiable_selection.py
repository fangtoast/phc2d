"""Screen and rank hard Fourier geometries from a surrogate trajectory.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

import numpy as np

from phc2d.inverse.fourier_design import update_fourier_binary
from phc2d.inverse.fourier_screen import screen_fourier_candidate, screen_fourier_geometry


def rank_hard_trajectory(source: Any, predictor: Any, history: Sequence[dict[str, Any]], *,
                         coefficient_keys: Sequence[tuple[int, str]], max_mode_radius: float,
                         raster_method: str, source_resolution: int, input_resolution: int,
                         material_fraction_bounds: tuple[float, float], lower_band: int,
                         target_frequency: float) -> dict[str, Any]:
    """Rank each eligible hard design by the exact sampled surrogate margin.

    Every trajectory record is retained. Identical model inputs are evaluated
    once and recorded as duplicates of their earliest candidate.
    """
    import hashlib
    import torch

    keys = tuple(coefficient_keys)
    rows = []
    seen: dict[str, int] = {}
    best = None
    for index, entry in enumerate(history):
        values = entry['fourier_parameters']
        if len(values) != len(keys) + 1:
            raise ValueError('trajectory parameter count mismatch')
        structure = update_fourier_binary(source, sample_id=f'{source.sample_id}-candidate-{index}',
            coefficients={key: float(values[i]) for i, key in enumerate(keys)},
            threshold=float(values[-1]), max_mode_radius=max_mode_radius)
        geometry = screen_fourier_geometry(structure, raster_method=raster_method,
            source_resolution=source_resolution, input_resolution=input_resolution,
            material_fraction_bounds=material_fraction_bounds)
        raster = np.ascontiguousarray(geometry.raster, dtype=np.float32)
        digest = hashlib.sha256(raster.tobytes()).hexdigest()
        row = {'index': index, 'structure': structure.to_mapping(), 'input_sha256': digest,
               'material_fraction': geometry.material_fraction, 'status': geometry.status}
        if geometry.status == 'eligible' and digest in seen:
            row.update(status='duplicate_input', duplicate_of=seen[digest])
        elif geometry.status == 'eligible':
            seen[digest] = index
            result = screen_fourier_candidate(structure, predictor, raster_method=raster_method,
                source_resolution=source_resolution, input_resolution=input_resolution,
                material_fraction_bounds=material_fraction_bounds, lower_band=lower_band,
                target_frequency=target_frequency)
            row.update(asdict(result))
            if result.status == 'surrogate_only':
                with torch.no_grad():
                    bands = predictor.predict_tensor(torch.as_tensor(raster[None])).detach().cpu().numpy()
                row['bands'] = bands[0].tolist()
                margin = float(result.predicted_gap['target_margin'])
                if best is None or margin > best[0]:
                    best = (margin, index)
        rows.append(row)
    return {'status': 'selected' if best else 'no_valid_candidates',
            'selected_index': None if best is None else best[1], 'candidates': rows}
