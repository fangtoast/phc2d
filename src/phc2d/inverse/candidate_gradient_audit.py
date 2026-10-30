"""Compare input-gradient directions with realizable local candidate changes.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from phc2d.inverse.input_gradient_audit import sampled_target_margin


def score_candidate_directions(predictor: Any, anchor: Any, candidates: Sequence[Mapping[str, Any]], *,
                               lower_band: int, target_frequency: float) -> dict[str, Any]:
    """Score actual candidate raster displacements using one anchor gradient.

    This is a first-order surrogate input-space comparison; a finite candidate
    displacement is not a derivative of the binary geometry. Callers compare
    the chosen candidate and a matched random baseline with numerical MPB.
    """
    import torch

    if not math.isfinite(target_frequency) or target_frequency <= 0 or not candidates:
        raise ValueError('positive target and nonempty candidates required')
    x = torch.as_tensor(anchor, dtype=torch.float32, device=predictor.device).detach().clone()
    if x.ndim < 2 or x.shape[0] != 1 or not torch.isfinite(x).all():
        raise ValueError('one finite anchor input is required')
    names = [str(c['label']) for c in candidates]
    if len(set(names)) != len(names):
        raise ValueError('candidate labels must be unique')

    def objective(inputs: Any) -> Any:
        return sampled_target_margin(predictor, inputs, lower_band=lower_band,
                                     target_frequency=target_frequency)

    x.requires_grad_(True)
    anchor_score = objective(x)[0]
    gradient, = torch.autograd.grad(anchor_score, x)
    if not torch.isfinite(gradient).all():
        raise ValueError('nonfinite input gradient')
    records = []
    with torch.no_grad():
        for candidate in candidates:
            y = torch.as_tensor(candidate['input'], dtype=torch.float32, device=predictor.device)
            if y.shape != x.shape or not torch.isfinite(y).all():
                raise ValueError('candidate input shape or value mismatch')
            delta = y - x
            linear_gain = float(torch.sum(gradient * delta).cpu())
            actual_gain = float((objective(y)[0] - anchor_score).cpu())
            distance = float(torch.linalg.vector_norm(delta).cpu())
            if not all(math.isfinite(v) for v in (linear_gain, actual_gain, distance)):
                raise ValueError('nonfinite candidate direction score')
            records.append({'label': str(candidate['label']), 'parameter_hash': str(candidate['parameter_hash']),
                            'linearized_gain': linear_gain, 'proxy_gain': actual_gain,
                            'input_displacement_norm': distance})
    records.sort(key=lambda r: r['label'])
    selected = max(records, key=lambda r: r['linearized_gain'])
    return {'status': 'continuous_input_gradient_on_discrete_candidates',
            'anchor_proxy_margin': float(anchor_score.detach().cpu()),
            'input_gradient_norm': float(torch.linalg.vector_norm(gradient).detach().cpu()),
            'selected_label': selected['label'], 'selected_parameter_hash': selected['parameter_hash'],
            'rows': records}
