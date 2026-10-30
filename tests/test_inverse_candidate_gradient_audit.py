"""Tests for local discrete-candidate gradient ranking.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import pytest

torch = pytest.importorskip('torch')

from phc2d.inverse.candidate_gradient_audit import score_candidate_directions


class LinearPredictor:
    device = torch.device('cpu')
    def predict_tensor(self, inputs):
        value = inputs.reshape(len(inputs), -1).sum(dim=1)
        return torch.stack((.2 + .01 * value, .8 + .02 * value), dim=1).unsqueeze(1)


def test_realizable_candidate_directions_are_ranked_by_input_gradient():
    anchor = torch.ones(1, 2, 2)
    candidates = [{'label': 'plus', 'parameter_hash': 'p', 'input': anchor + .1},
                  {'label': 'minus', 'parameter_hash': 'm', 'input': anchor - .1}]
    result = score_candidate_directions(LinearPredictor(), anchor, candidates,
                                        lower_band=1, target_frequency=.5)
    assert result['selected_label'] == 'minus'
    assert result['selected_parameter_hash'] == 'm'
    for row in result['rows']:
        assert row['linearized_gain'] == pytest.approx(row['proxy_gain'], abs=1e-6)


def test_duplicate_directions_are_rejected():
    anchor = torch.ones(1, 2, 2)
    candidates = [{'label': 'same', 'parameter_hash': 'a', 'input': anchor},
                  {'label': 'same', 'parameter_hash': 'b', 'input': anchor}]
    with pytest.raises(ValueError, match='unique'):
        score_candidate_directions(LinearPredictor(), anchor, candidates,
                                   lower_band=1, target_frequency=.5)
