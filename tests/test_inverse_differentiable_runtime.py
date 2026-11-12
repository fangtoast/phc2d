"""Check that inverse inference honors the frozen FP32 runtime contract.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import pytest

torch = pytest.importorskip('torch')
from phc2d.inverse.differentiable_runtime import configure_design_runtime


def test_runtime_disables_tf32_and_rejects_incompatible_policy(monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', True)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', True)
    policy = {'precision': 'fp32', 'amp_enabled': False, 'tf32_enabled': False}
    result = configure_design_runtime(policy, device='cpu')
    assert result['contract'] == policy
    assert result['matmul_tf32_enabled'] is False and result['cudnn_tf32_enabled'] is False
    assert result['device']['resolved_device'] == 'cpu'
    with pytest.raises(ValueError, match='TF32'):
        configure_design_runtime({**policy, 'tf32_enabled': True}, device='cpu')
