"""Apply the frozen inverse surrogate's floating-point runtime contract.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

from typing import Any, Mapping

from phc2d.training.device import DeviceConfig, resolve_device


def configure_design_runtime(policy: Mapping[str, Any], *, device: str) -> dict[str, Any]:
    """Apply explicit FP32, no autocast, no TF32 for one design process."""
    import torch

    expected = {'precision': 'fp32', 'amp_enabled': False, 'tf32_enabled': False}
    if not isinstance(policy, Mapping) or any(policy.get(key) != value for key, value in expected.items()):
        raise ValueError('inverse runtime requires fp32 with AMP and TF32 disabled')
    runtime = DeviceConfig.from_mapping({**expected, 'device': device,
                                         'require_accelerator': device.startswith('cuda')})
    info = resolve_device(runtime, torch_module=torch)
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError('unable to disable TF32 in CUDA backends')
    return {'contract': expected, 'device': info.metadata(),
            'matmul_tf32_enabled': bool(torch.backends.cuda.matmul.allow_tf32),
            'cudnn_tf32_enabled': bool(torch.backends.cudnn.allow_tf32)}
