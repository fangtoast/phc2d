"""Training-loop interfaces shared by forward surrogate experiments.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from phc2d.training.config import ForwardTrainingConfig
from phc2d.training.device import (
    DeviceConfig,
    ResolvedDeviceInfo,
    autocast_context,
    collect_device_metadata,
    create_grad_scaler,
    move_batch_to_device,
    peak_gpu_memory_mb,
    reset_peak_gpu_memory,
    resolve_device,
    synchronize_device,
)

__all__ = [
    "DeviceConfig",
    "ForwardTrainingConfig",
    "ResolvedDeviceInfo",
    "autocast_context",
    "collect_device_metadata",
    "create_grad_scaler",
    "move_batch_to_device",
    "peak_gpu_memory_mb",
    "reset_peak_gpu_memory",
    "resolve_device",
    "synchronize_device",
]
