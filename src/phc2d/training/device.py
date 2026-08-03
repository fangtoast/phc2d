"""Lazy GPU/CPU device resolution helpers for phc2d training runtimes.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if value is None:
        return default
    return bool(value)


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("runtime config must be a mapping")
    return dict(value)


@dataclass(frozen=True)
class DeviceConfig:
    """Serializable runtime device policy shared by forward and inverse runs."""

    device: str = "auto"
    require_accelerator: bool = False
    device_index: int = 0
    precision: str = "fp32"
    amp_enabled: bool = False
    amp_dtype: str = "fp16"
    tf32_enabled: bool = False
    non_blocking_transfer: bool = True
    record_peak_memory: bool = True
    num_workers: int = 0
    pin_memory: bool | None = None

    @classmethod
    def from_mapping(cls, value: Any, *, fallback_device: str | None = None) -> "DeviceConfig":
        payload = _as_mapping(value)
        if fallback_device is not None and "device" not in payload:
            payload["device"] = fallback_device
        return cls(
            device=str(payload.get("device", "auto")),
            require_accelerator=_coerce_bool(payload.get("require_accelerator"), default=False),
            device_index=int(payload.get("device_index", 0)),
            precision=str(payload.get("precision", "fp32")),
            amp_enabled=_coerce_bool(payload.get("amp_enabled"), default=False),
            amp_dtype=str(payload.get("amp_dtype", "fp16")),
            tf32_enabled=_coerce_bool(payload.get("tf32_enabled"), default=False),
            non_blocking_transfer=_coerce_bool(payload.get("non_blocking_transfer"), default=True),
            record_peak_memory=_coerce_bool(payload.get("record_peak_memory"), default=True),
            num_workers=max(0, int(payload.get("num_workers", 0))),
            pin_memory=None if payload.get("pin_memory") is None else _coerce_bool(payload.get("pin_memory")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "require_accelerator": self.require_accelerator,
            "device_index": self.device_index,
            "precision": self.precision,
            "amp_enabled": self.amp_enabled,
            "amp_dtype": self.amp_dtype,
            "tf32_enabled": self.tf32_enabled,
            "non_blocking_transfer": self.non_blocking_transfer,
            "record_peak_memory": self.record_peak_memory,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }


@dataclass(frozen=True)
class ResolvedDeviceInfo:
    """Runtime device resolution plus serializable metadata."""

    requested_device: str
    resolved_device: str
    accelerator_required: bool
    device_index: int | None
    device_name: str
    torch_version: str | None
    torch_cuda_version: str | None
    cudnn_version: int | None
    precision: str
    amp_enabled: bool
    amp_dtype: str
    tf32_enabled: bool
    non_blocking_transfer: bool
    record_peak_memory: bool
    num_workers: int
    pin_memory: bool
    fallback_reason: str | None = None
    torch_device: Any | None = None

    @property
    def is_cuda(self) -> bool:
        return str(self.resolved_device).startswith("cuda")

    def metadata(self, *, peak_gpu_memory_mb: float | None = None) -> dict[str, Any]:
        return {
            "requested_device": self.requested_device,
            "resolved_device": self.resolved_device,
            "accelerator_required": self.accelerator_required,
            "device_name": self.device_name,
            "device_index": self.device_index,
            "torch_version": self.torch_version,
            "torch_cuda_version": self.torch_cuda_version,
            "cudnn_version": self.cudnn_version,
            "precision": self.precision,
            "amp_enabled": self.amp_enabled,
            "amp_dtype": self.amp_dtype,
            "tf32_enabled": self.tf32_enabled,
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "non_blocking_transfer": self.non_blocking_transfer,
            "device_fallback_reason": self.fallback_reason,
        }


def _normalize_requested_device(config: DeviceConfig) -> str:
    requested = str(config.device).strip().lower()
    if requested == "cuda":
        return f"cuda:{int(config.device_index)}"
    return requested


def _cuda_index_from_device_name(device_name: str, fallback: int) -> int:
    if ":" not in device_name:
        return int(fallback)
    return int(device_name.split(":", 1)[1])


def resolve_device(config: DeviceConfig | Mapping[str, Any] | None = None, *, torch_module: Any | None = None) -> ResolvedDeviceInfo:
    """Resolve a DeviceConfig to a torch device without importing torch at module import time."""

    runtime = config if isinstance(config, DeviceConfig) else DeviceConfig.from_mapping(config)
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            if runtime.require_accelerator or str(runtime.device).lower().startswith("cuda"):
                raise ImportError("CUDA was requested, but optional dependency 'torch' is not installed.") from exc
            raise ImportError("Device resolution requires optional dependency 'torch'.") from exc

    requested = _normalize_requested_device(runtime)
    cuda_available = bool(torch_module.cuda.is_available())
    fallback_reason: str | None = None
    if requested == "auto":
        if cuda_available:
            resolved = f"cuda:{int(runtime.device_index)}"
        elif runtime.require_accelerator:
            raise RuntimeError("runtime.device=auto with require_accelerator=true, but torch.cuda.is_available() is false")
        else:
            resolved = "cpu"
            fallback_reason = "cuda_unavailable_auto_cpu_allowed"
    elif requested == "cpu":
        if runtime.require_accelerator:
            raise RuntimeError("runtime.require_accelerator=true forbids resolving runtime.device=cpu")
        resolved = "cpu"
    elif requested.startswith("cuda"):
        if not cuda_available:
            raise RuntimeError(f"runtime.device={runtime.device!r} requested CUDA, but torch.cuda.is_available() is false")
        resolved = requested
    else:
        raise ValueError("runtime.device must be one of auto, cpu, cuda, or cuda:<index>")

    device = torch_module.device(resolved)
    is_cuda = str(device).startswith("cuda")
    device_index = _cuda_index_from_device_name(str(device), runtime.device_index) if is_cuda else None
    if is_cuda and device_index is not None and device_index >= int(torch_module.cuda.device_count()):
        raise RuntimeError(f"runtime.device resolved to cuda:{device_index}, but only {torch_module.cuda.device_count()} CUDA device(s) are visible")
    if is_cuda:
        device_name = str(torch_module.cuda.get_device_name(device_index))
    else:
        device_name = "cpu"

    try:
        torch_module.backends.cuda.matmul.allow_tf32 = bool(runtime.tf32_enabled)
        torch_module.backends.cudnn.allow_tf32 = bool(runtime.tf32_enabled)
    except Exception:
        pass
    pin_memory = bool(runtime.pin_memory) if runtime.pin_memory is not None else is_cuda
    cudnn_version = None
    try:
        cudnn_version = torch_module.backends.cudnn.version()
    except Exception:
        cudnn_version = None
    return ResolvedDeviceInfo(
        requested_device=str(runtime.device),
        resolved_device=str(device),
        accelerator_required=bool(runtime.require_accelerator),
        device_index=device_index,
        device_name=device_name,
        torch_version=str(getattr(torch_module, "__version__", "")) or None,
        torch_cuda_version=getattr(torch_module.version, "cuda", None),
        cudnn_version=cudnn_version,
        precision=str(runtime.precision),
        amp_enabled=bool(runtime.amp_enabled),
        amp_dtype=str(runtime.amp_dtype),
        tf32_enabled=bool(runtime.tf32_enabled),
        non_blocking_transfer=bool(runtime.non_blocking_transfer),
        record_peak_memory=bool(runtime.record_peak_memory),
        num_workers=int(runtime.num_workers),
        pin_memory=pin_memory,
        fallback_reason=fallback_reason,
        torch_device=device,
    )


def move_batch_to_device(batch: Any, device: Any, *, non_blocking: bool = True) -> Any:
    """Recursively move tensors in a batch to one device."""

    if hasattr(batch, "to"):
        try:
            return batch.to(device, non_blocking=non_blocking)
        except TypeError:
            return batch.to(device)
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(item, device, non_blocking=non_blocking) for item in batch)
    if isinstance(batch, list):
        return [move_batch_to_device(item, device, non_blocking=non_blocking) for item in batch]
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device, non_blocking=non_blocking) for key, value in batch.items()}
    return batch


def _amp_dtype(torch_module: Any, config: DeviceConfig | ResolvedDeviceInfo) -> Any:
    dtype_name = str(getattr(config, "amp_dtype", "fp16")).lower()
    if dtype_name in {"bf16", "bfloat16", "amp_bf16"}:
        return torch_module.bfloat16
    return torch_module.float16


def autocast_context(config: DeviceConfig | ResolvedDeviceInfo, device: Any, *, torch_module: Any | None = None) -> Any:
    """Return an autocast context when AMP is enabled on CUDA."""

    if torch_module is None:
        import torch as torch_module
    enabled = bool(getattr(config, "amp_enabled", False)) and str(device).startswith("cuda")
    if not enabled:
        return nullcontext()
    return torch_module.autocast(device_type="cuda", dtype=_amp_dtype(torch_module, config), enabled=True)


def create_grad_scaler(config: DeviceConfig | ResolvedDeviceInfo, device: Any, *, torch_module: Any | None = None) -> Any:
    """Create an AMP grad scaler when fp16 AMP is active."""

    if torch_module is None:
        import torch as torch_module
    enabled = bool(getattr(config, "amp_enabled", False)) and str(device).startswith("cuda") and str(getattr(config, "amp_dtype", "fp16")).lower() in {"fp16", "float16", "amp_fp16"}
    try:
        return torch_module.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch_module.cuda.amp.GradScaler(enabled=enabled)


def synchronize_device(device: Any, *, torch_module: Any | None = None) -> None:
    """Synchronize CUDA timings; no-op on CPU."""

    if not str(device).startswith("cuda"):
        return
    if torch_module is None:
        import torch as torch_module
    torch_module.cuda.synchronize(device)


def peak_gpu_memory_mb(device: Any, *, torch_module: Any | None = None) -> float | None:
    """Return CUDA peak allocated memory in MiB when available."""

    if not str(device).startswith("cuda"):
        return None
    if torch_module is None:
        import torch as torch_module
    return float(torch_module.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))


def reset_peak_gpu_memory(device: Any, *, torch_module: Any | None = None) -> None:
    """Reset CUDA peak memory counter when available."""

    if not str(device).startswith("cuda"):
        return
    if torch_module is None:
        import torch as torch_module
    torch_module.cuda.reset_peak_memory_stats(device)


def collect_device_metadata(device: ResolvedDeviceInfo, config: DeviceConfig | None = None) -> dict[str, Any]:
    """Collect serializable device metadata for run.meta.json and metrics.json."""

    _ = config
    return device.metadata()


__all__ = [
    "DeviceConfig",
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
