"""Lazy runtime adapter boundary for portable HPC physics campaigns.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from phc2d.hpc.adapters import get_adapter_spec


class PhysicalTaskAdapter(Protocol):
    """Minimal executable interface implemented by each migrated task family."""

    adapter_id: str

    def plan(
        self,
        config_path: str | Path,
        *,
        run_dir: str | Path,
        workers: int | str | None,
        shard_count: int,
        platform_id: str,
    ) -> Path: ...

    def solve_shard(
        self,
        config_path: str | Path,
        *,
        run_dir: str | Path,
        shard_index: int,
        shard_count: int,
        workers: int | str | None,
        allow_resume: bool,
        skip_environment_check: bool,
        **kwargs: Any,
    ) -> Path: ...

    def finalize(
        self,
        config_path: str | Path,
        *,
        run_dir: str | Path,
        workers: int | str,
    ) -> Path: ...


class AdapterNotMigratedError(RuntimeError):
    """Raised instead of silently falling back to a legacy single-node runner."""


def get_runtime_adapter(adapter_id: str) -> PhysicalTaskAdapter:
    """Load one executable adapter without importing solver libraries eagerly."""

    spec = get_adapter_spec(adapter_id)
    if spec.adapter_id == "full_cell_band":
        from phc2d.data.full_cell_hpc import FullCellHPCAdapter

        return FullCellHPCAdapter()
    raise AdapterNotMigratedError(
        f"Adapter {adapter_id} is registered but not migrated to plan/solve-shard/finalize"
    )


__all__ = ["AdapterNotMigratedError", "PhysicalTaskAdapter", "get_runtime_adapter"]
