"""Registry describing physical workloads that share the HPC control plane.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PhysicalAdapterSpec:
    """Stable identity and migration contract for one physical task family."""

    adapter_id: str
    output_kind: str
    legacy_module: str
    executable_stages: tuple[str, ...]
    migration_status: str

    def to_mapping(self) -> dict[str, object]:
        payload = asdict(self)
        payload["executable_stages"] = list(self.executable_stages)
        return payload


_ADAPTERS = {
    item.adapter_id: item
    for item in (
        PhysicalAdapterSpec(
            "full_cell_band",
            "full_cell_hdf5",
            "phc2d.data.full_cell_generation",
            ("plan", "solve-shard", "finalize", "status", "audit"),
            "reference_adapter",
        ),
        PhysicalAdapterSpec(
            "query_k_path",
            "query_band_cases",
            "phc2d.query",
            ("status", "audit"),
            "registered_for_migration",
        ),
        PhysicalAdapterSpec(
            "field_eigen",
            "field_eigen_hdf5",
            "phc2d.data.field_eigen",
            ("status", "audit"),
            "registered_for_migration",
        ),
        PhysicalAdapterSpec(
            "resolution_convergence",
            "convergence_metrics",
            "phc2d.data.full_cell_convergence",
            ("status", "audit"),
            "registered_for_migration",
        ),
        PhysicalAdapterSpec(
            "inverse_mpb_replay",
            "replay_metrics",
            "phc2d.inverse.benchmark",
            ("status", "audit"),
            "registered_for_migration",
        ),
    )
}


def get_adapter_spec(adapter_id: str) -> PhysicalAdapterSpec:
    """Return one registered adapter or fail closed."""

    try:
        return _ADAPTERS[str(adapter_id)]
    except KeyError as exc:
        raise KeyError(f"Unknown physical HPC adapter: {adapter_id}") from exc


def list_adapter_specs() -> tuple[PhysicalAdapterSpec, ...]:
    """List adapters in stable identifier order."""

    return tuple(_ADAPTERS[key] for key in sorted(_ADAPTERS))


__all__ = ["PhysicalAdapterSpec", "get_adapter_spec", "list_adapter_specs"]
