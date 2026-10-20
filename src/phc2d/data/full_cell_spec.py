"""Analytic full-cell structure contract and material evaluators.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from phc2d.data.full_cell_fourier import (
    evaluate_real_fourier_field,
    wrapped_periodic_delta,
)
from phc2d.data.full_cell_hashing import (
    hash_structure_geometry,
    hash_structure_parameters,
    jsonable_full_cell_value,
)
from phc2d.data.single_hole import evaluate_single_hole


FULL_CELL_SCHEMA_VERSION = "phc2d.full_cell.v1"
FULL_CELL_FOURIER_SCHEMA_VERSION = "phc2d.full_cell_fourier.v2"


def _graded_rbf_weight(
    x: np.ndarray,
    y: np.ndarray,
    payload: Mapping[str, Any],
) -> np.ndarray:
    x_arr, y_arr = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
    )
    field = np.zeros_like(x_arr, dtype=np.float64)
    for blob in payload.get("blobs", []):
        dx = wrapped_periodic_delta(x_arr, float(blob["center_x"]))
        dy = wrapped_periodic_delta(y_arr, float(blob["center_y"]))
        angle = float(blob["angle"])
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        xr = cos_a * dx + sin_a * dy
        yr = -sin_a * dx + cos_a * dy
        sigma_x = max(float(blob["sigma_x"]), 1.0e-8)
        sigma_y = max(float(blob["sigma_y"]), 1.0e-8)
        field += float(blob["amplitude"]) * np.exp(
            -0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2)
        )
    field += float(payload.get("fourier_strength", 0.0)) * evaluate_real_fourier_field(
        x_arr,
        y_arr,
        payload.get("modes", []),
    )
    temperature = max(float(payload["temperature"]), 1.0e-8)
    logits = np.clip((field - float(payload["bias"])) / temperature, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _polar_radius(theta: np.ndarray, hole: Mapping[str, Any]) -> np.ndarray:
    perturbation = np.zeros_like(theta, dtype=np.float64)
    for row in hole.get("modes", []):
        order = int(row["order"])
        perturbation += float(row.get("cos", 0.0)) * np.cos(order * theta)
        perturbation += float(row.get("sin", 0.0)) * np.sin(order * theta)
    return float(hole["radius"]) * np.exp(perturbation)


def _binary_polar_weight(
    x: np.ndarray,
    y: np.ndarray,
    payload: Mapping[str, Any],
) -> np.ndarray:
    x_arr, y_arr = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
    )
    air = np.zeros_like(x_arr, dtype=bool)
    for hole in payload.get("holes", []):
        dx = wrapped_periodic_delta(x_arr, float(hole["center_x"]))
        dy = wrapped_periodic_delta(y_arr, float(hole["center_y"]))
        theta = np.arctan2(dy, dx)
        radius = np.sqrt(dx * dx + dy * dy)
        air |= radius <= _polar_radius(theta, hole)
    return (~air).astype(np.float64)


def _binary_spectral_weight(
    x: np.ndarray,
    y: np.ndarray,
    payload: Mapping[str, Any],
) -> np.ndarray:
    field = evaluate_real_fourier_field(x, y, payload.get("modes", []))
    return (field >= float(payload["threshold"])).astype(np.float64)


def _normalized_latent_field(
    x: np.ndarray,
    y: np.ndarray,
    payload: Mapping[str, Any],
) -> np.ndarray:
    field = evaluate_real_fourier_field(x, y, payload.get("modes", []))
    mean = float(payload["normalization_mean"])
    std = float(payload["normalization_std"])
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError("normalization_std must be positive and finite")
    return (field - mean) / std


def _unified_fourier_binary_weight(
    x: np.ndarray,
    y: np.ndarray,
    payload: Mapping[str, Any],
) -> np.ndarray:
    latent = _normalized_latent_field(x, y, payload)
    return (latent >= float(payload["threshold"])).astype(np.float64)


def _unified_fourier_graded_weight(
    x: np.ndarray,
    y: np.ndarray,
    payload: Mapping[str, Any],
) -> np.ndarray:
    latent = _normalized_latent_field(x, y, payload)
    temperature = max(float(payload["temperature"]), 1.0e-8)
    logits = np.clip((latent - float(payload["bias"])) / temperature, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


MaterialWeightEvaluator = Callable[[np.ndarray, np.ndarray, Mapping[str, Any]], np.ndarray]
MATERIAL_EVALUATORS: Mapping[str, MaterialWeightEvaluator] = MappingProxyType(
    {
        "analytic_single_hole_v1": evaluate_single_hole,
        "graded_rbf_fourier": _graded_rbf_weight,
        "binary_polar_holes": _binary_polar_weight,
        "binary_spectral_threshold": _binary_spectral_weight,
        "unified_fourier_binary": _unified_fourier_binary_weight,
        "unified_fourier_graded": _unified_fourier_graded_weight,
    }
)
SUPPORTED_FULL_CELL_FAMILIES = frozenset(MATERIAL_EVALUATORS)


@dataclass(frozen=True)
class FullCellStructureSpec:
    """One periodic full-cell material distribution reconstructed from parameters."""

    sample_id: str
    generator_family: str
    material_kind: str
    parameters: dict[str, Any]
    parameter_names: tuple[str, ...]
    parameter_vector: tuple[float, ...]
    parameter_mask: tuple[bool, ...]
    epsilon_air: float = 1.0
    epsilon_material: float = 8.9
    schema_version: str = FULL_CELL_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.generator_family not in MATERIAL_EVALUATORS:
            raise ValueError(
                f"Unsupported full-cell generator family: {self.generator_family}"
            )
        if self.material_kind not in {"continuous", "binary"}:
            raise ValueError("material_kind must be continuous or binary")
        if (
            len(self.parameter_names) != len(self.parameter_vector)
            or len(self.parameter_mask) != len(self.parameter_vector)
        ):
            raise ValueError("parameter names, values, and mask must have equal length")
        if not float(self.epsilon_material) > float(self.epsilon_air) > 0.0:
            raise ValueError("epsilon_material must exceed positive epsilon_air")

    @property
    def geometry_hash(self) -> str:
        """Return the compatibility-preserving structure record hash."""

        return hash_structure_geometry(self.to_mapping())

    @property
    def parameter_hash(self) -> str:
        """Return a sample-ID-independent analytic parameter hash."""

        return hash_structure_parameters(
            {
                "schema_version": self.schema_version,
                "generator_family": self.generator_family,
                "material_kind": self.material_kind,
                "parameters": self.parameters,
                "epsilon_air": float(self.epsilon_air),
                "epsilon_material": float(self.epsilon_material),
            }
        )

    def evaluate_material_weight(self, x: Any, y: Any) -> np.ndarray:
        """Evaluate the analytic material weight in the closed interval [0,1]."""

        evaluator = MATERIAL_EVALUATORS.get(self.generator_family)
        if evaluator is None:  # defensive fail-closed guard for deserialized inputs
            raise ValueError(
                f"Unsupported full-cell generator family: {self.generator_family}"
            )
        weight = evaluator(np.asarray(x), np.asarray(y), self.parameters)
        return np.clip(np.asarray(weight, dtype=np.float64), 0.0, 1.0)

    def rasterize(self, grid_shape: tuple[int, int] | list[int]) -> np.ndarray:
        """Directly sample the analytic material field at cell centres."""

        from phc2d.data.full_cell_raster import rasterize_full_cell

        return rasterize_full_cell(self, grid_shape)

    def epsilon_grid(self, grid_shape: tuple[int, int] | list[int]) -> np.ndarray:
        """Return float32 permittivity reconstructed from direct material weights."""

        weight = self.rasterize(grid_shape)
        return (
            float(self.epsilon_air)
            + (float(self.epsilon_material) - float(self.epsilon_air)) * weight
        ).astype(np.float32)

    def epsilon_at(self, x: float, y: float) -> float:
        """Evaluate scalar analytic permittivity for MPB material callbacks."""

        weight = float(
            np.asarray(self.evaluate_material_weight(float(x), float(y))).reshape(())
        )
        return float(
            self.epsilon_air
            + (self.epsilon_material - self.epsilon_air) * weight
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return the stable JSON-compatible structure record."""

        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "generator_family": self.generator_family,
            "material_kind": self.material_kind,
            "parameters": jsonable_full_cell_value(self.parameters),
            "parameter_names": list(self.parameter_names),
            "parameter_vector": list(self.parameter_vector),
            "parameter_mask": list(self.parameter_mask),
            "epsilon_air": float(self.epsilon_air),
            "epsilon_material": float(self.epsilon_material),
            "metadata": jsonable_full_cell_value(self.metadata),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FullCellStructureSpec":
        """Rebuild a structure from a stored mapping and fail on unknown families."""

        return cls(
            schema_version=str(value.get("schema_version", FULL_CELL_SCHEMA_VERSION)),
            sample_id=str(value["sample_id"]),
            generator_family=str(value["generator_family"]),
            material_kind=str(value["material_kind"]),
            parameters=dict(value.get("parameters") or {}),
            parameter_names=tuple(
                str(item) for item in value.get("parameter_names", ())
            ),
            parameter_vector=tuple(
                float(item) for item in value.get("parameter_vector", ())
            ),
            parameter_mask=tuple(
                bool(item) for item in value.get("parameter_mask", ())
            ),
            epsilon_air=float(value.get("epsilon_air", 1.0)),
            epsilon_material=float(value.get("epsilon_material", 8.9)),
            metadata=dict(value.get("metadata") or {}),
        )


__all__ = [
    "FULL_CELL_FOURIER_SCHEMA_VERSION",
    "FULL_CELL_SCHEMA_VERSION",
    "MATERIAL_EVALUATORS",
    "SUPPORTED_FULL_CELL_FAMILIES",
    "FullCellStructureSpec",
    "MaterialWeightEvaluator",
]
