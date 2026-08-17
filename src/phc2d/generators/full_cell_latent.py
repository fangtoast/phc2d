"""Unified hard-cutoff Fourier latent-field generators for B/G data v2.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from phc2d.data.full_cell_fourier import (
    evaluate_real_fourier_field,
    independent_half_plane_modes,
    periodic_cell_centers,
)
from phc2d.data.full_cell_spec import (
    FULL_CELL_FOURIER_SCHEMA_VERSION,
    FullCellStructureSpec,
)
from phc2d.generators.full_cell_legacy import full_cell_structure_stats


UNIFIED_FOURIER_ALGORITHM_VERSION = "phc2d.unified_fourier.v1"
UNIFIED_FOURIER_SEED_VERSION = "sha256_u64_be_v1"


def derive_full_cell_seed64(master_seed: int, *parts: Any) -> int:
    """Derive a deterministic unsigned 64-bit seed from stable identity parts."""

    raw = ":".join([str(int(master_seed)), *(str(item) for item in parts)])
    return int.from_bytes(hashlib.sha256(raw.encode("utf-8")).digest()[:8], "big")


def _numeric_parameters(
    value: Any,
    *,
    prefix: str = "",
) -> tuple[list[str], list[float]]:
    names: list[str] = []
    values: list[float] = []
    if isinstance(value, Mapping):
        for key in sorted(value):
            child_names, child_values = _numeric_parameters(
                value[key],
                prefix=f"{prefix}.{key}" if prefix else str(key),
            )
            names.extend(child_names)
            values.extend(child_values)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            child_names, child_values = _numeric_parameters(
                item,
                prefix=f"{prefix}[{index}]",
            )
            names.extend(child_names)
            values.extend(child_values)
    elif isinstance(value, (bool, int, float, np.integer, np.floating, np.bool_)):
        names.append(prefix)
        values.append(float(value))
    return names, values


def _range(cfg: Mapping[str, Any], key: str, default: tuple[float, float]) -> tuple[float, float]:
    raw = cfg.get(key, default)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
        raise ValueError(f"{key} must contain exactly two numeric bounds")
    low, high = float(raw[0]), float(raw[1])
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError(f"{key} must contain finite ascending bounds")
    return low, high


def _stratified_uniform(
    rng: np.random.Generator,
    bounds: tuple[float, float],
    *,
    bin_index: int,
    bin_count: int = 4,
) -> float:
    index = int(bin_index)
    count = int(bin_count)
    if count <= 0 or not 0 <= index < count:
        raise ValueError("bin_index must select one configured stratum")
    low, high = bounds
    edges = np.linspace(low, high, count + 1, dtype=np.float64)
    return float(rng.uniform(float(edges[index]), float(edges[index + 1])))


def _sample_latent_payload(
    rng: np.random.Generator,
    *,
    cutoff: int,
    audit_resolution: int,
) -> tuple[dict[str, Any], np.ndarray]:
    modes = [
        {
            "mx": int(mx),
            "my": int(my),
            "cos": float(rng.normal(0.0, 1.0)),
            "sin": float(rng.normal(0.0, 1.0)),
        }
        for mx, my in independent_half_plane_modes(cutoff)
    ]
    x, y = periodic_cell_centers((audit_resolution, audit_resolution))
    field = evaluate_real_fourier_field(x, y, modes)
    mean = float(np.mean(field, dtype=np.float64))
    std = float(np.std(field, dtype=np.float64))
    if not np.isfinite(field).all() or not np.isfinite(std) or std <= 1.0e-12:
        raise RuntimeError("unified Fourier latent field failed finite/std gate")
    payload = {
        "algorithm_version": UNIFIED_FOURIER_ALGORITHM_VERSION,
        "mode_set": "independent_half_plane_hard_radial_cutoff",
        "mode_order": "mx_then_my_ascending",
        "zero_mode": "excluded",
        "coefficient_distribution": "iid_standard_normal",
        "fourier_cutoff": int(cutoff),
        "modes": modes,
        "normalization_grid_resolution": int(audit_resolution),
        "normalization_mean": mean,
        "normalization_std": std,
    }
    return payload, (field - mean) / std


def _build_spec(
    *,
    sample_id: str,
    family: str,
    material_kind: str,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    epsilon_air: float,
    epsilon_material: float,
) -> FullCellStructureSpec:
    names, values = _numeric_parameters(payload)
    return FullCellStructureSpec(
        sample_id=sample_id,
        generator_family=family,
        material_kind=material_kind,
        parameters=dict(payload),
        parameter_names=tuple(names),
        parameter_vector=tuple(values),
        parameter_mask=tuple(True for _ in values),
        epsilon_air=float(epsilon_air),
        epsilon_material=float(epsilon_material),
        schema_version=FULL_CELL_FOURIER_SCHEMA_VERSION,
        metadata=dict(metadata),
    )


def _extended_stats(
    spec: FullCellStructureSpec,
    *,
    audit_resolution: int,
    latent: np.ndarray,
) -> dict[str, Any]:
    stats = full_cell_structure_stats(spec, audit_resolution=audit_resolution)
    weight = spec.rasterize((audit_resolution, audit_resolution)).astype(np.float64)
    edge_x = np.abs(weight - np.roll(weight, 1, axis=1))
    edge_y = np.abs(weight - np.roll(weight, 1, axis=0))
    centered = weight - float(np.mean(weight))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    frequency = np.fft.fftshift(np.fft.fftfreq(audit_resolution)) * audit_resolution
    fx, fy = np.meshgrid(frequency, frequency, indexing="xy")
    radial = np.hypot(fx, fy)
    total = max(float(np.sum(spectrum)), 1.0e-12)
    stats.update(
        {
            "boundary_variation_per_a": float(np.sum(edge_x + edge_y) / audit_resolution),
            "spectral_second_moment": float(np.sum(spectrum * radial * radial) / total),
            "latent_mean_audit": float(np.mean(latent)),
            "latent_std_audit": float(np.std(latent)),
        }
    )
    return stats


def generate_unified_fourier_binary(
    sample_id: str,
    *,
    master_seed: int,
    material_fraction_bin: int,
    config: Mapping[str, Any] | None = None,
) -> FullCellStructureSpec:
    """Generate one binary material field from the shared Fourier latent contract."""

    cfg = dict(config or {})
    seed = derive_full_cell_seed64(
        master_seed,
        sample_id,
        "unified_fourier_binary",
        UNIFIED_FOURIER_SEED_VERSION,
    )
    rng = np.random.default_rng(seed)
    cutoff = int(cfg.get("fourier_cutoff", 20))
    audit_resolution = int(cfg.get("audit_resolution", 256))
    target_range = _range(
        cfg,
        "binary_target_material_fraction_range",
        (0.50, 0.90),
    )
    target = _stratified_uniform(
        rng,
        target_range,
        bin_index=int(material_fraction_bin),
    )
    latent_payload, latent = _sample_latent_payload(
        rng,
        cutoff=cutoff,
        audit_resolution=audit_resolution,
    )
    threshold = float(np.quantile(latent, 1.0 - target))
    payload = {
        **latent_payload,
        "mapping": "binary_upper_quantile_v1",
        "target_material_fraction": target,
        "threshold": threshold,
    }
    spec = _build_spec(
        sample_id=sample_id,
        family="unified_fourier_binary",
        material_kind="binary",
        payload=payload,
        metadata={
            "seed": seed,
            "seed_derivation": UNIFIED_FOURIER_SEED_VERSION,
            "material_fraction_bin": int(material_fraction_bin),
        },
        epsilon_air=float(cfg.get("epsilon_air", 1.0)),
        epsilon_material=float(cfg.get("epsilon_material", 8.9)),
    )
    stats = _extended_stats(spec, audit_resolution=audit_resolution, latent=latent)
    tolerance = float(cfg.get("target_material_fraction_tolerance", 0.01))
    if abs(float(stats["material_fraction"]) - target) > tolerance:
        raise RuntimeError("binary unified Fourier sample failed target-fraction gate")
    if not {float(np.min(spec.rasterize((audit_resolution, audit_resolution)))), float(np.max(spec.rasterize((audit_resolution, audit_resolution))))} <= {0.0, 1.0}:
        raise RuntimeError("binary unified Fourier sample is not strictly binary")
    return FullCellStructureSpec.from_mapping(
        {**spec.to_mapping(), "metadata": {**spec.metadata, "stats": stats}}
    )


def generate_unified_fourier_graded(
    sample_id: str,
    *,
    master_seed: int,
    material_fraction_bin: int,
    temperature_bin: int,
    config: Mapping[str, Any] | None = None,
) -> FullCellStructureSpec:
    """Generate one graded material field from the shared Fourier latent contract."""

    cfg = dict(config or {})
    seed = derive_full_cell_seed64(
        master_seed,
        sample_id,
        "unified_fourier_graded",
        UNIFIED_FOURIER_SEED_VERSION,
    )
    rng = np.random.default_rng(seed)
    cutoff = int(cfg.get("fourier_cutoff", 20))
    audit_resolution = int(cfg.get("audit_resolution", 256))
    target = _stratified_uniform(
        rng,
        _range(cfg, "graded_target_material_fraction_range", (0.60, 0.78)),
        bin_index=int(material_fraction_bin),
    )
    temperature = _stratified_uniform(
        rng,
        _range(cfg, "graded_temperature_range", (0.08, 0.16)),
        bin_index=int(temperature_bin),
    )
    latent_payload, latent = _sample_latent_payload(
        rng,
        cutoff=cutoff,
        audit_resolution=audit_resolution,
    )
    low = float(np.min(latent) - 20.0 * temperature)
    high = float(np.max(latent) + 20.0 * temperature)
    iterations = int(cfg.get("bias_bisection_iterations", 80))
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        mean = float(
            np.mean(
                1.0
                / (
                    1.0
                    + np.exp(
                        np.clip((mid - latent) / temperature, -60.0, 60.0)
                    )
                )
            )
        )
        if mean > target:
            low = mid
        else:
            high = mid
    bias = float(0.5 * (low + high))
    payload = {
        **latent_payload,
        "mapping": "logistic_bias_temperature_v1",
        "target_material_fraction": target,
        "temperature": temperature,
        "bias": bias,
        "bias_bisection_iterations": iterations,
    }
    spec = _build_spec(
        sample_id=sample_id,
        family="unified_fourier_graded",
        material_kind="continuous",
        payload=payload,
        metadata={
            "seed": seed,
            "seed_derivation": UNIFIED_FOURIER_SEED_VERSION,
            "material_fraction_bin": int(material_fraction_bin),
            "temperature_bin": int(temperature_bin),
        },
        epsilon_air=float(cfg.get("epsilon_air", 1.0)),
        epsilon_material=float(cfg.get("epsilon_material", 8.9)),
    )
    stats = _extended_stats(spec, audit_resolution=audit_resolution, latent=latent)
    tolerance = float(cfg.get("target_material_fraction_tolerance", 1.0e-5))
    residual = abs(float(stats["material_fraction"]) - target)
    stats["bias_target_residual"] = residual
    if residual > tolerance:
        raise RuntimeError("graded unified Fourier sample failed bias target gate")
    if float(stats["gray_fraction"]) < float(cfg.get("minimum_gray_fraction", 0.01)):
        raise RuntimeError("graded unified Fourier sample failed gray-fraction gate")
    return FullCellStructureSpec.from_mapping(
        {**spec.to_mapping(), "metadata": {**spec.metadata, "stats": stats}}
    )


__all__ = [
    "UNIFIED_FOURIER_ALGORITHM_VERSION",
    "UNIFIED_FOURIER_SEED_VERSION",
    "derive_full_cell_seed64",
    "generate_unified_fourier_binary",
    "generate_unified_fourier_graded",
]
