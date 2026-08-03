"""Parameter-driven full-cell material generators without symmetry projection.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.ndimage import label

from phc2d.data.full_cell import FullCellStructureSpec


def _stable_seed(master_seed: int, *parts: Any) -> int:
    raw = ":".join([str(int(master_seed)), *(str(item) for item in parts)])
    return int.from_bytes(hashlib.sha256(raw.encode("utf-8")).digest()[:8], "big") % (2**32)


def _numeric_parameters(value: Any, *, prefix: str = "") -> tuple[list[str], list[float]]:
    names: list[str] = []
    values: list[float] = []
    if isinstance(value, Mapping):
        for key in sorted(value):
            child_names, child_values = _numeric_parameters(value[key], prefix=f"{prefix}.{key}" if prefix else str(key))
            names.extend(child_names)
            values.extend(child_values)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            child_names, child_values = _numeric_parameters(item, prefix=f"{prefix}[{index}]")
            names.extend(child_names)
            values.extend(child_values)
    elif isinstance(value, (bool, int, float, np.integer, np.floating, np.bool_)):
        names.append(prefix)
        values.append(float(value))
    return names, values


def _real_fourier_modes(rng: np.random.Generator, *, cutoff: int, envelope_power: float) -> list[dict[str, float | int]]:
    modes: list[dict[str, float | int]] = []
    for mx in range(-int(cutoff), int(cutoff) + 1):
        for my in range(-int(cutoff), int(cutoff) + 1):
            if (mx, my) == (0, 0) or not (mx > 0 or (mx == 0 and my > 0)):
                continue
            radius = float(np.hypot(mx, my))
            scale = (1.0 + radius * radius) ** (-0.5 * float(envelope_power))
            modes.append(
                {
                    "mx": int(mx),
                    "my": int(my),
                    "cos": float(rng.normal(0.0, scale)),
                    "sin": float(rng.normal(0.0, scale)),
                }
            )
    return modes


def _field_from_modes(x: np.ndarray, y: np.ndarray, modes: Sequence[Mapping[str, Any]]) -> np.ndarray:
    field = np.zeros(np.broadcast_shapes(x.shape, y.shape), dtype=np.float64)
    x_arr, y_arr = np.broadcast_arrays(x, y)
    for row in modes:
        phase = 2.0 * np.pi * (int(row["mx"]) * x_arr + int(row["my"]) * y_arr)
        field += float(row["cos"]) * np.cos(phase) + float(row["sin"]) * np.sin(phase)
    return field


def _grid(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(resolution)
    axis = -0.5 + (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    return np.meshgrid(axis, axis, indexing="xy")


def _float_range(
    cfg: Mapping[str, Any],
    key: str,
    default: tuple[float, float],
) -> tuple[float, float]:
    raw = cfg.get(key, default)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
        raise ValueError(f"{key} must contain exactly two numeric bounds")
    low, high = float(raw[0]), float(raw[1])
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError(f"{key} must contain finite ascending bounds")
    return low, high


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
        metadata=dict(metadata),
    )


def _periodic_component_count(binary: np.ndarray) -> int:
    """Count four-neighbour components on a periodic square grid."""

    mask = np.asarray(binary, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("periodic component count expects a 2D grid")
    rows, cols = mask.shape
    parent: dict[int, int] = {}

    def root(value: int) -> int:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a: int, b: int) -> None:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[rb] = ra

    for row in range(rows):
        for col in range(cols):
            if not mask[row, col]:
                continue
            current = row * cols + col
            parent.setdefault(current, current)
            for nr, nc in ((row, (col + 1) % cols), ((row + 1) % rows, col)):
                if mask[nr, nc]:
                    union(current, nr * cols + nc)
    return len({root(value) for value in parent})


def _p4m_symmetry_error(grid: np.ndarray) -> float:
    array = np.asarray(grid, dtype=np.float64)
    variants = (
        np.rot90(array, 1),
        np.rot90(array, 2),
        np.rot90(array, 3),
        np.fliplr(array),
        np.flipud(array),
        array.T,
        np.flipud(np.rot90(array, 1)),
    )
    return float(np.mean([np.mean(np.abs(array - item)) for item in variants]))


def full_cell_structure_stats(spec: FullCellStructureSpec, *, audit_resolution: int = 256) -> dict[str, Any]:
    """Return generation-gate statistics derived from one canonical structure."""

    weight = spec.rasterize((audit_resolution, audit_resolution)).astype(np.float64)
    centered = weight - float(np.mean(weight))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    freq = np.fft.fftshift(np.fft.fftfreq(int(audit_resolution))) * int(audit_resolution)
    fx, fy = np.meshgrid(freq, freq, indexing="xy")
    high = np.hypot(fx, fy) > max(8.0, float(audit_resolution) / 8.0)
    total_energy = float(np.sum(spectrum))
    binary_air = weight < 0.5
    labels, n_components_plain = label(binary_air)
    component_sizes = [int(np.sum(labels == index)) for index in range(1, int(n_components_plain) + 1)]
    min_equivalent_diameter = (
        float(2.0 * np.sqrt(min(component_sizes) / np.pi) / float(audit_resolution)) if component_sizes else 0.0
    )
    return {
        "material_fraction": float(np.mean(weight)),
        "material_weight_p10": float(np.quantile(weight, 0.10)),
        "material_weight_p50": float(np.quantile(weight, 0.50)),
        "material_weight_p90": float(np.quantile(weight, 0.90)),
        "dense_material_fraction": float(np.mean(weight >= 0.60)),
        "air_fraction": float(np.mean(binary_air)),
        "gray_fraction": float(np.mean((weight > 0.05) & (weight < 0.95))),
        "p4m_symmetry_error": _p4m_symmetry_error(weight),
        "high_frequency_energy_ratio": float(np.sum(spectrum[high]) / max(total_energy, 1.0e-12)),
        "air_component_count_periodic": int(_periodic_component_count(binary_air)),
        "air_component_count_plain": int(n_components_plain),
        "min_air_component_equivalent_diameter_a": min_equivalent_diameter,
        "minimum": float(np.min(weight)),
        "maximum": float(np.max(weight)),
    }


def generate_graded_rbf_fourier(
    sample_id: str,
    *,
    master_seed: int,
    core_count: int,
    config: Mapping[str, Any] | None = None,
) -> FullCellStructureSpec:
    """Generate a smooth periodic graded medium from radial kernels and low modes."""

    cfg = dict(config or {})
    if int(core_count) not in {1, 2}:
        raise ValueError("graded_rbf_fourier core_count must be 1 or 2")
    rng = np.random.default_rng(_stable_seed(master_seed, sample_id, "graded_rbf_fourier"))
    epsilon_air = float(cfg.get("epsilon_air", 1.0))
    epsilon_material = float(cfg.get("epsilon_material", 8.9))
    cutoff = int(cfg.get("fourier_cutoff", 3))
    audit_resolution = int(cfg.get("audit_resolution", 256))
    max_attempts = int(cfg.get("max_attempts", 256))
    target_range = _float_range(cfg, "graded_target_material_fraction_range", (0.60, 0.82))
    temperature_range = _float_range(cfg, "graded_temperature_range", (0.20, 0.36))
    strength_range = _float_range(cfg, "graded_fourier_strength_range", (0.03, 0.14))
    for attempt in range(max_attempts):
        blobs: list[dict[str, float]] = []
        for _ in range(int(core_count)):
            base = float(rng.uniform(0.08, 0.22))
            aspect = float(rng.uniform(0.75, 1.33))
            blobs.append(
                {
                    "center_x": float(rng.uniform(-0.5, 0.5)),
                    "center_y": float(rng.uniform(-0.5, 0.5)),
                    "sigma_x": float(base * np.sqrt(aspect)),
                    "sigma_y": float(base / np.sqrt(aspect)),
                    "angle": float(rng.uniform(0.0, 2.0 * np.pi)),
                    "amplitude": float(rng.uniform(0.8, 1.2)),
                }
            )
        modes = _real_fourier_modes(rng, cutoff=cutoff, envelope_power=2.5)
        strength = float(rng.uniform(*strength_range))
        temperature = float(rng.uniform(*temperature_range))
        target = float(rng.uniform(*target_range))
        x, y = _grid(audit_resolution)
        raw_payload = {
            "blobs": blobs,
            "modes": modes,
            "fourier_strength": strength,
            "temperature": temperature,
            "bias": 0.0,
            "target_material_fraction": target,
        }
        probe = _build_spec(
            sample_id=sample_id,
            family="graded_rbf_fourier",
            material_kind="continuous",
            payload=raw_payload,
            metadata={},
            epsilon_air=epsilon_air,
            epsilon_material=epsilon_material,
        )
        raw = np.asarray(probe.evaluate_material_weight(x, y))
        # Evaluate the pre-sigmoid field by undoing the current logistic map.
        clipped = np.clip(raw, 1.0e-12, 1.0 - 1.0e-12)
        field = temperature * np.log(clipped / (1.0 - clipped))
        low = float(np.min(field) - 20.0 * temperature)
        high = float(np.max(field) + 20.0 * temperature)
        for _ in range(80):
            mid = 0.5 * (low + high)
            mean = float(np.mean(1.0 / (1.0 + np.exp(np.clip((mid - field) / temperature, -60.0, 60.0)))))
            if mean > target:
                low = mid
            else:
                high = mid
        payload = {**raw_payload, "bias": float(0.5 * (low + high))}
        spec = _build_spec(
            sample_id=sample_id,
            family="graded_rbf_fourier",
            material_kind="continuous",
            payload=payload,
            metadata={"core_count": int(core_count), "attempt": attempt},
            epsilon_air=epsilon_air,
            epsilon_material=epsilon_material,
        )
        stats = full_cell_structure_stats(spec, audit_resolution=audit_resolution)
        if (
            target_range[0] - 0.01 <= stats["material_fraction"] <= target_range[1] + 0.01
            and stats["dense_material_fraction"] >= 0.35
            and stats["gray_fraction"] >= 0.20
            and stats["high_frequency_energy_ratio"] <= 0.05
        ):
            return FullCellStructureSpec.from_mapping({**spec.to_mapping(), "metadata": {**spec.metadata, "stats": stats}})
    raise RuntimeError(f"Unable to generate accepted graded structure {sample_id} after {max_attempts} attempts")


def _sample_hole(rng: np.random.Generator, *, radius_range: tuple[float, float]) -> tuple[dict[str, Any], float]:
    radius = float(rng.uniform(*radius_range))
    modes = [
        {
            "order": order,
            "cos": float(rng.normal(0.0, 0.12 / float(order))),
            "sin": float(rng.normal(0.0, 0.12 / float(order))),
        }
        for order in range(2, 6)
    ]
    theta = np.linspace(-np.pi, np.pi, 2048, endpoint=False)
    perturbation = sum(
        float(row["cos"]) * np.cos(int(row["order"]) * theta)
        + float(row["sin"]) * np.sin(int(row["order"]) * theta)
        for row in modes
    )
    max_radius = float(radius * np.max(np.exp(perturbation)))
    return {"radius": radius, "modes": modes}, max_radius


def generate_binary_polar_holes(
    sample_id: str,
    *,
    master_seed: int,
    hole_count: int,
    config: Mapping[str, Any] | None = None,
) -> FullCellStructureSpec:
    """Generate one or two star-shaped air holes in a dielectric background."""

    cfg = dict(config or {})
    if int(hole_count) not in {1, 2}:
        raise ValueError("binary_polar_holes hole_count must be 1 or 2")
    rng = np.random.default_rng(_stable_seed(master_seed, sample_id, "binary_polar_holes"))
    epsilon_air = float(cfg.get("epsilon_air", 1.0))
    epsilon_material = float(cfg.get("epsilon_material", 8.9))
    audit_resolution = int(cfg.get("audit_resolution", 256))
    max_attempts = int(cfg.get("max_attempts", 512))
    for attempt in range(max_attempts):
        radius_range = (0.14, 0.25) if int(hole_count) == 1 else (0.10, 0.18)
        holes: list[dict[str, Any]] = []
        extents: list[float] = []
        for _ in range(int(hole_count)):
            hole, extent = _sample_hole(rng, radius_range=radius_range)
            margin = extent + 0.03
            if margin >= 0.49:
                break
            hole["center_x"] = float(rng.uniform(-0.5 + margin, 0.5 - margin))
            hole["center_y"] = float(rng.uniform(-0.5 + margin, 0.5 - margin))
            holes.append(hole)
            extents.append(extent)
        if len(holes) != int(hole_count):
            continue
        if int(hole_count) == 2:
            distance = float(np.hypot(holes[0]["center_x"] - holes[1]["center_x"], holes[0]["center_y"] - holes[1]["center_y"]))
            radius_sum = float(extents[0] + extents[1])
            if not 0.60 * radius_sum <= distance <= 1.50 * radius_sum:
                continue
        payload = {"holes": holes, "hole_count": int(hole_count)}
        spec = _build_spec(
            sample_id=sample_id,
            family="binary_polar_holes",
            material_kind="binary",
            payload=payload,
            metadata={"hole_count": int(hole_count), "attempt": attempt},
            epsilon_air=epsilon_air,
            epsilon_material=epsilon_material,
        )
        stats = full_cell_structure_stats(spec, audit_resolution=audit_resolution)
        if (
            0.10 <= stats["air_fraction"] <= 0.50
            and 1 <= stats["air_component_count_periodic"] <= int(hole_count)
            and stats["min_air_component_equivalent_diameter_a"] >= 0.04
        ):
            return FullCellStructureSpec.from_mapping({**spec.to_mapping(), "metadata": {**spec.metadata, "stats": stats}})
    raise RuntimeError(f"Unable to generate accepted polar-hole structure {sample_id} after {max_attempts} attempts")


def generate_binary_spectral_threshold(
    sample_id: str,
    *,
    master_seed: int,
    config: Mapping[str, Any] | None = None,
) -> FullCellStructureSpec:
    """Generate a periodic low-pass random field and threshold it to two media."""

    cfg = dict(config or {})
    rng = np.random.default_rng(_stable_seed(master_seed, sample_id, "binary_spectral_threshold"))
    epsilon_air = float(cfg.get("epsilon_air", 1.0))
    epsilon_material = float(cfg.get("epsilon_material", 8.9))
    audit_resolution = int(cfg.get("audit_resolution", 256))
    cutoff = int(cfg.get("fourier_cutoff", 6))
    max_attempts = int(cfg.get("max_attempts", 512))
    cutoff_scale_range = _float_range(cfg, "spectral_cutoff_scale_range", (2.5, 4.5))
    x, y = _grid(audit_resolution)
    for attempt in range(max_attempts):
        k_c = float(rng.uniform(*cutoff_scale_range))
        modes: list[dict[str, float | int]] = []
        for mx in range(-cutoff, cutoff + 1):
            for my in range(-cutoff, cutoff + 1):
                if (mx, my) == (0, 0) or not (mx > 0 or (mx == 0 and my > 0)):
                    continue
                radius = float(np.hypot(mx, my))
                scale = float(np.exp(-((radius / k_c) ** 4)))
                modes.append(
                    {
                        "mx": int(mx),
                        "my": int(my),
                        "cos": float(rng.normal(0.0, scale)),
                        "sin": float(rng.normal(0.0, scale)),
                    }
                )
        field = _field_from_modes(x, y, modes)
        std = float(np.std(field))
        if std <= 1.0e-12:
            continue
        for row in modes:
            row["cos"] = float(row["cos"] / std)
            row["sin"] = float(row["sin"] / std)
        field /= std
        target_air = float(rng.uniform(0.10, 0.50))
        quantile = float(np.clip(target_air + rng.uniform(-0.03, 0.03), 0.07, 0.53))
        threshold = float(np.quantile(field, quantile))
        payload = {
            "modes": modes,
            "threshold": threshold,
            "cutoff_scale": k_c,
            "target_air_fraction": target_air,
            "threshold_quantile": quantile,
        }
        spec = _build_spec(
            sample_id=sample_id,
            family="binary_spectral_threshold",
            material_kind="binary",
            payload=payload,
            metadata={"attempt": attempt},
            epsilon_air=epsilon_air,
            epsilon_material=epsilon_material,
        )
        stats = full_cell_structure_stats(spec, audit_resolution=audit_resolution)
        if (
            0.07 <= stats["air_fraction"] <= 0.53
            and 1 <= stats["air_component_count_periodic"] <= 3
            and stats["min_air_component_equivalent_diameter_a"] >= 0.04
        ):
            return FullCellStructureSpec.from_mapping({**spec.to_mapping(), "metadata": {**spec.metadata, "stats": stats}})
    raise RuntimeError(f"Unable to generate accepted spectral structure {sample_id} after {max_attempts} attempts")


__all__ = [
    "full_cell_structure_stats",
    "generate_binary_polar_holes",
    "generate_binary_spectral_threshold",
    "generate_graded_rbf_fourier",
]
