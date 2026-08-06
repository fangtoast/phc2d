"""Finite-array Meep scattering responses for analytic full-cell structures.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from phc2d.data.full_cell import FullCellStructureSpec


@dataclass(frozen=True)
class MeepScatteringConfig:
    periods_x: int
    periods_y: int
    resolution: int
    pml_thickness: float
    monitor_padding: float
    source_gap: float
    fractional_bandwidth: float
    dft_decay_tolerance: float
    dft_minimum_run_time: float
    dft_maximum_run_time: float
    require_decay_before_max: bool
    farfield_radius_wavelengths: float
    farfield_angle_count: int
    greencyl_tolerance: float = 1.0e-6

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MeepScatteringConfig":
        return cls(**{key: value[key] for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class MeepScatteringResult:
    frequency: float
    x: np.ndarray
    y: np.ndarray
    epsilon: np.ndarray
    electric_energy: np.ndarray
    poynting_x: np.ndarray
    poynting_y: np.ndarray
    far_angles_rad: np.ndarray
    far_radial_flux: np.ndarray
    runtime_s: float
    metadata: dict[str, Any]


def _validate_config(config: MeepScatteringConfig) -> None:
    if min(config.periods_x, config.periods_y, config.resolution, config.farfield_angle_count) <= 0:
        raise ValueError("period counts, resolution, and farfield_angle_count must be positive")
    if min(config.pml_thickness, config.monitor_padding, config.source_gap) <= 0.0:
        raise ValueError("PML, monitor padding, and source gap must be positive")
    if not 0.0 < config.fractional_bandwidth < 1.0:
        raise ValueError("fractional_bandwidth must lie in (0, 1)")
    if not 0.0 < config.dft_decay_tolerance < 1.0:
        raise ValueError("dft_decay_tolerance must lie in (0, 1)")


def _radial_poynting_from_farfield(fields: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Return outward radial time-averaged Poynting flux from Meep far fields."""

    values = np.asarray(fields, dtype=np.complex128)
    theta = np.asarray(angles, dtype=np.float64)
    if values.shape != (theta.size, 6):
        raise ValueError(f"far fields must have shape [n_angles, 6], got {values.shape}")
    ex, ey, ez, hx, hy, hz = (values[:, index] for index in range(6))
    sx = 0.5 * np.real(ey * np.conj(hz) - ez * np.conj(hy))
    sy = 0.5 * np.real(ez * np.conj(hx) - ex * np.conj(hz))
    return sx * np.cos(theta) + sy * np.sin(theta)


def solve_finite_array_scattering_meep(
    structure: FullCellStructureSpec,
    *,
    frequency: float,
    config: MeepScatteringConfig,
) -> MeepScatteringResult:
    """Solve a normally incident +x TE (Hz) pulse for a finite tiled array."""

    _validate_config(config)
    if not isinstance(structure, FullCellStructureSpec):
        structure = FullCellStructureSpec.from_mapping(structure)
    fcen = float(frequency)
    if not np.isfinite(fcen) or fcen <= 0.0:
        raise ValueError("frequency must be finite and positive")

    import meep as mp

    started = time.perf_counter()
    span_x, span_y = float(config.periods_x), float(config.periods_y)
    air_shell = float(config.monitor_padding + config.source_gap)
    cell_x = span_x + 2.0 * (air_shell + config.pml_thickness)
    cell_y = span_y + 2.0 * (config.monitor_padding + config.pml_thickness)
    monitor_x = span_x + 2.0 * config.monitor_padding
    monitor_y = span_y + 2.0 * config.monitor_padding
    source_x = -0.5 * span_x - config.monitor_padding - 0.5 * config.source_gap

    def finite_material(position: Any) -> Any:
        x, y = float(position.x), float(position.y)
        if abs(x) <= 0.5 * span_x and abs(y) <= 0.5 * span_y:
            return mp.Medium(epsilon=structure.epsilon_at(x, y))
        return mp.air

    sim = mp.Simulation(
        cell_size=mp.Vector3(cell_x, cell_y),
        boundary_layers=[mp.PML(config.pml_thickness)],
        sources=[
            mp.Source(
                src=mp.GaussianSource(fcen, fwidth=config.fractional_bandwidth * fcen, is_integrated=True),
                component=mp.Hz,
                center=mp.Vector3(source_x, 0.0),
                size=mp.Vector3(0.0, monitor_y),
            )
        ],
        default_material=finite_material,
        resolution=config.resolution,
        dimensions=2,
    )
    half_x, half_y = 0.5 * monitor_x, 0.5 * monitor_y
    near2far = sim.add_near2far(
        fcen,
        0.0,
        1,
        mp.Near2FarRegion(center=mp.Vector3(0.0, half_y), size=mp.Vector3(monitor_x, 0.0)),
        mp.Near2FarRegion(center=mp.Vector3(0.0, -half_y), size=mp.Vector3(monitor_x, 0.0), weight=-1.0),
        mp.Near2FarRegion(center=mp.Vector3(half_x, 0.0), size=mp.Vector3(0.0, monitor_y)),
        mp.Near2FarRegion(center=mp.Vector3(-half_x, 0.0), size=mp.Vector3(0.0, monitor_y), weight=-1.0),
    )
    dft = sim.add_dft_fields(
        [mp.Ex, mp.Ey, mp.Hz],
        fcen,
        0.0,
        1,
        center=mp.Vector3(),
        size=mp.Vector3(monitor_x, monitor_y),
    )
    sim.run(
        until_after_sources=mp.stop_when_dft_decayed(
            tol=config.dft_decay_tolerance,
            minimum_run_time=config.dft_minimum_run_time,
            maximum_run_time=config.dft_maximum_run_time,
        )
    )
    simulation_end_time = float(sim.meep_time())
    decay_reached_before_max = simulation_end_time < config.dft_maximum_run_time - 1.0 / config.resolution
    if config.require_decay_before_max and not decay_reached_before_max:
        raise RuntimeError(
            f"Meep DFT fields did not satisfy the decay gate before maximum_run_time={config.dft_maximum_run_time}"
        )

    ex = np.asarray(sim.get_dft_array(dft, mp.Ex, 0), dtype=np.complex128)
    ey = np.asarray(sim.get_dft_array(dft, mp.Ey, 0), dtype=np.complex128)
    hz = np.asarray(sim.get_dft_array(dft, mp.Hz, 0), dtype=np.complex128)
    epsilon = np.asarray(
        sim.get_array(center=mp.Vector3(), size=mp.Vector3(monitor_x, monitor_y), component=mp.Dielectric),
        dtype=np.float64,
    )
    x, y, _, _ = sim.get_array_metadata(dft_cell=dft)
    if not (ex.shape == ey.shape == hz.shape == epsilon.shape):
        raise RuntimeError(f"Meep response arrays are misaligned: {ex.shape}, {ey.shape}, {hz.shape}, {epsilon.shape}")
    electric_energy = 0.25 * epsilon * (np.abs(ex) ** 2 + np.abs(ey) ** 2)
    poynting_x = 0.5 * np.real(ey * np.conj(hz))
    poynting_y = -0.5 * np.real(ex * np.conj(hz))

    angles = np.linspace(-math.pi, math.pi, config.farfield_angle_count, endpoint=False, dtype=np.float64)
    radius = config.farfield_radius_wavelengths / fcen
    fields = np.asarray(
        [
            sim.get_farfield(
                near2far,
                mp.Vector3(radius * math.cos(theta), radius * math.sin(theta)),
                config.greencyl_tolerance,
            )
            for theta in angles
        ],
        dtype=np.complex128,
    )
    radial = _radial_poynting_from_farfield(fields, angles)
    negative_fraction = float(np.mean(radial < 0.0))
    radial = np.maximum(radial, 0.0)
    runtime_s = time.perf_counter() - started
    metadata = {
        "solver_backend": "meep.Simulation",
        "meep_version": str(getattr(mp, "__version__", "unknown")),
        "problem": "finite_periodic_array_plane_wave_scattering",
        "polarization": "2d_te_hz",
        "incident_direction": "+x",
        "frequency_a_over_lambda": fcen,
        "periods": [config.periods_x, config.periods_y],
        "resolution_pixels_per_a": config.resolution,
        "pml_thickness_a": config.pml_thickness,
        "near2far_surface": "closed_box_in_air",
        "farfield_radius_a": radius,
        "farfield_negative_flux_fraction_before_clipping": negative_fraction,
        "source_inside_near2far_box": False,
        "simulation_end_time": simulation_end_time,
        "dft_decay_reached_before_max": decay_reached_before_max,
        "geometry_hash": structure.geometry_hash,
        "response_config": asdict(config),
    }
    sim.reset_meep()
    return MeepScatteringResult(
        frequency=fcen,
        x=np.asarray(x, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        epsilon=epsilon.T,
        electric_energy=electric_energy.T,
        poynting_x=poynting_x.T,
        poynting_y=poynting_y.T,
        far_angles_rad=angles,
        far_radial_flux=radial,
        runtime_s=runtime_s,
        metadata=metadata,
    )


__all__ = ["MeepScatteringConfig", "MeepScatteringResult", "solve_finite_array_scattering_meep"]
