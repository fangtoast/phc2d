"""Real-MPB solvers for analytic full-cell material distributions.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from phc2d.data.full_cell import FullCellStructureSpec
from phc2d.solvers.mpb import MpbBandSolverConfig, _normalize_freqs, load_mpb_modules


@dataclass(frozen=True)
class FullCellMpbBandSolverConfig(MpbBandSolverConfig):
    """Full-cell-only MPB options kept out of legacy protocol hashes."""

    mesh_size: int = 3


def _run_mode_solver(
    *,
    k_points: Sequence[Sequence[float]],
    config: FullCellMpbBandSolverConfig,
    geometry: Sequence[Any],
    default_material: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    if config.eigensolver_tolerance is None:
        raise ValueError("eigensolver_tolerance is required for real-MPB full-cell generation")
    if int(config.mesh_size) <= 0:
        raise ValueError("mesh_size must be positive")
    mp, mpb = load_mpb_modules()
    points = np.asarray(k_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("k_points must have shape [n_k_points, 2]")
    mp_points = [mp.Vector3(float(row[0]), float(row[1]), 0.0) for row in points]
    requested = int(config.num_bands)
    guard = max(0, int(config.band_guard_count))
    solved = requested + guard
    mode_solver = mpb.ModeSolver(
        num_bands=solved,
        k_points=mp_points,
        geometry=list(geometry),
        geometry_lattice=mp.Lattice(size=mp.Vector3(1.0, 1.0, 0.0)),
        default_material=default_material,
        resolution=int(config.resolution),
        mesh_size=int(config.mesh_size),
        tolerance=float(config.eigensolver_tolerance),
    )
    polarization = str(config.polarization).upper()
    if polarization == "TE":
        mode_solver.run_te()
    elif polarization == "TM":
        mode_solver.run_tm()
    else:
        mode_solver.run()
    frequencies = _normalize_freqs(
        np.asarray(mode_solver.all_freqs, dtype=np.float64),
        len(mp_points),
        solved,
        band_order_policy=str(config.band_order_policy),
    )[:, :requested]
    metadata = {
        "solver_backend": "meep.mpb.ModeSolver",
        "solver_version": {
            "meep": str(getattr(mp, "__version__", "unknown")),
            "mpb": str(getattr(mpb, "__version__", "unknown")),
        },
        "polarization": polarization,
        "resolution": int(config.resolution),
        "mesh_size": int(config.mesh_size),
        "num_bands": requested,
        "band_guard_count": guard,
        "solved_num_bands": solved,
        "band_order_policy": str(config.band_order_policy),
        "eigensolver_tolerance": float(config.eigensolver_tolerance),
        "frequency_normalization": "MPB dimensionless frequency a/lambda, a=1",
    }
    return frequencies.astype(np.float32), metadata


def build_full_cell_material_function(mp: Any, structure: FullCellStructureSpec):
    """Build a lazy MPB material function from a serializable structure spec."""

    def material(position: Any):
        epsilon = structure.epsilon_at(float(position.x), float(position.y))
        return mp.Medium(epsilon=float(epsilon))

    return material


def solve_bands_for_full_cell_mpb(
    structure: FullCellStructureSpec,
    *,
    k_points: Sequence[Sequence[float]],
    config: FullCellMpbBandSolverConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve one analytic periodic full-cell structure using real MPB."""

    if not isinstance(structure, FullCellStructureSpec):
        if isinstance(structure, dict):
            structure = FullCellStructureSpec.from_mapping(structure)
        else:
            raise TypeError("full-cell MPB solver expects FullCellStructureSpec or mapping")
    mp, _ = load_mpb_modules()
    material = build_full_cell_material_function(mp, structure)
    bands, metadata = _run_mode_solver(
        k_points=k_points,
        config=config,
        geometry=[],
        default_material=material,
    )
    return bands, {
        **metadata,
        "geometry_convention": "analytic_periodic_full_cell_material_function",
        "geometry_hash": structure.geometry_hash,
        "sample_id": structure.sample_id,
        "generator_family": structure.generator_family,
        "material_kind": structure.material_kind,
        "epsilon_air": float(structure.epsilon_air),
        "epsilon_material": float(structure.epsilon_material),
    }


def solve_uniform_medium_mpb(
    epsilon: float,
    *,
    k_points: Sequence[Sequence[float]],
    config: FullCellMpbBandSolverConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reference solve using MPB's native uniform default material."""

    mp, _ = load_mpb_modules()
    bands, metadata = _run_mode_solver(
        k_points=k_points,
        config=config,
        geometry=[],
        default_material=mp.Medium(epsilon=float(epsilon)),
    )
    return bands, {**metadata, "geometry_convention": "native_uniform_medium", "epsilon": float(epsilon)}


def solve_circular_air_hole_mpb(
    *,
    center: Sequence[float],
    radius: float,
    k_points: Sequence[Sequence[float]],
    config: FullCellMpbBandSolverConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reference solve using one native MPB cylinder in a dielectric background."""

    mp, _ = load_mpb_modules()
    center_values = tuple(float(value) for value in center)
    if len(center_values) != 2:
        raise ValueError("circle center must contain two coordinates")
    geometry = [
        mp.Cylinder(
            radius=float(radius),
            center=mp.Vector3(center_values[0], center_values[1], 0.0),
            material=mp.Medium(epsilon=float(config.epsilon_air)),
        )
    ]
    bands, metadata = _run_mode_solver(
        k_points=k_points,
        config=config,
        geometry=geometry,
        default_material=mp.Medium(epsilon=float(config.epsilon_material)),
    )
    return bands, {
        **metadata,
        "geometry_convention": "native_circular_air_hole",
        "center": list(center_values),
        "radius": float(radius),
    }


__all__ = [
    "FullCellMpbBandSolverConfig",
    "build_full_cell_material_function",
    "solve_bands_for_full_cell_mpb",
    "solve_circular_air_hole_mpb",
    "solve_uniform_medium_mpb",
]
