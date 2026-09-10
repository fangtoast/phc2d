"""Lazy MPB band-solver adapter for physical replay checks.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from phc2d.data.lift import GridShape, infer_square_grid_shape_from_wedge_length, lift_rho_to_grid


@dataclass(frozen=True)
class MpbBandSolverConfig:
    """Authoritative MPB parameters required for a band recomputation."""

    num_bands: int
    resolution: int
    polarization: str
    epsilon_air: float
    epsilon_material: float
    lattice_constant: float = 1.0
    eigensolver_tolerance: float | None = None
    grid_shape: GridShape | None = None
    band_guard_count: int = 0
    band_order_policy: str = "frequency_ascending"
    field_band_index_one_based: int | None = None
    field_band_indices_one_based: tuple[int, ...] | None = None


@dataclass(frozen=True)
class MpbEigenfieldResult:
    """One real-MPB Bloch eigenfield with its canonical/solver band mapping."""

    eigenfrequency: float
    efield: np.ndarray
    hfield: np.ndarray
    requested_band_index_one_based: int
    solver_band_index_one_based: int
    k_point: tuple[float, ...]
    metadata: dict[str, Any]


def load_mpb_modules() -> tuple[Any, Any]:
    """Import Meep/MPB only when a real MPB solve is requested."""

    try:
        import meep as mp  # type: ignore
        from meep import mpb  # type: ignore
    except Exception as exc:  # pragma: no cover - only exercised when MPB is absent
        raise ImportError("Real MPB validation requires Meep/MPB Python bindings.") from exc
    try:
        mp.verbosity(0)
    except Exception:
        pass
    return mp, mpb


def build_block_geometry_from_rho(
    mp: Any,
    rho_wedge: np.ndarray,
    *,
    eps_air: float,
    eps_material: float,
    grid_shape: GridShape | None = None,
) -> tuple[list[Any], Any, dict[str, Any]]:
    """Build the historical pixel-block MPB geometry from a p4m rho wedge."""

    rho = np.asarray(rho_wedge, dtype=np.float64).reshape(-1)
    resolved_grid_shape = grid_shape or infer_square_grid_shape_from_wedge_length(int(rho.shape[0]))
    grid = lift_rho_to_grid(rho, grid_shape=resolved_grid_shape)
    rows, cols = int(resolved_grid_shape[0]), int(resolved_grid_shape[1])
    if rows != cols:
        raise ValueError(f"MPB square block geometry requires square grid, got {resolved_grid_shape}")
    cell = 1.0 / float(rows)
    blocks: list[Any] = []
    for i in range(rows):
        for j in range(cols):
            if float(grid[i, j]) <= 0.5:
                continue
            x = -0.5 + (j + 0.5) * cell
            y = -0.5 + (i + 0.5) * cell
            blocks.append(
                mp.Block(
                    center=mp.Vector3(float(x), float(y), 0.0),
                    size=mp.Vector3(float(cell), float(cell), mp.inf),
                    material=mp.Medium(epsilon=float(eps_material)),
                )
            )
    metadata = {
        "grid_shape": [rows, cols],
        "cell_size": float(cell),
        "occupied_blocks": int(len(blocks)),
        "geometry_convention": "p4m_wedge_binary_pixels_as_mpb_blocks_air_background",
    }
    return blocks, mp.Medium(epsilon=float(eps_air)), metadata


def _normalize_freqs(freqs: np.ndarray, n_k: int, n_bands: int, *, band_order_policy: str = "frequency_ascending") -> np.ndarray:
    arr = np.asarray(freqs, dtype=np.float64)
    if arr.ndim != 2:
        raise RuntimeError(f"Unexpected MPB frequency array shape: {arr.shape}")
    # MPB returns [k, band]. Accept only an unambiguous transposed adapter
    # result; a square array must remain in MPB's native orientation.
    if arr.shape[0] != int(n_k) and arr.shape[1] == int(n_k):
        arr = arr.T
    if arr.shape[0] != int(n_k):
        raise RuntimeError(f"Unexpected MPB frequency array shape after normalization: {arr.shape}")
    if arr.shape[1] < int(n_bands):
        raise RuntimeError(f"MPB returned {arr.shape[1]} bands, expected at least {n_bands}")
    policy = str(band_order_policy or "frequency_ascending").lower()
    if policy in {"frequency_ascending", "ascending_frequency", "sorted_frequency"}:
        arr = np.sort(arr, axis=1)
    elif policy in {"mpb_output", "as_returned"}:
        arr = arr.copy()
    else:
        raise ValueError(f"Unsupported MPB band_order_policy: {band_order_policy!r}")
    return arr[:, : int(n_bands)]


def solve_bands_for_rho_mpb(
    rho_wedge: np.ndarray,
    *,
    k_points: Sequence[Sequence[float]],
    config: MpbBandSolverConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run a real MPB TE/TM solve for one rho sample and return ``(Nk, Nb)`` bands."""

    if config.eigensolver_tolerance is None:
        raise ValueError("eigensolver_tolerance is required for formal physical validation.")
    mp, mpb = load_mpb_modules()
    geometry, default_material, geometry_metadata = build_block_geometry_from_rho(
        mp,
        np.asarray(rho_wedge, dtype=np.float64),
        eps_air=float(config.epsilon_air),
        eps_material=float(config.epsilon_material),
        grid_shape=config.grid_shape,
    )
    mp_k_points = [
        mp.Vector3(float(row[0]), float(row[1]), 0.0)
        for row in np.asarray(k_points, dtype=np.float64)
    ]
    requested_num_bands = int(config.num_bands)
    band_guard_count = max(0, int(config.band_guard_count))
    solved_num_bands = requested_num_bands + band_guard_count
    solver_kwargs = {
        "num_bands": solved_num_bands,
        "k_points": mp_k_points,
        "geometry": geometry,
        "geometry_lattice": mp.Lattice(size=mp.Vector3(1.0, 1.0, 0.0)),
        "default_material": default_material,
        "resolution": int(config.resolution),
        "tolerance": float(config.eigensolver_tolerance),
    }
    try:
        mode_solver = mpb.ModeSolver(**solver_kwargs)
    except TypeError as exc:  # pragma: no cover - depends on installed MPB binding signature
        raise TypeError("Installed MPB ModeSolver did not accept explicit eigensolver tolerance.") from exc

    polarization = str(config.polarization).upper()
    if polarization == "TE":
        mode_solver.run_te()
    elif polarization == "TM":
        mode_solver.run_tm()
    else:
        mode_solver.run()
    band_order_policy = str(config.band_order_policy or "frequency_ascending")
    solved_bands = _normalize_freqs(
        np.asarray(mode_solver.all_freqs, dtype=np.float64),
        len(mp_k_points),
        solved_num_bands,
        band_order_policy=band_order_policy,
    )
    bands = solved_bands[:, :requested_num_bands]
    metadata = {
        **geometry_metadata,
        "solver_backend": "meep.mpb.ModeSolver",
        "solver_version": {
            "meep": str(getattr(mp, "__version__", "unknown")),
            "mpb": str(getattr(mpb, "__version__", "unknown")),
        },
        "mpb_backend": "meep.mpb.ModeSolver",
        "polarization": polarization,
        "resolution": int(config.resolution),
        "num_bands": requested_num_bands,
        "band_guard_count": band_guard_count,
        "band_order_policy": band_order_policy,
        "solved_num_bands": solved_num_bands,
        "eigensolver_tolerance": float(config.eigensolver_tolerance),
        "frequency_normalization": "MPB dimensionless frequency a/lambda, a=1",
    }
    return bands, metadata


def solve_eigenfield_for_rho_mpb(
    rho_wedge: np.ndarray,
    *,
    k_point: Sequence[float],
    band_index_one_based: int,
    config: MpbBandSolverConfig,
) -> MpbEigenfieldResult:
    """Solve one Bloch eigenmode and retain the exact MPB field-band lineage.

    MPB field access uses its original one-based band index.  The project,
    however, defines retained bands by ascending frequency.  This function
    records both indices so a sorted-frequency contract never silently reads
    a field from the wrong MPB mode.
    """

    if config.eigensolver_tolerance is None:
        raise ValueError("eigensolver_tolerance is required for real-MPB eigenfield generation.")
    requested_band = int(band_index_one_based)
    if requested_band < 1 or requested_band > int(config.num_bands):
        raise ValueError(f"band_index_one_based={requested_band} is outside 1..{config.num_bands}")

    mp, mpb = load_mpb_modules()
    geometry, default_material, geometry_metadata = build_block_geometry_from_rho(
        mp,
        np.asarray(rho_wedge, dtype=np.float64),
        eps_air=float(config.epsilon_air),
        eps_material=float(config.epsilon_material),
        grid_shape=config.grid_shape,
    )
    point = tuple(float(value) for value in np.asarray(k_point, dtype=np.float64).reshape(-1))
    if len(point) != 2:
        raise ValueError(f"k_point must contain two reduced coordinates, got {point}")
    solved_num_bands = int(config.num_bands) + max(0, int(config.band_guard_count))
    mode_solver = mpb.ModeSolver(
        num_bands=solved_num_bands,
        k_points=[mp.Vector3(point[0], point[1], 0.0)],
        geometry=geometry,
        geometry_lattice=mp.Lattice(size=mp.Vector3(1.0, 1.0, 0.0)),
        default_material=default_material,
        resolution=int(config.resolution),
        tolerance=float(config.eigensolver_tolerance),
    )
    polarization = str(config.polarization).upper()
    if polarization == "TE":
        mode_solver.run_te()
    elif polarization == "TM":
        mode_solver.run_tm()
    else:
        raise ValueError(f"Bloch eigenfield generation requires TE or TM polarization, got {config.polarization!r}")

    raw_freqs = _normalize_freqs(
        np.asarray(mode_solver.all_freqs, dtype=np.float64),
        n_k=1,
        n_bands=solved_num_bands,
        band_order_policy="mpb_output",
    )[0]
    order = np.argsort(raw_freqs, kind="stable")
    canonical_position = requested_band - 1
    solver_band_zero_based = int(order[canonical_position])
    solver_band_one_based = solver_band_zero_based + 1
    try:
        efield = np.asarray(mode_solver.get_efield(solver_band_one_based, bloch_phase=True), dtype=np.complex128)
        hfield = np.asarray(mode_solver.get_hfield(solver_band_one_based, bloch_phase=True), dtype=np.complex128)
    except TypeError as exc:  # pragma: no cover - depends on the installed MPB binding
        raise TypeError("Installed MPB binding must support bloch_phase=True for the field contract.") from exc
    if efield.size == 0 or hfield.size == 0:
        raise RuntimeError("MPB returned an empty E or H eigenfield.")

    metadata = {
        **geometry_metadata,
        "mpb_backend": "meep.mpb.ModeSolver",
        "solver_backend": "meep.mpb.ModeSolver",
        "solver_version": {
            "meep": str(getattr(mp, "__version__", "unknown")),
            "mpb": str(getattr(mpb, "__version__", "unknown")),
        },
        "mpb_version": str(getattr(mpb, "__version__", "unknown")),
        "polarization": polarization,
        "resolution": int(config.resolution),
        "material_parameters": {
            "epsilon_air": float(config.epsilon_air),
            "epsilon_material": float(config.epsilon_material),
        },
        "num_bands": int(config.num_bands),
        "band_guard_count": max(0, int(config.band_guard_count)),
        "band_order_policy": "frequency_ascending",
        "solved_num_bands": solved_num_bands,
        "eigensolver_tolerance": float(config.eigensolver_tolerance),
        "frequency_normalization": "MPB dimensionless frequency a/lambda, a=1",
        "k_point_reduced": list(point),
        "bloch_phase_applied_by_mpb": True,
        "requested_band_index_one_based": requested_band,
        "solver_band_index_one_based": solver_band_one_based,
        "raw_solver_frequencies": [float(value) for value in raw_freqs],
        "field_component_order": "mpb_cartesian_xyz",
        "efield_shape": [int(value) for value in efield.shape],
        "hfield_shape": [int(value) for value in hfield.shape],
    }
    return MpbEigenfieldResult(
        eigenfrequency=float(raw_freqs[solver_band_zero_based]),
        efield=efield,
        hfield=hfield,
        requested_band_index_one_based=requested_band,
        solver_band_index_one_based=solver_band_one_based,
        k_point=point,
        metadata=metadata,
    )


def solve_eigenfield_bundle_for_rho_mpb(
    rho_wedge: np.ndarray,
    *,
    k_points: Sequence[Sequence[float]],
    config: MpbBandSolverConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Adapt one eigenfield solve to the resume-aware MPB worker interface.

    The first array axis is ``(E, H)``; spatial and component axes remain
    exactly as returned by MPB and are never assumed by the worker layer.
    """

    points = np.asarray(k_points, dtype=np.float64)
    requested_band = config.field_band_index_one_based
    if points.shape != (1, 2):
        raise ValueError("An eigenfield worker job must provide exactly one two-dimensional k-point.")
    if requested_band is None:
        raise ValueError("field_band_index_one_based is required for an eigenfield worker job.")
    result = solve_eigenfield_for_rho_mpb(
        rho_wedge,
        k_point=points[0],
        band_index_one_based=int(requested_band),
        config=config,
    )
    if result.efield.shape != result.hfield.shape:
        raise RuntimeError(f"MPB returned incompatible E/H field schemas: {result.efield.shape} versus {result.hfield.shape}")
    return np.stack([result.efield, result.hfield], axis=0), {
        **result.metadata,
        "eigenfrequency": float(result.eigenfrequency),
        "requested_band_index_one_based": int(result.requested_band_index_one_based),
        "solver_band_index_one_based": int(result.solver_band_index_one_based),
        "worker_payload": "stacked_efield_hfield_complex",
    }


def solve_eigenfield_multiband_bundle_for_rho_mpb(
    rho_wedge: np.ndarray,
    *,
    k_points: Sequence[Sequence[float]],
    config: MpbBandSolverConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve one structure/k pair once and return all requested eigenfields.

    The worker payload has shape ``(band, E_or_H, spatial..., component)``.
    Frequencies and the canonical-to-solver band mapping are stored in the
    matching order in the returned metadata.
    """

    if config.eigensolver_tolerance is None:
        raise ValueError("eigensolver_tolerance is required for real-MPB eigenfield generation.")
    points = np.asarray(k_points, dtype=np.float64)
    if points.shape != (1, 2):
        raise ValueError("A multiband eigenfield job requires exactly one two-dimensional k-point.")
    requested = tuple(int(value) for value in (config.field_band_indices_one_based or ()))
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("field_band_indices_one_based must contain unique requested bands.")
    if min(requested) < 1 or max(requested) > int(config.num_bands):
        raise ValueError("Requested field band lies outside the configured MPB band range.")

    mp, mpb = load_mpb_modules()
    geometry, default_material, geometry_metadata = build_block_geometry_from_rho(
        mp,
        np.asarray(rho_wedge, dtype=np.float64),
        eps_air=float(config.epsilon_air),
        eps_material=float(config.epsilon_material),
        grid_shape=config.grid_shape,
    )
    point = (float(points[0, 0]), float(points[0, 1]))
    solved_num_bands = int(config.num_bands) + max(0, int(config.band_guard_count))
    mode_solver = mpb.ModeSolver(
        num_bands=solved_num_bands,
        k_points=[mp.Vector3(point[0], point[1], 0.0)],
        geometry=geometry,
        geometry_lattice=mp.Lattice(size=mp.Vector3(1.0, 1.0, 0.0)),
        default_material=default_material,
        resolution=int(config.resolution),
        tolerance=float(config.eigensolver_tolerance),
    )
    polarization = str(config.polarization).upper()
    if polarization == "TE":
        mode_solver.run_te()
    elif polarization == "TM":
        mode_solver.run_tm()
    else:
        raise ValueError(f"Bloch eigenfield generation requires TE or TM polarization, got {config.polarization!r}")

    raw_freqs = _normalize_freqs(
        np.asarray(mode_solver.all_freqs, dtype=np.float64),
        n_k=1,
        n_bands=solved_num_bands,
        band_order_policy="mpb_output",
    )[0]
    order = np.argsort(raw_freqs, kind="stable")
    payloads: list[np.ndarray] = []
    frequencies: list[float] = []
    solver_indices: list[int] = []
    for canonical_band in requested:
        solver_zero = int(order[canonical_band - 1])
        solver_one = solver_zero + 1
        efield = np.asarray(mode_solver.get_efield(solver_one, bloch_phase=True), dtype=np.complex128)
        hfield = np.asarray(mode_solver.get_hfield(solver_one, bloch_phase=True), dtype=np.complex128)
        if efield.shape != hfield.shape or efield.size == 0:
            raise RuntimeError("MPB returned incompatible or empty multiband E/H fields.")
        payloads.append(np.stack([efield, hfield], axis=0))
        frequencies.append(float(raw_freqs[solver_zero]))
        solver_indices.append(solver_one)

    metadata = {
        **geometry_metadata,
        "mpb_backend": "meep.mpb.ModeSolver",
        "solver_backend": "meep.mpb.ModeSolver",
        "solver_version": {"meep": str(getattr(mp, "__version__", "unknown")), "mpb": str(getattr(mpb, "__version__", "unknown"))},
        "mpb_version": str(getattr(mpb, "__version__", "unknown")),
        "polarization": polarization,
        "resolution": int(config.resolution),
        "material_parameters": {"epsilon_air": float(config.epsilon_air), "epsilon_material": float(config.epsilon_material)},
        "num_bands": int(config.num_bands),
        "band_guard_count": max(0, int(config.band_guard_count)),
        "band_order_policy": "frequency_ascending",
        "solved_num_bands": solved_num_bands,
        "eigensolver_tolerance": float(config.eigensolver_tolerance),
        "frequency_normalization": "MPB dimensionless frequency a/lambda, a=1",
        "k_point_reduced": list(point),
        "bloch_phase_applied_by_mpb": True,
        "requested_band_indices_one_based": list(requested),
        "solver_band_indices_one_based": solver_indices,
        "eigenfrequencies": frequencies,
        "raw_solver_frequencies": [float(value) for value in raw_freqs],
        "field_component_order": "mpb_cartesian_xyz",
        "worker_payload": "band_stacked_efield_hfield_complex",
    }
    return np.stack(payloads, axis=0).astype(np.complex64), metadata


__all__ = [
    "MpbBandSolverConfig",
    "MpbEigenfieldResult",
    "build_block_geometry_from_rho",
    "load_mpb_modules",
    "solve_bands_for_rho_mpb",
    "solve_eigenfield_bundle_for_rho_mpb",
    "solve_eigenfield_multiband_bundle_for_rho_mpb",
    "solve_eigenfield_for_rho_mpb",
]
