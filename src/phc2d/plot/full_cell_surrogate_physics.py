"""Evidence-grounded Full-Wh spectral and downstream response sensitivity workflow.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
import math
import os
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from phc2d.data.full_cell import FullCellStructureSpec
from phc2d.plot.full_cell_gallery import normalized_electric_energy_density
from phc2d.plot.full_cell_surrogate_figures import render_eigenfield_plates, render_response_plates
from phc2d.plot.full_cell_surrogate_spectral import prepare_spectral_evidence as _prepare_spectral
from phc2d.plot.full_cell_surrogate_support import (
    SOURCES,
    decode_text as _decode,
    hash_file as _hash,
    jsonable as _jsonable,
    load_surrogate_physics_config as _load_config,
    paired_bootstrap_mean_difference,
    read_csv_rows as _read_csv,
    relative_path as _relative,
    repo_path as _repo_path,
    select_quantile_positions,
    write_csv_rows as _write_csv,
    write_history as _write_history,
)
from phc2d.runs import (
    ensure_figures_dir,
    make_run_dir,
    setup_logging,
    write_json,
    write_metrics,
    write_resolved_config,
    write_results,
    write_run_meta,
)
from phc2d.solvers import FullCellMpbBandSolverConfig, MpbEigenfieldResult, solve_eigenfield_for_full_cell_mpb
from phc2d.solvers.meep_scattering import (
    MeepScatteringConfig,
    MeepScatteringResult,
    solve_finite_array_scattering_meep,
)


EigenfieldSolverFn = Callable[..., MpbEigenfieldResult]
ScatteringSolverFn = Callable[..., MeepScatteringResult]


def _circular_angle_difference_deg(left: float, right: float) -> float:
    return abs(math.degrees(math.atan2(math.sin(left - right), math.cos(left - right))))


def response_difference_metrics(truth: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    """Quantify within-structure response-shape differences without absolute-power claims."""

    truth_near = np.asarray(truth["electric_energy"], dtype=np.float64)
    candidate_near = np.asarray(candidate["electric_energy"], dtype=np.float64)
    if truth_near.shape != candidate_near.shape:
        raise ValueError("near-field response shapes do not match")
    common_max = max(float(np.max(truth_near)), float(np.max(candidate_near)))
    if not np.isfinite(common_max) or common_max <= 0.0:
        raise ValueError("near-field response lacks positive finite energy")
    truth_scaled, candidate_scaled = truth_near / common_max, candidate_near / common_max
    near_delta = candidate_scaled - truth_scaled
    truth_norm = max(float(np.linalg.norm(truth_scaled)), 1.0e-12)
    candidate_norm = max(float(np.linalg.norm(candidate_scaled)), 1.0e-12)
    near_cosine = float(np.vdot(truth_scaled.reshape(-1), candidate_scaled.reshape(-1)).real / (truth_norm * candidate_norm))

    x, y = np.asarray(truth["x"], dtype=np.float64), np.asarray(truth["y"], dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    if truth_scaled.shape != xx.shape:
        raise ValueError(f"near-field coordinates do not match response grid: {truth_scaled.shape} != {xx.shape}")

    def centroid(values: np.ndarray) -> tuple[float, float]:
        total = max(float(np.sum(values)), 1.0e-12)
        return float(np.sum(xx * values) / total), float(np.sum(yy * values) / total)

    truth_center, candidate_center = centroid(truth_scaled), centroid(candidate_scaled)
    centroid_shift = math.hypot(candidate_center[0] - truth_center[0], candidate_center[1] - truth_center[1])

    truth_angles = np.asarray(truth["far_angles_rad"], dtype=np.float64)
    candidate_angles = np.asarray(candidate["far_angles_rad"], dtype=np.float64)
    if not np.allclose(truth_angles, candidate_angles, rtol=0.0, atol=1.0e-12):
        raise ValueError("far-field angle grids do not match")
    truth_far = np.asarray(truth["far_radial_flux"], dtype=np.float64)
    candidate_far = np.asarray(candidate["far_radial_flux"], dtype=np.float64)
    truth_area = float(np.trapezoid(truth_far, x=truth_angles))
    candidate_area = float(np.trapezoid(candidate_far, x=truth_angles))
    if min(truth_area, candidate_area) <= 0.0:
        raise ValueError("far-field response lacks positive angular integral")
    truth_far, candidate_far = truth_far / truth_area, candidate_far / candidate_area
    far_l1 = float(np.trapezoid(np.abs(candidate_far - truth_far), x=truth_angles))
    far_cosine = float(
        np.vdot(truth_far, candidate_far).real
        / max(float(np.linalg.norm(truth_far) * np.linalg.norm(candidate_far)), 1.0e-12)
    )
    truth_lobe = float(truth_angles[int(np.argmax(truth_far))])
    candidate_lobe = float(truth_angles[int(np.argmax(candidate_far))])
    truth_frequency, candidate_frequency = float(truth["frequency"]), float(candidate["frequency"])
    return {
        "frequency_relative_error": abs(candidate_frequency - truth_frequency) / max(abs(truth_frequency), 1.0e-12),
        "near_relative_l2": float(np.linalg.norm(near_delta) / truth_norm),
        "near_cosine_similarity": near_cosine,
        "near_energy_centroid_shift_a": float(centroid_shift),
        "far_angular_l1": far_l1,
        "far_cosine_similarity": far_cosine,
        "far_lobe_shift_deg": _circular_angle_difference_deg(candidate_lobe, truth_lobe),
    }


def _validate_physics_environment(config: Mapping[str, Any]) -> None:
    if not bool(config.get("runtime", {}).get("authorized_solver_execution")):
        raise RuntimeError("EXP-PHY-004 physics execution lacks explicit authorization in the resolved config")
    is_wsl = "microsoft" in platform.release().lower() or bool(os.environ.get("WSL_DISTRO_NAME"))
    if not is_wsl or os.environ.get("CONDA_DEFAULT_ENV") != "meep_env":
        raise RuntimeError("EXP-PHY-004 physics execution requires WSL Ubuntu with conda environment meep_env")


def _write_mpb_shard(path: Path, record: Mapping[str, Any]) -> None:
    np.savez_compressed(
        path,
        energy=record["energy"],
        epsilon=record["epsilon"],
        efield=record["efield"],
        hfield=record["hfield"],
        replay_frequency=np.asarray(record["replay_frequency"]),
        dataset_frequency=np.asarray(record["dataset_frequency"]),
        frequency_mre=np.asarray(record["frequency_mre"]),
        metadata_json=np.asarray(json.dumps(_jsonable(record["metadata"]), sort_keys=True)),
    )


def _read_mpb_shard(path: Path, *, selected: Mapping[str, Any], edge: Mapping[str, Any]) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        if metadata.get("geometry_hash") != selected["geometry_hash"]:
            raise RuntimeError("MPB field shard geometry hash mismatch")
        if int(metadata.get("band_index_one_based")) != int(edge["upper_band_one_based"]):
            raise RuntimeError("MPB field shard band mismatch")
        if int(metadata.get("k_index")) != int(edge["k_index"]):
            raise RuntimeError("MPB field shard k-index mismatch")
        return {
            "lane": selected["lane"],
            "sample_label": selected["sample_label"],
            "sample_id": selected["sample_id"],
            "dataset_index": int(selected["dataset_index"]),
            "band_index_one_based": int(edge["upper_band_one_based"]),
            "k_index": int(edge["k_index"]),
            "energy": np.asarray(payload["energy"]),
            "epsilon": np.asarray(payload["epsilon"]),
            "efield": np.asarray(payload["efield"]),
            "hfield": np.asarray(payload["hfield"]),
            "replay_frequency": float(payload["replay_frequency"].item()),
            "dataset_frequency": float(payload["dataset_frequency"].item()),
            "frequency_mre": float(payload["frequency_mre"].item()),
            "metadata": metadata,
        }


def _write_response_shard(path: Path, record: Mapping[str, Any]) -> None:
    np.savez_compressed(
        path,
        frequency=np.asarray(record["frequency"]),
        x=record["x"],
        y=record["y"],
        epsilon=record["epsilon"],
        electric_energy=record["electric_energy"],
        poynting_x=record["poynting_x"],
        poynting_y=record["poynting_y"],
        far_angles_rad=record["far_angles_rad"],
        far_radial_flux=record["far_radial_flux"],
        runtime_s=np.asarray(record["runtime_s"]),
        metadata_json=np.asarray(json.dumps(_jsonable(record["metadata"]), sort_keys=True)),
    )


def _read_response_shard(
    path: Path,
    *,
    selected: Mapping[str, Any],
    source: str,
    frequency: float,
    response_config: MeepScatteringConfig,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        if metadata.get("geometry_hash") != selected["geometry_hash"]:
            raise RuntimeError("Meep response shard geometry hash mismatch")
        if metadata.get("response_config") != asdict(response_config):
            raise RuntimeError("Meep response shard config mismatch")
        if not np.isclose(float(payload["frequency"].item()), frequency, rtol=0.0, atol=1.0e-10):
            raise RuntimeError("Meep response shard frequency mismatch")
        return {
            "lane": selected["lane"],
            "sample_label": selected["sample_label"],
            "sample_id": selected["sample_id"],
            "dataset_index": int(selected["dataset_index"]),
            "source": source,
            "frequency": float(payload["frequency"].item()),
            "x": np.asarray(payload["x"]),
            "y": np.asarray(payload["y"]),
            "epsilon": np.asarray(payload["epsilon"]),
            "electric_energy": np.asarray(payload["electric_energy"]),
            "poynting_x": np.asarray(payload["poynting_x"]),
            "poynting_y": np.asarray(payload["poynting_y"]),
            "far_angles_rad": np.asarray(payload["far_angles_rad"]),
            "far_radial_flux": np.asarray(payload["far_radial_flux"]),
            "runtime_s": float(payload["runtime_s"].item()),
            "metadata": metadata,
        }


def _solver_configs(config: Mapping[str, Any]) -> tuple[FullCellMpbBandSolverConfig, MeepScatteringConfig]:
    mpb = dict(config["mpb_eigenfield"])
    return (
        FullCellMpbBandSolverConfig(
            epsilon_air=float(mpb["epsilon_air"]),
            epsilon_material=float(mpb["epsilon_material"]),
            resolution=int(mpb["resolution"]),
            polarization=str(mpb["polarization"]),
            num_bands=int(mpb["num_bands"]),
            band_guard_count=int(mpb["band_guard_count"]),
            band_order_policy=str(mpb["band_order_policy"]),
            eigensolver_tolerance=float(mpb["eigensolver_tolerance"]),
            mesh_size=int(mpb["mesh_size"]),
        ),
        MeepScatteringConfig.from_mapping(dict(config["meep_response"])),
    )


def _load_selected_rows(out: Path, quantiles: Sequence[float] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [dict(row) for row in _read_csv(out / "selected_samples.csv")]
    if quantiles is not None:
        requested = {round(float(value), 8) for value in quantiles}
        rows = [row for row in rows if round(float(row["quantile"]), 8) in requested]
    if not rows:
        raise ValueError("physics selection is empty")
    return rows


def _load_band_edges(out: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    rows = _read_csv(out / "band_edge_frequency_manifest.csv")
    return {(row["lane"], row["sample_label"], row["source"]): row for row in rows}


def _run_or_load_physics(
    config: Mapping[str, Any],
    out: Path,
    *,
    quantiles: Sequence[float] | None,
    eigenfield_solver: EigenfieldSolverFn,
    scattering_solver: ScatteringSolverFn,
    solve_missing: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise ImportError("h5py is required for EXP-PHY-004 physics evidence") from exc

    selected_rows = _load_selected_rows(out, quantiles)
    edges = _load_band_edges(out)
    mpb_config, response_config = _solver_configs(config)
    field_dir, response_dir = out / "mpb_field_shards", out / "response_shards"
    field_dir.mkdir(exist_ok=True)
    response_dir.mkdir(exist_ok=True)
    datasets = dict(config["datasets"])
    field_records: list[dict[str, Any]] = []
    response_records: list[dict[str, Any]] = []
    mpb_executed = 0
    meep_executed = 0
    for selected in selected_rows:
        lane = str(selected["lane"])
        dataset = dict(datasets[lane])
        path = _repo_path(dataset["path"])
        with h5py.File(path, "r") as handle:
            structure = FullCellStructureSpec.from_mapping(json.loads(_decode(handle["structure_json"][int(selected["dataset_index"])])))
            material = np.asarray(handle[str(dataset["grid_field"])][int(selected["dataset_index"])], dtype=np.float64)
            epsilon = 1.0 + 7.9 * np.clip(material, 0.0, 1.0)
            k_points = np.asarray(handle["k_points"], dtype=np.float64)
        if structure.geometry_hash != selected["geometry_hash"]:
            raise RuntimeError("selected structure geometry hash drift")
        truth_edge = edges[(lane, str(selected["sample_label"]), "real_mpb")]
        field_path = field_dir / f"{selected['sample_label']}_real_mpb.npz"
        if not field_path.is_file():
            if not solve_missing:
                raise FileNotFoundError(f"MPB field shard is missing: {field_path}")
            solver_started = time.perf_counter()
            result = eigenfield_solver(
                structure,
                k_point=k_points[int(truth_edge["k_index"])],
                band_index_one_based=int(truth_edge["upper_band_one_based"]),
                config=mpb_config,
            )
            efield, hfield, energy, normalization = normalized_electric_energy_density(
                result.efield,
                result.hfield,
                epsilon=epsilon,
            )
            expected_frequency = float(truth_edge["frequency"])
            frequency_mre = abs(float(result.eigenfrequency) - expected_frequency) / max(abs(expected_frequency), 1.0e-12)
            if frequency_mre > float(config["mpb_eigenfield"]["frequency_mre_max"]):
                raise RuntimeError(f"real-MPB eigenfield replay frequency gate failed for {selected['sample_label']}: {frequency_mre}")
            record = {
                "energy": energy,
                "epsilon": epsilon,
                "efield": efield,
                "hfield": hfield,
                "replay_frequency": float(result.eigenfrequency),
                "dataset_frequency": expected_frequency,
                "frequency_mre": frequency_mre,
                "metadata": {
                    **dict(result.metadata),
                    "runtime_s": time.perf_counter() - solver_started,
                    "geometry_hash": structure.geometry_hash,
                    "band_index_one_based": int(truth_edge["upper_band_one_based"]),
                    "k_index": int(truth_edge["k_index"]),
                    "normalization": normalization,
                },
            }
            _write_mpb_shard(field_path, record)
            mpb_executed += 1
        field_records.append(_read_mpb_shard(field_path, selected=selected, edge=truth_edge))

        for source in SOURCES:
            edge = edges[(lane, str(selected["sample_label"]), source)]
            frequency = float(edge["frequency"])
            response_path = response_dir / f"{selected['sample_label']}_{source}.npz"
            if not response_path.is_file():
                if not solve_missing:
                    raise FileNotFoundError(f"Meep response shard is missing: {response_path}")
                result = scattering_solver(structure, frequency=frequency, config=response_config)
                record = {
                    "frequency": result.frequency,
                    "x": result.x,
                    "y": result.y,
                    "epsilon": result.epsilon,
                    "electric_energy": result.electric_energy,
                    "poynting_x": result.poynting_x,
                    "poynting_y": result.poynting_y,
                    "far_angles_rad": result.far_angles_rad,
                    "far_radial_flux": result.far_radial_flux,
                    "runtime_s": result.runtime_s,
                    "metadata": result.metadata,
                }
                _write_response_shard(response_path, record)
                meep_executed += 1
            response_records.append(
                _read_response_shard(
                    response_path,
                    selected=selected,
                    source=source,
                    frequency=frequency,
                    response_config=response_config,
                )
            )
    return field_records, response_records, mpb_executed, meep_executed


def _complete_physics(
    config: Mapping[str, Any],
    out: Path,
    *,
    quantiles: Sequence[float] | None,
    eigenfield_solver: EigenfieldSolverFn,
    scattering_solver: ScatteringSolverFn,
    solve_missing: bool,
) -> None:
    started = time.perf_counter()
    field_records, response_records, mpb_executed, meep_executed = _run_or_load_physics(
        config,
        out,
        quantiles=quantiles,
        eigenfield_solver=eigenfield_solver,
        scattering_solver=scattering_solver,
        solve_missing=solve_missing,
    )
    by_sample: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for record in response_records:
        by_sample.setdefault((str(record["lane"]), str(record["sample_label"])), {})[str(record["source"])] = record
    metric_rows: list[dict[str, Any]] = []
    for (lane, sample_label), records in sorted(by_sample.items()):
        if set(records) != set(SOURCES):
            raise RuntimeError(f"incomplete response triplet for {lane}/{sample_label}")
        for source in ("convnext", "joint"):
            metrics = response_difference_metrics(records["real_mpb"], records[source])
            records[source]["difference_metrics"] = metrics
            metric_rows.append({"lane": lane, "sample_label": sample_label, "source": source, **metrics})
        records["real_mpb"]["difference_metrics"] = {
            "frequency_relative_error": 0.0,
            "near_relative_l2": 0.0,
            "near_cosine_similarity": 1.0,
            "near_energy_centroid_shift_a": 0.0,
            "far_angular_l1": 0.0,
            "far_cosine_similarity": 1.0,
            "far_lobe_shift_deg": 0.0,
        }
    _write_csv(out / "response_difference_metrics.csv", metric_rows)
    figures_dir = ensure_figures_dir(out)
    output = dict(config["output"])
    formats = tuple(str(value) for value in output["formats"])
    dpi = int(output["dpi"])
    figure_paths = render_eigenfield_plates(field_records, figures_dir=figures_dir, formats=formats, dpi=dpi)
    figure_paths += render_response_plates(response_records, figures_dir=figures_dir, formats=formats, dpi=dpi)
    elapsed = time.perf_counter() - started
    previous_metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    full_selection_count = len(_read_csv(out / "selected_samples.csv"))
    completed_sample_count = len(field_records)
    complete = completed_sample_count == full_selection_count
    metrics = {
        **previous_metrics,
        "physics_wall_time_latest_invocation_s": elapsed,
        "physical_sample_count_completed": completed_sample_count,
        "mpb_eigenfield_logical_case_count": completed_sample_count,
        "meep_response_logical_case_count": len(response_records),
        "mpb_executed_this_invocation_count": mpb_executed,
        "meep_executed_this_invocation_count": meep_executed,
        "mpb_solver_time_s": float(sum(float(row["metadata"].get("runtime_s", 0.0)) for row in field_records)),
        "meep_solver_time_s": float(sum(float(row["runtime_s"]) for row in response_records)),
        "physics_figure_file_count": len(figure_paths),
    }
    write_metrics(out, metrics)
    write_results(
        out,
        {
            "status": "completed" if complete else "p50_preflight_complete",
            "evidence_level": "stored_real_mpb_plus_surrogate_held_out_plus_real_mpb_eigenfield_plus_real_meep_sensitivity",
            "selected_sample_count": full_selection_count,
            "physical_sample_count_completed": completed_sample_count,
            "physical_solver_execution_completed": complete,
            "field_surrogate_prediction_included": False,
            "response_interpretation": "real_meep_at_real_and_surrogate_selected_frequencies_not_surrogate_field_prediction",
            "spectral_ground_truth": "frozen_hdf5_stored_real_mpb_Wh_idx_test",
            "response_reference": "real_meep_at_real_mpb_selected_frequency",
            "surrogate_role_in_response": "predict_bands_then_select_band_k_frequency_frequency_only_drives_real_meep",
            "absolute_scattering_power_claim_allowed": False,
        },
    )
    history = _read_csv(out / "history.csv")
    history = [row for row in history if row.get("phase") not in {"physics_preflight", "physics_formal"}]
    history.append(
        {
            "phase": "physics_formal" if complete else "physics_preflight",
            "wall_time_s": elapsed,
            "record_count": len(response_records),
        }
    )
    _write_history(out / "history.csv", history)
    figure_paths_all = sorted(path for path in figures_dir.iterdir() if path.is_file())
    write_json(
        out / "figure_manifest.json",
        {
            "schema_version": "phc2d.exp_phy_004_figure_manifest.v1",
            "core_conclusion": "Ground truth is frozen real-MPB Wh; all response panels are real-Meep solves, and surrogates affect them only through selected frequency.",
            "archetype": "asymmetric_mixed_modality_figure",
            "backend": "python_matplotlib_only",
            "field_surrogate_prediction_included": False,
            "figures": [
                {"path": _relative(path), "sha256": _hash(path), "format": path.suffix.lstrip(".")}
                for path in figure_paths_all
            ],
        },
    )


def build_full_cell_surrogate_physics(
    config_path: str | Path,
    *,
    run_dir: str | Path | None = None,
    stage: str = "spectral",
    physics_quantiles: Sequence[float] | None = None,
    eigenfield_solver: EigenfieldSolverFn | None = None,
    scattering_solver: ScatteringSolverFn | None = None,
    skip_environment_check: bool = False,
) -> Path:
    """Build EXP-PHY-004 spectral figures and optional real-solver sensitivity evidence."""

    config_file = _repo_path(config_path)
    config = _load_config(config_file)
    output = dict(config["output"])
    if stage not in {"spectral", "physics", "render", "all"}:
        raise ValueError(f"unsupported EXP-PHY-004 stage: {stage}")
    if stage in {"physics", "render"} and run_dir is None:
        raise ValueError(f"stage={stage} requires an explicit existing run_dir")
    out = make_run_dir(
        str(config["experiment"]["experiment_id"]),
        runs_dir=_repo_path(output.get("runs_dir", "runs")),
        run_dir=run_dir,
    )
    setup_logging(log_file=out / "run.log", run_meta={"stage": stage, "experiment_id": "EXP-PHY-004"})
    write_resolved_config(out, yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    write_run_meta(
        out,
        {
            "stage": stage,
            "experiment_id": "EXP-PHY-004",
            "primary_layer": "L3",
            "config_path": _relative(config_file),
            "validation_boundary": "surrogate_spectral_then_wsl_real_mpb_meep_response_sensitivity",
        },
    )
    if stage in {"spectral", "all"}:
        _prepare_spectral(config, out)
    if stage in {"physics", "all", "render"}:
        if not (out / "selected_samples.csv").is_file():
            raise RuntimeError("physics/render stage requires completed spectral evidence in the same run directory")
        if not skip_environment_check and stage != "render":
            _validate_physics_environment(config)
        eigen_solver = eigenfield_solver or solve_eigenfield_for_full_cell_mpb
        meep_solver = scattering_solver or solve_finite_array_scattering_meep
        _complete_physics(
            config,
            out,
            quantiles=physics_quantiles,
            eigenfield_solver=eigen_solver,
            scattering_solver=meep_solver,
            solve_missing=stage != "render",
        )
    return out


__all__ = [
    "build_full_cell_surrogate_physics",
    "paired_bootstrap_mean_difference",
    "response_difference_metrics",
    "select_quantile_positions",
]
