"""EXP-PHY-005 TE effective-index and folded-gap physical audit.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from phc2d.data.full_cell import FullCellStructureSpec
from phc2d.data.sampling import (
    select_nested_sample_indices,
    select_stratified_sample_indices,
)
from phc2d.hpc import sha256_file
from phc2d.physics.effective_index_validation import (
    CANDIDATES,
    build_near_gamma_k_points,
)
from phc2d.plotting.effective_index import write_effective_index_figures
from phc2d.replay.effective_index_stages import (
    StructureSolver,
    UniformSolver,
    collect_gap_evidence,
    collect_slope_evidence,
    run_uniform_sentinels,
)
from phc2d.replay.effective_index_gates import evaluate_effective_index_gates
from phc2d.runs import (
    RUN_LOG_NAME,
    ensure_figures_dir,
    make_run_dir,
    setup_logging,
    validate_run_artifacts,
    write_json,
    write_metrics,
    write_resolved_config,
    write_results,
    write_run_meta,
)
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig


LANES = ("binary", "graded")


@dataclass(frozen=True)
class LaneContract:
    lane: str
    path: Path
    dataset_sha256: str
    raster_field: str
    selected_train: np.ndarray
    validation: np.ndarray
    slope_train: np.ndarray
    slope_validation: np.ndarray
    solver: dict[str, Any]
    target_shape: tuple[int, int]
    sample_manifest: tuple[dict[str, Any], ...]


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in keys:
                keys.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _write_solver_blocked_run(
    out: Path,
    *,
    base_meta: Mapping[str, Any],
    started: float,
    k_points: np.ndarray,
    failures: Sequence[Mapping[str, Any]],
    reason: str,
) -> Path:
    """Close a solver-blocked run with explicit case-level failures."""

    _write_jsonl(out / "solver_failures.jsonl", failures)
    _write_csv(out / "history.csv", [{"stage": "solver", "status": "blocked"}])
    np.savez_compressed(out / "near_gamma_bands.npz", k_points=k_points)
    _write_csv(out / "effective_index_samples.csv", [])
    write_json(out / "slope_metrics.json", {"status": "not_completed"})
    write_json(out / "gap_alignment_metrics.json", {"status": "not_run"})
    write_metrics(
        out,
        {
            "status": "blocked",
            "failure_count": len(failures),
            "wall_time_total_s": time.perf_counter() - started,
        },
    )
    write_results(
        out,
        {
            "status": "blocked",
            "reason": reason,
            "formal_physical_validation_completed": False,
            "training_started": False,
            "exp_fwd_017_unlocked": False,
        },
    )
    write_run_meta(out, dict(base_meta))
    missing = validate_run_artifacts(out)
    if missing:
        raise RuntimeError(f"incomplete blocked EXP-PHY-005 artifacts: {missing}")
    return out


def _validate_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    experiment = dict(payload.get("experiment") or {})
    datasets = dict(payload.get("datasets") or {})
    sampling = dict(payload.get("sampling") or {})
    mpb = dict(payload.get("mpb") or {})
    audit = dict(payload.get("audit") or {})
    gates = dict(payload.get("gates") or {})
    execution = dict(payload.get("execution") or {})
    output = dict(payload.get("output") or {})
    if experiment.get("experiment_id") != "EXP-PHY-005":
        raise ValueError("effective-index audit requires experiment_id=EXP-PHY-005")
    if set(datasets) != set(LANES):
        raise ValueError("datasets must contain exactly binary and graded")
    if str(mpb.get("polarization", "")).upper() != "TE":
        raise ValueError("EXP-PHY-005 is frozen to TE")
    if [float(value) for value in mpb.get("kappa", [])] != [
        0.0,
        1.0 / 256.0,
        1.0 / 128.0,
        1.0 / 64.0,
        1.0 / 32.0,
    ]:
        raise ValueError("EXP-PHY-005 kappa contract drift")
    mode = str(execution.get("mode", "preflight"))
    if mode not in {"preflight", "engineering_smoke", "real_mpb"}:
        raise ValueError("execution.mode must be preflight, engineering_smoke, or real_mpb")
    if mode == "real_mpb" and execution.get("formal_solver_authorized") is not True:
        raise ValueError("formal real-MPB execution requires formal_solver_authorized=true")
    expected_counts = (64, 32) if mode in {"preflight", "real_mpb"} else None
    counts = (
        int(sampling.get("slope_train_per_lane", 0)),
        int(sampling.get("slope_validation_per_lane", 0)),
    )
    if expected_counts is not None and counts != expected_counts:
        raise ValueError("formal EXP-PHY-005 requires 64 train + 32 validation per lane")
    if min(counts) < 0 or sum(counts) < 1:
        raise ValueError("slope sample counts must be nonnegative and nonempty")
    gap_counts = (
        int(audit.get("gap_train_size_per_lane", 0)),
        int(audit.get("gap_validation_size_per_lane", 0)),
    )
    if mode in {"preflight", "real_mpb"} and gap_counts != (4000, 1000):
        raise ValueError("formal EXP-PHY-005 requires 4,000 train + 1,000 validation gap samples per lane")
    if min(gap_counts) < 1:
        raise ValueError("gap-audit sample counts must be positive")
    if int(audit.get("lower_band_min_one_based", 0)) != 1:
        raise ValueError("EXP-PHY-005 must include the Band 1-2 gap")
    required_gates = {
        "sentinel_relative_error_max": 0.005,
        "sentinel_direction_difference_max": 0.005,
        "slope_r2_min": 0.999,
        "slope_r2_pass_rate_min": 0.95,
        "slope_window_drift_max": 0.02,
        "slope_window_drift_pass_rate_min": 0.90,
        "anisotropy_median_max": 0.02,
        "anisotropy_p90_max": 0.05,
        "oracle_pair_agreement_min": 0.80,
        "candidate_out_of_range_rate_max": 0.05,
        "train_validation_tv_max": 0.10,
        "complete_subset_min": 30,
        "complete_pair_agreement_min": 0.75,
    }
    for key, expected in required_gates.items():
        if float(gates.get(key, -1)) != float(expected):
            raise ValueError(f"EXP-PHY-005 gate drift for {key}")
    return {
        "experiment": experiment,
        "datasets": {lane: dict(datasets[lane]) for lane in LANES},
        "sampling": sampling,
        "mpb": mpb,
        "audit": audit,
        "gates": gates,
        "execution": execution,
        "output": output,
    }


def _read_values(handle: Any, name: str, indices: np.ndarray) -> np.ndarray:
    return np.asarray(handle[name][np.sort(indices)])


def _stratified_indices(
    handle: Any,
    indices: np.ndarray,
    *,
    size: int,
) -> np.ndarray:
    selected = np.sort(np.asarray(indices, dtype=np.int64))
    strata = list(
        zip(
            [_decode(value) for value in _read_values(handle, "family_id", selected)],
            [_decode(value) for value in _read_values(handle, "material_kind", selected)],
            strict=True,
        )
    )
    stable_ids = [_decode(value) for value in _read_values(handle, "geometry_hash", selected)]
    return select_stratified_sample_indices(
        selected,
        strata,
        stable_ids,
        size=size,
    )


def _lane_contract(lane: str, dataset: Mapping[str, Any], resolved: Mapping[str, Any]) -> LaneContract:
    import h5py

    path = Path(str(dataset["path"]))
    expected_sha = str(dataset.get("expected_dataset_sha256", ""))
    actual_sha = sha256_file(path)
    if not expected_sha or actual_sha != expected_sha:
        raise ValueError(f"{lane} dataset SHA-256 mismatch: {actual_sha} != {expected_sha}")
    raster_field = str(dataset["raster_field"])
    sampling = dict(resolved["sampling"])
    audit = dict(resolved["audit"])
    manifest: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        required = {
            "Wh",
            "idx_train",
            "idx_val",
            "sample_id",
            "family_id",
            "material_kind",
            "geometry_hash",
            "structure_json",
            raster_field,
            str(dataset["content_hash_field"]),
        }
        missing = sorted(required.difference(handle.keys()))
        if missing:
            raise KeyError(f"{lane} dataset missing fields: {missing}")
        metadata = json.loads(_decode(handle.attrs["metadata_json"]))
        solver = dict(metadata.get("solver") or {})
        if str(solver.get("polarization", "")).upper() != "TE":
            raise ValueError(f"{lane} solver metadata is not TE")
        idx_train = np.asarray(handle["idx_train"], dtype=np.int64)
        idx_val = np.asarray(handle["idx_val"], dtype=np.int64)
        sample_ids = [_decode(value) for value in handle["sample_id"][np.sort(idx_train)]]
        selected_train = select_nested_sample_indices(
            np.sort(idx_train),
            sample_ids,
            size=int(audit["gap_train_size_per_lane"]),
            salt=str(audit["train_subset_salt"]),
        )
        selected_validation = _stratified_indices(
            handle,
            idx_val,
            size=int(audit["gap_validation_size_per_lane"]),
        )
        slope_train = _stratified_indices(
            handle,
            selected_train,
            size=int(sampling["slope_train_per_lane"]),
        )
        slope_validation = _stratified_indices(
            handle,
            selected_validation,
            size=int(sampling["slope_validation_per_lane"]),
        )
        for split, indices in (("train", slope_train), ("validation", slope_validation)):
            for index in indices:
                raw = json.loads(_decode(handle["structure_json"][int(index)]))
                structure = FullCellStructureSpec.from_mapping(raw)
                stored_geometry_hash = _decode(handle["geometry_hash"][int(index)])
                if structure.geometry_hash != stored_geometry_hash:
                    raise ValueError(f"{lane} geometry hash mismatch at index {index}")
                raster = np.asarray(handle[raster_field][int(index)])
                manifest.append(
                    {
                        "lane": lane,
                        "split": split,
                        "dataset_index": int(index),
                        "sample_id": _decode(handle["sample_id"][int(index)]),
                        "family_id": _decode(handle["family_id"][int(index)]),
                        "material_kind": _decode(handle["material_kind"][int(index)]),
                        "geometry_hash": stored_geometry_hash,
                        "content_hash": _decode(
                            handle[str(dataset["content_hash_field"])][int(index)]
                        ),
                        "raster_sha256": _sha256_array(raster),
                    }
                )
        target_shape = tuple(int(value) for value in handle["Wh"].shape[1:])
        if len(target_shape) != 2 or target_shape[1] < 2:
            raise ValueError(f"{lane} invalid Wh target shape: {target_shape}")
    return LaneContract(
        lane=lane,
        path=path,
        dataset_sha256=actual_sha,
        raster_field=raster_field,
        selected_train=selected_train,
        validation=selected_validation,
        slope_train=slope_train,
        slope_validation=slope_validation,
        solver=solver,
        target_shape=target_shape,
        sample_manifest=tuple(manifest),
    )


def _solver_config(mpb: Mapping[str, Any]) -> FullCellMpbBandSolverConfig:
    return FullCellMpbBandSolverConfig(
        num_bands=int(mpb["num_bands"]),
        resolution=int(mpb["resolution"]),
        polarization=str(mpb["polarization"]),
        epsilon_air=float(mpb["epsilon_air"]),
        epsilon_material=float(mpb["epsilon_material"]),
        eigensolver_tolerance=float(mpb["eigensolver_tolerance"]),
        band_guard_count=int(mpb["band_guard_count"]),
        band_order_policy=str(mpb["band_order_policy"]),
        mesh_size=int(mpb["mesh_size"]),
    )


def run_effective_index_audit(
    config_path: str | Path,
    *,
    run_dir: str | Path | None = None,
    preflight: bool = False,
    structure_solver: StructureSolver | None = None,
    uniform_solver: UniformSolver | None = None,
) -> Path:
    """Run the registered EXP-PHY-005 preflight, smoke, or formal audit."""

    config_file = Path(config_path)
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("EXP-PHY-005 config root must be a mapping")
    resolved = _validate_config(raw)
    mode = "preflight" if preflight else str(resolved["execution"]["mode"])
    out = make_run_dir(
        "EXP-PHY-005",
        runs_dir=resolved["output"].get("runs_dir", "runs"),
        run_dir=run_dir,
    )
    if run_dir is None and any(out.iterdir()):
        raise FileExistsError(
            f"automatic EXP-PHY-005 run directory already contains evidence: {out}; "
            "wait for a new timestamp or pass a new --run-dir"
        )
    logger = setup_logging(
        log_file=out / RUN_LOG_NAME,
        run_meta={
            "script": "phc2d replay effective-index",
            "config_path": str(config_file),
            "execution_mode": mode,
        },
    )
    started = time.perf_counter()
    write_resolved_config(out, yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))
    ensure_figures_dir(out)
    contracts = {
        lane: _lane_contract(lane, resolved["datasets"][lane], resolved)
        for lane in LANES
    }
    manifest = [row for lane in LANES for row in contracts[lane].sample_manifest]
    _write_jsonl(out / "sample_manifest.jsonl", manifest)
    base_meta = {
        "script": "phc2d replay effective-index",
        "config_path": str(config_file),
        "execution_mode": mode,
        "dataset_paths": {lane: str(contracts[lane].path) for lane in LANES},
        "dataset_sha256": {lane: contracts[lane].dataset_sha256 for lane in LANES},
        "dataset_solver_metadata_sha256": {
            lane: hashlib.sha256(
                json.dumps(contracts[lane].solver, sort_keys=True).encode("utf-8")
            ).hexdigest()
            for lane in LANES
        },
        "resolved_config_sha256": sha256_file(out / "config.resolved.yaml"),
        "solver_contract_sha256": hashlib.sha256(
            json.dumps(resolved["mpb"], sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "selected_train_sha256": {
            lane: _sha256_array(contracts[lane].selected_train) for lane in LANES
        },
        "validation_sha256": {
            lane: _sha256_array(contracts[lane].validation) for lane in LANES
        },
        "sample_manifest_sha256": sha256_file(out / "sample_manifest.jsonl"),
        "test_targets_loaded": False,
        "training_started": False,
        "n40_used": False,
    }
    if mode == "preflight":
        _write_jsonl(out / "solver_failures.jsonl", [])
        _write_csv(out / "history.csv", [{"stage": "preflight", "status": "pass"}])
        write_metrics(
            out,
            {
                "status": "preflight_pass",
                "sample_manifest_count": len(manifest),
                "wall_time_total_s": time.perf_counter() - started,
            },
        )
        write_results(
            out,
            {
                "status": "preflight_pass",
                "formal_physical_validation_completed": False,
                "next_gate": "separate_192_structure_real_mpb_execution_authorization",
            },
        )
        write_run_meta(out, base_meta)
        missing = validate_run_artifacts(out)
        if missing:
            raise RuntimeError(f"incomplete EXP-PHY-005 preflight artifacts: {missing}")
        return out
    if structure_solver is None or uniform_solver is None:
        from phc2d.solvers.full_cell import (
            solve_bands_for_full_cell_mpb,
            solve_uniform_medium_mpb,
        )

        structure_solver = structure_solver or solve_bands_for_full_cell_mpb
        uniform_solver = uniform_solver or solve_uniform_medium_mpb
    mpb = dict(resolved["mpb"])
    kappa = np.asarray(mpb["kappa"], dtype=np.float64)
    k_points, direction_indices = build_near_gamma_k_points(kappa)
    solver_config = _solver_config(mpb)
    try:
        sentinel_rows, sentinel_provenance = run_uniform_sentinels(
            k_points=k_points,
            direction_indices=direction_indices,
            kappa=kappa,
            config=solver_config,
            uniform_solver=uniform_solver,
        )
    except Exception as exc:
        return _write_solver_blocked_run(
            out,
            base_meta=base_meta,
            started=started,
            k_points=k_points,
            failures=[
                {
                    "stage": "uniform_sentinel",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ],
            reason="uniform_medium_sentinel_solver_error",
        )
    gates = dict(resolved["gates"])
    sentinel_pass = all(
        max(row["x_relative_error"], row["m_relative_error"])
        <= float(gates["sentinel_relative_error_max"])
        and row["direction_difference"] <= float(gates["sentinel_direction_difference_max"])
        for row in sentinel_rows
    )
    if not sentinel_pass:
        _write_jsonl(out / "solver_failures.jsonl", [])
        _write_csv(out / "history.csv", sentinel_rows)
        write_json(out / "slope_metrics.json", {"sentinels": sentinel_rows})
        write_json(out / "gap_alignment_metrics.json", {"status": "not_run"})
        np.savez_compressed(out / "near_gamma_bands.npz", k_points=k_points)
        _write_csv(out / "effective_index_samples.csv", [])
        write_metrics(out, {"status": "blocked", "sentinel_pass": False})
        write_results(
            out,
            {
                "status": "blocked",
                "reason": "uniform_medium_sentinel_fail",
                "formal_physical_validation_completed": False,
            },
        )
        write_run_meta(out, {**base_meta, "sentinel_provenance": sentinel_provenance})
        return out
    slope_rows, bands_keys, bands_records, solver_provenance, solver_failures = collect_slope_evidence(
        contracts,
        lanes=LANES,
        k_points=k_points,
        direction_indices=direction_indices,
        kappa=kappa,
        solver_config=solver_config,
        structure_solver=structure_solver,
    )
    _write_jsonl(out / "solver_failures.jsonl", solver_failures)
    if solver_failures:
        return _write_solver_blocked_run(
            out,
            base_meta={**base_meta, "solver_provenance": solver_provenance},
            started=started,
            k_points=k_points,
            failures=solver_failures,
            reason="near_gamma_structure_solver_failures",
        )
    np.savez_compressed(
        out / "near_gamma_bands.npz",
        keys=np.asarray(bands_keys),
        k_points=k_points,
        bands=np.stack(bands_records),
    )
    _write_csv(out / "effective_index_samples.csv", slope_rows)
    spacing_floor, gap_rows, oracle_rows = collect_gap_evidence(
        contracts,
        slope_rows,
        lanes=LANES,
    )
    _write_csv(out / "gap_alignment_rows.csv", gap_rows)
    decision = evaluate_effective_index_gates(
        mode=mode,
        sentinel_rows=sentinel_rows,
        slope_rows=slope_rows,
        gap_rows=gap_rows,
        oracle_rows=oracle_rows,
        spacing_floor=spacing_floor,
        gates=gates,
        lanes=LANES,
        band_count=contracts["binary"].target_shape[1],
    )
    write_json(out / "slope_metrics.json", decision.slope_metrics)
    write_json(out / "gap_alignment_metrics.json", decision.gap_payload)
    write_effective_index_figures(
        ensure_figures_dir(out),
        slope_rows,
        gap_rows,
        candidates=CANDIDATES,
        lanes=LANES,
    )
    history = [
        {"stage": "uniform_sentinel", "status": "pass"},
        {"stage": "near_gamma_slope", "status": "pass" if decision.numeric_pass else "fail"},
        {"stage": "scalar_anisotropy", "status": "pass" if decision.anisotropy_pass else "fail"},
        {"stage": "effective_medium", "status": "pass" if decision.estimator_pass else "fail"},
        {"stage": "gap_alignment", "status": "pass" if decision.gap_pass else "fail"},
    ]
    _write_csv(out / "history.csv", history)
    write_metrics(
        out,
        {
            "status": decision.status,
            "sentinel_pass": sentinel_pass,
            "numeric_pass": decision.numeric_pass,
            "anisotropy_pass": decision.anisotropy_pass,
            "estimator_pass": decision.estimator_pass,
            "gap_pass": decision.gap_pass,
            "candidate_gates": decision.candidate_gates,
            "slope_sample_count": len(slope_rows),
            "wall_time_total_s": time.perf_counter() - started,
        },
    )
    write_results(
        out,
        {
            "status": decision.status,
            "formal_physical_validation_completed": mode == "real_mpb",
            "test_executed": False,
            "training_started": False,
            "n40_used": False,
            "exp_fwd_017_unlocked": decision.status == "prior_physical_gate_pass",
            "next_gate": (
                "separate_192_structure_real_mpb_execution_authorization"
                if mode == "engineering_smoke"
                else (
                    "human_acceptance_before_exp_fwd_017_registration"
                    if decision.status == "prior_physical_gate_pass"
                    else "human_review_or_stop_resonance_prior_route"
                )
            ),
        },
    )
    write_run_meta(
        out,
        {
            **base_meta,
            "sentinel_provenance": sentinel_provenance,
            "solver_provenance": solver_provenance,
            "k_points_sha256": _sha256_array(k_points),
            "near_gamma_bands_sha256": sha256_file(out / "near_gamma_bands.npz"),
        },
    )
    missing = validate_run_artifacts(out)
    if missing:
        raise RuntimeError(f"incomplete EXP-PHY-005 artifacts: {missing}")
    logger.info("terminal_status=%s run_dir=%s", decision.status, out)
    return out


__all__ = [
    "build_near_gamma_k_points",
    "run_effective_index_audit",
]
