"""Audit train-only Bragg priors against Full-Wh train/validation evidence.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from phc2d.forward.dual_lane_contracts import LANES, read_dual_lane_contracts
from phc2d.forward.full_cell_metrics import select_largest_gap_edge
from phc2d.forward.work_frequency import (
    bragg_frequency,
    nearest_frequency_distance,
    positive_adjacent_band_spacings,
)
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


def _validate_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    experiment = dict(payload.get("experiment") or {})
    datasets = dict(payload.get("datasets") or {})
    audit = dict(payload.get("audit") or {})
    output = dict(payload.get("output") or {})
    if str(experiment.get("experiment_id")) != "EXP-FWD-012":
        raise ValueError("work-frequency audit requires experiment_id=EXP-FWD-012")
    if set(datasets) != set(LANES):
        raise ValueError("datasets must contain exactly binary and graded")
    if int(audit.get("train_size", 0)) != 4000:
        raise ValueError("EXP-FWD-012 requires 4,000 train samples per lane")
    if not str(audit.get("train_subset_salt", "")):
        raise ValueError("train_subset_salt is required")
    if int(audit.get("lower_band_min_one_based", 0)) < 1:
        raise ValueError("lower_band_min_one_based must be positive")
    gates = dict(audit.get("gates") or {})
    if float(gates.get("coordinate_coverage_min", -1.0)) != 0.85:
        raise ValueError("EXP-FWD-012 coordinate coverage gate must equal 0.85")
    if float(gates.get("gap_hit_improvement_min", -1.0)) != 0.05:
        raise ValueError("EXP-FWD-012 gap-hit improvement gate must equal 0.05")
    return {
        "experiment": {
            **experiment,
            "formal_training_started": False,
            "test_executed": False,
            "mpb_meep_used": False,
        },
        "datasets": {lane: dict(datasets[lane]) for lane in LANES},
        "representation": {"sampling": "continuous", "threshold": None, "layout": "image"},
        "training": {
            "train_size": 4000,
            "train_subset_salt": str(audit["train_subset_salt"]),
        },
        "audit": audit,
        "output": output,
    }


def _gap_rows(
    frequencies: np.ndarray,
    *,
    lower_band_min_one_based: int,
    lower_band_max_one_based: int | None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    centers = np.empty(frequencies.shape[0], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for index, values in enumerate(frequencies):
        selected = select_largest_gap_edge(
            values,
            lower_band_min_one_based=lower_band_min_one_based,
            lower_band_max_one_based=lower_band_max_one_based,
        )
        centers[index] = selected.frequency - 0.5 * selected.gap
        rows.append({"sample_position": index, "gap_center": centers[index], **asdict(selected)})
    return centers, rows


def _load_prior(
    path: Path,
    *,
    grid_field: str,
    indices: np.ndarray,
    epsilon_air: float,
    epsilon_material: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    x_parts: list[np.ndarray] = []
    m_parts: list[np.ndarray] = []
    with h5py.File(path, "r") as handle:
        for start in range(0, len(indices), batch_size):
            selected = np.sort(indices[start : start + batch_size])
            grid = np.asarray(handle[grid_field][selected], dtype=np.float32)
            x_parts.append(
                bragg_frequency(
                    grid,
                    epsilon_air=epsilon_air,
                    epsilon_material=epsilon_material,
                    direction="X",
                )
            )
            m_parts.append(
                bragg_frequency(
                    grid,
                    epsilon_air=epsilon_air,
                    epsilon_material=epsilon_material,
                    direction="M",
                )
            )
    return np.concatenate(x_parts), np.concatenate(m_parts)


def _load_frequencies(path: Path, indices: np.ndarray) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as handle:
        return np.asarray(handle["Wh"][np.sort(indices)], dtype=np.float32)


def _dataset_solver(path: Path) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as handle:
        metadata = json.loads(str(handle.attrs["metadata_json"]))
    solver = dict(metadata.get("solver") or {})
    for key in ("epsilon_air", "epsilon_material"):
        if key not in solver:
            raise KeyError(f"dataset solver metadata is missing {key}")
    return solver


def _split_payload(
    frequencies: np.ndarray,
    x_prior: np.ndarray,
    m_prior: np.ndarray,
    *,
    delta_f: float,
    fixed_center: float,
    lower_band_min_one_based: int,
    lower_band_max_one_based: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gap_centers, gap_rows = _gap_rows(
        frequencies,
        lower_band_min_one_based=lower_band_min_one_based,
        lower_band_max_one_based=lower_band_max_one_based,
    )
    coordinate_coverage = nearest_frequency_distance(frequencies, x_prior) <= delta_f
    x_hit = np.abs(gap_centers - x_prior) <= delta_f
    m_hit = np.abs(gap_centers - m_prior) <= delta_f
    fixed_hit = np.abs(gap_centers - fixed_center) <= delta_f
    x_gap_distance = np.abs(gap_centers - x_prior)
    quantiles = (0.05, 0.50, 0.95)
    metrics = {
        "sample_count": int(frequencies.shape[0]),
        "coordinate_coverage_rate_x": float(np.mean(coordinate_coverage)),
        "gap_center_hit_rate_x": float(np.mean(x_hit)),
        "gap_center_hit_rate_m": float(np.mean(m_hit)),
        "fixed_center_gap_hit_rate": float(np.mean(fixed_hit)),
        "x_vs_fixed_gap_hit_improvement": float(np.mean(x_hit) - np.mean(fixed_hit)),
        "complete_gap_rate": float(np.mean([row["status"] == "complete_gap" for row in gap_rows])),
        "x_prior_quantiles": {str(q): float(np.quantile(x_prior, q)) for q in quantiles},
        "m_prior_quantiles": {str(q): float(np.quantile(m_prior, q)) for q in quantiles},
        "gap_center_quantiles": {str(q): float(np.quantile(gap_centers, q)) for q in quantiles},
        "x_to_gap_center_distance_quantiles": {str(q): float(np.quantile(x_gap_distance, q)) for q in quantiles},
    }
    return metrics, gap_rows


def run_work_frequency_audit(config_path: Path, *, run_dir: Path | None = None) -> Path:
    """Run EXP-FWD-012 without loading test targets or starting training."""

    config_file = Path(config_path)
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("work-frequency config root must be a mapping")
    resolved = _validate_config(raw)
    output = resolved["output"]
    out = make_run_dir(
        "EXP-FWD-012",
        runs_dir=output.get("runs_dir", "runs"),
        run_dir=run_dir,
    )
    logger = setup_logging(
        log_file=out / RUN_LOG_NAME,
        run_meta={
            "script": "python -m phc2d.forward.work_frequency_audit",
            "config_path": str(config_file),
            "validation_boundary": "surrogate_train_validation_prior_audit_no_test",
        },
    )
    started = time.perf_counter()
    write_resolved_config(out, yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))
    ensure_figures_dir(out)
    contracts = read_dual_lane_contracts(resolved)
    audit = resolved["audit"]
    batch_size = int(audit.get("batch_size", 256))
    lower_min = int(audit["lower_band_min_one_based"])
    lower_max_raw = audit.get("lower_band_max_one_based")
    lower_max = int(lower_max_raw) if lower_max_raw is not None else None
    lane_data: dict[str, dict[str, Any]] = {}
    spacing_parts: list[np.ndarray] = []
    nearest_parts: list[np.ndarray] = []
    train_gap_parts: list[np.ndarray] = []
    for lane in LANES:
        contract = contracts[lane]
        path = Path(contract["path"])
        solver = _dataset_solver(path)
        split_payload: dict[str, Any] = {}
        for split, indices in (
            ("train", np.asarray(contract["selected_train"], dtype=np.int64)),
            ("validation", np.asarray(contract["idx_val"], dtype=np.int64)),
        ):
            frequencies = _load_frequencies(path, indices)
            x_prior, m_prior = _load_prior(
                path,
                grid_field=str(contract["grid_field"]),
                indices=indices,
                epsilon_air=float(solver["epsilon_air"]),
                epsilon_material=float(solver["epsilon_material"]),
                batch_size=batch_size,
            )
            split_payload[split] = {
                "frequencies": frequencies,
                "x_prior": x_prior,
                "m_prior": m_prior,
            }
            if split == "train":
                spacing_parts.append(positive_adjacent_band_spacings(frequencies))
                nearest_parts.append(nearest_frequency_distance(frequencies, x_prior))
                gap_centers, _ = _gap_rows(
                    frequencies,
                    lower_band_min_one_based=lower_min,
                    lower_band_max_one_based=lower_max,
                )
                train_gap_parts.append(gap_centers)
        lane_data[lane] = split_payload
    epsilon_f = float(np.quantile(np.concatenate(spacing_parts), 0.25))
    delta_f = float(np.quantile(np.concatenate(nearest_parts), 0.90))
    fixed_center = float(np.median(np.concatenate(train_gap_parts)))
    if min(epsilon_f, delta_f) <= 0.0:
        raise RuntimeError("train-only frequency scales must be positive")
    metrics_by_lane: dict[str, Any] = {}
    history_rows: list[dict[str, Any]] = []
    gates = dict(audit["gates"])
    gate_failures: list[str] = []
    for lane in LANES:
        metrics_by_lane[lane] = {}
        for split in ("train", "validation"):
            data = lane_data[lane][split]
            metrics, _ = _split_payload(
                data["frequencies"],
                data["x_prior"],
                data["m_prior"],
                delta_f=delta_f,
                fixed_center=fixed_center,
                lower_band_min_one_based=lower_min,
                lower_band_max_one_based=lower_max,
            )
            metrics_by_lane[lane][split] = metrics
            history_rows.append({"lane": lane, "split": split, **metrics})
        validation = metrics_by_lane[lane]["validation"]
        if validation["coordinate_coverage_rate_x"] < float(gates["coordinate_coverage_min"]):
            gate_failures.append(f"{lane}:coordinate_coverage")
        if validation["x_vs_fixed_gap_hit_improvement"] < float(gates["gap_hit_improvement_min"]):
            gate_failures.append(f"{lane}:gap_hit_improvement")
    status = "prior_audit_pass" if not gate_failures else "prior_audit_fail"
    prior = {
        "frequency_unit": "mpb_dimensionless_a_over_lambda",
        "primary_direction": "X",
        "sensitivity_direction": "M",
        "epsilon_f_train_q25_positive_adjacent_band_spacing": epsilon_f,
        "delta_f_train_q90_nearest_coordinate_distance": delta_f,
        "fixed_center_train_median_gap_center": fixed_center,
        "test_influence": False,
    }
    write_json(out / "work_frequency_prior.json", prior)
    write_json(out / "work_frequency_metrics.json", metrics_by_lane)
    with (out / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0]))
        writer.writeheader()
        writer.writerows(history_rows)
    write_metrics(
        out,
        {
            "status": status,
            "gate_failures": gate_failures,
            "epsilon_f": epsilon_f,
            "delta_f": delta_f,
            "wall_time_total_s": time.perf_counter() - started,
            "lane_metrics": metrics_by_lane,
        },
    )
    write_results(
        out,
        {
            "status": status,
            "formal_training_started": False,
            "test_executed": False,
            "mpb_meep_used": False,
            "evidence_boundary": "surrogate_train_validation_prior_audit_no_test",
            "gate_failures": gate_failures,
            "next_gate": "EXP-FWD-013_loss_screen_authorization" if not gate_failures else "human_review_or_stop_work_frequency_route",
        },
    )
    write_run_meta(
        out,
        {
            "script": "python -m phc2d.forward.work_frequency_audit",
            "config_path": str(config_file),
            "dataset_paths": {lane: str(contracts[lane]["path"]) for lane in LANES},
            "dataset_sha256": {lane: contracts[lane]["dataset_sha256"] for lane in LANES},
            "train_subset_hash": {lane: contracts[lane]["train_subset_hash"] for lane in LANES},
            "idx_val_hash": {lane: contracts[lane]["idx_val_hash"] for lane in LANES},
            "test_targets_loaded": False,
        },
    )
    missing = validate_run_artifacts(out)
    if missing:
        raise RuntimeError(f"incomplete work-frequency audit artifacts: {missing}")
    logger.info("terminal_status=%s gate_failures=%s run_dir=%s", status, gate_failures, out)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    run_work_frequency_audit(args.config, run_dir=args.run_dir)


if __name__ == "__main__":
    main()


__all__ = ["run_work_frequency_audit"]
