"""Validation-only target-frequency residual diagnostic for frozen B4 PINO models.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from phc2d.forward.field_eigen_b4 import (
    ROOT,
    _atomic_write_json,
    _hash,
    _json_fingerprint,
    _load_b3_structures,
    _load_checkpoint_case,
    _load_resumable_partial,
    _read_sidecar,
    _root_path,
    _source_index,
    _validate_dataset_lineage,
    _verify_v2_source_lineage,
    _write_csv,
)
from phc2d.forward.field_eigen_b4_selector import _candidate_cases
from phc2d.forward.field_eigen_v2 import _periodic_grid, _torch_active_complex
from phc2d.physics import evaluate_te_bloch_eigen_residual_torch
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
from phc2d.training import (
    DeviceConfig,
    collect_device_metadata,
    peak_gpu_memory_mb,
    resolve_device,
    synchronize_device,
)


def is_field_eigen_b4_target_residual_config(path: str | Path) -> bool:
    """Return whether a config selects the target-residual diagnostic."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    execution = str(dict(raw.get("experiment") or {}).get("execution", "")).lower()
    return execution in {
        "field_eigen_te_b4_target_residual_64",
        "field_eigen_te_p6_seed0_target_residual_pilot",
    }


def _diagnostic_scope(raw: Mapping[str, Any]) -> str:
    """Return the explicitly isolated formal or seed-0 pilot scope."""

    execution = str(dict(raw.get("experiment") or {}).get("execution", "")).lower()
    return "seed0_pilot" if execution == "field_eigen_te_p6_seed0_target_residual_pilot" else "formal"


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for group in np.flatnonzero(counts > 1):
        indices = np.flatnonzero(inverse == group)
        ranks[indices] = ranks[indices].mean()
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 3:
        return float("nan")
    x_rank = _rank(x[finite])
    y_rank = _rank(y[finite])
    if float(np.std(x_rank)) == 0.0 or float(np.std(y_rank)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _unique_k_indices(sidecar: Mapping[str, Any]) -> list[int]:
    points = np.asarray(sidecar["k_points"], dtype=np.float64)
    duplicate_indices = {
        int(index)
        for group in sidecar.get("duplicate_coordinate_groups", [])
        for index in list(group)[1:]
    }
    unique = [index for index in range(len(points)) if index not in duplicate_indices]
    if len(unique) != len(np.unique(points, axis=0)):
        raise ValueError("Frozen k-path duplicate groups do not define one row per physical coordinate.")
    return unique


def _validate_contract(raw: Mapping[str, Any]) -> tuple[str, set[int], int]:
    contract = dict(raw["validation_contract"])
    split = str(contract["evaluation_split"])
    required_seeds = {int(seed) for seed in contract["required_seeds"]}
    if split != "val" or bool(contract.get("allow_test", False)):
        raise ValueError("Target-residual diagnostic must be validation-only and reject test access.")
    scope = _diagnostic_scope(raw)
    expected_seeds = {0} if scope == "seed0_pilot" else {0, 1, 2}
    if required_seeds != expected_seeds:
        raise ValueError(
            f"Target-residual {scope} diagnostic requires frozen seeds {sorted(expected_seeds)}."
        )
    if scope == "seed0_pilot":
        if not bool(contract.get("pilot_only", False)):
            raise ValueError("Seed-0 target-residual diagnostic must declare pilot_only=true.")
        if not bool(contract.get("single_seed_inverse_admission_allowed", False)):
            raise ValueError(
                "Seed-0 target-residual diagnostic must explicitly allow conditional admission."
            )
        if bool(contract.get("formal_three_seed_claim_allowed", True)):
            raise ValueError(
                "Seed-0 target-residual diagnostic must forbid a formal three-seed claim."
            )
    resolution = int(dict(raw["evaluation"])["resolution"])
    if resolution <= 1:
        raise ValueError("Target-residual evaluation resolution must be greater than one.")
    return split, required_seeds, resolution


def _validate_p6_pilot_source(
    raw: Mapping[str, Any],
    *,
    candidates: Mapping[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate the frozen P6 config/run lineage without claiming a formal B4 pass."""

    source = dict(raw["source_p6_training"])
    config_path = _root_path(str(source["config_path"]))
    config_hash = _hash(config_path)
    if config_hash != str(source["config_sha256"]).lower():
        raise ValueError("Frozen P6 source config SHA256 mismatch.")
    source_raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    training = dict(source_raw.get("training") or {})
    if str(training.get("physics_frequency_source", "")).lower() != "target":
        raise ValueError("P6 pilot source config must use target-frequency physics.")
    run_dir = _root_path(str(source["run_dir"])).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Frozen P6 source run does not exist: {run_dir}")
    for cases in candidates.values():
        for case in cases:
            checkpoint = _root_path(str(case["checkpoint_path"])).resolve()
            try:
                checkpoint.relative_to(run_dir / "cases")
            except ValueError as exc:
                raise ValueError("P6 pilot checkpoint is outside the frozen source run.") from exc
    return {
        "scope": "seed0_pilot",
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "candidate_names": sorted(candidates),
    }, {
        "source_p6_config": config_hash,
        "source_p6_run": str(run_dir),
    }


def _validate_source_decision(
    raw: Mapping[str, Any],
    *,
    candidates: Mapping[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, str]]:
    source = dict(raw["source_b4_decision"])
    path = _root_path(str(source["path"]))
    expected_hash = str(source["sha256"]).lower()
    actual_hash = _hash(path)
    if actual_hash != expected_hash:
        raise ValueError("Frozen B4 source decision SHA256 mismatch.")
    decision = json.loads(path.read_text(encoding="utf-8"))
    source_candidates = dict(decision.get("candidate_decisions") or {})
    if set(source_candidates) != set(candidates):
        raise ValueError("Target-residual candidates do not match the frozen B4 decision.")
    for candidate in candidates:
        frequency = dict(source_candidates[candidate].get("physics_frequency_crosscheck_ready") or {})
        if str(frequency.get("status")) != "pass":
            raise ValueError(f"Candidate {candidate!r} lacks the required frozen frequency cross-check pass.")
    if str(decision.get("selection_split")) != "val":
        raise ValueError("Frozen B4 source decision is not validation-only.")
    return decision, {"source_b4_decision": actual_hash}


def _case_rows(
    *,
    case: Mapping[str, Any],
    model: Any,
    identity: Mapping[str, Any],
    structures: Mapping[str, Mapping[str, Any]],
    split: str,
    k_points: np.ndarray,
    unique_k_indices: list[int],
    bands: np.ndarray,
    truth: np.ndarray,
    resolution: int,
    residual_solver_floor_scale: float,
    normalization: Mapping[str, Any],
    batch_size: int,
    device: Any,
    torch: Any,
) -> tuple[list[dict[str, Any]], float]:
    epsilon_range = tuple(float(value) for value in normalization["epsilon_range"])
    k_scale = float(normalization["k_scale"])
    band_min, band_max = (float(value) for value in normalization["band_range_one_based"])
    condition_k_indices = np.repeat(np.asarray(unique_k_indices, dtype=np.int64), len(bands))
    condition_bands = np.tile(bands, len(unique_k_indices))
    condition_k_points = k_points[condition_k_indices]
    contexts = np.stack(
        (
            condition_k_points[:, 0] / k_scale,
            condition_k_points[:, 1] / k_scale,
            (condition_bands.astype(np.float32) - band_min) / max(band_max - band_min, 1.0),
        ),
        axis=1,
    ).astype(np.float32)
    rows: list[dict[str, Any]] = []
    synchronize_device(device, torch_module=torch)
    started = time.perf_counter()
    for source_id, payload in structures.items():
        grid = _periodic_grid(
            payload["epsilon"][None],
            resolution,
            resolution,
            include_coordinates=True,
            epsilon_range=epsilon_range,
        )
        grid_tensor = torch.as_tensor(grid, device=device)
        source_index = _source_index(source_id)
        target_values = truth[
            source_index,
            condition_k_indices,
            condition_bands - 1,
        ].astype(np.float32)
        for start in range(0, len(contexts), batch_size):
            stop = min(start + batch_size, len(contexts))
            count = stop - start
            context = torch.as_tensor(contexts[start:stop], device=device)
            target_frequency = torch.as_tensor(target_values[start:stop], device=device)
            batch_k = torch.as_tensor(condition_k_points[start:stop], device=device)
            batch_epsilon = torch.as_tensor(payload["epsilon"], device=device).expand(count, -1, -1)
            spacing = torch.full((count,), 1.0 / float(resolution), device=device)
            with torch.no_grad():
                predicted, frequency_z = model(grid_tensor.expand(count, -1, -1, -1), context)
                predicted_frequency = torch.exp(
                    frequency_z.reshape(count) * float(identity["frequency_std"])
                    + float(identity["frequency_mean"])
                )
                active = _torch_active_complex(torch, predicted)
                self_residual = evaluate_te_bloch_eigen_residual_torch(
                    active[..., :2],
                    epsilon=batch_epsilon,
                    eigenfrequency=predicted_frequency,
                    dx=spacing,
                    dy=spacing,
                    bloch_k_point=batch_k,
                )
                target_residual = evaluate_te_bloch_eigen_residual_torch(
                    active[..., :2],
                    epsilon=batch_epsilon,
                    eigenfrequency=target_frequency,
                    dx=spacing,
                    dy=spacing,
                    bloch_k_point=batch_k,
                )
            predicted_np = predicted_frequency.detach().cpu().numpy()
            target_np = target_frequency.detach().cpu().numpy()
            self_np = self_residual["relative_residual"].detach().cpu().numpy()
            target_np_residual = target_residual["relative_residual"].detach().cpu().numpy()
            rayleigh_np = target_residual["rayleigh_frequency"].detach().cpu().numpy()
            rayleigh_valid_np = target_residual["rayleigh_valid"].detach().cpu().numpy()
            for offset in range(count):
                condition = start + offset
                predicted_value = float(predicted_np[offset])
                target_value = float(target_np[offset])
                self_value = float(self_np[offset]) / residual_solver_floor_scale
                target_value_residual = float(target_np_residual[offset]) / residual_solver_floor_scale
                rayleigh_value = float(rayleigh_np[offset])
                rayleigh_valid = bool(rayleigh_valid_np[offset]) and math.isfinite(rayleigh_value)
                k_index = int(condition_k_indices[condition])
                k_point = condition_k_points[condition]
                rows.append(
                    {
                        "case": str(case["name"]),
                        "candidate": str(case["candidate"]),
                        "kind": "pino",
                        "seed": int(case["seed"]),
                        "split": split,
                        "resolution": resolution,
                        "source_sample_id": source_id,
                        "source_sample_hash": str(payload["source_hash"]),
                        "k_index": k_index,
                        "k_x": float(k_point[0]),
                        "k_y": float(k_point[1]),
                        "band_index_one_based": int(condition_bands[condition]),
                        "frequency_prediction": predicted_value,
                        "frequency_truth": target_value,
                        "frequency_mre_pct": 100.0
                        * abs(predicted_value - target_value)
                        / max(abs(target_value), 1.0e-12),
                        "self_relative_residual": self_value,
                        "target_relative_residual": target_value_residual,
                        "target_residual_score_component": math.log1p(target_value_residual**2),
                        "rayleigh_frequency": rayleigh_value if rayleigh_valid else float("nan"),
                        "rayleigh_valid": rayleigh_valid,
                        "rayleigh_target_log_error": (
                            abs(math.log(max(rayleigh_value, 1.0e-12)) - math.log(target_value))
                            if rayleigh_valid
                            else float("nan")
                        ),
                        "residual_solver_floor_scale": residual_solver_floor_scale,
                    }
                )
    synchronize_device(device, torch_module=torch)
    return rows, time.perf_counter() - started


def _structure_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["case"]),
                str(row["candidate"]),
                int(row["seed"]),
                str(row["source_sample_id"]),
                str(row["source_sample_hash"]),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        target_components = np.asarray(
            [float(row["target_residual_score_component"]) for row in values],
            dtype=np.float64,
        )
        self_components = np.log1p(
            np.square(
                np.asarray([float(row["self_relative_residual"]) for row in values], dtype=np.float64)
            )
        )
        frequency_errors = np.asarray(
            [float(row["frequency_mre_pct"]) for row in values],
            dtype=np.float64,
        )
        rayleigh_errors = np.asarray(
            [float(row["rayleigh_target_log_error"]) for row in values],
            dtype=np.float64,
        )
        all_finite = bool(
            np.isfinite(target_components).all()
            and np.isfinite(self_components).all()
            and np.isfinite(frequency_errors).all()
        )
        output.append(
            {
                "case": key[0],
                "candidate": key[1],
                "kind": "pino",
                "seed": key[2],
                "source_sample_id": key[3],
                "source_sample_hash": key[4],
                "condition_count": len(values),
                "target_residual_score": float(np.median(target_components)),
                "self_residual_score": float(np.median(self_components)),
                "frequency_mre_pct": float(np.mean(frequency_errors)),
                "rayleigh_target_log_error_median": (
                    float(np.nanmedian(rayleigh_errors))
                    if bool(np.isfinite(rayleigh_errors).any())
                    else float("nan")
                ),
                "all_finite": all_finite,
            }
        )
    return output


def _bootstrap_structure_ci(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    repeats: int,
) -> tuple[float, float, float]:
    residual = np.asarray([float(row["target_residual_score"]) for row in rows], dtype=np.float64)
    frequency = np.asarray([float(row["frequency_mre_pct"]) for row in rows], dtype=np.float64)
    if len(rows) < 3:
        return float("nan"), float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    bootstrapped: list[float] = []
    for _ in range(repeats):
        sampled = generator.integers(0, len(rows), size=len(rows))
        bootstrapped.append(_spearman(residual[sampled], frequency[sampled]))
    finite = np.asarray([value for value in bootstrapped if np.isfinite(value)], dtype=np.float64)
    if len(finite) == 0:
        return _spearman(residual, frequency), float("nan"), float("nan")
    return (
        _spearman(residual, frequency),
        float(np.quantile(finite, 0.025)),
        float(np.quantile(finite, 0.975)),
    )


def _calibration_rows(
    structure_rows: list[dict[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_repeats: int,
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in structure_rows:
        grouped[(str(row["case"]), str(row["candidate"]), int(row["seed"]))].append(row)
    output: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        residual = np.asarray([float(row["target_residual_score"]) for row in values], dtype=np.float64)
        frequency = np.asarray([float(row["frequency_mre_pct"]) for row in values], dtype=np.float64)
        all_finite = bool(
            all(bool(row["all_finite"]) for row in values)
            and np.isfinite(residual).all()
            and np.isfinite(frequency).all()
        )
        correlation, lower, upper = _bootstrap_structure_ci(
            values,
            seed=bootstrap_seed + int(key[2]),
            repeats=bootstrap_repeats,
        )
        if all_finite:
            low_threshold, high_threshold = np.quantile(residual, [0.25, 0.75])
            low_mre = float(np.mean(frequency[residual <= low_threshold]))
            high_mre = float(np.mean(frequency[residual >= high_threshold]))
            reduction = 100.0 * (high_mre - low_mre) / max(high_mre, 1.0e-12)
        else:
            low_mre = high_mre = reduction = float("nan")
        passed = bool(
            all_finite
            and np.isfinite(lower)
            and lower > float(thresholds["spearman_ci_lower_min_exclusive"])
            and np.isfinite(reduction)
            and reduction >= float(thresholds["quartile_frequency_mre_reduction_min_pct"])
        )
        output.append(
            {
                "case": key[0],
                "candidate": key[1],
                "kind": "pino",
                "seed": key[2],
                "structure_count": len(values),
                "all_finite": all_finite,
                "spearman_target_residual_frequency_error": correlation,
                "spearman_bootstrap_ci_lower": lower,
                "spearman_bootstrap_ci_upper": upper,
                "lowest_residual_quartile_frequency_mre_pct": low_mre,
                "highest_residual_quartile_frequency_mre_pct": high_mre,
                "quartile_frequency_mre_reduction_pct": reduction,
                "physics_target_residual_64_ready": "pass" if passed else "blocked",
            }
        )
    return output


def _candidate_decision(
    calibration: list[dict[str, Any]],
    structure_rows: list[dict[str, Any]],
    *,
    candidates: Mapping[str, list[dict[str, Any]]],
    source_decision: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    required_seeds: set[int],
) -> dict[str, Any]:
    source_candidates = dict(source_decision["candidate_decisions"])
    candidate_decisions: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    eligible: list[str] = []
    for candidate, cases in sorted(candidates.items()):
        names = {str(case["name"]) for case in cases}
        candidate_calibration = [row for row in calibration if str(row["case"]) in names]
        calibration_by_seed = {int(row["seed"]): row for row in candidate_calibration}
        target_ready = bool(
            set(calibration_by_seed) == required_seeds
            and all(
                str(row["physics_target_residual_64_ready"]) == "pass"
                for row in calibration_by_seed.values()
            )
        )
        frequency_values = [
            float(row["frequency_mre_pct"])
            for row in structure_rows
            if str(row["case"]) in names
        ]
        score = float(np.mean(frequency_values)) if frequency_values else float("inf")
        scores[candidate] = score
        if target_ready and math.isfinite(score):
            eligible.append(candidate)
        source = dict(source_candidates[candidate])
        candidate_decisions[candidate] = {
            "physics_frequency_crosscheck_ready": dict(source["physics_frequency_crosscheck_ready"]),
            "physics_target_residual_64_ready": {
                "status": "pass" if target_ready else "blocked",
                "complete_required_seeds": set(calibration_by_seed) == required_seeds,
                "all_seed_calibrations_pass": target_ready,
                "thresholds": dict(thresholds),
            },
            "cross_resolution_zero_shot_pass": dict(source["cross_resolution_zero_shot_pass"]),
            "pino_inverse_adapter_preflight_ready": {"status": "blocked", "reason": "not_run"},
            "pino_paired_mpb_validation_ready": {"status": "blocked", "reason": "not_run"},
            "pino_inverse_rerank_64_ready": {"status": "blocked", "reason": "later_gates_not_run"},
            "pino_inverse_rerank_multires_ready": {
                "status": "blocked",
                "reason": "outside_fixed_resolution_diagnostic_scope",
            },
            "field_inverse_generation_ready": {
                "status": "blocked",
                "reason": "field_generation_is_out_of_scope",
            },
            "mean_validation_frequency_mre_pct": score,
        }
    selected = min(eligible, key=lambda name: (scores[name], name)) if eligible else None
    return {
        "status": "selected" if selected else "blocked",
        "selection_scope": "fixed_resolution_target_residual_diagnostic",
        "eligible_candidates": sorted(eligible),
        "selected_candidate": selected,
        "selection_rule": "all_three_seed_target_residual_gates_then_lowest_mean_frequency_mre",
        "selection_scores": {
            name: {"mean_validation_frequency_mre_pct": scores[name]}
            for name in sorted(scores)
        },
        "candidate_decisions": candidate_decisions,
        "physics_target_residual_64_ready": {
            "status": "pass" if selected else "blocked",
            "selected_candidate": selected,
        },
        "pino_inverse_adapter_preflight_ready": {"status": "blocked", "reason": "not_run"},
        "pino_paired_mpb_validation_ready": {"status": "blocked", "reason": "not_run"},
        "pino_inverse_rerank_64_ready": {"status": "blocked", "reason": "later_gates_not_run"},
        "pino_inverse_rerank_multires_ready": {
            "status": "blocked",
            "reason": "cross_resolution_gate_is_independent",
        },
        "field_inverse_generation_ready": {
            "status": "blocked",
            "reason": "field_generation_is_out_of_scope",
        },
        "next_required_stage": (
            "pino_inverse_adapter_preflight_64"
            if selected
            else "p6_target_frequency_pde_training"
        ),
        "test_accessed": False,
        "mpb_accessed": False,
    }


def _pilot_candidate_decision(
    calibration: list[dict[str, Any]],
    structure_rows: list[dict[str, Any]],
    *,
    candidates: Mapping[str, list[dict[str, Any]]],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the user-authorized seed-0 conditional adapter-admission rule."""

    candidate_decisions: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    eligible: list[str] = []
    for candidate, cases in sorted(candidates.items()):
        names = {str(case["name"]) for case in cases}
        candidate_calibration = [
            row for row in calibration if str(row["case"]) in names
        ]
        pilot_pass = bool(
            len(candidate_calibration) == 1
            and int(candidate_calibration[0]["seed"]) == 0
            and str(candidate_calibration[0]["physics_target_residual_64_ready"]) == "pass"
        )
        frequency_values = [
            float(row["frequency_mre_pct"])
            for row in structure_rows
            if str(row["case"]) in names
        ]
        score = float(np.mean(frequency_values)) if frequency_values else float("inf")
        scores[candidate] = score
        if pilot_pass and math.isfinite(score):
            eligible.append(candidate)
        candidate_decisions[candidate] = {
            "seed0_target_residual_pilot": {
                "status": "pass" if pilot_pass else "blocked",
                "calibration": candidate_calibration[0] if candidate_calibration else None,
                "thresholds": dict(thresholds),
            },
            "mean_validation_frequency_mre_pct": score,
            "physics_target_residual_64_ready": {
                "status": "pass" if pilot_pass else "blocked",
                "evidence_level": "single_seed_conditional",
                "reason": (
                    "user_authorized_seed0_admission"
                    if pilot_pass
                    else "seed0_target_residual_gate_failed"
                ),
            },
        }
    selected = min(eligible, key=lambda name: (scores[name], name)) if eligible else None
    return {
        "status": "conditional_inverse_admission" if selected else "pilot_blocked",
        "pilot_only": True,
        "evidence_level": "single_seed_conditional",
        "selection_scope": "fixed_resolution_seed0_target_residual_pilot",
        "eligible_candidates": sorted(eligible),
        "selected_candidate": selected,
        "selection_rule": "seed0_target_residual_gate_then_lowest_validation_frequency_mre",
        "selection_scores": {
            name: {"mean_validation_frequency_mre_pct": scores[name]}
            for name in sorted(scores)
        },
        "candidate_decisions": candidate_decisions,
        "physics_target_residual_64_ready": {
            "status": "pass" if selected else "blocked",
            "evidence_level": "single_seed_conditional",
            "selected_candidate": selected,
            "reason": (
                "user_authorized_seed0_admission"
                if selected
                else "seed0_target_residual_gate_failed"
            ),
        },
        "pino_inverse_adapter_preflight_ready": {
            "status": "pass" if selected else "blocked",
            "reason": (
                "seed0_conditional_admission_allows_preflight"
                if selected
                else "target_residual_gate_failed"
            ),
        },
        "pino_paired_mpb_validation_ready": {"status": "blocked", "reason": "not_run"},
        "pino_inverse_rerank_64_ready": {
            "status": "blocked",
            "reason": "adapter_and_paired_mpb_gates_not_run",
        },
        "pino_inverse_rerank_multires_ready": {
            "status": "blocked",
            "reason": "outside_fixed_resolution_pilot_scope",
        },
        "field_inverse_generation_ready": {
            "status": "blocked",
            "reason": "field_generation_is_out_of_scope",
        },
        "next_required_stage": (
            "pino_inverse_adapter_preflight_64"
            if selected
            else "analyze_seed0_pilot_failure"
        ),
        "test_accessed": False,
        "mpb_accessed": False,
    }


def run_field_eigen_b4_target_residual(
    path: str | Path,
    *,
    run_dir: Path | None = None,
) -> Path:
    """Run the frozen validation-only target-frequency residual diagnostic."""

    import torch

    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    scope = _diagnostic_scope(raw)
    split, required_seeds, resolution = _validate_contract(raw)
    runtime = dict(raw["runtime"])
    device_info = resolve_device(DeviceConfig.from_mapping(runtime), torch_module=torch)
    device = device_info.torch_device
    output = dict(raw.get("output") or {})
    resume_value = output.get("resume_run_dir")
    if run_dir is not None and resume_value:
        raise ValueError("Specify either run_dir or output.resume_run_dir, not both.")
    out = make_run_dir(
        str(dict(raw["experiment"])["experiment_id"]),
        runs_dir=ROOT / str(output.get("runs_dir", "runs")),
        run_dir=run_dir or (_root_path(str(resume_value)) if resume_value else None),
    )
    logger = setup_logging(
        log_file=out / "run.log",
        run_meta={"stage": "field_eigen_te_b4_target_residual_64"},
    )
    ensure_figures_dir(out)
    write_resolved_config(out, yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
    started = time.perf_counter()

    dataset = dict(raw["dataset"])
    input_hashes = _validate_dataset_lineage(dataset)
    candidates = _candidate_cases(raw, required_seeds=required_seeds)
    cases: list[dict[str, Any]] = []
    for candidate, candidate_cases in candidates.items():
        for case in candidate_cases:
            cases.append({**case, "candidate": candidate})
    if scope == "seed0_pilot":
        source_decision, decision_lineage = _validate_p6_pilot_source(
            raw,
            candidates=candidates,
        )
    else:
        source_decision, decision_lineage = _validate_source_decision(
            raw,
            candidates=candidates,
        )
    input_lineage = {**input_hashes, **decision_lineage}

    sidecar = _read_sidecar(_root_path(str(dataset["k_path_sidecar_path"])))
    k_points = np.asarray(sidecar["k_points"], dtype=np.float32)
    unique_k_indices = _unique_k_indices(sidecar)
    evaluation = dict(raw["evaluation"])
    if len(k_points) != int(evaluation["path_index_count"]):
        raise ValueError("Configured path-index count does not match the frozen k-path.")
    if len(unique_k_indices) != int(evaluation["unique_coordinate_count"]):
        raise ValueError("Configured unique-coordinate count does not match the frozen k-path.")
    band_bounds = tuple(int(value) for value in dict(raw["normalization"])["band_range_one_based"])
    bands = np.arange(band_bounds[0], band_bounds[1] + 1, dtype=np.int64)

    structures = _load_b3_structures(
        _root_path(str(dataset["field_hdf5_path"])),
        splits=(split,),
    )
    if len(structures[split]) != int(evaluation["structures_per_split"]):
        raise ValueError("Configured validation structure count does not match the frozen field dataset.")
    with np.load(_root_path(str(dataset["v2_frequency_dataset_path"])), allow_pickle=True) as bundle:
        truth = np.asarray(bundle["Wh"], dtype=np.float32)
        metadata_jsonl = bundle["metadata_jsonl"].item()
    _verify_v2_source_lineage(structures, metadata_jsonl)
    if int(bands[-1]) > int(truth.shape[2]) or len(k_points) != int(truth.shape[1]):
        raise ValueError("V2 frequency target shape does not match the frozen k-path and band range.")

    floor_scale = float(evaluation["residual_solver_floor_scale"])
    if not math.isfinite(floor_scale) or floor_scale <= 0.0:
        raise ValueError("Residual solver-floor scale must be positive and finite.")
    batch_size = int(evaluation["batch_size"])
    if batch_size <= 0:
        raise ValueError("Target-residual batch size must be positive.")

    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    expected_rows = len(structures[split]) * len(unique_k_indices) * len(bands)
    for case in cases:
        model, identity = _load_checkpoint_case(
            case,
            device,
            torch,
            expected_dataset_hash=input_hashes["field_hdf5"],
        )
        identities.append(
            {
                "case": case["name"],
                "candidate": case["candidate"],
                "seed": case["seed"],
                **identity,
            }
        )
        fingerprint = _json_fingerprint(
            {
                "config": {key: value for key, value in raw.items() if key != "output"},
                "input_lineage": input_lineage,
                "case": case,
                "identity": identity,
                "split": split,
                "resolution": resolution,
            }
        )
        partial = out / "partials" / f"{case['name']}__{split}__r{resolution}.json"
        case_rows = _load_resumable_partial(
            partial,
            fingerprint=fingerprint,
            expected_rows=expected_rows,
        )
        resumed = case_rows is not None
        elapsed = 0.0
        if case_rows is None:
            case_rows, elapsed = _case_rows(
                case=case,
                model=model,
                identity=identity,
                structures=structures[split],
                split=split,
                k_points=k_points,
                unique_k_indices=unique_k_indices,
                bands=bands,
                truth=truth,
                resolution=resolution,
                residual_solver_floor_scale=floor_scale,
                normalization=dict(raw["normalization"]),
                batch_size=batch_size,
                device=device,
                torch=torch,
            )
            _atomic_write_json(
                partial,
                {
                    "input_fingerprint": fingerprint,
                    "case": str(case["name"]),
                    "split": split,
                    "resolution": resolution,
                    "row_count": len(case_rows),
                    "rows": case_rows,
                },
            )
        per_condition_ms = 1000.0 * elapsed / max(len(case_rows), 1)
        for row in case_rows:
            row["eval_time_s_case"] = elapsed
            row["inference_time_per_condition_ms"] = per_condition_ms
        rows.extend(case_rows)
        history.append(
            {
                "case": case["name"],
                "candidate": case["candidate"],
                "seed": case["seed"],
                "split": split,
                "resolution": resolution,
                "row_count": len(case_rows),
                "resumed": resumed,
                "eval_time_s": elapsed,
                "inference_time_per_condition_ms": per_condition_ms,
            }
        )
        logger.info(
            "%s target-residual case %s rows=%d",
            "Resumed" if resumed else "Completed",
            case["name"],
            len(case_rows),
        )

    structure_rows = _structure_scores(rows)
    statistics = dict(raw["statistics"])
    thresholds = dict(raw["promotion_thresholds"]["target_residual_calibration"])
    calibration = _calibration_rows(
        structure_rows,
        bootstrap_seed=int(statistics["bootstrap_seed"]),
        bootstrap_repeats=int(statistics["bootstrap_repeats"]),
        thresholds=thresholds,
    )
    if scope == "seed0_pilot":
        decision = _pilot_candidate_decision(
            calibration,
            structure_rows,
            candidates=candidates,
            thresholds=thresholds,
        )
    else:
        decision = _candidate_decision(
            calibration,
            structure_rows,
            candidates=candidates,
            source_decision=source_decision,
            thresholds=thresholds,
            required_seeds=required_seeds,
        )
    wall_time = time.perf_counter() - started

    _write_csv(out / "target_residual_per_case.csv", rows)
    _write_csv(out / "target_residual_structure_scores.csv", structure_rows)
    _write_csv(out / "target_residual_calibration.csv", calibration)
    _write_csv(out / "history.csv", history)
    write_json(
        out / "lineage.json",
        {
            "inputs": input_lineage,
            "diagnostic_scope": scope,
            "source_decision": source_decision,
            "model_identities": identities,
            "evaluation_split": split,
            "resolution": resolution,
            "unique_k_indices": unique_k_indices,
            "band_indices_one_based": [int(value) for value in bands],
            "test_accessed": False,
            "mpb_accessed": False,
        },
    )
    write_json(out / "candidate_decision.json", decision)
    write_metrics(
        out,
        {
            "case_count": len(cases),
            "structure_score_count": len(structure_rows),
            "condition_row_count": len(rows),
            "partial_artifact_count": len(history),
            "resumed_partial_count": sum(bool(row["resumed"]) for row in history),
            "eval_time_s": float(sum(float(row["eval_time_s"]) for row in history)),
            "wall_time_total_s": wall_time,
            "peak_gpu_memory_mb": peak_gpu_memory_mb(device, torch_module=torch),
            "decision": decision,
        },
    )
    write_results(
        out,
        {
            "status": (
                "p6_seed0_target_residual_pilot_completed"
                if scope == "seed0_pilot"
                else "target_residual_64_diagnostic_completed"
            ),
            "decision": decision,
            "test_accessed": False,
            "mpb_accessed": False,
        },
    )
    write_run_meta(
        out,
        {
            **collect_device_metadata(device_info),
            "status": (
                "p6_seed0_target_residual_pilot_completed"
                if scope == "seed0_pilot"
                else "target_residual_64_diagnostic_completed"
            ),
            "test_accessed": False,
            "mpb_accessed": False,
            "wall_time_total_s": wall_time,
        },
    )
    return out


__all__ = [
    "is_field_eigen_b4_target_residual_config",
    "run_field_eigen_b4_target_residual",
]
