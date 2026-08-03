"""Validation-only dense-k selection for repaired B4 PINO candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from phc2d.forward.field_eigen_b4 import (
    ROOT,
    _atomic_write_json,
    _calibration_rows,
    _case_rows,
    _hash,
    _json_fingerprint,
    _load_b3_structures,
    _load_checkpoint_case,
    _load_resumable_partial,
    _partial_path,
    _read_sidecar,
    _root_path,
    _support_rows,
    _validate_configured_cases,
    _validate_dataset_lineage,
    _verify_v2_source_lineage,
    _write_csv,
)
from phc2d.runs import ensure_figures_dir, make_run_dir, setup_logging, write_json, write_metrics, write_resolved_config, write_results, write_run_meta
from phc2d.training import DeviceConfig, collect_device_metadata, peak_gpu_memory_mb, resolve_device


def _candidate_cases(raw: Mapping[str, Any], *, required_seeds: set[int]) -> dict[str, list[dict[str, Any]]]:
    model = dict(raw["candidate_model"])
    resolved: dict[str, list[dict[str, Any]]] = {}
    for candidate, payload in dict(raw["candidates"]).items():
        cases: list[dict[str, Any]] = []
        for row in list(dict(payload).get("cases") or []):
            cases.append({"name": str(row["name"]), "kind": "pino", "seed": int(row["seed"]), "checkpoint_path": str(row["checkpoint_path"]), "checkpoint_sha256": str(row["checkpoint_sha256"]), "model_name": str(model["name"]), "model_params": dict(model["params"])})
        seeds = [int(case["seed"]) for case in cases]
        if set(seeds) != required_seeds or len(seeds) != len(required_seeds):
            raise ValueError(f"Validation candidate {candidate!r} must freeze exactly seeds {sorted(required_seeds)}.")
        if len({str(case["name"]) for case in cases}) != len(cases):
            raise ValueError(f"Validation candidate {candidate!r} has duplicate case names.")
        resolved[str(candidate)] = cases
    if not resolved:
        raise ValueError("B4 validation selector requires at least one named candidate.")
    return resolved


def _transfer_rows(support: list[dict[str, Any]], *, split: str, thresholds: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    base_resolution = int(dict(thresholds["resolution_transfer"])["base_resolution"])
    for case in sorted({str(row["case"]) for row in support}):
        base_rows = [row for row in support if str(row["case"]) == case and str(row["split"]) == split and int(row["resolution"]) == base_resolution and str(row["metric_scope"]) == "all_k"]
        for resolution in (int(value) for value in dict(thresholds["resolution_transfer"])["evaluation_resolutions"]):
            current_rows = [row for row in support if str(row["case"]) == case and str(row["split"]) == split and int(row["resolution"]) == resolution and str(row["metric_scope"]) == "all_k"]
            if len(base_rows) != 1 or len(current_rows) != 1:
                continue
            base, current = base_rows[0], current_rows[0]
            frequency_change = 100.0 * (float(current["frequency_mre_pct"]) / max(float(base["frequency_mre_pct"]), 1.0e-12) - 1.0)
            residual_change = 100.0 * (float(current["maxwell_relative_residual_median"]) / max(float(base["maxwell_relative_residual_median"]), 1.0e-12) - 1.0)
            limits = dict(thresholds["resolution_transfer"])
            output.append({"case": case, "kind": base["kind"], "seed": base["seed"], "resolution": resolution, "frequency_mre_change_pct": frequency_change, "residual_median_change_pct": residual_change, "frequency_pass": frequency_change <= float(limits["frequency_mre_degradation_max_pct"]), "residual_pass": residual_change <= float(limits["residual_median_degradation_max_pct"]), "all_finite": bool(np.isfinite([frequency_change, residual_change]).all())})
    return output


def _candidate_decision(support: list[dict[str, Any]], calibration: list[dict[str, Any]], transfer: list[dict[str, Any]], *, candidate_cases: list[dict[str, Any]], thresholds: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    from phc2d.forward.field_eigen_b4 import _decision

    names = {str(case["name"]) for case in candidate_cases}
    subset_support = [row for row in support if str(row["kind"]) == "fno" or str(row["case"]) in names]
    subset_calibration = [row for row in calibration if str(row["kind"]) == "fno" or str(row["case"]) in names]
    subset_transfer = [row for row in transfer if str(row["kind"]) == "fno" or str(row["case"]) in names]
    return _decision(subset_support, subset_calibration, subset_transfer, thresholds=thresholds, admission_contract=contract)


def _selection_score(support: list[dict[str, Any]], transfer: list[dict[str, Any]], *, candidate_cases: list[dict[str, Any]], split: str) -> tuple[float, float]:
    candidate_by_seed = {int(case["seed"]): str(case["name"]) for case in candidate_cases}
    factors: list[float] = []
    for seed, name in candidate_by_seed.items():
        pino = [row for row in support if str(row["case"]) == name and str(row["split"]) == split and int(row["resolution"]) == 64 and str(row["metric_scope"]) == "support" and str(row["scope_value"]) == "unseen_k"]
        fno = [row for row in support if str(row["kind"]) == "fno" and int(row["seed"]) == seed and str(row["split"]) == split and int(row["resolution"]) == 64 and str(row["metric_scope"]) == "support" and str(row["scope_value"]) == "unseen_k"]
        if len(pino) != 1 or len(fno) != 1:
            return (float("inf"), float("inf"))
        factors.append(float(pino[0]["frequency_mre_pct"]) / max(float(fno[0]["frequency_mre_pct"]), 1.0e-12))
    degradation = [float(row["residual_median_change_pct"]) for row in transfer if str(row["case"]) in set(candidate_by_seed.values()) and int(row["resolution"]) == 128]
    return (max(factors) if factors else float("inf"), max(degradation) if degradation else float("inf"))


def _select_candidate(decisions: Mapping[str, Mapping[str, Any]], scores: Mapping[str, tuple[float, float]]) -> dict[str, Any]:
    eligible = [name for name, decision in decisions.items() if all(dict(decision[key])["status"] == "pass" for key in ("physics_frequency_crosscheck_ready", "physics_residual_rerank_ready", "cross_resolution_zero_shot_pass"))]
    selected = min(eligible, key=lambda name: (scores[name][0], scores[name][1], name)) if eligible else None
    return {"status": "selected" if selected else "blocked", "eligible_candidates": sorted(eligible), "selected_candidate": selected, "tie_break_scores": {name: {"worst_seed_unseen_k_frequency_factor": scores[name][0], "worst_seed_128_residual_degradation_pct": scores[name][1]} for name in sorted(scores)}, "inverse_status": "blocked"}


def run_field_eigen_b4_validation_selector(path: str | Path, *, run_dir: Path | None = None) -> Path:
    import torch

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    contract = dict(raw["validation_contract"]); split = str(contract["evaluation_split"]); required_seeds = {int(seed) for seed in contract["required_seeds"]}
    if split != "val" or required_seeds != {0, 1, 2}:
        raise ValueError("B4 validation selector must be validation-only with seeds {0,1,2}.")
    if bool(contract.get("allow_test", False)):
        raise ValueError("B4 validation selector must reject test access.")
    runtime = dict(raw["runtime"]); device_info = resolve_device(DeviceConfig.from_mapping(runtime), torch_module=torch); device = device_info.torch_device
    output = dict(raw.get("output") or {}); resume_value = output.get("resume_run_dir")
    if run_dir is not None and resume_value:
        raise ValueError("Specify either run_dir or output.resume_run_dir, not both.")
    out = make_run_dir(str(dict(raw["experiment"])["experiment_id"]), runs_dir=ROOT / str(output.get("runs_dir", "runs")), run_dir=run_dir or (_root_path(str(resume_value)) if resume_value else None))
    logger = setup_logging(log_file=out / "run.log", run_meta={"stage": "field_eigen_te_b4_validation_selector"}); ensure_figures_dir(out); write_resolved_config(out, yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
    dataset = dict(raw["dataset"]); input_hashes = _validate_dataset_lineage(dataset); thresholds = dict(raw["promotion_thresholds"]); dense = dict(raw["dense_k"])
    reference = dict(raw["residual_solver_floor_reference"]); factors_raw = dict(reference["factors"]); factors = {int(resolution): float(factors_raw.get(str(resolution), factors_raw.get(resolution))) for resolution in dense["resolutions"]}
    if any(not np.isfinite(value) or value <= 0.0 for value in factors.values()):
        raise ValueError("Validation selector solver-floor factors must be positive and finite.")
    sidecar = _read_sidecar(_root_path(str(dataset["k_path_sidecar_path"]))); k_points = np.asarray(sidecar["k_points"], dtype=np.float32); band_bounds = dict(raw["normalization"])["band_range_one_based"]; bands = np.arange(int(band_bounds[0]), int(band_bounds[1]) + 1, dtype=np.int64)
    seen = {int(value) for value in dense["seen_k_indices"]}; duplicate = {int(value) for group in sidecar["duplicate_coordinate_groups"] for value in group[1:]}; unique = set(range(len(k_points))) - duplicate
    if len(k_points) != int(dense["path_index_count"]) or len(unique) != int(dense["unique_coordinate_count"]) or len(seen) != int(dense["seen_k_count"]):
        raise ValueError("Validation selector dense-k cardinalities do not match the frozen sidecar.")
    structures = _load_b3_structures(_root_path(str(dataset["field_hdf5_path"])), splits=(split,))
    if len(structures[split]) != int(dense["structures_per_split"]):
        raise ValueError("Validation selector structure count does not match its frozen contract.")
    v2 = np.load(_root_path(str(dataset["v2_frequency_dataset_path"])), allow_pickle=True); truth = np.asarray(v2["Wh"], dtype=np.float32); _verify_v2_source_lineage(structures, v2["metadata_jsonl"].item())
    fno_cases = list(dict(raw["fno"])["cases"]); candidates = _candidate_cases(raw, required_seeds=required_seeds); all_cases = fno_cases + [case for cases in candidates.values() for case in cases]; _validate_configured_cases(all_cases, required_seeds=required_seeds)
    input_lineage = {**input_hashes, "candidate_model": dict(raw["candidate_model"]), "residual_solver_floor_reference": reference}; rows: list[dict[str, Any]] = []; history: list[dict[str, Any]] = []; identities: list[dict[str, Any]] = []
    for case in all_cases:
        model, identity = _load_checkpoint_case(case, device, torch, expected_dataset_hash=input_hashes["field_hdf5"]); identities.append({"case": case["name"], "kind": case["kind"], "seed": case["seed"], **identity})
        for resolution in (int(value) for value in dense["resolutions"]):
            expected = len(structures[split]) * len(k_points) * len(bands); fingerprint = _json_fingerprint({"config": {key: value for key, value in raw.items() if key != "output"}, "input_lineage": input_lineage, "case": case, "identity": identity, "split": split, "resolution": resolution})
            partial = _partial_path(out, case=str(case["name"]), split=split, resolution=resolution); partial_rows = _load_resumable_partial(partial, fingerprint=fingerprint, expected_rows=expected); resumed = partial_rows is not None; elapsed = 0.0
            if partial_rows is None:
                partial_rows, elapsed = _case_rows(case=case, model=model, identity=identity, structures=structures[split], split=split, k_points=k_points, bands=bands, truth=truth, resolution=resolution, residual_solver_floor_factors=factors, normalization=dict(raw["normalization"]), device=device, torch=torch)
                for row in partial_rows:
                    row["eval_time_s_split"] = elapsed; row["inference_time_per_sample_ms"] = 1000.0 * elapsed / max(len(partial_rows), 1)
                _atomic_write_json(partial, {"input_fingerprint": fingerprint, "case": str(case["name"]), "split": split, "resolution": resolution, "row_count": len(partial_rows), "rows": partial_rows})
            rows.extend(partial_rows); history.append({"case": case["name"], "kind": case["kind"], "seed": case["seed"], "split": split, "resolution": resolution, "row_count": len(partial_rows), "resumed": resumed, "eval_time_s": elapsed}); logger.info("%s validation dense-k %s resolution=%d", "Resumed" if resumed else "Completed", case["name"], resolution)
    support = _support_rows(rows, seen=seen, unique_indices=unique, sidecar=sidecar); calibration = _calibration_rows([row for row in rows if int(row["resolution"]) == 64 and int(row["k_index"]) not in seen and int(row["k_index"]) in unique], bootstrap_seed=int(dict(raw["statistics"])["bootstrap_seed"]), bootstrap_repeats=int(dict(raw["statistics"])["bootstrap_repeats"])); transfer = _transfer_rows(support, split=split, thresholds=thresholds)
    decisions = {name: _candidate_decision(support, calibration, transfer, candidate_cases=cases, thresholds=thresholds, contract=contract) for name, cases in candidates.items()}; scores = {name: _selection_score(support, transfer, candidate_cases=cases, split=split) for name, cases in candidates.items()}; selection = _select_candidate(decisions, scores)
    selection["candidate_decisions"] = decisions; selection["selection_split"] = split
    _write_csv(out / "candidate_dense_k_metrics.csv", rows); _write_csv(out / "candidate_k_support_metrics.csv", support); _write_csv(out / "candidate_residual_calibration.csv", calibration); _write_csv(out / "candidate_resolution_transfer.csv", transfer); _write_csv(out / "history.csv", history)
    write_json(out / "lineage.json", {"inputs": input_lineage, "model_identities": identities, "seen_k_indices": sorted(seen), "unique_k_indices": sorted(unique), "test_accessed": False})
    write_json(out / "candidate_selection.json", selection)
    if selection["selected_candidate"]:
        selected = str(selection["selected_candidate"]); manifest = {"selection": selection, "selection_split": split, "evaluation_split": "test", "cases": [{"seed": case["seed"], "checkpoint_path": case["checkpoint_path"], "checkpoint_sha256": case["checkpoint_sha256"]} for case in candidates[selected]], "model": dict(raw["candidate_model"]), "dataset_hash": input_hashes["field_hdf5"]}
        write_json(out / "frozen_candidate_manifest.json", manifest)
    write_metrics(out, {"case_count": len(all_cases), "partial_artifact_count": len(history), "resumed_partial_count": sum(bool(row["resumed"]) for row in history), "peak_gpu_memory_mb": peak_gpu_memory_mb(device, torch_module=torch), "selection": selection}); write_results(out, {"status": "b4_validation_selector_completed", "selection": selection, "inverse_status": "blocked"}); write_run_meta(out, {**collect_device_metadata(device_info), "status": "b4_validation_selector_completed", "test_accessed": False})
    return out


__all__ = ["run_field_eigen_b4_validation_selector"]
