"""B4 dense-k and resolution-transfer evaluator for TE Bloch eigenfields.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from phc2d.forward.field_eigen_v2 import ROOT, _epsilon_on_field_grid_numpy, _hash, _periodic_grid, _phase_align_numpy, _relative_l2_numpy, _torch_active_complex, active_channels_to_complex, reconstruct_full_te_fields
from phc2d.models import architecture_fingerprint, build_model
from phc2d.physics import canonicalize_complex_efield, evaluate_te_bloch_eigen_residual
from phc2d.runs import ensure_figures_dir, make_run_dir, setup_logging, write_json, write_metrics, write_resolved_config, write_results, write_run_meta
from phc2d.training import DeviceConfig, collect_device_metadata, peak_gpu_memory_mb, resolve_device, synchronize_device


def is_field_eigen_b4_dense_config(path: str | Path) -> bool:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return str(dict(raw.get("experiment") or {}).get("execution", "")).lower() in {
        "field_eigen_te_b4_dense_k_gate",
        "field_eigen_te_b4_validation_selector",
        "field_eigen_te_b4_target_residual_64",
        "field_eigen_te_p6_seed0_target_residual_pilot",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist one recoverable B4 artifact without exposing a partial write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    temporary.replace(path)


def _json_fingerprint(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA256 for resume provenance comparisons."""

    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    import hashlib

    return hashlib.sha256(encoded).hexdigest()


def _root_path(value: str | Path) -> Path:
    """Resolve repository-relative paths while accepting an explicit absolute resume path."""

    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for group in range(len(unique)):
            indices = np.flatnonzero(inverse == group)
            ranks[indices] = ranks[indices].mean()
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return float("nan")
    xr, yr = _rank(x[finite]), _rank(y[finite])
    if np.std(xr) == 0.0 or np.std(yr) == 0.0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _bootstrap_structure_ci(rows: list[dict[str, Any]], *, seed: int, repeats: int) -> tuple[float, float, float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["source_sample_id"])].append(row)
    source_ids = sorted(groups)
    if len(source_ids) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repeats):
        sampled = rng.choice(source_ids, size=len(source_ids), replace=True)
        materialized = [item for source_id in sampled for item in groups[str(source_id)]]
        values.append(_spearman(np.asarray([float(item["maxwell_relative_residual"]) for item in materialized]), np.asarray([float(item["frequency_mre_pct"]) for item in materialized])))
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return (_spearman(np.asarray([float(item["maxwell_relative_residual"]) for item in rows]), np.asarray([float(item["frequency_mre_pct"]) for item in rows])), float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))) if len(finite) else (float("nan"), float("nan"), float("nan"))


def _read_sidecar(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    points = np.asarray(raw["k_points"], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) != int(raw["Nk"]):
        raise ValueError("Invalid frozen k-path sidecar.")
    duplicate_groups = [tuple(int(value) for value in group) for group in raw.get("duplicate_coordinate_groups", [])]
    if len(points) - sum(len(group) - 1 for group in duplicate_groups) != len(np.unique(points, axis=0)):
        raise ValueError("k-path duplicate-coordinate metadata is inconsistent.")
    return raw


def _source_index(source_id: str) -> int:
    try:
        return int(source_id.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Cannot map source_sample_id {source_id!r} to V2 row.") from exc


def _load_b3_structures(path: Path, *, splits: tuple[str, ...]) -> dict[str, dict[str, dict[str, Any]]]:
    import h5py

    result: dict[str, dict[str, dict[str, Any]]] = {split: {} for split in splits}
    with h5py.File(path, "r") as handle:
        decode = lambda values: np.asarray([value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values])
        source_id = decode(np.asarray(handle["source_sample_id"]))
        source_split = decode(np.asarray(handle["source_split"]))
        source_hash = decode(np.asarray(handle["source_sample_hash"]))
        k_points = np.asarray(handle["k_points"], dtype=np.float32)
        bands = np.asarray(handle["band_index"], dtype=np.int64)
        epsilon = np.asarray(handle["epsilon"], dtype=np.float32)
        dx = np.asarray(handle["dx"], dtype=np.float32)
        dy = np.asarray(handle["dy"], dtype=np.float32)
        efield = np.asarray(handle["efield_complex"])
        hfield = np.asarray(handle["hfield_complex"])
    for index, split in enumerate(source_split):
        if split not in result:
            continue
        source = str(source_id[index])
        payload = result[split].setdefault(source, {"epsilon": epsilon[index], "source_hash": str(source_hash[index]), "dx": float(dx[index]), "dy": float(dy[index]), "fields": {}})
        key = (tuple(float(value) for value in k_points[index]), int(bands[index]))
        payload["fields"][key] = np.concatenate((efield[index, ..., 0, :2], hfield[index, ..., 0, 2:3]), axis=-1)
    return result


def _verify_v2_source_lineage(structures: Mapping[str, Mapping[str, Mapping[str, Any]]], metadata_jsonl: Any) -> None:
    """Require B3 source IDs, split names, V2 rows, and rho hashes to agree."""

    records = [json.loads(line) for line in str(metadata_jsonl).splitlines() if str(line).strip()]
    by_id = {str(record["sample_id"]): record for record in records}
    for split, items in structures.items():
        for source_id, payload in items.items():
            record = by_id.get(str(source_id))
            if record is None:
                raise ValueError(f"B3 source {source_id!r} is absent from V2 metadata_jsonl.")
            if str(record.get("split")) != str(split):
                raise ValueError(f"B3/V2 split mismatch for {source_id!r}.")
            if str(record.get("rho_hash")) != str(payload["source_hash"]):
                raise ValueError(f"B3/V2 rho hash mismatch for {source_id!r}.")
            if int(record.get("sample_index")) != _source_index(str(source_id)):
                raise ValueError(f"B3/V2 source index mismatch for {source_id!r}.")


def _validate_required_hash(*, label: str, path: Path, expected: str) -> str:
    """Require a configured SHA256 to match its immutable B4 input."""

    actual = _hash(path)
    if not expected or actual != str(expected):
        raise ValueError(f"{label} SHA256 mismatch for {path}.")
    return actual


def _load_checkpoint_case(
    case: Mapping[str, Any],
    device: Any,
    torch: Any,
    *,
    expected_dataset_hash: str,
) -> tuple[Any, dict[str, Any]]:
    checkpoint_path = _root_path(str(case["checkpoint_path"]))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Frozen B4 checkpoint does not exist: {checkpoint_path}")
    checkpoint_sha256 = _validate_required_hash(
        label=f"{case['name']} checkpoint",
        path=checkpoint_path,
        expected=str(case.get("checkpoint_sha256", "")),
    )
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    dataset_hash = str(payload.get("dataset_hash", ""))
    if dataset_hash != expected_dataset_hash:
        raise ValueError(f"{case['name']} checkpoint dataset lineage does not match the frozen B3 HDF5.")
    model = build_model(str(case["model_name"]), **dict(case["model_params"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_hash": dataset_hash,
        "frequency_mean": float(payload["log_frequency_mean"]),
        "frequency_std": float(payload["log_frequency_std"]),
        "architecture_fingerprint": architecture_fingerprint(model),
    }


def _resolve_pino_cases(raw: Mapping[str, Any], *, expected_dataset_hash: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load exactly one frozen frequency-guard manifest and validate its lineage."""

    pino = dict(raw.get("pino") or {})
    checkpoint_manifest = str(pino.get("checkpoint_manifest", ""))
    if not checkpoint_manifest or checkpoint_manifest == "auto":
        raise ValueError("B4 dense-k requires pino.checkpoint_manifest to name the exact frozen manifest; auto-discovery is forbidden.")
    manifest_path = _root_path(checkpoint_manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Frozen B4 manifest does not exist: {manifest_path}")
    manifest_sha256 = _validate_required_hash(
        label="Frozen B4 manifest",
        path=manifest_path,
        expected=str(pino.get("checkpoint_manifest_sha256", "")),
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(payload.get("evaluation_split")) != str(dict(raw["admission_contract"])["evaluation_split"]):
        raise ValueError("Frozen B4 manifest evaluation_split does not match the dense-k admission contract.")
    if str(payload.get("dataset_hash", "")) != expected_dataset_hash:
        raise ValueError("Frozen B4 manifest dataset_hash does not match the frozen B3 HDF5.")
    required_seeds = {int(seed) for seed in dict(raw["admission_contract"])["required_seeds"]}
    manifest_cases = list(payload.get("cases") or [])
    manifest_seeds = [int(row["seed"]) for row in manifest_cases]
    if set(manifest_seeds) != required_seeds or len(manifest_seeds) != len(required_seeds):
        raise ValueError("Frozen B4 manifest must contain exactly the required three distinct seeds.")
    model = dict(payload["model"])
    cases = [
        {
            "name": f"pino_seed{row['seed']}",
            "kind": "pino",
            "seed": int(row["seed"]),
            "checkpoint_path": str(row["checkpoint_path"]),
            "checkpoint_sha256": str(row.get("checkpoint_sha256", "")),
            "model_name": model["name"],
            "model_params": model["params"],
        }
        for row in manifest_cases
    ]
    return cases, {"manifest_path": str(manifest_path), "manifest_sha256": manifest_sha256, "evaluation_split": str(payload["evaluation_split"]), "dataset_hash": str(payload["dataset_hash"])}


def _validate_configured_cases(cases: list[dict[str, Any]], *, required_seeds: set[int]) -> None:
    """Reject aliases, missing seeds, or unpinned FNO checkpoint inputs before GPU work."""

    fno_cases = [case for case in cases if str(case["kind"]) == "fno"]
    fno_seeds = [int(case["seed"]) for case in fno_cases]
    if set(fno_seeds) != required_seeds or len(fno_seeds) != len(required_seeds):
        raise ValueError("B4 FNO cases must contain exactly the required three distinct seeds.")
    case_names = [str(case["name"]) for case in cases]
    if len(case_names) != len(set(case_names)):
        raise ValueError("B4 checkpoint case names must be unique.")
    for case in cases:
        if not str(case.get("checkpoint_path", "")) or not str(case.get("checkpoint_sha256", "")):
            raise ValueError(f"{case['name']} must freeze checkpoint_path and checkpoint_sha256.")


def _case_rows(
    *,
    case: Mapping[str, Any],
    model: Any,
    identity: Mapping[str, Any],
    structures: Mapping[str, Mapping[str, Any]],
    split: str,
    k_points: np.ndarray,
    bands: np.ndarray,
    truth: np.ndarray,
    resolution: int,
    residual_solver_floor_factors: Mapping[int, float],
    normalization: Mapping[str, Any],
    device: Any,
    torch: Any,
) -> tuple[list[dict[str, Any]], float]:
    epsilon_range = tuple(float(value) for value in normalization["epsilon_range"])
    k_scale = float(normalization["k_scale"])
    band_range = tuple(float(value) for value in normalization["band_range_one_based"])
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for source_id, payload in structures.items():
        grid = _periodic_grid(payload["epsilon"][None], resolution, resolution, include_coordinates=True, epsilon_range=epsilon_range)
        grid_tensor = torch.as_tensor(grid, device=device)
        source_index = _source_index(source_id)
        for k_index, k_point in enumerate(k_points):
            for band in bands:
                context = np.asarray([[k_point[0] / k_scale, k_point[1] / k_scale, (float(band) - band_range[0]) / max(band_range[1] - band_range[0], 1.0)]], dtype=np.float32)
                with torch.no_grad():
                    predicted, frequency_z = model(grid_tensor, torch.as_tensor(context, device=device))
                frequency = float(torch.exp(frequency_z * identity["frequency_std"] + identity["frequency_mean"]).item())
                active = _torch_active_complex(torch, predicted).detach().cpu().numpy()[0]
                predicted_e, _ = reconstruct_full_te_fields(active[None])
                residual_value = boundary_value = raw_residual_value = raw_boundary_value = float("nan")
                try:
                    canonical = canonicalize_complex_efield(
                        predicted_e[0], epsilon=payload["epsilon"], dx=1.0 / float(resolution), dy=1.0 / float(resolution)
                    )
                    residual = evaluate_te_bloch_eigen_residual(canonical, epsilon=payload["epsilon"], eigenfrequency=frequency, dx=1.0 / float(resolution), dy=1.0 / float(resolution), bloch_k_point=k_point)
                    raw_residual_value, raw_boundary_value = float(residual["relative_residual"]), float(residual["boundary_relative_residual"])
                    floor = float(residual_solver_floor_factors[int(resolution)])
                    residual_value, boundary_value = raw_residual_value / floor, raw_boundary_value / floor
                except ValueError:
                    pass
                target_frequency = float(truth[source_index, k_index, int(band) - 1])
                row = {"case": case["name"], "kind": case["kind"], "seed": int(case["seed"]), "split": split, "resolution": resolution, "source_sample_id": source_id, "source_sample_hash": payload["source_hash"], "k_index": k_index, "k_x": float(k_point[0]), "k_y": float(k_point[1]), "band_index_one_based": int(band), "frequency_prediction": frequency, "frequency_truth": target_frequency, "frequency_mre_pct": 100.0 * abs(frequency - target_frequency) / max(abs(target_frequency), 1.0e-12), "frequency_mae": abs(frequency - target_frequency), "maxwell_relative_residual": residual_value, "boundary_relative_residual": boundary_value, "raw_maxwell_relative_residual": raw_residual_value, "raw_boundary_relative_residual": raw_boundary_value, "residual_solver_floor_scale": float(residual_solver_floor_factors[int(resolution)])}
                field_truth = payload["fields"].get((tuple(float(value) for value in k_point), int(band)))
                if field_truth is not None and resolution == field_truth.shape[0]:
                    predicted_complex = active_channels_to_complex(predicted.detach().cpu().numpy())[0]
                    aligned = _phase_align_numpy(predicted_complex[None], field_truth[None], payload["epsilon"][None])[0]
                    epsilon_field = _epsilon_on_field_grid_numpy(payload["epsilon"][None], field_truth.shape[:2])
                    row["e_relative_l2"] = float(_relative_l2_numpy(field_truth[None, ..., :2], aligned[None, ..., :2], epsilon_field)[0])
                    row["h_relative_l2"] = float(_relative_l2_numpy(field_truth[None, ..., 2:], aligned[None, ..., 2:])[0])
                else:
                    row["e_relative_l2"] = float("nan"); row["h_relative_l2"] = float("nan")
                rows.append(row)
    synchronize_device(device, torch_module=torch)
    return rows, time.perf_counter() - started


def _support_rows(rows: list[dict[str, Any]], *, seen: set[int], unique_indices: set[int], sidecar: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    segment = np.asarray(sidecar["segment_id"], dtype=np.int64)
    for row in rows:
        if int(row["k_index"]) not in unique_indices:
            continue
        support = "seen_k" if int(row["k_index"]) in seen else "unseen_k"
        for label, value in (("all_k", "all"), ("support", support), ("segment", f"segment_{int(segment[int(row['k_index'])])}"), ("band", f"band_{int(row['band_index_one_based'])}")):
            groups[(row["case"], row["kind"], row["seed"], row["split"], row["resolution"], label, value)].append(row)
    output: list[dict[str, Any]] = []
    for key, values in groups.items():
        output.append({"case": key[0], "kind": key[1], "seed": key[2], "split": key[3], "resolution": key[4], "metric_scope": key[5], "scope_value": key[6], "case_count": len(values), "frequency_mre_pct": float(np.mean([item["frequency_mre_pct"] for item in values])), "frequency_mae": float(np.mean([item["frequency_mae"] for item in values])), "e_relative_l2": float(np.nanmean([item["e_relative_l2"] for item in values])), "h_relative_l2": float(np.nanmean([item["h_relative_l2"] for item in values])), "maxwell_relative_residual_median": float(np.nanmedian([item["maxwell_relative_residual"] for item in values])), "boundary_relative_residual_median": float(np.nanmedian([item["boundary_relative_residual"] for item in values]))})
    return output


def _calibration_rows(rows: list[dict[str, Any]], *, bootstrap_seed: int, bootstrap_repeats: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["case"], row["kind"], row["seed"], row["split"], row["resolution"])].append(row)
    for key, values in groups.items():
        correlation, lower, upper = _bootstrap_structure_ci(values, seed=bootstrap_seed, repeats=bootstrap_repeats)
        residuals = np.asarray([float(item["maxwell_relative_residual"]) for item in values])
        frequencies = np.asarray([float(item["frequency_mre_pct"]) for item in values])
        all_finite = bool(np.isfinite(residuals).all() and np.isfinite(frequencies).all())
        if not all_finite:
            output.append({"case": key[0], "kind": key[1], "seed": key[2], "split": key[3], "resolution": key[4], "calibration_status": "blocked_nonfinite", "all_finite": False, "spearman_residual_frequency_error": correlation, "spearman_bootstrap_ci_lower": lower, "spearman_bootstrap_ci_upper": upper, "lowest_residual_quartile_frequency_mre_pct": float("nan"), "highest_residual_quartile_frequency_mre_pct": float("nan"), "quartile_frequency_mre_reduction_pct": float("nan")})
            continue
        low, high = np.quantile(residuals, [0.25, 0.75])
        low_mre = float(np.mean(frequencies[residuals <= low]))
        high_mre = float(np.mean(frequencies[residuals >= high]))
        output.append({"case": key[0], "kind": key[1], "seed": key[2], "split": key[3], "resolution": key[4], "calibration_status": "ready", "all_finite": True, "spearman_residual_frequency_error": correlation, "spearman_bootstrap_ci_lower": lower, "spearman_bootstrap_ci_upper": upper, "lowest_residual_quartile_frequency_mre_pct": low_mre, "highest_residual_quartile_frequency_mre_pct": high_mre, "quartile_frequency_mre_reduction_pct": 100.0 * (high_mre - low_mre) / max(high_mre, 1.0e-12)})
    return output


def _finite_values(rows: list[dict[str, Any]], *keys: str) -> bool:
    return bool(rows) and all(np.isfinite(float(row[key])) for row in rows for key in keys)


def _decision(
    support: list[dict[str, Any]],
    calibration: list[dict[str, Any]],
    transfer: list[dict[str, Any]],
    *,
    thresholds: Mapping[str, Any],
    admission_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate only frozen B4 criteria and retain every criterion in the artifact."""

    split = str(admission_contract["evaluation_split"])
    required_seeds = {int(seed) for seed in admission_contract["required_seeds"]}
    base_resolution = int(thresholds["resolution_transfer"]["base_resolution"])

    def scoped(kind: str, resolution: int, scope: str) -> list[dict[str, Any]]:
        return [row for row in support if row["kind"] == kind and row["resolution"] == resolution and row["metric_scope"] == "support" and row["scope_value"] == scope and row["split"] == split]

    fno = scoped("fno", base_resolution, "unseen_k")
    pino = scoped("pino", base_resolution, "unseen_k")
    fno_by_seed = {int(row["seed"]): row for row in fno}; pino_by_seed = {int(row["seed"]): row for row in pino}
    paired = [(pino_by_seed[seed], fno_by_seed[seed]) for seed in sorted(required_seeds) if seed in pino_by_seed and seed in fno_by_seed]
    paired_complete = set(pino_by_seed) == required_seeds and set(fno_by_seed) == required_seeds and len(paired) == len(required_seeds)
    frequency_finite = _finite_values([row for pair in paired for row in pair], "frequency_mre_pct")
    frequency_pass = paired_complete and frequency_finite and float(np.mean([item[0]["frequency_mre_pct"] for item in paired])) <= float(thresholds["frequency"]["mean_fno_factor_max"]) * float(np.mean([item[1]["frequency_mre_pct"] for item in paired])) and all(float(pino_row["frequency_mre_pct"]) <= float(thresholds["frequency"]["per_seed_fno_factor_max"]) * float(fno_row["frequency_mre_pct"]) for pino_row, fno_row in paired)
    fno_seen = {int(row["seed"]): row for row in scoped("fno", base_resolution, "seen_k")}; pino_seen = {int(row["seed"]): row for row in scoped("pino", base_resolution, "seen_k")}
    field_pairs = [(pino_seen[seed], fno_seen[seed]) for seed in sorted(required_seeds) if seed in pino_seen and seed in fno_seen]
    field_pass = set(fno_seen) == required_seeds and set(pino_seen) == required_seeds and _finite_values([row for pair in field_pairs for row in pair], "e_relative_l2") and all(float(pino_row["e_relative_l2"]) <= float(thresholds["field"]["seen_k_e_relative_l2_fno_factor_max"]) * float(fno_row["e_relative_l2"]) for pino_row, fno_row in field_pairs)
    pino_cal = [row for row in calibration if row["kind"] == "pino" and row["split"] == split and row["resolution"] == base_resolution]
    cal_by_seed = {int(row["seed"]): row for row in pino_cal}
    residual_finite = _finite_values([row for pair in paired for row in pair], "maxwell_relative_residual_median")
    calibration_pass = set(cal_by_seed) == required_seeds and all(bool(row["all_finite"]) and np.isfinite(float(row["spearman_bootstrap_ci_lower"])) and np.isfinite(float(row["quartile_frequency_mre_reduction_pct"])) and float(row["spearman_bootstrap_ci_lower"]) > float(thresholds["residual_calibration"]["spearman_ci_lower_min_exclusive"]) and float(row["quartile_frequency_mre_reduction_pct"]) >= float(thresholds["residual_calibration"]["quartile_frequency_mre_reduction_min_pct"]) for row in cal_by_seed.values())
    residual_pass = paired_complete and residual_finite and all(float(pino_row["maxwell_relative_residual_median"]) <= float(thresholds["residual"]["per_seed_fno_factor_max"]) * float(fno_row["maxwell_relative_residual_median"]) for pino_row, fno_row in paired) and 100.0 * float(np.mean([1.0 - float(pino_row["maxwell_relative_residual_median"]) / float(fno_row["maxwell_relative_residual_median"]) for pino_row, fno_row in paired])) >= float(thresholds["residual"]["mean_reduction_min_pct"]) and calibration_pass
    required_resolutions = {int(value) for value in thresholds["resolution_transfer"]["evaluation_resolutions"]}
    pino_transfer = [row for row in transfer if row["kind"] == "pino"]
    transfer_keys = {(int(row["seed"]), int(row["resolution"])) for row in pino_transfer}
    transfer_pass = transfer_keys == {(seed, resolution) for seed in required_seeds for resolution in required_resolutions} and all(bool(row["all_finite"]) and bool(row["frequency_pass"]) and bool(row["residual_pass"]) for row in pino_transfer)
    return {
        "admission_contract": dict(admission_contract),
        "thresholds": dict(thresholds),
        "physics_frequency_crosscheck_ready": {"status": "pass" if frequency_pass and field_pass else "blocked", "frequency_pass": frequency_pass, "field_pass": field_pass, "complete_required_seeds": paired_complete},
        "physics_residual_rerank_ready": {"status": "pass" if residual_pass else "blocked", "criteria_pass": residual_pass, "calibration_pass": calibration_pass, "all_required_calibration_finite": all(bool(row.get("all_finite")) for row in cal_by_seed.values()) and set(cal_by_seed) == required_seeds},
        "cross_resolution_zero_shot_pass": {"status": "pass" if transfer_pass else "blocked", "criteria_pass": transfer_pass, "complete_required_seed_resolution_pairs": transfer_keys == {(seed, resolution) for seed in required_seeds for resolution in required_resolutions}},
        "inverse_status": "blocked",
        "real_solver_benefit_status": "blocked",
    }


def _validate_dataset_lineage(dataset: Mapping[str, Any]) -> dict[str, str]:
    """Verify every frozen dense-k dataset and sidecar hash before evaluation."""

    bindings = {
        "field_hdf5": ("field_hdf5_path", "field_hdf5_sha256"),
        "v2_frequency": ("v2_frequency_dataset_path", "v2_frequency_dataset_sha256"),
        "k_path_sidecar": ("k_path_sidecar_path", "k_path_sidecar_sha256"),
    }
    return {
        name: _validate_required_hash(label=name, path=_root_path(str(dataset[path_key])), expected=str(dataset.get(hash_key, "")))
        for name, (path_key, hash_key) in bindings.items()
    }


def _partial_path(run_dir: Path, *, case: str, split: str, resolution: int) -> Path:
    safe_case = "".join(character if character.isalnum() or character in "-_" else "_" for character in case)
    return run_dir / "partials" / f"{safe_case}__{split}__r{int(resolution)}.json"


def _load_resumable_partial(path: Path, *, fingerprint: str, expected_rows: int) -> list[dict[str, Any]] | None:
    """Return a complete compatible partial, otherwise fail closed on stale input."""

    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"B4 partial artifact is not valid JSON: {path}") from exc
    if str(payload.get("input_fingerprint", "")) != fingerprint:
        raise ValueError(f"B4 partial artifact fingerprint mismatch; refusing unsafe resume: {path}")
    rows = list(payload.get("rows") or [])
    if len(rows) != int(expected_rows):
        raise ValueError(f"B4 partial artifact has {len(rows)} rows, expected {expected_rows}: {path}")
    return [dict(row) for row in rows]


def _partial_fingerprint(
    *,
    raw: Mapping[str, Any],
    input_lineage: Mapping[str, Any],
    case: Mapping[str, Any],
    identity: Mapping[str, Any],
    split: str,
    resolution: int,
) -> str:
    fingerprint_config = dict(raw)
    output = dict(fingerprint_config.get("output") or {})
    # The resume directory selects where compatible partials live; it is not an
    # experimental input and must not invalidate a first-run partial.
    output.pop("resume_run_dir", None)
    fingerprint_config["output"] = output
    return _json_fingerprint({
        "config": fingerprint_config,
        "input_lineage": input_lineage,
        "case": dict(case),
        "identity": dict(identity),
        "split": split,
        "resolution": int(resolution),
    })


def run_field_eigen_b4_dense_gate(path: str | Path, *, run_dir: Path | None = None) -> Path:
    import torch

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if str(dict(raw.get("experiment") or {}).get("execution", "")).lower() in {
        "field_eigen_te_b4_target_residual_64",
        "field_eigen_te_p6_seed0_target_residual_pilot",
    }:
        from phc2d.forward.field_eigen_b4_target_residual import run_field_eigen_b4_target_residual

        return run_field_eigen_b4_target_residual(path, run_dir=run_dir)
    if str(dict(raw.get("experiment") or {}).get("execution", "")).lower() == "field_eigen_te_b4_validation_selector":
        from phc2d.forward.field_eigen_b4_selector import run_field_eigen_b4_validation_selector

        return run_field_eigen_b4_validation_selector(path, run_dir=run_dir)
    runtime = dict(raw["runtime"]); device_info = resolve_device(DeviceConfig.from_mapping(runtime), torch_module=torch); device = device_info.torch_device
    output = dict(raw.get("output") or {})
    resume_value = output.get("resume_run_dir")
    if run_dir is not None and resume_value:
        raise ValueError("Specify either run_dir or output.resume_run_dir, not both.")
    requested_run_dir = run_dir or (_root_path(str(resume_value)) if resume_value else None)
    out = make_run_dir(str(dict(raw["experiment"])["experiment_id"]), runs_dir=ROOT / str(output.get("runs_dir", "runs")), run_dir=requested_run_dir)
    logger = setup_logging(log_file=out / "run.log", run_meta={"stage": "field_eigen_te_b4_dense_k_gate"}); ensure_figures_dir(out)
    write_resolved_config(out, yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
    dataset = dict(raw["dataset"]); input_hashes = _validate_dataset_lineage(dataset)
    admission_contract = dict(raw["admission_contract"]); thresholds = dict(raw["promotion_thresholds"])
    reference_cfg = dict(raw.get("residual_solver_floor_reference") or {})
    raw_factors = dict(reference_cfg.get("factors") or {})
    residual_solver_floor_factors = {int(resolution): float(raw_factors.get(str(resolution), raw_factors.get(resolution, 1.0))) for resolution in dict(raw["dense_k"])["resolutions"]}
    if any(value <= 0.0 or not np.isfinite(value) for value in residual_solver_floor_factors.values()):
        raise ValueError("B4 residual solver-floor factors must be finite and positive.")
    required_seeds = {int(seed) for seed in admission_contract["required_seeds"]}
    if str(admission_contract["evaluation_split"]) != "test" or required_seeds != {0, 1, 2}:
        raise ValueError("B4 admission contract must explicitly require evaluation_split=test and seeds {0,1,2}.")
    sidecar = _read_sidecar(_root_path(str(dataset["k_path_sidecar_path"])))
    k_points = np.asarray(sidecar["k_points"], dtype=np.float32); bands = np.asarray(dict(raw["normalization"])["band_range_one_based"], dtype=np.int64)
    bands = np.arange(int(bands[0]), int(bands[1]) + 1, dtype=np.int64)
    seen = set(int(value) for value in dict(raw["dense_k"])["seen_k_indices"]); duplicate = {int(value) for group in sidecar["duplicate_coordinate_groups"] for value in group[1:]}; unique_indices = set(range(len(k_points))) - duplicate
    if len(k_points) != int(dict(raw["dense_k"])["path_index_count"]) or len(unique_indices) != int(dict(raw["dense_k"])["unique_coordinate_count"]) or len(seen) != int(dict(raw["dense_k"])["seen_k_count"]):
        raise ValueError("Dense-k configuration does not match frozen sidecar cardinalities.")
    structures = _load_b3_structures(_root_path(str(dataset["field_hdf5_path"])), splits=("val", "test"))
    if any(len(structures[split]) != int(dict(raw["dense_k"])["structures_per_split"]) for split in structures):
        raise ValueError("B3 source structure counts do not match frozen dense-k configuration.")
    v2 = np.load(_root_path(str(dataset["v2_frequency_dataset_path"])), allow_pickle=True); truth = np.asarray(v2["Wh"], dtype=np.float32)
    _verify_v2_source_lineage(structures, v2["metadata_jsonl"].item())
    pino_cases, manifest_lineage = _resolve_pino_cases(raw, expected_dataset_hash=input_hashes["field_hdf5"])
    cases = list(dict(raw["fno"])["cases"]) + pino_cases
    _validate_configured_cases(cases, required_seeds=required_seeds)
    input_lineage = {**input_hashes, "pino_manifest": manifest_lineage}
    all_rows: list[dict[str, Any]] = []; identity_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    for case in cases:
        model, identity = _load_checkpoint_case(case, device, torch, expected_dataset_hash=input_hashes["field_hdf5"])
        identity_rows.append({"case": case["name"], "kind": case["kind"], "seed": case["seed"], **identity})
        for resolution in tuple(int(value) for value in dict(raw["dense_k"])["resolutions"]):
            for split, split_structures in structures.items():
                expected_rows = len(split_structures) * len(k_points) * len(bands)
                fingerprint = _partial_fingerprint(raw=raw, input_lineage=input_lineage, case=case, identity=identity, split=split, resolution=resolution)
                partial = _partial_path(out, case=str(case["name"]), split=split, resolution=resolution)
                rows = _load_resumable_partial(partial, fingerprint=fingerprint, expected_rows=expected_rows)
                resumed = rows is not None
                elapsed = 0.0
                if rows is None:
                    rows, elapsed = _case_rows(case=case, model=model, identity=identity, structures=split_structures, split=split, k_points=k_points, bands=bands, truth=truth, resolution=resolution, residual_solver_floor_factors=residual_solver_floor_factors, normalization=dict(raw["normalization"]), device=device, torch=torch)
                    for row in rows:
                        row["eval_time_s_split"] = elapsed; row["inference_time_per_sample_ms"] = 1000.0 * elapsed / max(len(rows), 1)
                    _atomic_write_json(partial, {"input_fingerprint": fingerprint, "case": str(case["name"]), "split": split, "resolution": resolution, "row_count": len(rows), "rows": rows})
                history_rows.append({"case": case["name"], "kind": case["kind"], "seed": case["seed"], "split": split, "resolution": resolution, "row_count": len(rows), "resumed": resumed, "eval_time_s": elapsed})
                all_rows.extend(rows)
                logger.info("%s dense-k %s split=%s resolution=%d", "Resumed" if resumed else "Completed", case["name"], split, resolution)
    support = _support_rows(all_rows, seen=seen, unique_indices=unique_indices, sidecar=sidecar)
    calibration = _calibration_rows([row for row in all_rows if row["resolution"] == 64 and int(row["k_index"]) not in seen and int(row["k_index"]) in unique_indices], bootstrap_seed=int(dict(raw["statistics"])["bootstrap_seed"]), bootstrap_repeats=int(dict(raw["statistics"])["bootstrap_repeats"]))
    transfer: list[dict[str, Any]] = []
    for case in sorted({row["case"] for row in support}):
        rows64 = [row for row in support if row["case"] == case and row["split"] == "test" and row["resolution"] == 64 and row["metric_scope"] == "all_k"]
        for resolution in tuple(int(value) for value in thresholds["resolution_transfer"]["evaluation_resolutions"]):
            rowsn = [row for row in support if row["case"] == case and row["split"] == "test" and row["resolution"] == resolution and row["metric_scope"] == "all_k"]
            if len(rows64) != 1 or len(rowsn) != 1:
                continue
            base, current = rows64[0], rowsn[0]
            mre_change = 100.0 * (float(current["frequency_mre_pct"]) / max(float(base["frequency_mre_pct"]), 1.0e-12) - 1.0)
            residual_change = 100.0 * (float(current["maxwell_relative_residual_median"]) / max(float(base["maxwell_relative_residual_median"]), 1.0e-12) - 1.0)
            transfer.append({"case": case, "kind": base["kind"], "seed": base["seed"], "resolution": resolution, "frequency_mre_change_pct": mre_change, "residual_median_change_pct": residual_change, "frequency_pass": mre_change <= float(thresholds["resolution_transfer"]["frequency_mre_degradation_max_pct"]), "residual_pass": residual_change <= float(thresholds["resolution_transfer"]["residual_median_degradation_max_pct"]), "all_finite": bool(np.isfinite([mre_change, residual_change]).all())})
    decision = _decision(support, calibration, transfer, thresholds=thresholds, admission_contract=admission_contract)
    _write_csv(out / "dense_k_per_case_metrics.csv", all_rows); _write_csv(out / "k_support_metrics.csv", support); _write_csv(out / "residual_calibration.csv", calibration); _write_csv(out / "resolution_transfer.csv", transfer)
    _write_csv(out / "history.csv", history_rows)
    write_json(out / "lineage.json", {"inputs": input_lineage, "model_identities": identity_rows, "seen_k_indices": sorted(seen), "unique_k_indices": sorted(unique_indices), "residual_solver_floor_reference": {"factors": {str(key): value for key, value in residual_solver_floor_factors.items()}, **reference_cfg}})
    write_json(out / "promotion_decision.json", decision); write_metrics(out, {"path_index_cases_per_split": len(structures["test"]) * len(k_points) * len(bands), "unique_coordinate_cases_per_split": len(structures["test"]) * len(unique_indices) * len(bands), "partial_artifact_count": len(history_rows), "resumed_partial_count": sum(bool(row["resumed"]) for row in history_rows), "peak_gpu_memory_mb": peak_gpu_memory_mb(device, torch_module=torch)}); write_results(out, {"status": "b4_dense_k_gate_completed", "promotion_decision": decision, "inverse_status": "blocked"}); write_run_meta(out, {**collect_device_metadata(device_info), "status": "b4_dense_k_gate_completed"})
    return out


__all__ = ["is_field_eigen_b4_dense_config", "run_field_eigen_b4_dense_gate"]
