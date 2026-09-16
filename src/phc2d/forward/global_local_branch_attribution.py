"""Frozen Global/Local branch attribution for EXP-FWD-022.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from phc2d.forward.global_local_bridge_evaluation import cell_metrics, paired_bootstrap


EXPERIMENT_ID = "EXP-FWD-022"
INTERVENTIONS = (
    "full",
    "g_off_raw",
    "l_off_raw",
    "g_off_renorm",
    "l_off_renorm",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {destination}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected YAML mapping: {path}")
    return dict(value)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_checkpoint(path: Path, device: Any, torch: Any) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    required = {
        "state_dict",
        "model_name",
        "model_params",
        "input_shape",
        "output_shape",
        "target_mean",
        "target_std",
    }
    if not isinstance(payload, Mapping) or not required.issubset(payload):
        raise RuntimeError(f"checkpoint payload is incomplete: {path}")
    if payload["model_name"] != "global_local_band_operator_v1":
        raise RuntimeError(f"unexpected checkpoint model: {payload['model_name']}")
    return dict(payload)


def _normalizer(payload: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mean = payload["target_mean"]
    std = payload["target_std"]
    if hasattr(mean, "detach"):
        mean = mean.detach().cpu().numpy()
    if hasattr(std, "detach"):
        std = std.detach().cpu().numpy()
    return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


def _build_model(payload: Mapping[str, Any], device: Any) -> Any:
    from phc2d.models import build_model

    model = build_model(
        str(payload["model_name"]),
        input_shape=tuple(payload["input_shape"]),
        output_shape=tuple(payload["output_shape"]),
        **dict(payload["model_params"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()


def _predict(
    model: Any,
    values: np.ndarray,
    k_points: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    intervention: str,
    batch_size: int,
    device: Any,
    torch: Any,
) -> np.ndarray:
    model.set_branch_intervention(intervention)
    points = torch.as_tensor(k_points, dtype=torch.float32, device=device)
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(
                values[start : start + batch_size], dtype=torch.float32, device=device
            )
            prediction = model(batch, k_points=points)
            parts.append((prediction.cpu().numpy() * std + mean).astype(np.float32))
    output = np.concatenate(parts, axis=0)
    if not np.isfinite(output).all():
        raise RuntimeError(f"nonfinite prediction under {intervention}")
    return output


def output_distance(candidate: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Return per-sample relative Frobenius prediction distance for Bands 2--10."""

    if candidate.shape != baseline.shape or candidate.ndim != 3:
        raise ValueError("output distance requires matching [N,K,B] predictions")
    delta = candidate[:, :, 1:10] - baseline[:, :, 1:10]
    numerator = np.sqrt(np.square(delta, dtype=np.float64).sum(axis=(1, 2)))
    denominator = np.sqrt(
        np.square(baseline[:, :, 1:10], dtype=np.float64).sum(axis=(1, 2))
    )
    return numerator / np.maximum(denominator, 1.0e-12)


def _sample_features(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = np.abs(values - np.roll(values, 1, axis=-1)).mean(axis=(-2, -1))
    dy = np.abs(values - np.roll(values, 1, axis=-2)).mean(axis=(-2, -1))
    boundary = (dx + dy) / 2.0
    gray = ((values > 0.05) & (values < 0.95)).mean(axis=(-2, -1))
    return boundary.reshape(-1), gray.reshape(-1)


def _rng(key: str) -> np.random.Generator:
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
    return np.random.default_rng(seed)


def diagnostic_indices(values: np.ndarray, *, count: int, key: str) -> dict[str, np.ndarray]:
    """Freeze random and structure-stratified diagnostic positions."""

    size = len(values)
    take = min(int(count), size)
    random = np.sort(_rng(f"{key}:random").choice(size, size=take, replace=False))
    boundary, gray = _sample_features(values)
    boundary_bin = np.digitize(boundary, np.quantile(boundary, (0.25, 0.5, 0.75)))
    gray_bin = np.digitize(gray, np.quantile(gray, (0.25, 0.5, 0.75)))
    groups = boundary_bin * 4 + gray_bin
    selected: list[int] = []
    generator = _rng(f"{key}:stratified")
    per_group = max(1, take // max(len(np.unique(groups)), 1))
    for group in sorted(np.unique(groups)):
        available = np.flatnonzero(groups == group)
        chosen = generator.choice(available, size=min(per_group, len(available)), replace=False)
        selected.extend(int(value) for value in chosen)
    remaining = np.setdiff1d(np.arange(size), np.asarray(selected, dtype=np.int64))
    if len(selected) < take:
        selected.extend(
            int(value)
            for value in generator.choice(remaining, size=take - len(selected), replace=False)
        )
    stratified = np.sort(np.asarray(selected[:take], dtype=np.int64))
    return {"random": random.astype(np.int64), "structure_stratified": stratified}


def _internal_attribution(
    model: Any,
    values: np.ndarray,
    targets: np.ndarray,
    k_points: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    sample_set: str,
    batch_size: int,
    device: Any,
    torch: Any,
) -> list[dict[str, Any]]:
    model.set_branch_intervention("full")
    points = torch.as_tensor(k_points, dtype=torch.float32, device=device)
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std_tensor = torch.as_tensor(std, dtype=torch.float32, device=device)
    totals: dict[tuple[int, str], dict[str, float]] = defaultdict(
        lambda: {"activation_rms_sum": 0.0, "grad_activation_abs_sum": 0.0, "count": 0.0}
    )
    captures: dict[tuple[int, str], Any] = {}
    handles = []
    for layer_index, block in enumerate(model.blocks):
        for branch in ("spectral", "local"):
            module = getattr(block, branch)

            def capture(_module: Any, _inputs: Any, output: Any, *, key: tuple[int, str] = (layer_index, branch)) -> None:
                output.retain_grad()
                captures[key] = output

            handles.append(module.register_forward_hook(capture))
    try:
        for start in range(0, len(values), batch_size):
            captures.clear()
            batch_values = torch.as_tensor(
                values[start : start + batch_size], dtype=torch.float32, device=device
            )
            batch_targets = torch.as_tensor(
                targets[start : start + batch_size], dtype=torch.float32, device=device
            )
            model.zero_grad(set_to_none=True)
            prediction = model(batch_values, k_points=points)
            normalized_target = (batch_targets - mean_tensor) / std_tensor
            loss = torch.square(prediction - normalized_target).mean()
            loss.backward()
            actual_batch = int(batch_values.shape[0])
            for key, activation in captures.items():
                gradient = activation.grad
                if gradient is None:
                    raise RuntimeError(f"missing branch gradient: {key}")
                per_sample_a = activation.detach().square().mean(dim=(1, 2, 3)).sqrt()
                per_sample_c = (activation.detach() * gradient.detach() * actual_batch).abs().mean(
                    dim=(1, 2, 3)
                )
                if not bool(torch.isfinite(per_sample_a).all()) or not bool(torch.isfinite(per_sample_c).all()):
                    raise RuntimeError(f"nonfinite internal attribution: {key}")
                totals[key]["activation_rms_sum"] += float(per_sample_a.sum().cpu())
                totals[key]["grad_activation_abs_sum"] += float(per_sample_c.sum().cpu())
                totals[key]["count"] += actual_batch
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
    rows: list[dict[str, Any]] = []
    for layer_index in range(len(model.blocks)):
        values_by_branch: dict[str, dict[str, float]] = {}
        for branch in ("spectral", "local"):
            item = totals[(layer_index, branch)]
            values_by_branch[branch] = {
                "activation_rms": item["activation_rms_sum"] / item["count"],
                "grad_activation_abs": item["grad_activation_abs_sum"] / item["count"],
            }
        denominator = sum(item["grad_activation_abs"] for item in values_by_branch.values())
        for branch, item in values_by_branch.items():
            rows.append(
                {
                    "sample_set": sample_set,
                    "layer_index": layer_index,
                    "branch": branch,
                    **item,
                    "global_saliency_share": values_by_branch["spectral"]["grad_activation_abs"]
                    / max(denominator, 1.0e-12),
                    "sample_count": len(values),
                }
            )
    return rows


def freeze_campaign(
    config_path: str | Path,
    checkpoint_manifest: str | Path,
    output_dir: str | Path,
    *,
    repository_root: str | Path = ".",
) -> Path:
    """Freeze the 36-checkpoint by six-target validation attribution matrix."""

    root = Path(repository_root).resolve()
    config = _load_yaml(_resolve(root, config_path))
    checkpoints = _read_csv(_resolve(root, checkpoint_manifest))
    if len(checkpoints) != 36:
        raise RuntimeError(f"EXP-FWD-022 requires 36 checkpoints, found {len(checkpoints)}")
    identities = {(row["domain"], row["variant"], row["seed"]) for row in checkpoints}
    if len(identities) != 36:
        raise RuntimeError("checkpoint manifest contains identity collisions")
    resolved_checkpoints: list[dict[str, Any]] = []
    for row in checkpoints:
        path = _resolve(root, row["checkpoint_path"])
        expected = row.get("checkpoint_sha256", "")
        observed = _sha256(path)
        if expected and observed != expected:
            raise RuntimeError(f"checkpoint hash drift: {path}")
        resolved_checkpoints.append({**row, "checkpoint_path": str(path), "checkpoint_sha256": observed})
    datasets: list[dict[str, Any]] = []
    for domain, contract in dict(config["datasets"]).items():
        path = _resolve(root, contract["path"])
        observed = _sha256(path)
        if observed != contract["sha256"]:
            raise RuntimeError(f"dataset hash drift: {domain}")
        datasets.append({"domain": domain, **contract, "path": str(path)})
    if len(datasets) != 6:
        raise RuntimeError("EXP-FWD-022 requires six validation targets")
    plan: list[dict[str, Any]] = []
    for checkpoint in sorted(resolved_checkpoints, key=lambda row: (row["domain"], row["variant"], int(row["seed"]))):
        for dataset in datasets:
            plan.append(
                {
                    "cell_id": f"attr_{len(plan):03d}",
                    "checkpoint_id": checkpoint["case_id"],
                    "source_domain": checkpoint["domain"],
                    "source_lane": checkpoint["lane"],
                    "source_variant": checkpoint["variant"],
                    "source_seed": checkpoint["seed"],
                    "checkpoint_path": checkpoint["checkpoint_path"],
                    "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                    "target_domain": dataset["domain"],
                    "target_lane": "binary" if dataset["domain"].endswith("binary") else "graded",
                    "shard_id": len(plan) % int(config["matrix"]["shard_count"]),
                }
            )
    if len(plan) != 216:
        raise RuntimeError("EXP-FWD-022 freeze requires exactly 216 checkpoint-target cells")
    output = _resolve(root, output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "checkpoint_manifest.csv", resolved_checkpoints)
    _write_csv(output / "dataset_manifest.csv", datasets)
    _write_csv(output / "evaluation_plan.csv", plan)
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (output / "campaign_status.json").write_text(
        json.dumps(
            {
                "status": "frozen_preflight_required",
                "experiment_id": EXPERIMENT_ID,
                "checkpoint_count": 36,
                "target_count": 6,
                "planned_checkpoint_target_cells": 216,
                "planned_intervention_cells": 1080,
                "test_executed": False,
                "next_gate": "g0_gpu_preflight",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def preflight_campaign(
    campaign_dir: str | Path,
    *,
    repository_root: str | Path = ".",
    device: str = "cuda:0",
) -> Path:
    """Check all checkpoint identities and one finite intervention batch."""

    import h5py
    import torch

    root = Path(repository_root).resolve()
    campaign = _resolve(root, campaign_dir)
    checkpoints = _read_csv(campaign / "checkpoint_manifest.csv")
    datasets = _read_csv(campaign / "dataset_manifest.csv")
    device_object = torch.device(device)
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        path = Path(checkpoint["checkpoint_path"])
        if _sha256(path) != checkpoint["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash drift: {path}")
        payload = _load_checkpoint(path, device_object, torch)
        model = _build_model(payload, device_object)
        rows.append({"kind": "checkpoint", "identity": checkpoint["case_id"], "status": "passed"})
        del model, payload
    dataset = datasets[0]
    with h5py.File(dataset["path"], "r") as handle:
        indices = np.asarray(handle["idx_val"][:4], dtype=np.int64)
        values = np.asarray(handle[dataset["grid_field"]][indices], dtype=np.float32)
        targets = np.asarray(handle["Wh"][indices], dtype=np.float32)
        k_points = np.asarray(handle["k_points"], dtype=np.float32)
    payload = _load_checkpoint(Path(checkpoints[0]["checkpoint_path"]), device_object, torch)
    model = _build_model(payload, device_object)
    mean, std = _normalizer(payload)
    for mode in INTERVENTIONS:
        prediction = _predict(
            model, values, k_points, mean, std,
            intervention=mode, batch_size=4, device=device_object, torch=torch,
        )
        rows.append({"kind": "intervention", "identity": mode, "status": "passed", "finite": bool(np.isfinite(prediction).all())})
    sample_indices = diagnostic_indices(values, count=4, key="preflight")
    for name, selected in sample_indices.items():
        attribution = _internal_attribution(
            model, values[selected], targets[selected], k_points, mean, std,
            sample_set=name, batch_size=4, device=device_object, torch=torch,
        )
        rows.append({"kind": "attribution", "identity": name, "status": "passed", "rows": len(attribution)})
    _write_csv(campaign / "preflight_manifest.csv", rows)
    (campaign / "campaign_status.json").write_text(
        json.dumps({"status": "g0_preflight_passed", "next_gate": "g1_validation_shards", "test_executed": False}, indent=2) + "\n",
        encoding="utf-8",
    )
    return campaign


def run_validation_shard(
    campaign_dir: str | Path,
    shard_id: int,
    *,
    repository_root: str | Path = ".",
    device: str = "cuda:0",
) -> Path:
    """Evaluate one frozen validation shard with all five interventions."""

    import h5py
    import torch

    root = Path(repository_root).resolve()
    campaign = _resolve(root, campaign_dir)
    status = json.loads((campaign / "campaign_status.json").read_text(encoding="utf-8"))
    if status.get("status") not in {"g0_preflight_passed", "g1_submitted"}:
        raise RuntimeError(f"validation shards are not admitted: {status.get('status')}")
    plan = [row for row in _read_csv(campaign / "evaluation_plan.csv") if int(row["shard_id"]) == int(shard_id)]
    datasets = {row["domain"]: row for row in _read_csv(campaign / "dataset_manifest.csv")}
    if not plan:
        raise RuntimeError(f"empty validation shard: {shard_id}")
    output = campaign / "shards" / f"shard_{int(shard_id):02d}"
    output.mkdir(parents=True, exist_ok=False)
    device_object = torch.device(device)
    intervention_rows: list[dict[str, Any]] = []
    per_band_rows: list[dict[str, Any]] = []
    per_k_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    for cell in plan:
        checkpoint_path = Path(cell["checkpoint_path"])
        if _sha256(checkpoint_path) != cell["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash drift: {checkpoint_path}")
        dataset = datasets[cell["target_domain"]]
        with h5py.File(dataset["path"], "r") as handle:
            indices = np.asarray(handle["idx_val"], dtype=np.int64)
            values = np.asarray(handle[dataset["grid_field"]][indices], dtype=np.float32)
            targets = np.asarray(handle["Wh"][indices], dtype=np.float32)
            k_points = np.asarray(handle["k_points"], dtype=np.float32)
        payload = _load_checkpoint(checkpoint_path, device_object, torch)
        model = _build_model(payload, device_object)
        mean, std = _normalizer(payload)
        predictions: dict[str, np.ndarray] = {}
        for mode in INTERVENTIONS:
            started = time.perf_counter()
            prediction = _predict(
                model, values, k_points, mean, std,
                intervention=mode,
                batch_size=int(dataset.get("batch_size", 64)),
                device=device_object,
                torch=torch,
            )
            predictions[mode] = prediction
            metrics, per_band, per_k, arrays = cell_metrics(prediction, targets)
            sample_distance = np.zeros(len(values), dtype=np.float64)
            if mode != "full":
                sample_distance = output_distance(prediction, predictions["full"])
            np.savez_compressed(
                output / f"{cell['cell_id']}_{mode}.npz",
                sample_mre_bands_2_10_pct=arrays["sample_mre_bands_2_10_pct"],
                sample_output_distance=sample_distance,
            )
            identity = {**cell, "intervention": mode}
            intervention_rows.append(
                {
                    **identity,
                    **metrics,
                    "output_distance_bands_2_10": float(sample_distance.mean()),
                    "validation_samples": len(values),
                    "elapsed_s": time.perf_counter() - started,
                    "status": "completed",
                }
            )
            per_band_rows.extend({**identity, **row} for row in per_band)
            per_k_rows.extend({**identity, **row} for row in per_k)
        selected_sets = diagnostic_indices(
            values,
            count=int(dataset.get("diagnostic_samples", 512)),
            key=f"{cell['target_domain']}:{cell['checkpoint_id']}",
        )
        for sample_set, selected in selected_sets.items():
            index_rows.append(
                {
                    "cell_id": cell["cell_id"],
                    "sample_set": sample_set,
                    "count": len(selected),
                    "positions_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
                    "dataset_indices_sha256": hashlib.sha256(indices[selected].tobytes()).hexdigest(),
                    "overlap_count": int(len(np.intersect1d(selected_sets["random"], selected_sets["structure_stratified"]))),
                }
            )
            attribution = _internal_attribution(
                model,
                values[selected],
                targets[selected],
                k_points,
                mean,
                std,
                sample_set=sample_set,
                batch_size=min(16, len(selected)),
                device=device_object,
                torch=torch,
            )
            attribution_rows.extend({**cell, **row} for row in attribution)
        del model, payload, predictions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _write_csv(output / "intervention_cells.csv", intervention_rows)
    _write_csv(output / "per_band_metrics.csv", per_band_rows)
    _write_csv(output / "per_k_metrics.csv", per_k_rows)
    _write_csv(output / "branch_internal_metrics.csv", attribution_rows)
    _write_csv(output / "diagnostic_index_manifest.csv", index_rows)
    (output / "results.json").write_text(
        json.dumps({"status": "validation_shard_success", "base_cells": len(plan), "intervention_cells": len(intervention_rows), "test_executed": False}, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def status_campaign(campaign_dir: str | Path, *, repository_root: str | Path = ".") -> dict[str, Any]:
    campaign = _resolve(Path(repository_root).resolve(), campaign_dir)
    plan = _read_csv(campaign / "evaluation_plan.csv")
    observed = [
        row
        for shard in (campaign / "shards").glob("shard_*")
        if (shard / "intervention_cells.csv").is_file()
        for row in _read_csv(shard / "intervention_cells.csv")
    ]
    result = {
        "planned_checkpoint_target_cells": len(plan),
        "planned_intervention_cells": len(plan) * len(INTERVENTIONS),
        "completed_intervention_cells": len(observed),
        "completed_checkpoint_target_cells": len({row["cell_id"] for row in observed}),
    }
    (campaign / "status_snapshot.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def finalize_campaign(campaign_dir: str | Path, *, repository_root: str | Path = ".") -> Path:
    """Fail closed unless all 1080 validation interventions are complete."""

    root = Path(repository_root).resolve()
    campaign = _resolve(root, campaign_dir)
    plan = _read_csv(campaign / "evaluation_plan.csv")
    shards = sorted((campaign / "shards").glob("shard_*"))
    interventions = [row for shard in shards for row in _read_csv(shard / "intervention_cells.csv")]
    expected = {(row["cell_id"], mode) for row in plan for mode in INTERVENTIONS}
    observed = {(row["cell_id"], row["intervention"]) for row in interventions}
    if len(interventions) != 1080 or observed != expected:
        raise RuntimeError("finalizer requires exactly 1080 unique intervention cells")
    if any(row["status"] != "completed" or not np.isfinite(float(row["bands_2_10_mre_pct"])) for row in interventions):
        raise RuntimeError("finalizer found incomplete or nonfinite intervention cells")
    _write_csv(campaign / "intervention_cells.csv", interventions)
    for name in ("per_band_metrics.csv", "per_k_metrics.csv", "branch_internal_metrics.csv", "diagnostic_index_manifest.csv"):
        _write_csv(campaign / name, [row for shard in shards for row in _read_csv(shard / name)])
    by_cell: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in interventions:
        by_cell[row["cell_id"]][row["intervention"]] = row
    comparisons: list[dict[str, Any]] = []
    for cell_id, modes in sorted(by_cell.items()):
        baseline_archive = next(shard / f"{cell_id}_full.npz" for shard in shards if (shard / f"{cell_id}_full.npz").is_file())
        with np.load(baseline_archive) as archive:
            baseline = np.asarray(archive["sample_mre_bands_2_10_pct"], dtype=np.float64)
        for mode in INTERVENTIONS[1:]:
            archive_path = baseline_archive.with_name(f"{cell_id}_{mode}.npz")
            with np.load(archive_path) as archive:
                candidate = np.asarray(archive["sample_mre_bands_2_10_pct"], dtype=np.float64)
                distance = np.asarray(archive["sample_output_distance"], dtype=np.float64)
            comparisons.append(
                {
                    **{key: value for key, value in modes[mode].items() if key not in {"bands_2_10_mre_pct"}},
                    "comparison": f"{mode}_minus_full",
                    **paired_bootstrap(candidate, baseline, key=f"{EXPERIMENT_ID}:{cell_id}:{mode}"),
                    "output_distance_mean": float(distance.mean()),
                }
            )
    _write_csv(campaign / "paired_bootstrap.csv", comparisons)
    scale_rows: list[dict[str, Any]] = []
    for cell_id, modes in sorted(by_cell.items()):
        full = float(modes["full"]["bands_2_10_mre_pct"])
        row: dict[str, Any] = {key: modes["full"][key] for key in ("cell_id", "checkpoint_id", "source_domain", "source_variant", "source_seed", "target_domain")}
        for branch, prefix in (("g", "g_off"), ("l", "l_off")):
            raw = float(modes[f"{prefix}_raw"]["bands_2_10_mre_pct"]) - full
            controlled = float(modes[f"{prefix}_renorm"]["bands_2_10_mre_pct"]) - full
            row[f"i_{branch}_raw_pp"] = raw
            row[f"i_{branch}_ctrl_pp"] = controlled
            row[f"s_{branch}_pp"] = raw - controlled
            row[f"d_{branch}_ctrl"] = float(modes[f"{prefix}_renorm"]["output_distance_bands_2_10"])
        scale_rows.append(row)
    _write_csv(campaign / "branch_dependency_metrics.csv", scale_rows)
    (campaign / "metrics.json").write_text(
        json.dumps({"checkpoint_target_cells": 216, "intervention_cells": 1080, "paired_bootstrap_rows": len(comparisons), "test_executed": False}, indent=2) + "\n",
        encoding="utf-8",
    )
    (campaign / "results.json").write_text(
        json.dumps({"status": "g1_validation_complete_pending_human_review", "evidence_scope": "frozen_validation_branch_attribution", "model_promotion": False, "physical_claim": False, "test_executed": False}, indent=2) + "\n",
        encoding="utf-8",
    )
    (campaign / "campaign_status.json").write_text(
        json.dumps({"status": "g1_validation_complete_pending_human_review", "next_gate": "human_review_before_independent_training", "test_executed": False}, indent=2) + "\n",
        encoding="utf-8",
    )
    return campaign


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--checkpoint-manifest", required=True)
    freeze.add_argument("--output-dir", required=True)
    freeze.add_argument("--repository-root", default=".")
    for name in ("preflight", "run-shard", "status", "finalize"):
        item = sub.add_parser(name)
        item.add_argument("--campaign-dir", required=True)
        item.add_argument("--repository-root", default=".")
        if name in {"preflight", "run-shard"}:
            item.add_argument("--device", default="cuda:0")
        if name == "run-shard":
            item.add_argument("--shard-id", required=True, type=int)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze_campaign(args.config, args.checkpoint_manifest, args.output_dir, repository_root=args.repository_root)
    elif args.command == "preflight":
        result = preflight_campaign(args.campaign_dir, repository_root=args.repository_root, device=args.device)
    elif args.command == "run-shard":
        result = run_validation_shard(args.campaign_dir, args.shard_id, repository_root=args.repository_root, device=args.device)
    elif args.command == "status":
        result = status_campaign(args.campaign_dir, repository_root=args.repository_root)
    else:
        result = finalize_campaign(args.campaign_dir, repository_root=args.repository_root)
    print(result)


__all__ = [
    "INTERVENTIONS",
    "diagnostic_indices",
    "finalize_campaign",
    "freeze_campaign",
    "output_distance",
    "preflight_campaign",
    "run_validation_shard",
    "status_campaign",
]


if __name__ == "__main__":
    main()
