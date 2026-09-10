"""Freeze, train, evaluate, and finalize the K=20 B/G seed-0 campaign.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

MODELS = (
    ("mlp", "B"), ("mlp", "G"), ("res_mlp_wide", "B"), ("res_mlp_wide", "G"),
    ("lifted_cnn", "B"), ("lifted_cnn", "G"), ("fno2d_band_v2", "B"), ("fno2d_band_v2", "G"),
    ("mlp_mixer_band_v1", "B"), ("mlp_mixer_band_v1", "G"), ("res_mlp_gated_v2", "B"), ("res_mlp_gated_v2", "G"),
    ("convnext_band_v1", "B"), ("convnext_band_v1", "G"), ("tfno2d_band_v3", "B"), ("tfno2d_band_v3", "G"),
)
LANES = {"B": "binary", "G": "graded"}
CONFIGS = {"B": "configs/research/full_cell_k20_forward_binary_seed0_v1.yaml", "G": "configs/research/full_cell_k20_forward_graded_seed0_v1.yaml", "J": "configs/research/full_cell_k20_forward_dual_lane_seed0_v1.yaml"}
DATASETS = {"B": ("data/full_cell_binary_fourier_bz9_10000_xingyi_v1.h5", "material_area_fraction_grid_64"), "G": ("data/full_cell_graded_fourier_bz9_10000_xingyi_v1.h5", "material_weight_grid_64")}
VALIDATION_FROZEN_TRAINING_STATUSES = frozenset({"formal_training_success", "dual_lane_training_success"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_hash(value: Any) -> str:
    return hashlib.sha256(np.asarray(value, dtype=np.int64).tobytes()).hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(str(key))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected mapping: {path}")
    return dict(payload)


def _commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        marker = root / "DEPLOYED_GIT_COMMIT"
        if marker.is_file():
            value = marker.read_text(encoding="utf-8").strip()
            if value:
                return value
        return os.environ.get("PHC2D_DEPLOYED_GIT_COMMIT", "unknown")


def _audit_dataset(root: Path, lane: str) -> dict[str, Any]:
    import h5py

    path_text, grid_field = DATASETS[lane]
    path = root / path_text
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        required = {grid_field, "Wh", "k_points", "idx_train", "idx_val", "idx_test", "sample_id"}
        missing = sorted(required.difference(handle.keys()))
        if missing:
            raise ValueError(f"{lane} missing HDF5 fields: {missing}")
        shape = tuple(int(value) for value in handle["Wh"].shape)
        grid_shape = tuple(int(value) for value in handle[grid_field].shape)
        if shape != (10000, 81, 10) or grid_shape != (10000, 64, 64):
            raise ValueError(f"{lane} unexpected shapes: Wh={shape}, grid={grid_shape}")
        indices = {name: np.asarray(handle[name], dtype=np.int64) for name in ("idx_train", "idx_val", "idx_test")}
        if {name: int(value.size) for name, value in indices.items()} != {"idx_train": 8000, "idx_val": 1000, "idx_test": 1000}:
            raise ValueError(f"{lane} split sizes are not 8000/1000/1000")
        if np.unique(np.concatenate(tuple(indices.values()))).size != 10000:
            raise ValueError(f"{lane} splits overlap or do not cover all samples")
        k_points = np.asarray(handle["k_points"], dtype=np.float32)
        if k_points.shape != (81, 2) or not np.isfinite(k_points).all():
            raise ValueError(f"{lane} invalid k_points")
        if not np.isfinite(handle["Wh"][:]).all() or not np.isfinite(handle[grid_field][:]).all():
            raise ValueError(f"{lane} contains non-finite values")
        return {
            "lane": lane, "dataset_path": path_text, "grid_field": grid_field,
            "dataset_sha256": _sha256(path), "k_points_sha256": hashlib.sha256(np.ascontiguousarray(k_points).tobytes()).hexdigest(),
            "idx_train_hash": _array_hash(indices["idx_train"]), "idx_val_hash": _array_hash(indices["idx_val"]), "idx_test_hash": _array_hash(indices["idx_test"]),
            "shape": list(shape), "grid_shape": list(grid_shape), "schema_version": str(handle.attrs.get("schema_version", "")),
        }


def _task_rows(campaign: Path, *, stage: str, audits: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tasks = list(MODELS) + [("dual_lane_band_operator_v1", "J")]
    for index, (model, lane) in enumerate(tasks):
        audit = audits["B" if lane in {"B", "J"} else "G"]
        config = CONFIGS[lane if lane in CONFIGS else "J"]
        task_id = f"{stage}_{index:02d}_{model}_{lane}_s0"
        rows.append({
            "task_index": index, "task_id": task_id, "model": model, "train_lane": lane,
            "seed": 0, "head": "dense", "grid_field": audit["grid_field"], "config": config,
            "config_sha256": _sha256(Path(config)), "dataset_sha256": audit["dataset_sha256"],
            "idx_train_hash": audit["idx_train_hash"], "idx_val_hash": audit["idx_val_hash"], "idx_test_hash": audit["idx_test_hash"],
            "k_points_sha256": audit["k_points_sha256"], "run_dir": str(campaign / task_id).replace("\\", "/"),
            "gpu_uuid": "", "deployed_commit": _commit(campaign.parents[2] if len(campaign.parents) > 2 else Path.cwd()), "stage": stage,
        })
    return rows


def freeze_campaign(campaign_dir: str | Path, *, repository_root: str | Path = ".") -> Path:
    root = Path(repository_root).resolve()
    campaign = Path(campaign_dir)
    if not campaign.is_absolute():
        campaign = root / campaign
    campaign.mkdir(parents=True, exist_ok=False)
    audits = {lane: _audit_dataset(root, lane) for lane in ("B", "G")}
    if audits["B"]["k_points_sha256"] != audits["G"]["k_points_sha256"]:
        raise ValueError("B/G k_points hash mismatch; campaign is fail-closed")
    rows = _task_rows(campaign, stage="formal", audits=audits)
    admission_rows = [dict(row, task_id=row["task_id"].replace("formal", "admission", 1), run_dir=row["run_dir"].replace("/formal_", "/admission_"), stage="admission") for row in rows]
    _write_csv(campaign / "task_manifest.csv", rows)
    _write_csv(campaign / "admission_manifest.csv", admission_rows)
    _write_json(campaign / "dataset_audit.json", {"status": "pass", "lanes": audits, "k_points_shared": True})
    _write_json(campaign / "freeze.json", {"schema_version": "phc2d.full_cell_k20_campaign.v1", "status": "frozen", "seed": 0, "head": "dense", "task_count": 17, "admission_task_count": 17, "test_read_before_checkpoint_freeze": False, "tasks": "task_manifest.csv", "admission_tasks": "admission_manifest.csv", "deployed_commit": _commit(root), "created_at": time.time()})
    return campaign


def run_task(*, manifest: str | Path, task_index: int, repository_root: str | Path = ".", stage: str = "formal") -> Path:
    root = Path(repository_root).resolve()
    rows = _read_csv(Path(manifest))
    row = rows[int(task_index)]
    if int(row["task_index"]) != int(task_index) or row["stage"] != stage:
        raise ValueError("manifest/task stage mismatch")
    command = ["python", "-m", "phc2d.forward.full_cell_k20_campaign", "_run-one", "--manifest", str(Path(manifest).resolve()), "--task-index", str(task_index), "--repository-root", str(root), "--stage", stage]
    if stage == "admission":
        command.append("--admission-probe")
    head_args = ["--head-kind", "dense"] if _model_accepts_head_override(row["model"]) else []
    if row["train_lane"] == "J":
        command = ["python", "-m", "phc2d.forward.full_cell_dual_lane", "--config", row["config"], "--seed", "0"] + head_args + ["--run-dir", row["run_dir"]] + (["--admission-probe"] if stage == "admission" else [])
    else:
        command = ["python", "-m", "phc2d.forward.full_cell", "--config", row["config"], "--model", row["model"], "--seed", "0"] + head_args + ["--run-dir", row["run_dir"]] + (["--admission-probe"] if stage == "admission" else [])
    subprocess.run(command, cwd=root, check=True)
    return Path(row["run_dir"])


def _model_accepts_head_override(model_name: str) -> bool:
    from phc2d.models import get_model_spec

    return bool(get_model_spec(model_name).accepts_k_points)


def _checkpoint_inventory(campaign: Path) -> list[dict[str, Any]]:
    rows = _read_csv(campaign / "task_manifest.csv")
    inventory: list[dict[str, Any]] = []
    for row in rows:
        run = Path(row["run_dir"])
        result_path = run / "results.json"
        if not result_path.is_file():
            raise RuntimeError(f"missing training result: {run}")
        result = _read_json(result_path)
        if result.get("status") not in VALIDATION_FROZEN_TRAINING_STATUSES or result.get("test_executed"):
            raise RuntimeError(f"checkpoint is not frozen validation-only success: {run}")
        checkpoint = run / "model.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"missing checkpoint: {checkpoint}")
        inventory.append(dict(row, checkpoint_path=str(checkpoint).replace("\\", "/"), checkpoint_sha256=_sha256(checkpoint), checkpoint_role="best_validation", selection_split="validation"))
    return inventory


def plan_evaluation(campaign_dir: str | Path) -> Path:
    campaign = Path(campaign_dir).resolve()
    inventory = _checkpoint_inventory(campaign)
    _write_csv(campaign / "checkpoint_inventory.csv", inventory)
    cells: list[dict[str, Any]] = []
    for source in inventory:
        if source["train_lane"] == "J":
            for target in ("B", "G"):
                cells.append({"evaluation_id": f"eval_{len(cells):03d}", "source_task_id": source["task_id"], "source_model": source["model"], "source_lane": "J", "target_lane": target, "domain": "native", "status": "planned", "checkpoint_path": source["checkpoint_path"], "checkpoint_sha256": source["checkpoint_sha256"]})
        else:
            for target, domain in ((source["train_lane"], "native"), ("G" if source["train_lane"] == "B" else "B", "cross")):
                cells.append({"evaluation_id": f"eval_{len(cells):03d}", "source_task_id": source["task_id"], "source_model": source["model"], "source_lane": source["train_lane"], "target_lane": target, "domain": domain, "status": "planned", "checkpoint_path": source["checkpoint_path"], "checkpoint_sha256": source["checkpoint_sha256"]})
    if len(cells) != 34:
        raise RuntimeError(f"expected 34 evaluation cells, got {len(cells)}")
    _write_csv(campaign / "evaluation_cells_plan.csv", cells)
    _write_json(campaign / "evaluation_plan.json", {"status": "evaluation_plan_frozen", "cell_count": len(cells), "checkpoint_inventory_sha256": _sha256(campaign / "checkpoint_inventory.csv"), "test_read_allowed": True})
    return campaign


def _metric_rows(prediction: np.ndarray, target: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    diff = prediction - target
    rel = np.abs(diff[..., 1:]) / np.maximum(np.abs(target[..., 1:]), 1.0e-8)
    sample_mre = np.mean(rel, axis=(1, 2)) * 100.0
    r2 = 1.0 - np.sum(diff * diff, axis=0) / np.maximum(np.sum((target - np.mean(target, axis=0, keepdims=True)) ** 2, axis=0), 1.0e-12)
    band_order = np.mean(np.any(np.diff(prediction, axis=2) < 0, axis=2))
    return ({"bands_2_10_mre_pct": float(np.mean(sample_mre)), "full_matrix_mre_pct": float(np.mean(np.abs(diff) / np.maximum(np.abs(target), 1.0e-8)) * 100.0), "mae": float(np.mean(np.abs(diff))), "rmse": float(np.sqrt(np.mean(diff * diff))), "r2_centered_mean": float(np.mean(r2)), "band_order_violation_rate": float(band_order), "sample_count": int(target.shape[0])}, sample_mre)


def _eval_one(root: Path, row: Mapping[str, str], out: Path) -> tuple[dict[str, Any], np.ndarray]:
    import h5py
    import torch
    from phc2d.models import build_model, get_model_spec

    payload = torch.load(Path(row["checkpoint_path"]), map_location="cpu", weights_only=True)
    if _sha256(Path(row["checkpoint_path"])) != row["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint hash changed: {row['checkpoint_path']}")
    target_lane = row["target_lane"]
    data_path, grid_field = DATASETS[target_lane]
    with h5py.File(root / data_path, "r") as handle:
        test_idx = np.asarray(handle["idx_test"], dtype=np.int64)
        x = np.asarray(handle[grid_field][test_idx], dtype=np.float32)
        y = np.asarray(handle["Wh"][test_idx], dtype=np.float32)
        k_points = np.asarray(handle["k_points"], dtype=np.float32)
    if row["source_lane"] == "J":
        normalizer_key = "binary" if target_lane == "B" else "graded"
        normalizer = payload["target_normalizers"][normalizer_key]
        mean = np.asarray(normalizer["mean"], dtype=np.float32)
        std = np.asarray(normalizer["std"], dtype=np.float32)
    else:
        mean = np.asarray(payload["target_mean"], dtype=np.float32)
        std = np.asarray(payload["target_std"], dtype=np.float32)
    spec = get_model_spec(str(payload["model_name"]))
    model = build_model(str(payload["model_name"]), input_shape=tuple(payload["input_shape"]), output_shape=tuple(payload["output_shape"]), **dict(payload["model_params"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    batch = 64
    predictions: list[np.ndarray] = []
    start = time.perf_counter()
    with torch.no_grad():
        for begin in range(0, len(x), batch):
            xb = torch.from_numpy(x[begin:begin + batch])
            kwargs: dict[str, Any] = {}
            if spec.accepts_k_points:
                kwargs["k_points"] = torch.from_numpy(k_points)
            if spec.accepts_lane_id:
                kwargs["lane_id"] = torch.full((len(xb),), 0 if target_lane == "B" else 1, dtype=torch.long)
            pred_norm = model(xb, **kwargs)
            predictions.append((pred_norm.numpy() * std + mean).astype(np.float32))
    prediction = np.concatenate(predictions, axis=0)
    metrics, sample_mre = _metric_rows(prediction, y)
    metrics.update({"evaluation_id": row["evaluation_id"], "source_model": row["source_model"], "source_lane": row["source_lane"], "target_lane": target_lane, "domain": row["domain"], "source_normalization_reused": True, "inference_time_total_s": time.perf_counter() - start, "inference_time_per_sample_ms": (time.perf_counter() - start) * 1000.0 / max(len(x), 1)})
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{row['evaluation_id']}.npz", sample_mre=sample_mre, prediction=prediction, target=y, test_indices=test_idx)
    return metrics, sample_mre


def run_evaluation_shard(campaign_dir: str | Path, shard_id: int, *, repository_root: str | Path = ".", shard_count: int = 8) -> Path:
    root = Path(repository_root).resolve()
    campaign = Path(campaign_dir).resolve()
    rows = _read_csv(campaign / "evaluation_cells_plan.csv")
    selected = [row for i, row in enumerate(rows) if i % int(shard_count) == int(shard_id)]
    out = campaign / "evaluation_shards" / str(shard_id)
    results: list[dict[str, Any]] = []
    for row in selected:
        metrics, _ = _eval_one(root, row, out)
        results.append(metrics)
    _write_csv(out / "evaluation_cells.csv", results)
    _write_json(out / "results.json", {"status": "evaluation_shard_success", "shard_id": int(shard_id), "cell_count": len(results)})
    return out


def _bootstrap(left: np.ndarray, right: np.ndarray, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    observed = float(np.mean(left - right))
    values = np.empty(10000, dtype=np.float64)
    for begin in range(0, 10000, 250):
        indices = rng.integers(0, len(left), size=(250, len(left)))
        values[begin:begin + 250] = np.mean((left - right)[indices], axis=1)
    return {"paired_delta_pp": observed, "ci95_low_pp": float(np.quantile(values, 0.025)), "ci95_high_pp": float(np.quantile(values, 0.975)), "bootstrap_replicates": 10000}


def finalize_campaign(campaign_dir: str | Path, *, repository_root: str | Path = ".") -> Path:
    root = Path(repository_root).resolve()
    campaign = Path(campaign_dir).resolve()
    plan = _read_csv(campaign / "evaluation_cells_plan.csv")
    shard_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for shard in sorted((campaign / "evaluation_shards").glob("*/evaluation_cells.csv")):
        shard_rows.extend(_read_csv(shard))
        for npz in shard.parent.glob("eval_*.npz"):
            arrays[npz.stem] = np.load(npz)["sample_mre"]
    if {row["evaluation_id"] for row in shard_rows} != {row["evaluation_id"] for row in plan} or len(shard_rows) != len(plan):
        raise RuntimeError("evaluation shards do not exactly cover the frozen 34-cell plan")
    _write_csv(campaign / "evaluation_cells.csv", shard_rows)
    native = [row for row in shard_rows if row["domain"] == "native"]
    cross = [row for row in shard_rows if row["domain"] == "cross"]
    _write_csv(campaign / "native_summary.csv", native)
    _write_csv(campaign / "cross_lane_summary.csv", cross)
    _write_csv(campaign / "per_band_metrics.csv", [{"evaluation_id": row["evaluation_id"], "bands_2_10_mre_pct": row["bands_2_10_mre_pct"]} for row in shard_rows])
    _write_csv(campaign / "r2_summary.csv", [{"evaluation_id": row["evaluation_id"], "r2_centered_mean": row["r2_centered_mean"]} for row in shard_rows])
    comparisons: list[dict[str, Any]] = []
    for lane in ("B", "G"):
        for upgrade, baseline in (("mlp_mixer_band_v1", "mlp"), ("res_mlp_gated_v2", "res_mlp_wide"), ("convnext_band_v1", "lifted_cnn"), ("tfno2d_band_v3", "fno2d_band_v2")):
            for domain in ("native", "cross"):
                left = next((row for row in shard_rows if row["source_model"] == upgrade and row["source_lane"] == lane and row["target_lane"] == (lane if domain == "native" else ("G" if lane == "B" else "B")) and row["domain"] == domain), None)
                right = next((row for row in shard_rows if row["source_model"] == baseline and row["source_lane"] == lane and row["target_lane"] == (lane if domain == "native" else ("G" if lane == "B" else "B")) and row["domain"] == domain), None)
                if left and right:
                    comparisons.append({"lane": lane, "domain": domain, "upgrade_model": upgrade, "baseline_model": baseline, **_bootstrap(arrays[left["evaluation_id"]], arrays[right["evaluation_id"]], seed=20260820)})
    _write_csv(campaign / "paired_bootstrap_comparisons.csv", comparisons)
    resource_rows: list[dict[str, Any]] = []
    for row in _read_csv(campaign / "task_manifest.csv"):
        profile = Path(row["run_dir"]) / "n40_profile.json"
        resource_rows.append({"task_id": row["task_id"], "profile": str(profile), "profile_present": profile.is_file()})
    _write_csv(campaign / "resource_summary.csv", resource_rows)
    _write_json(campaign / "results.json", {"status": "evaluation_complete_pending_report", "training_task_count": 17, "evaluation_cell_count": 34, "native_cell_count": len(native), "cross_cell_count": len(cross), "test_executed_after_checkpoint_freeze": True, "evidence_status": "seed0_preliminary_candidate"})
    return campaign


def _main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze"); freeze.add_argument("--campaign-dir", required=True); freeze.add_argument("--repository-root", default=".")
    task = sub.add_parser("run-task"); task.add_argument("--manifest", required=True); task.add_argument("--task-index", type=int, required=True); task.add_argument("--stage", choices=("admission", "formal"), default="formal"); task.add_argument("--repository-root", default=".")
    one = sub.add_parser("_run-one"); one.add_argument("--manifest", required=True); one.add_argument("--task-index", type=int, required=True); one.add_argument("--stage", choices=("admission", "formal"), default="formal"); one.add_argument("--repository-root", default="."); one.add_argument("--admission-probe", action="store_true")
    plan = sub.add_parser("plan-eval"); plan.add_argument("--campaign-dir", required=True)
    shard = sub.add_parser("run-eval-shard"); shard.add_argument("--campaign-dir", required=True); shard.add_argument("--shard-id", type=int, required=True); shard.add_argument("--repository-root", default=".")
    final = sub.add_parser("finalize"); final.add_argument("--campaign-dir", required=True); final.add_argument("--repository-root", default=".")
    args = parser.parse_args(argv)
    if args.command == "freeze": print(freeze_campaign(args.campaign_dir, repository_root=args.repository_root))
    elif args.command == "run-task": print(run_task(manifest=args.manifest, task_index=args.task_index, repository_root=args.repository_root, stage=args.stage))
    elif args.command == "_run-one":
        rows = _read_csv(Path(args.manifest)); row = rows[args.task_index]; root = Path(args.repository_root).resolve()
        command = ["python", "-m", "phc2d.forward.full_cell_dual_lane" if row["train_lane"] == "J" else "phc2d.forward.full_cell"]
        if row["train_lane"] != "J": command += ["--model", row["model"]]
        command += ["--config", row["config"], "--seed", "0"]
        if _model_accepts_head_override(row["model"]):
            command += ["--head-kind", "dense"]
        command += ["--run-dir", row["run_dir"]]
        if args.admission_probe: command.append("--admission-probe")
        subprocess.run(command, cwd=root, check=True)
    elif args.command == "plan-eval": print(plan_evaluation(args.campaign_dir))
    elif args.command == "run-eval-shard": print(run_evaluation_shard(args.campaign_dir, args.shard_id, repository_root=args.repository_root))
    elif args.command == "finalize": print(finalize_campaign(args.campaign_dir, repository_root=args.repository_root))


if __name__ == "__main__":
    _main()


__all__ = ["freeze_campaign", "run_task", "plan_evaluation", "run_evaluation_shard", "finalize_campaign"]
