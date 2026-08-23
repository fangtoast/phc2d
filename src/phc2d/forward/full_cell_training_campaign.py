"""Config-driven frozen dual-lane Full-Wh training campaign control plane.

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

SINGLE_LANES = ("B", "G")
JOINT_LANE = "J"
SUCCESS = frozenset({"formal_training_success", "dual_lane_training_success"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _profile(root: Path, profile: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(profile, Mapping):
        return dict(profile)
    path = Path(profile)
    path = path if path.is_absolute() else root / path
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"campaign profile must be a mapping: {path}")
    return dict(value)


def _commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        marker = root / "DEPLOYED_GIT_COMMIT"
        return marker.read_text(encoding="utf-8").strip() if marker.is_file() else "unknown"


def _array_hash(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value, dtype=np.int64)).tobytes()).hexdigest()


def _audit_dataset(root: Path, lane: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    import h5py

    path = root / str(contract["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if actual != str(contract["sha256"]):
        raise RuntimeError(f"{lane} dataset hash mismatch: {actual}")
    grid = str(contract["grid_field"])
    with h5py.File(path, "r") as data:
        required = {grid, "Wh", "k_points", "idx_train", "idx_val", "idx_test", "sample_id"}
        missing = sorted(required.difference(data.keys()))
        if missing:
            raise RuntimeError(f"{lane} missing fields: {missing}")
        splits = {key: np.asarray(data[key], dtype=np.int64) for key in ("idx_train", "idx_val", "idx_test")}
        if tuple(data["Wh"].shape) != (10000, 81, 10) or tuple(data[grid].shape) != (10000, 64, 64):
            raise RuntimeError(f"{lane} shape contract mismatch")
        if {key: len(value) for key, value in splits.items()} != {"idx_train": 8000, "idx_val": 1000, "idx_test": 1000}:
            raise RuntimeError(f"{lane} split contract mismatch")
        if len(np.unique(np.concatenate(tuple(splits.values())))) != 10000:
            raise RuntimeError(f"{lane} split overlap")
        k_points = np.asarray(data["k_points"], dtype=np.float32)
        if k_points.shape != (81, 2) or not np.isfinite(k_points).all():
            raise RuntimeError(f"{lane} invalid k grid")
        return {"lane": lane, "dataset_path": str(contract["path"]), "grid_field": grid, "dataset_sha256": actual,
                "idx_train_hash": _array_hash(splits["idx_train"]), "idx_val_hash": _array_hash(splits["idx_val"]),
                "idx_test_hash": _array_hash(splits["idx_test"]), "k_points_sha256": hashlib.sha256(np.ascontiguousarray(k_points).tobytes()).hexdigest(),
                "shape": [10000, 81, 10], "grid_shape": [10000, 64, 64]}


def _task_rows(campaign: Path, profile: Mapping[str, Any], audits: Mapping[str, Mapping[str, Any]], *, stage: str) -> list[dict[str, Any]]:
    models = profile["models"]
    tasks = [(model, lane, "W1") for model in models["baseline"] for lane in SINGLE_LANES]
    tasks += [(model, lane, "W2") for model in models["single_lane_upgrade"] for lane in SINGLE_LANES]
    tasks += [(str(models["joint"]), JOINT_LANE, "W3")]
    if len(tasks) != 17:
        raise RuntimeError(f"profile must resolve to exactly 17 tasks, got {len(tasks)}")
    rows: list[dict[str, Any]] = []
    for index, (model, lane, wave) in enumerate(tasks):
        audit = audits["B" if lane == JOINT_LANE else lane]
        config = str(profile["configs"][lane])
        rows.append({"task_index": index, "task_id": f"{stage.lower()}_{index:02d}_{model}_{lane}_s0", "model": model,
                     "train_lane": lane, "wave": wave, "seed": int(profile["training"]["seed"]), "head": profile["training"]["head"],
                     "config": config, "config_sha256": _sha256(Path(config)), "dataset_sha256": audit["dataset_sha256"],
                     "idx_train_hash": audit["idx_train_hash"], "idx_val_hash": audit["idx_val_hash"], "idx_test_hash": audit["idx_test_hash"],
                     "k_points_sha256": audit["k_points_sha256"], "run_dir": (campaign / f"{stage.lower()}_{index:02d}_{model}_{lane}_s0").as_posix(),
                     "stage": stage.lower(), "evidence_status": profile["evidence_status"]})
    return rows


def freeze_campaign(profile: str | Path | Mapping[str, Any], campaign_dir: str | Path, *, repository_root: str | Path = ".") -> Path:
    root = Path(repository_root).resolve(); resolved = _profile(root, profile); campaign = Path(campaign_dir)
    campaign = campaign if campaign.is_absolute() else root / campaign
    campaign.mkdir(parents=True, exist_ok=False)
    audits = {lane: _audit_dataset(root, lane, resolved["datasets"][lane]) for lane in SINGLE_LANES}
    if audits["B"]["k_points_sha256"] != audits["G"]["k_points_sha256"]:
        raise RuntimeError("B/G k-grid hash mismatch")
    formal = _task_rows(campaign, resolved, audits, stage="FORMAL")
    admission = _task_rows(campaign, resolved, audits, stage="ADMISSION")
    _write_csv(campaign / "task_manifest.csv", formal); _write_csv(campaign / "admission_manifest.csv", admission)
    _write_json(campaign / "dataset_audit.json", {"status": "pass", "lanes": audits, "shared_k_points": True})
    frozen = {**resolved, "deployed_git_commit": _commit(root), "frozen_at": time.time()}
    (campaign / "campaign_profile.resolved.yaml").write_text(yaml.safe_dump(frozen, sort_keys=False, allow_unicode=True), encoding="utf-8")
    _write_json(campaign / "freeze.json", {"status": "frozen", "experiment_id": resolved["experiment_id"], "task_count": 17,
        "admission_task_count": 17, "test_read_before_checkpoint_freeze": False, "deployed_git_commit": _commit(root),
        "evidence_status": resolved["evidence_status"], "next_gate": "17_one_batch_admission"})
    return campaign


def _accepts_head(model_name: str) -> bool:
    from phc2d.models import get_model_spec
    return bool(get_model_spec(model_name).accepts_k_points)


def run_task(*, manifest: str | Path, task_index: int, repository_root: str | Path = ".", stage: str = "formal") -> Path:
    root = Path(repository_root).resolve(); row = _read_csv(Path(manifest))[int(task_index)]
    if int(row["task_index"]) != int(task_index) or row["stage"] != stage.lower():
        raise RuntimeError("task manifest stage/index mismatch")
    command = ["python", "-m", "phc2d.forward.full_cell_dual_lane" if row["train_lane"] == JOINT_LANE else "phc2d.forward.full_cell"]
    if row["train_lane"] != JOINT_LANE:
        command += ["--model", row["model"]]
    command += ["--config", row["config"], "--seed", row["seed"], "--run-dir", row["run_dir"]]
    if _accepts_head(row["model"]): command += ["--head-kind", row["head"]]
    if stage.lower() == "admission": command.append("--admission-probe")
    subprocess.run(command, cwd=root, check=True)
    return Path(row["run_dir"])


def finalize_training(campaign_dir: str | Path, *, repository_root: str | Path = ".") -> Path:
    root = Path(repository_root).resolve(); campaign = Path(campaign_dir).resolve(); rows = _read_csv(campaign / "task_manifest.csv")
    inventory: list[dict[str, Any]] = []; summaries: list[dict[str, Any]] = []
    for row in rows:
        run = Path(row["run_dir"]); result_path = run / "results.json"; checkpoint = run / "model.pt"
        if not result_path.is_file() or not checkpoint.is_file(): raise RuntimeError(f"missing formal artifacts: {run}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") not in SUCCESS or result.get("test_executed"):
            raise RuntimeError(f"not validation-only training success: {run}")
        stop, best = int(result["stop_epoch"]), int(result["best_epoch"])
        if not 1 <= stop <= 100 or not 1 <= best <= stop: raise RuntimeError(f"invalid early-stop record: {run}")
        payload = {**row, "checkpoint_path": checkpoint.as_posix(), "checkpoint_sha256": _sha256(checkpoint), "checkpoint_role": "best_validation", "selection_split": "validation"}
        identity = result.get("model_identity")
        if not isinstance(identity, Mapping): raise RuntimeError(f"model identity missing: {run}")
        required = {"canonical_model_name", "factory_function_identity", "state_dict_schema_hash", "architecture_fingerprint", "parameter_count"}
        if not required.issubset(identity): raise RuntimeError(f"incomplete model identity: {run}")
        inventory.append({**payload, **{key: identity[key] for key in required}})
        summaries.append({"task_id": row["task_id"], "model": row["model"], "train_lane": row["train_lane"], "best_epoch": best, "stop_epoch": stop, "best_validation_mre_pct": result.get("best_validation_bands_2_10_mre_pct", result.get("best_validation_equal_lane_mre_pct"))})
    if len(inventory) != 17 or len({(x["model"], x["train_lane"], x["seed"]) for x in inventory}) != 17: raise RuntimeError("training checkpoint identity collision")
    _write_csv(campaign / "checkpoint_inventory.csv", inventory); _write_csv(campaign / "training_summary.csv", summaries)
    curve_rows = [{"task_id": row["task_id"], "history_csv": str(Path(row["run_dir"]) / "history.csv"), "png": str(Path(row["run_dir"]) / "figures" / "training_curve.png"), "pdf": str(Path(row["run_dir"]) / "figures" / "training_curve.pdf")} for row in rows]
    _write_csv(campaign / "curve_manifest.csv", curve_rows)
    _write_json(campaign / "results.json", {"status": "training_complete_pending_six_domain_evaluation", "training_task_count": 17, "checkpoint_count": 17, "evidence_status": "user_authorized_surrogate_training_pending_wsl_acceptance"})
    return campaign


def _main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze"); freeze.add_argument("--profile", required=True); freeze.add_argument("--campaign-dir", required=True); freeze.add_argument("--repository-root", default=".")
    task = sub.add_parser("run-task"); task.add_argument("--manifest", required=True); task.add_argument("--task-index", type=int, required=True); task.add_argument("--stage", choices=("admission", "formal"), default="formal"); task.add_argument("--repository-root", default=".")
    final = sub.add_parser("finalize-training"); final.add_argument("--campaign-dir", required=True); final.add_argument("--repository-root", default=".")
    args = parser.parse_args(argv)
    if args.command == "freeze": print(freeze_campaign(args.profile, args.campaign_dir, repository_root=args.repository_root))
    elif args.command == "run-task": print(run_task(manifest=args.manifest, task_index=args.task_index, stage=args.stage, repository_root=args.repository_root))
    else: print(finalize_training(args.campaign_dir, repository_root=args.repository_root))


if __name__ == "__main__": _main()


__all__ = ["freeze_campaign", "run_task", "finalize_training"]
