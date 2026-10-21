"""Frozen data and task contracts for fair multi-domain Full-Wh campaigns.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from phc2d.data.sampling import select_nested_sample_indices
from phc2d.models import get_model_spec
from phc2d.runs import ensure_figures_dir, write_metrics, write_results, write_run_meta

SCHEMA_VERSION = "phc2d.full_cell_multidomain.v1"
TRAINING_SUCCESS_STATUSES = frozenset(
    {"formal_training_success", "training_complete_convergence_not_met"}
)


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: Any, *, dtype: Any | None = None) -> str:
    """Hash one contiguous array with an optional canonical dtype."""

    array = np.asarray(values, dtype=dtype) if dtype is not None else np.asarray(values)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Atomically write a non-empty manifest CSV."""

    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(str(key) for row in rows for key in row))
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)
    return output


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a manifest by field name."""

    with Path(path).open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_profile(path: str | Path) -> dict[str, Any]:
    """Load and validate the top-level multi-domain profile mapping."""

    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise TypeError("multi-domain profile must be a mapping")
    profile = dict(payload)
    if profile.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must equal {SCHEMA_VERSION}")
    if str((profile.get("experiment") or {}).get("experiment_id")) != "EXP-FWD-021":
        raise ValueError("multi-domain v1 is frozen to EXP-FWD-021")
    return profile


def repository_commit(root: str | Path) -> str:
    """Return the checkout or deployed commit used to freeze evidence."""

    repository = Path(root)
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        marker = repository / "DEPLOYED_GIT_COMMIT"
        if marker.is_file():
            return marker.read_text(encoding="utf-8").strip()
        return os.environ.get("PHC2D_DEPLOYED_GIT_COMMIT", "unknown")


def resolve_dataset_path(
    repository_root: str | Path,
    contract: Mapping[str, Any],
    *,
    path_key: str,
) -> Path:
    """Resolve a configured local or platform-specific dataset path."""

    raw = contract.get(path_key)
    if not raw:
        raise KeyError(f"dataset contract is missing path key {path_key!r}")
    path = Path(str(raw))
    return path if path.is_absolute() else Path(repository_root) / path


def audit_dataset(
    domain: str,
    contract: Mapping[str, Any],
    path: str | Path,
    *,
    verify_file_hash: bool = True,
) -> dict[str, Any]:
    """Audit one HDF5 dataset without reading held-out labels."""

    import h5py

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    actual_hash = sha256_file(source) if verify_file_hash else str(contract["sha256"])
    if actual_hash != str(contract["sha256"]):
        raise RuntimeError(f"{domain} dataset hash mismatch: {actual_hash}")
    grid_field = str(contract["grid_field"])
    expected_target_shape = tuple(int(value) for value in contract["target_shape"])
    expected_grid_shape = tuple(int(value) for value in contract["grid_shape"])
    with h5py.File(source, "r") as handle:
        required = {
            grid_field,
            "Wh",
            "k_points",
            "idx_train",
            "idx_val",
            "idx_test",
            "sample_id",
        }
        missing = sorted(required.difference(handle.keys()))
        if missing:
            raise RuntimeError(f"{domain} missing fields: {missing}")
        sample_count = int(contract["sample_count"])
        if tuple(handle["Wh"].shape) != (sample_count, *expected_target_shape):
            raise RuntimeError(f"{domain} target shape mismatch")
        if tuple(handle[grid_field].shape) != (sample_count, *expected_grid_shape):
            raise RuntimeError(f"{domain} grid shape mismatch")
        splits = {
            key: np.asarray(handle[key], dtype=np.int64)
            for key in ("idx_train", "idx_val", "idx_test")
        }
        expected_counts = {
            "idx_train": int(contract["train_count"]),
            "idx_val": int(contract["validation_count"]),
            "idx_test": int(contract["test_count"]),
        }
        if {key: len(value) for key, value in splits.items()} != expected_counts:
            raise RuntimeError(f"{domain} split count mismatch")
        combined = np.concatenate(tuple(splits.values()))
        if len(np.unique(combined)) != len(combined):
            raise RuntimeError(f"{domain} split overlap")
        if int(combined.min()) < 0 or int(combined.max()) >= sample_count:
            raise RuntimeError(f"{domain} split index out of range")
        k_points = np.asarray(handle["k_points"], dtype=np.float32)
        if k_points.shape != (expected_target_shape[0], 2) or not np.isfinite(k_points).all():
            raise RuntimeError(f"{domain} invalid k-point contract")
        sample_ids = np.asarray(handle["sample_id"]).astype("U")
        if sample_ids.shape != (sample_count,) or len(np.unique(sample_ids)) != sample_count:
            raise RuntimeError(f"{domain} sample_id values must be one-to-one and unique")
    return {
        "domain": domain,
        "label": str(contract.get("label", domain)),
        "dataset_path": str(source),
        "grid_field": grid_field,
        "dataset_sha256": actual_hash,
        "sample_count": int(contract["sample_count"]),
        "train_count": len(splits["idx_train"]),
        "validation_count": len(splits["idx_val"]),
        "test_count": len(splits["idx_test"]),
        "target_shape": json.dumps(list(expected_target_shape)),
        "grid_shape": json.dumps(list(expected_grid_shape)),
        "idx_train_hash": sha256_array(splits["idx_train"], dtype=np.int64),
        "idx_val_hash": sha256_array(splits["idx_val"], dtype=np.int64),
        "idx_test_hash": sha256_array(splits["idx_test"], dtype=np.int64),
        "k_points_sha256": sha256_array(k_points, dtype=np.float32),
        "sample_id_sha256": sha256_array(sample_ids.astype("S")),
        "test_targets_loaded": False,
    }


def _atomic_numpy(path: Path, values: Any, *, compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        if compressed:
            np.savez_compressed(handle, **values)
        else:
            np.save(handle, values)
    os.replace(temporary, path)


def _selected_indices(
    dataset_path: Path,
    *,
    count: int,
    salt: str,
) -> np.ndarray:
    import h5py

    with h5py.File(dataset_path, "r") as handle:
        idx_train = np.sort(np.asarray(handle["idx_train"], dtype=np.int64))
        sample_ids = np.asarray(handle["sample_id"][idx_train]).astype("U").tolist()
    return select_nested_sample_indices(idx_train, sample_ids, size=count, salt=salt)


def _pooled_normalizer(
    datasets: Mapping[str, Mapping[str, Any]],
    selections: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    values: list[np.ndarray] = []
    for domain, indices in selections.items():
        with h5py.File(str(datasets[domain]["dataset_path"]), "r") as handle:
            values.append(np.asarray(handle["Wh"][indices], dtype=np.float32))
    pooled = np.concatenate(values, axis=0)
    mean = np.mean(pooled, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(pooled, axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, np.float32(1.0e-6))


def _subset_provenance(dataset_path: Path, indices: np.ndarray) -> dict[str, Any]:
    """Verify split isolation and hash the selected stable sample identities."""

    import h5py

    with h5py.File(dataset_path, "r") as handle:
        validation = np.asarray(handle["idx_val"], dtype=np.int64)
        test = np.asarray(handle["idx_test"], dtype=np.int64)
        selected_sample_ids = np.asarray(handle["sample_id"][indices]).astype("U")
    validation_overlap = int(np.intersect1d(indices, validation).size)
    test_overlap = int(np.intersect1d(indices, test).size)
    if validation_overlap or test_overlap:
        raise RuntimeError("selected subset overlaps validation or test")
    if len(np.unique(selected_sample_ids)) != len(indices):
        raise RuntimeError("selected sample identities are not unique")
    return {
        "selected_sample_ids_sha256": sha256_array(selected_sample_ids.astype("S")),
        "validation_overlap_count": validation_overlap,
        "test_overlap_count": test_overlap,
    }


def _validate_campaign_shape(profile: Mapping[str, Any]) -> None:
    models = dict(profile.get("models") or {})
    regimes = dict(profile.get("regimes") or {})
    datasets = dict(profile.get("datasets") or {})
    training = dict(profile.get("training") or {})
    evaluation = dict(profile.get("evaluation") or {})
    if len(models) != 9 or len(regimes) != 7 or len(datasets) != 6:
        raise ValueError("EXP-FWD-021 requires 9 models, 7 regimes, and 6 domains")
    if int(training.get("seed", -1)) != 0 or int(training.get("max_epochs", -1)) != 200:
        raise ValueError("EXP-FWD-021 requires seed 0 and exactly 200 epochs")
    if int(training.get("train_size_per_task", -1)) != 8000:
        raise ValueError("EXP-FWD-021 requires 8,000 training samples per task")
    if set(evaluation.get("targets") or []) != set(datasets):
        raise ValueError("evaluation targets must equal the six configured domains")
    for regime, raw in regimes.items():
        domains = dict((raw or {}).get("domains") or {})
        if set(domains).difference(datasets):
            raise ValueError(f"{regime} references an unknown domain")
        if sum(int(value) for value in domains.values()) != 8000:
            raise ValueError(f"{regime} must contain exactly 8,000 samples")
    learning_rates = dict(training.get("learning_rates") or {})
    if set(learning_rates) != set(models):
        raise ValueError("every model requires exactly one frozen learning rate")
    for model in models:
        spec = get_model_spec(model)
        if spec.output_target != "Wh" or spec.accepts_lane_id:
            raise ValueError(f"model {model!r} is not an unconditioned Full-Wh model")


def freeze_campaign(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    repository_root: str | Path = ".",
    dataset_path_key: str | None = None,
    verify_file_hash: bool = True,
) -> Path:
    """Freeze all data, subset, normalizer, and 63 training identities."""

    root = Path(repository_root).resolve()
    config_file = Path(config_path)
    config_file = config_file if config_file.is_absolute() else root / config_file
    profile = load_profile(config_file)
    _validate_campaign_shape(profile)
    campaign = Path(output_dir).resolve()
    if campaign.exists() and any(campaign.iterdir()):
        raise FileExistsError(f"campaign directory is not empty: {campaign}")
    campaign.mkdir(parents=True, exist_ok=True)
    ensure_figures_dir(campaign)
    selected_path_key = dataset_path_key or str(
        (profile.get("runtime") or {}).get("dataset_path_key", "path")
    )
    audits: dict[str, dict[str, Any]] = {}
    resolved_datasets: dict[str, dict[str, Any]] = {}
    for domain, raw_contract in dict(profile["datasets"]).items():
        contract = dict(raw_contract)
        path = resolve_dataset_path(root, contract, path_key=selected_path_key)
        audit = audit_dataset(
            domain, contract, path, verify_file_hash=verify_file_hash
        )
        audits[domain] = audit
        resolved_datasets[domain] = {**contract, "resolved_path": str(path)}
    if len({row["target_shape"] for row in audits.values()}) != 1:
        raise RuntimeError("target shapes differ across domains")
    if len({row["grid_shape"] for row in audits.values()}) != 1:
        raise RuntimeError("grid shapes differ across domains")
    if len({row["k_points_sha256"] for row in audits.values()}) != 1:
        raise RuntimeError("k-point grids differ across domains")

    ranking_salt = str(profile["subset"]["ranking_salt"])
    subset_rows: list[dict[str, Any]] = []
    normalizer_rows: list[dict[str, Any]] = []
    selections_by_domain: dict[str, list[np.ndarray]] = {
        domain: [] for domain in audits
    }
    for regime, raw_regime in dict(profile["regimes"]).items():
        domain_counts = {
            domain: int(count)
            for domain, count in dict(raw_regime["domains"]).items()
        }
        selections: dict[str, np.ndarray] = {}
        for domain, count in domain_counts.items():
            salt = f"{ranking_salt}:{domain}"
            indices = _selected_indices(
                Path(audits[domain]["dataset_path"]), count=count, salt=salt
            )
            selections[domain] = indices
            selections_by_domain[domain].append(indices)
            indices_relative = Path("subsets") / regime / f"{domain}.npy"
            _atomic_numpy(campaign / indices_relative, indices)
            subset_provenance = _subset_provenance(
                Path(audits[domain]["dataset_path"]), indices
            )
            subset_rows.append(
                {
                    "regime": regime,
                    "stage": int(raw_regime["stage"]),
                    "domain": domain,
                    "sample_count": len(indices),
                    "ranking_salt": salt,
                    "indices_path": str(indices_relative).replace("\\", "/"),
                    "indices_sha256": sha256_array(indices, dtype=np.int64),
                    "selected_sample_ids_sha256": subset_provenance[
                        "selected_sample_ids_sha256"
                    ],
                    "dataset_sha256": audits[domain]["dataset_sha256"],
                    "idx_train_hash": audits[domain]["idx_train_hash"],
                    "validation_overlap_count": subset_provenance[
                        "validation_overlap_count"
                    ],
                    "test_overlap_count": subset_provenance["test_overlap_count"],
                }
            )
        mean, std = _pooled_normalizer(audits, selections)
        normalizer_relative = Path("normalizers") / f"{regime}.npz"
        _atomic_numpy(
            campaign / normalizer_relative,
            {"target_mean": mean, "target_std": std},
            compressed=True,
        )
        normalizer_rows.append(
            {
                "regime": regime,
                "training_sample_count": sum(domain_counts.values()),
                "normalizer_path": str(normalizer_relative).replace("\\", "/"),
                "target_mean_sha256": sha256_array(mean, dtype=np.float32),
                "target_std_sha256": sha256_array(std, dtype=np.float32),
                "normalizer_file_sha256": sha256_file(campaign / normalizer_relative),
                "fit_scope": "selected_training_samples_only",
            }
        )

    for domain, arrays in selections_by_domain.items():
        if not arrays:
            raise RuntimeError(f"domain {domain} is unused by every regime")
        largest = max(arrays, key=len)
        largest_set = set(largest.tolist())
        if any(not set(array.tolist()).issubset(largest_set) for array in arrays):
            raise RuntimeError(f"nested subset contract failed for {domain}")

    task_rows: list[dict[str, Any]] = []
    index = 0
    for regime, raw_regime in dict(profile["regimes"]).items():
        for model in dict(profile["models"]):
            task_id = f"{regime.lower()}_{model}_s0"
            task_rows.append(
                {
                    "task_index": index,
                    "task_id": task_id,
                    "experiment_id": "EXP-FWD-021",
                    "stage": int(raw_regime["stage"]),
                    "regime": regime,
                    "model": model,
                    "seed": 0,
                    "training_sample_count": 8000,
                    "max_epochs": 200,
                    "learning_rate": float(profile["training"]["learning_rates"][model]),
                    "run_dir": f"training/{regime}/{model}_s0",
                    "config_path": str(config_file),
                    "execution_start_git_commit": repository_commit(root),
                }
            )
            index += 1
    if len(task_rows) != 63 or len({row["task_id"] for row in task_rows}) != 63:
        raise RuntimeError("EXP-FWD-021 must freeze 63 unique training tasks")

    write_csv(campaign / "dataset_manifest.csv", list(audits.values()))
    write_csv(campaign / "subset_manifest.csv", subset_rows)
    write_csv(campaign / "normalizer_manifest.csv", normalizer_rows)
    write_csv(campaign / "task_manifest.csv", task_rows)
    resolved = {
        **profile,
        "datasets": resolved_datasets,
        "freeze": {
            "dataset_path_key": selected_path_key,
            "repository_root": str(root),
            "execution_start_git_commit": repository_commit(root),
            "test_targets_loaded": False,
        },
    }
    (campaign / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    write_run_meta(
        campaign,
        {
            "script": "python -m phc2d.forward.full_cell_multidomain_campaign freeze",
            "experiment_id": "EXP-FWD-021",
            "stage": "contract_freeze",
            "execution_start_git_commit": repository_commit(root),
            "scientific_boundary": "configuration_only_no_training_no_test",
        },
    )
    (campaign / "run.log").write_text(
        "Frozen 6 datasets, 7 regimes, 63 training tasks; held-out targets were not loaded.\n",
        encoding="utf-8",
    )
    write_csv(
        campaign / "history.csv",
        [{"phase": "freeze", "status": "complete", "training_started": False}],
    )
    write_metrics(
        campaign,
        {
            "dataset_count": 6,
            "regime_count": 7,
            "training_task_count": 63,
            "training_samples_per_task": 8000,
            "test_targets_loaded": False,
        },
    )
    write_results(
        campaign,
        {
            "status": "config_frozen_pending_admission",
            "scientific_boundary": "configuration_only_no_training_no_test",
            "model_promotion": False,
            "physical_claim": False,
        },
    )
    return campaign


def audit_training_runs(
    manifest_path: str | Path,
    *,
    start_index: int = 0,
    end_index: int | None = None,
) -> dict[str, Any]:
    """Fail closed unless a contiguous task slice has complete accepted artifacts."""

    manifest = Path(manifest_path).resolve()
    rows = read_csv(manifest)
    stop = len(rows) - 1 if end_index is None else int(end_index)
    selected = [
        row
        for row in rows
        if int(start_index) <= int(row["task_index"]) <= stop
    ]
    if len(selected) != stop - int(start_index) + 1:
        raise RuntimeError("training audit slice is incomplete")
    checkpoint_hashes: set[str] = set()
    for row in selected:
        run = manifest.parent / row["run_dir"]
        required = (
            "run.log",
            "run.meta.json",
            "config.resolved.yaml",
            "results.json",
            "metrics.json",
            "history.csv",
            "model.pt",
            "epoch200.pt",
            "model_summary.json",
        )
        missing = [name for name in required if not (run / name).is_file()]
        if missing:
            raise RuntimeError(f"{row['task_id']} missing artifacts: {missing}")
        result = json.loads((run / "results.json").read_text(encoding="utf-8"))
        if result.get("status") not in TRAINING_SUCCESS_STATUSES:
            raise RuntimeError(f"{row['task_id']} has unaccepted status {result.get('status')}")
        if bool(result.get("test_executed", True)):
            raise RuntimeError(f"{row['task_id']} accessed held-out test")
        if int(result.get("stop_epoch", -1)) != 200:
            raise RuntimeError(f"{row['task_id']} did not stop at epoch 200")
        if result.get("task_id") != row["task_id"] or result.get("model") != row["model"]:
            raise RuntimeError(f"{row['task_id']} result identity mismatch")
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        finite_fields = (
            "best_validation_bands_2_10_mre_pct",
            "train_time_s",
            "peak_gpu_memory_mb",
        )
        if not np.isfinite(
            np.asarray([metrics.get(name) for name in finite_fields], dtype=np.float64)
        ).all():
            raise RuntimeError(f"{row['task_id']} has non-finite training metrics")
        history = read_csv(run / "history.csv")
        if len(history) != 200 or {int(item["epoch"]) for item in history} != set(range(1, 201)):
            raise RuntimeError(f"{row['task_id']} history is not exactly epochs 1-200")
        if any(
            not np.isfinite(
                np.asarray(
                    [
                        item["train_loss"],
                        item["val_bands_2_10_mre_pct"],
                        item["val_bands_1_10_mre_pct"],
                    ],
                    dtype=np.float64,
                )
            ).all()
            for item in history
        ):
            raise RuntimeError(f"{row['task_id']} history contains non-finite metrics")
        checkpoint_hash = sha256_file(run / "model.pt")
        if checkpoint_hash != result.get("checkpoint_sha256"):
            raise RuntimeError(f"{row['task_id']} checkpoint hash mismatch")
        if checkpoint_hash in checkpoint_hashes:
            raise RuntimeError("checkpoint collision detected within training audit slice")
        checkpoint_hashes.add(checkpoint_hash)
    return {
        "status": "training_slice_complete",
        "start_index": int(start_index),
        "end_index": stop,
        "task_count": len(selected),
    }


__all__ = [
    "SCHEMA_VERSION",
    "TRAINING_SUCCESS_STATUSES",
    "audit_dataset",
    "audit_training_runs",
    "freeze_campaign",
    "load_profile",
    "read_csv",
    "repository_commit",
    "resolve_dataset_path",
    "sha256_array",
    "sha256_file",
    "write_csv",
]
