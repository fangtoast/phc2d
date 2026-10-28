"""Evaluate one frozen EXP-FWD-031 checkpoint on a frozen test dataset.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from phc2d.forward.budget_confirmation import file_digest


def ready_checkpoint(run: Path, completed: Path) -> dict:
    """Admit only a finished, validation-selected checkpoint."""
    result = json.loads((run / "results.json").read_text(encoding="utf-8"))
    marker = json.loads(completed.read_text(encoding="utf-8"))
    if result["status"] not in {"formal_training_success", "training_complete_convergence_not_met"}:
        raise ValueError("training has not reached an admissible terminal state")
    if result["selection_split"] != "validation" or result["checkpoint_role"] != "best_validation" or result["test_executed"]:
        raise ValueError("checkpoint selection contract changed")
    digest = file_digest(run / "model.pt")
    if marker["checkpoint_sha256"] != digest or result["checkpoint_sha256"] != digest:
        raise ValueError("completed marker and checkpoint hashes disagree")
    validation = json.loads((run / "per_band_validation.json").read_text(encoding="utf-8"))
    frozen_metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    if validation["checkpoint_sha256"] != digest or validation["status"] != "validation_per_band_verified":
        raise ValueError("selected checkpoint has no matching validation recomputation")
    for key in ("bands_1_10_mre_pct", "bands_2_10_mre_pct"):
        if abs(float(validation[key]) - float(frozen_metrics[key])) > 1e-3:
            raise ValueError("validation recomputation differs from frozen metrics")
    return result


def test_indices(handle, *, external: bool) -> np.ndarray:
    """Read the target test split and reject within-dataset overlap."""
    if external and "idx_test" not in handle:
        indices = np.arange(len(handle["Wh"]), dtype=np.int64)
    else:
        if "idx_test" not in handle:
            raise ValueError("native dataset has no frozen test split")
        indices = np.asarray(handle["idx_test"], dtype=np.int64)
    if not len(indices) or len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(handle["Wh"])):
        raise ValueError("invalid test indices")
    for name in ("idx_train", "idx_val"):
        if name in handle and np.intersect1d(indices, np.asarray(handle[name], dtype=np.int64)).size:
            raise ValueError(f"test overlaps {name}")
    return indices


def evaluate(run: Path, completed: Path, dataset: Path, dataset_sha256: str,
             grid_field: str, output: Path, *, external: bool = False) -> dict:
    """Write a new test cell without changing the source training run."""
    import h5py
    import torch

    from phc2d.models import build_model

    result = ready_checkpoint(run, completed)
    if output.exists():
        raise FileExistsError(output)
    if file_digest(dataset) != dataset_sha256:
        raise ValueError("target dataset SHA-256 mismatch")
    config = yaml.safe_load((run / "config.resolved.yaml").read_text(encoding="utf-8"))
    if config["experiment"]["experiment_id"].split("_", 1)[0] != "EXP-FWD-031":
        raise ValueError("unexpected source experiment")
    if file_digest(Path(config["dataset"]["path"])) != config["dataset"]["expected_dataset_sha256"]:
        raise ValueError("source dataset changed")
    payload = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    if payload["model_name"] != result["model"]:
        raise ValueError("model identity changed")
    model = build_model(payload["model_name"], input_shape=payload["input_shape"],
                        output_shape=payload["output_shape"], **payload["model_params"])
    model.load_state_dict(payload["state_dict"], strict=True)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("formal test requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model.to(device).eval()
    mean, std = payload["target_mean"].to(device), payload["target_std"].to(device)
    with h5py.File(dataset, "r") as handle:
        indices = test_indices(handle, external=external)
        if grid_field not in handle or "k_points" not in handle:
            raise ValueError("target inputs or k points missing")
        if tuple(handle[grid_field].shape[1:]) != tuple(payload["input_shape"][-2:]) or tuple(handle["Wh"].shape[1:]) != tuple(payload["output_shape"]):
            raise ValueError("target shape differs from checkpoint")
        k_hash = hashlib.sha256(np.asarray(handle["k_points"], dtype=np.float32).tobytes()).hexdigest()
        if payload.get("k_points") is None:
            raise ValueError("source checkpoint has no k-point identity")
        source_k_hash = hashlib.sha256(np.asarray(payload["k_points"], dtype=np.float32).tobytes()).hexdigest()
        if k_hash != source_k_hash:
            raise ValueError("k-point order changed")
        sums = np.zeros(int(payload["output_shape"][-1]), dtype=np.float64)
        rows = []
        torch.cuda.reset_peak_memory_stats(device)
        start = datetime.now(timezone.utc)
        with torch.no_grad():
            for offset in range(0, len(indices), 16):
                batch = np.sort(indices[offset:offset + 16])
                x = torch.from_numpy(np.asarray(handle[grid_field][batch], dtype=np.float32)).to(device)
                y = np.asarray(handle["Wh"][batch], dtype=np.float64)
                prediction = (model(x) * std + mean).cpu().numpy().astype(np.float64)
                if prediction.shape != y.shape or not np.isfinite(prediction).all() or not np.isfinite(y).all():
                    raise ValueError("nonfinite or shape-mismatched prediction")
                relative = np.abs(prediction - y) / np.maximum(np.abs(y), 1e-8)
                sums += relative.sum(axis=(0, 1))
                for index, sample in zip(batch, relative, strict=True):
                    rows.append(dict(dataset_index=int(index), full_matrix_mre_pct=float(sample.mean() * 100),
                                     bands_2_10_mre_pct=float(sample[..., 1:].mean() * 100)))
        torch.cuda.synchronize(device)
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        bands = (sums * (100 / (len(indices) * handle["Wh"].shape[1]))).tolist()
    metrics = dict(bands_1_10_mre_pct=float(np.mean(bands)), bands_2_10_mre_pct=float(np.mean(bands[1:])),
                   band_1_mre_pct=bands[0], per_band_mre_pct=bands,
                   sample_p95_mre_pct=float(np.percentile([r["full_matrix_mre_pct"] for r in rows], 95)),
                   samples=len(rows), inference_seconds=elapsed, samples_per_second=len(rows) / elapsed,
                   peak_gpu_bytes=torch.cuda.max_memory_allocated(device))
    output.mkdir(parents=True)
    with (output / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (output / "per_band_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["band_one_based", "mre_pct"])
        writer.writeheader()
        writer.writerows(dict(band_one_based=i + 1, mre_pct=value) for i, value in enumerate(bands))
    record = dict(status="test_complete", model=result["model"], source_status=result["status"],
                  checkpoint_sha256=result["checkpoint_sha256"], target_sha256=dataset_sha256,
                  target_split_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                  grid_field=grid_field, source_run=str(run), sample_count=len(rows), external=external,
                  evaluation_source_sha256=file_digest(Path(__file__)))
    (output / "results.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    (output / "config.resolved.yaml").write_text(yaml.safe_dump(dict(source_run=str(run), dataset=str(dataset),
        dataset_sha256=dataset_sha256, grid_field=grid_field, batch_size=16, precision="fp32")), encoding="utf-8")
    (output / "run.meta.json").write_text(json.dumps(dict(start_utc=start.isoformat(), device=torch.cuda.get_device_name(device))) + "\n", encoding="utf-8")
    (output / "run.log").write_text("Frozen test evaluation completed.\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    for name in ("run", "completed", "dataset", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--grid-field", required=True)
    parser.add_argument("--external", action="store_true")
    args = parser.parse_args()
    evaluate(args.run, args.completed, args.dataset, args.dataset_sha256, args.grid_field, args.output, external=args.external)


if __name__ == "__main__":
    main()
