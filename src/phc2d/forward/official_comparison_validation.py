"""Read-only selected-checkpoint per-band validation for EXP-FWD-031.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from phc2d.forward.budget_confirmation import file_digest, write_json


def per_band_mre_sum(prediction: Any, target: Any, torch: Any) -> Any:
    """Sum relative error across the sample and k-point axes."""
    return (torch.abs(prediction - target) / torch.clamp(torch.abs(target), min=1e-8)).sum(dim=(0, 1))


def report_per_band_validation(run: Path, *, batch_size: int = 16, device: str = "cpu",
                               expected_experiment: str = "EXP-FWD-031") -> dict:
    """Re-evaluate only idx_val using a frozen validation-selected checkpoint."""
    import h5py
    import torch

    from phc2d.models import build_model

    result = json.loads((run / "results.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((run / "config.resolved.yaml").read_text(encoding="utf-8"))
    experiment_id = config["experiment"]["experiment_id"]
    if expected_experiment not in {"EXP-FWD-031", "EXP-FWD-032"} or experiment_id.split("_", 1)[0] != expected_experiment:
        raise ValueError("per-band validation experiment identity mismatch")
    if result["selection_split"] != "validation" or result["test_executed"]:
        raise ValueError("only validation-selected checkpoints are admissible")
    if result["status"] not in {"formal_training_success", "training_complete_convergence_not_met"}:
        raise ValueError("training has not reached a complete terminal state")
    checkpoint = run / "model.pt"
    digest = file_digest(checkpoint)
    if digest != result["checkpoint_sha256"]:
        raise ValueError("selected checkpoint SHA-256 changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["model_name"] != result["model"]:
        raise ValueError("checkpoint model identity changed")
    model = build_model(payload["model_name"], input_shape=payload["input_shape"],
                        output_shape=payload["output_shape"], **payload["model_params"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    mean = payload["target_mean"].to(device)
    std = payload["target_std"].to(device)
    path = Path(config["dataset"]["path"])
    grid_field = config["dataset"]["grid_field"]
    total = torch.zeros(int(payload["output_shape"][-1]), dtype=torch.float64)
    count = 0
    with h5py.File(path, "r") as handle, torch.no_grad():
        indices = np.sort(np.asarray(handle["idx_val"], dtype=np.int64))
        for begin in range(0, len(indices), batch_size):
            positions = indices[begin:begin + batch_size]
            x = torch.from_numpy(np.asarray(handle[grid_field][positions], dtype=np.float32)).to(device)
            y = torch.from_numpy(np.asarray(handle["Wh"][positions], dtype=np.float32)).to(device)
            prediction = model(x) * std + mean
            if tuple(prediction.shape) != tuple(y.shape) or not bool(torch.isfinite(prediction).all()):
                raise ValueError("non-finite or shape-mismatched validation prediction")
            total += per_band_mre_sum(prediction, y, torch).double().cpu()
            count += int(y.shape[0] * y.shape[1])
    bands = (total * (100.0 / count)).tolist()
    all_mre = sum(bands) / len(bands)
    excluded_first = sum(bands[1:]) / (len(bands) - 1)
    for observed, expected in ((all_mre, metrics["bands_1_10_mre_pct"]),
                               (excluded_first, metrics["bands_2_10_mre_pct"])):
        if abs(observed - float(expected)) > 1e-3:
            raise ValueError("per-band validation disagrees with frozen aggregate metrics")
    record = dict(status="validation_per_band_verified", experiment_id=experiment_id,
                  model=payload["model_name"],
                  checkpoint_sha256=digest, dataset_sha256=config["dataset"]["expected_dataset_sha256"],
                  validation_count=len(indices), per_band_mre_pct=bands,
                  bands_1_10_mre_pct=all_mre, bands_2_10_mre_pct=excluded_first,
                  test_executed=False)
    write_json(run / "per_band_validation.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--experiment-id", choices=["EXP-FWD-031", "EXP-FWD-032"], default="EXP-FWD-031")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    report_per_band_validation(args.run, batch_size=args.batch_size, device=args.device,
                               expected_experiment=args.experiment_id)


if __name__ == "__main__":
    main()
