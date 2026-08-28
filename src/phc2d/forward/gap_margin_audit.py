"""Validation-only near-degeneracy audit for paired Full-Wh checkpoints.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from phc2d.forward.dual_lane_contracts import LANES, read_dual_lane_contracts
from phc2d.forward.full_cell_metrics import (
    compute_gap_margin_audit_metrics,
    derive_gap_decision_tolerances,
)
from phc2d.hpc import sha256_file
from phc2d.models import build_model, get_model_spec
from phc2d.runs import (
    RUN_LOG_NAME,
    ensure_figures_dir,
    make_run_dir,
    setup_logging,
    validate_run_artifacts,
    write_metrics,
    write_resolved_config,
    write_results,
    write_run_meta,
)


def _load_checkpoint(path: Path, expected_sha256: str, *, torch_module: Any) -> Any:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"checkpoint SHA256 mismatch for {path}: {actual_sha256}")
    payload = torch_module.load(path, map_location="cpu", weights_only=True)
    spec = get_model_spec(str(payload["model_name"]))
    model = build_model(
        spec.name,
        input_shape=payload["input_shape"],
        output_shape=payload["output_shape"],
        k_points=payload["k_points"],
        **payload["model_params"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _read_targets(path: Path, indices: np.ndarray) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as handle:
        return np.asarray(handle["Wh"][np.sort(indices)], dtype=np.float32)


def _read_validation(
    contract: Mapping[str, Any],
    *,
    grid_field: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one frozen validation split once for all audited models."""

    import h5py

    indices = np.sort(np.asarray(contract["idx_val"], dtype=np.int64))
    with h5py.File(contract["path"], "r") as handle:
        target = np.asarray(handle["Wh"][indices], dtype=np.float32)
        grids = np.asarray(handle[grid_field][indices], dtype=np.float32)
    return grids, target


def _predict(
    model: Any,
    grids: np.ndarray,
    contract: Mapping[str, Any],
    *,
    lane_index: int,
    batch_size: int,
    k_points: Any,
    torch_module: Any,
) -> np.ndarray:
    """Predict one already-loaded validation grid batch."""

    parts = []
    with torch_module.inference_mode():
        for start in range(0, len(grids), batch_size):
            stop = min(start + batch_size, len(grids))
            inputs = torch_module.as_tensor(grids[start:stop], dtype=torch_module.float32)
            lane_id = torch_module.full(
                (stop - start,), lane_index, dtype=torch_module.long
            )
            normalized = model(inputs, k_points=k_points, lane_id=lane_id).numpy()
            parts.append(
                normalized * contract["target_std"] + contract["target_mean"]
            )
    return np.concatenate(parts, axis=0)


def run_gap_margin_audit(
    config_path: str | Path,
    *,
    baseline_checkpoint: str | Path,
    gap_aware_checkpoint: str | Path,
    run_dir: str | Path | None = None,
) -> Path:
    """Run EXP-FWD-015 without loading held-out test targets."""

    config_file = Path(config_path)
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    experiment = dict(raw.get("experiment") or {})
    if experiment.get("experiment_id") != "EXP-FWD-015":
        raise ValueError("gap margin audit requires experiment_id=EXP-FWD-015")
    if bool(experiment.get("held_out_test_authorized", True)):
        raise ValueError("gap margin audit must forbid held-out test access")
    audit = dict(raw.get("audit") or {})
    checkpoint_paths = {
        "baseline": Path(baseline_checkpoint),
        "gap_aware": Path(gap_aware_checkpoint),
    }
    expected_hashes = dict(audit.get("checkpoint_sha256") or {})
    if set(expected_hashes) != set(checkpoint_paths):
        raise ValueError("audit.checkpoint_sha256 must contain baseline and gap_aware")
    resolved = {
        **raw,
        "audit": {
            **audit,
            "checkpoint_paths": {
                name: str(path.resolve()) for name, path in checkpoint_paths.items()
            },
        },
    }
    out = make_run_dir("EXP-FWD-015", runs_dir=raw["output"]["runs_dir"], run_dir=run_dir)
    logger = setup_logging(log_file=out / RUN_LOG_NAME)
    ensure_figures_dir(out)
    write_resolved_config(out, yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))
    wall_start = time.perf_counter()

    import torch

    contracts = read_dual_lane_contracts(raw)
    models = {
        name: _load_checkpoint(path, str(expected_hashes[name]), torch_module=torch)
        for name, path in checkpoint_paths.items()
    }
    k_points = torch.as_tensor(contracts["binary"]["k_points"], dtype=torch.float32)
    lower_min = int(audit["lower_band_min_one_based"])
    lower_max = int(audit["lower_band_max_one_based"])
    quantile = float(audit["train_margin_quantile"])
    batch_size = int(audit.get("batch_size", 32))
    lane_metrics: dict[str, Any] = {}
    history_rows: list[dict[str, Any]] = []
    for lane_index, lane in enumerate(LANES):
        contract = contracts[lane]
        train_target = _read_targets(
            Path(contract["path"]), np.asarray(contract["selected_train"], dtype=np.int64)
        )
        tolerances = derive_gap_decision_tolerances(
            train_target,
            quantile=quantile,
            lower_band_min_one_based=lower_min,
            lower_band_max_one_based=lower_max,
        )
        lane_metrics[lane] = {"train_only_tolerances": tolerances, "models": {}}
        grids, target = _read_validation(
            contract, grid_field=str(raw["datasets"][lane]["grid_field"])
        )
        for model_name, model in models.items():
            prediction = _predict(
                model,
                grids,
                contract,
                lane_index=lane_index,
                batch_size=batch_size,
                k_points=k_points,
                torch_module=torch,
            )
            metrics = compute_gap_margin_audit_metrics(
                prediction,
                target,
                gap_margin_tolerance=tolerances["gap_margin_tolerance"],
                upper_k_margin_tolerance=tolerances["upper_k_margin_tolerance"],
                lower_band_min_one_based=lower_min,
                lower_band_max_one_based=lower_max,
            )
            lane_metrics[lane]["models"][model_name] = metrics
            history_rows.append(
                {
                    "lane": lane,
                    "model": model_name,
                    "sample_count": metrics["sample_count"],
                    "exact_joint_accuracy": metrics["exact_joint_accuracy"],
                    "tolerant_joint_accuracy": metrics["tolerant_joint_accuracy"],
                    "non_tie_exact_joint_accuracy": metrics[
                        "non_tie_exact_joint_accuracy"
                    ],
                    "complete_gap_recall": metrics["complete_gap_recall"],
                }
            )
    tolerance_gains = {
        lane: {
            model: float(values["tolerant_joint_accuracy"])
            - float(values["exact_joint_accuracy"])
            for model, values in lane_metrics[lane]["models"].items()
        }
        for lane in LANES
    }
    maximum_gain = max(value for lane in tolerance_gains.values() for value in lane.values())
    tie_limited = maximum_gain >= float(audit["tie_limited_minimum_joint_gain"])
    status = "metric_tie_limited" if tie_limited else "model_decision_limited"
    with (out / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0]))
        writer.writeheader()
        writer.writerows(history_rows)
    write_metrics(
        out,
        {
            "status": status,
            "lane_metrics": lane_metrics,
            "tolerance_joint_accuracy_gains": tolerance_gains,
            "maximum_tolerance_joint_accuracy_gain": maximum_gain,
        },
    )
    write_results(
        out,
        {
            "status": status,
            "test_executed": False,
            "training_executed": False,
            "mpb_meep_used": False,
            "evidence_boundary": "validation_only_surrogate_decision_audit",
            "decision": (
                "revise_exact_metric_contract"
                if tie_limited
                else "revise_candidate_ranking_and_complete_gap_supervision"
            ),
        },
    )
    write_run_meta(
        out,
        {
            "script": "python -m phc2d.forward.gap_margin_audit",
            "config_path": str(config_file),
            "checkpoint_sha256": expected_hashes,
            "dataset_sha256": {
                lane: contracts[lane]["dataset_sha256"] for lane in LANES
            },
            "train_subset_hash": {
                lane: contracts[lane]["train_subset_hash"] for lane in LANES
            },
            "idx_val_hash": {lane: contracts[lane]["idx_val_hash"] for lane in LANES},
            "test_targets_loaded": False,
            "wall_time_s": time.perf_counter() - wall_start,
        },
    )
    missing = validate_run_artifacts(out)
    if missing:
        raise RuntimeError(f"incomplete gap-margin audit artifacts: {missing}")
    logger.info("terminal_status=%s maximum_tolerance_gain=%.6f", status, maximum_gain)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--gap-aware-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    print(
        run_gap_margin_audit(
            args.config,
            baseline_checkpoint=args.baseline_checkpoint,
            gap_aware_checkpoint=args.gap_aware_checkpoint,
            run_dir=args.run_dir,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["run_gap_margin_audit"]
