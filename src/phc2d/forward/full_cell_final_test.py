"""Evaluate the three final validation-frozen full-cell configurations on test.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from phc2d.forward.full_cell import _FullCellH5Dataset
from phc2d.models import build_model
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
from phc2d.training import DeviceConfig, resolve_device


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def resolve_final_run_dirs(
    final_rows: Sequence[Mapping[str, str]],
    main_rows: Sequence[Mapping[str, str]],
    ablation_rows: Sequence[Mapping[str, str]],
) -> dict[tuple[str, int], str]:
    """Resolve exactly three seed checkpoints for each frozen final arm."""

    resolved: dict[tuple[str, int], str] = {}
    for final in final_rows:
        representation = str(final["representation"])
        model = str(final["model"])
        resolution = int(final["resolution"])
        source_rows = main_rows if resolution == 64 else ablation_rows
        for seed in (0, 1, 2):
            matches = [
                row
                for row in source_rows
                if str(row.get("model")) == model
                and int(row.get("seed", -1)) == seed
                and (
                    (resolution == 64 and str(row.get("wave")) == representation)
                    or (
                        resolution != 64
                        and str(row.get("representation")) == representation
                        and int(row.get("resolution", -1)) == resolution
                    )
                )
            ]
            if len(matches) != 1:
                raise RuntimeError(f"cannot uniquely resolve {representation}/{model}/r{resolution}/seed{seed}")
            resolved[(representation, seed)] = str(matches[0]["run_dir"])
    if len(resolved) != 9:
        raise RuntimeError(f"final test requires 9 checkpoints, got {len(resolved)}")
    return resolved


def _predict(model: Any, loader: Any, *, device: Any, mean: Any, std: Any, torch: Any) -> tuple[np.ndarray, np.ndarray, float]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            prediction = model(xb) * std + mean
            target = yb * std + mean
            predictions.append(prediction.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
    return np.concatenate(predictions), np.concatenate(targets), time.perf_counter() - started


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    diff = prediction - target
    return {
        "mre_pct": float(100.0 * np.mean(np.abs(diff[..., 1:]) / np.maximum(np.abs(target[..., 1:]), 1.0e-8))),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }


def _bootstrap_mean_ci(values: np.ndarray, *, indices: np.ndarray) -> tuple[float, float, float]:
    """Return mean and percentile CI using caller-frozen resample indices."""

    sample_values = np.asarray(values, dtype=np.float64)
    if sample_values.ndim != 1 or indices.ndim != 2 or indices.shape[1] != sample_values.shape[0]:
        raise ValueError("bootstrap inputs must be one-dimensional samples and a matching 2D index matrix")
    bootstrap_means = np.mean(sample_values[indices], axis=1)
    return (
        float(np.mean(sample_values)),
        float(np.quantile(bootstrap_means, 0.025)),
        float(np.quantile(bootstrap_means, 0.975)),
    )


def run_full_cell_final_test(
    final_selection: str | Path,
    *,
    main_matrix_manifest: str | Path,
    ablation_manifest: str | Path,
    binary_dataset: str | Path,
    graded_dataset: str | Path,
    output_dir: str | Path,
    repository_root: str | Path = ".",
    device: str = "cuda:0",
) -> Path:
    """Run test only after all architecture/representation/resolution choices are frozen."""

    root = Path(repository_root).resolve()

    def resolve(value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    final_rows = _read_csv(resolve(final_selection))
    main_rows = _read_csv(resolve(main_matrix_manifest))
    ablation_rows = _read_csv(resolve(ablation_manifest))
    if len(final_rows) != 3 or any(row.get("test_driven_selection", "").lower() != "false" for row in final_rows):
        raise RuntimeError("final test requires exactly three validation-frozen configurations")
    run_dirs = resolve_final_run_dirs(final_rows, main_rows, ablation_rows)
    binary_path = resolve(binary_dataset)
    graded_path = resolve(graded_dataset)
    out = make_run_dir("full_cell_forward_final_test_v1", run_dir=output_dir)
    ensure_figures_dir(out)
    logger = setup_logging(
        log_file=out / RUN_LOG_NAME,
        run_meta={
            "script": "python -m phc2d.forward.full_cell_final_test",
            "validation_boundary": "post_validation_freeze_final_test",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_node": os.environ.get("SLURM_JOB_NODELIST"),
        },
    )
    write_resolved_config(
        out,
        yaml.safe_dump(
            {
                "final_selection": str(resolve(final_selection).relative_to(root)),
                "main_matrix_manifest": str(resolve(main_matrix_manifest).relative_to(root)),
                "ablation_manifest": str(resolve(ablation_manifest).relative_to(root)),
                "device": device,
                "selection_split": "validation",
                "evaluation_split": "test",
            },
            sort_keys=False,
        ),
    )

    import h5py
    import torch
    from torch.utils.data import DataLoader

    device_info = resolve_device(DeviceConfig(device=device, require_accelerator=device.startswith("cuda"), precision="fp32", num_workers=4, pin_memory=True), torch_module=torch)
    result_rows: list[dict[str, Any]] = []
    per_band_rows: list[dict[str, Any]] = []
    per_k_rows: list[dict[str, Any]] = []
    per_sample_errors: dict[tuple[str, str], list[np.ndarray]] = {}
    sample_ids_by_configuration: dict[tuple[str, str], np.ndarray] = {}
    total_inference = 0.0
    for final in final_rows:
        representation = final["representation"]
        model_name = final["model"]
        resolution = int(final["resolution"])
        threshold = None if not final.get("threshold") else float(final["threshold"])
        dataset_path = binary_path if final["lane"] == "binary" else graded_path
        with h5py.File(dataset_path, "r") as handle:
            indices = np.asarray(handle["idx_test"], dtype=np.int64)
            order = np.argsort(indices)
            sorted_ids = np.asarray(handle["sample_id"][indices[order]]).astype("U")
            sample_ids = np.empty_like(sorted_ids)
            sample_ids[order] = sorted_ids
        prediction_by_seed: dict[int, np.ndarray] = {}
        target_reference: np.ndarray | None = None
        for seed in (0, 1, 2):
            run = resolve(run_dirs[(representation, seed)])
            try:
                checkpoint = torch.load(run / "model.pt", map_location=device_info.torch_device, weights_only=True)
            except TypeError:
                checkpoint = torch.load(run / "model.pt", map_location=device_info.torch_device)
            model = build_model(
                model_name,
                input_shape=tuple(int(v) for v in checkpoint["input_shape"]),
                output_shape=tuple(int(v) for v in checkpoint["output_shape"]),
                **dict(checkpoint["model_params"]),
            ).to(device_info.torch_device)
            model.load_state_dict(checkpoint["state_dict"])
            mean = checkpoint["target_mean"].to(device_info.torch_device)
            std = checkpoint["target_std"].to(device_info.torch_device)
            dataset = _FullCellH5Dataset(
                dataset_path,
                indices,
                grid_field=f"material_weight_grid_{resolution}",
                layout="flat" if model_name in {"mlp", "res_mlp_wide"} else "image",
                sampling="deterministic_threshold" if threshold is not None else "continuous",
                threshold=threshold,
                target_mean=mean.detach().cpu().numpy(),
                target_std=std.detach().cpu().numpy(),
            )
            loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
            prediction, target, elapsed = _predict(model, loader, device=device_info.torch_device, mean=mean, std=std, torch=torch)
            total_inference += elapsed
            if target_reference is None:
                target_reference = target
            elif not np.allclose(target_reference, target, rtol=1.0e-6, atol=1.0e-7):
                raise RuntimeError(f"test targets changed across seeds for {representation}")
            metric = _metrics(target, prediction)
            if not all(math.isfinite(value) for value in metric.values()):
                raise RuntimeError(f"non-finite final test metric for {representation}/seed{seed}")
            result_rows.append(
                {
                    "lane": final["lane"],
                    "representation": representation,
                    "model": model_name,
                    "resolution": resolution,
                    "seed": seed,
                    **metric,
                    "inference_time_s": elapsed,
                }
            )
            relative_error = 100.0 * np.abs(prediction - target) / np.maximum(np.abs(target), 1.0e-8)
            for band_index, value in enumerate(np.mean(relative_error, axis=(0, 1)), start=1):
                per_band_rows.append(
                    {
                        "lane": final["lane"],
                        "representation": representation,
                        "model": model_name,
                        "resolution": resolution,
                        "seed": seed,
                        "band_index": band_index,
                        "mre_pct": float(value),
                    }
                )
            for k_index, value in enumerate(np.mean(relative_error[..., 1:], axis=(0, 2))):
                per_k_rows.append(
                    {
                        "lane": final["lane"],
                        "representation": representation,
                        "model": model_name,
                        "resolution": resolution,
                        "seed": seed,
                        "k_index": k_index,
                        "mre_bands_2_10_pct": float(value),
                    }
                )
            per_sample_errors.setdefault((final["lane"], representation), []).append(
                np.mean(relative_error[..., 1:], axis=(1, 2))
            )
            sample_ids_by_configuration[(final["lane"], representation)] = sample_ids.copy()
            prediction_by_seed[seed] = prediction
        assert target_reference is not None
        np.savez_compressed(
            out / f"predictions_test_{final['lane']}_{representation}.npz",
            indices=indices,
            sample_id=sample_ids,
            target_Wh=target_reference,
            prediction_seed0=prediction_by_seed[0],
            prediction_seed1=prediction_by_seed[1],
            prediction_seed2=prediction_by_seed[2],
        )
        logger.info("completed final test for %s/%s/r%d", final["lane"], representation, resolution)
    _write_csv(out / "history.csv", result_rows)
    _write_csv(out / "test_seed_metrics.csv", result_rows)
    _write_csv(out / "test_per_band.csv", per_band_rows)
    _write_csv(out / "test_per_k.csv", per_k_rows)
    summary_rows: list[dict[str, Any]] = []
    for final in final_rows:
        relevant = [row for row in result_rows if row["lane"] == final["lane"] and row["representation"] == final["representation"]]
        summary_rows.append(
            {
                "lane": final["lane"],
                "representation": final["representation"],
                "model": final["model"],
                "resolution": int(final["resolution"]),
                "mre_mean_pct": statistics.mean(float(row["mre_pct"]) for row in relevant),
                "mre_std_pct": statistics.stdev(float(row["mre_pct"]) for row in relevant),
                "mae_mean": statistics.mean(float(row["mae"]) for row in relevant),
                "rmse_mean": statistics.mean(float(row["rmse"]) for row in relevant),
                "seed_count": 3,
            }
        )
    _write_csv(out / "test_summary.csv", summary_rows)

    bootstrap_rows: list[dict[str, Any]] = []
    bootstrap_count = 2_000
    seed_sequence = np.random.SeedSequence(20260806)
    configuration_keys = sorted(per_sample_errors)
    child_sequences = seed_sequence.spawn(len(configuration_keys) + 1)
    mean_error_by_configuration: dict[tuple[str, str], np.ndarray] = {}
    for child_sequence, key in zip(child_sequences, configuration_keys):
        values = np.mean(np.stack(per_sample_errors[key], axis=0), axis=0)
        mean_error_by_configuration[key] = values
        rng = np.random.default_rng(child_sequence)
        indices = rng.integers(0, values.size, size=(bootstrap_count, values.size))
        estimate, ci_low, ci_high = _bootstrap_mean_ci(values, indices=indices)
        bootstrap_rows.append(
            {
                "comparison": "configuration_mean",
                "lane": key[0],
                "left_representation": key[1],
                "right_representation": "",
                "estimate_mre_pct_points": estimate,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "sample_count": values.size,
                "bootstrap_replicates": bootstrap_count,
                "paired": False,
            }
        )
    graded_keys = [key for key in configuration_keys if key[0] == "graded"]
    if len(graded_keys) == 2:
        left, right = sorted(graded_keys, key=lambda key: key[1] != "continuous")
        if not np.array_equal(sample_ids_by_configuration[left], sample_ids_by_configuration[right]):
            raise RuntimeError("graded final configurations do not share identical test sample order")
        difference = mean_error_by_configuration[left] - mean_error_by_configuration[right]
        rng = np.random.default_rng(child_sequences[-1])
        indices = rng.integers(0, difference.size, size=(bootstrap_count, difference.size))
        estimate, ci_low, ci_high = _bootstrap_mean_ci(difference, indices=indices)
        bootstrap_rows.append(
            {
                "comparison": "paired_difference",
                "lane": "graded",
                "left_representation": left[1],
                "right_representation": right[1],
                "estimate_mre_pct_points": estimate,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "sample_count": difference.size,
                "bootstrap_replicates": bootstrap_count,
                "paired": True,
            }
        )
    _write_csv(out / "paired_bootstrap_ci.csv", bootstrap_rows)

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.8))
    labels = [f"{row['lane']}:{row['representation']}:r{row['resolution']}" for row in summary_rows]
    axis.bar(labels, [float(row["mre_mean_pct"]) for row in summary_rows], yerr=[float(row["mre_std_pct"]) for row in summary_rows], capsize=4)
    axis.set_ylabel("test bands 2-10 MRE (%)")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(out / "figures" / f"final_test_comparison.{suffix}", dpi=180)
    plt.close(figure)
    metrics_payload = {
        "final_configuration_count": 3,
        "seed_evaluations": 9,
        "per_band_rows": len(per_band_rows),
        "per_k_rows": len(per_k_rows),
        "bootstrap_rows": len(bootstrap_rows),
        "inference_time_total_s": total_inference,
    }
    write_metrics(out, metrics_payload)
    write_results(
        out,
        {
            "status": "final_test_evaluation_success",
            "selection_split": "validation",
            "evaluation_split": "test",
            "test_driven_selection": False,
            "prediction_artifacts": [f"predictions_test_{row['lane']}_{row['representation']}.npz" for row in final_rows],
            "summary": summary_rows,
            "next_gate": "local_wsl_checkpoint_replay_and_final_acceptance",
        },
    )
    write_run_meta(out, {**device_info.metadata(), "final_configuration_count": 3, "seed_evaluations": 9})
    missing = validate_run_artifacts(out)
    if missing:
        raise RuntimeError(f"final test artifact contract incomplete: {missing}")
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-selection", type=Path, required=True)
    parser.add_argument("--main-matrix-manifest", type=Path, required=True)
    parser.add_argument("--ablation-manifest", type=Path, required=True)
    parser.add_argument("--binary-dataset", type=Path, required=True)
    parser.add_argument("--graded-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    print(
        run_full_cell_final_test(
            args.final_selection,
            main_matrix_manifest=args.main_matrix_manifest,
            ablation_manifest=args.ablation_manifest,
            binary_dataset=args.binary_dataset,
            graded_dataset=args.graded_dataset,
            output_dir=args.output_dir,
            repository_root=args.repository_root,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["_bootstrap_mean_ci", "resolve_final_run_dirs", "run_full_cell_final_test"]
