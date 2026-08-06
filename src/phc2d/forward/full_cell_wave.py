"""Aggregate one formal 4-model by 3-seed full-cell forward wave.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from phc2d.runs import write_json

MODELS = ("mlp", "res_mlp_wide", "lifted_cnn", "fno2d_band_v2")
SEEDS = (0, 1, 2)
REQUIRED_RUN_FILES = (
    "run.log",
    "run.meta.json",
    "config.resolved.yaml",
    "results.json",
    "metrics.json",
    "history.csv",
    "model.pt",
    "figures/training_curve.png",
    "figures/training_curve.pdf",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected JSON mapping: {path}")
    return dict(payload)


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_training_curves(output_dir: Path, histories: Mapping[str, Mapping[int, list[dict[str, str]]]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=False)
    for axis, model in zip(axes.flat, MODELS):
        seed_histories = histories[model]
        common_epochs = min(len(seed_histories[seed]) for seed in SEEDS)
        common_values: list[list[float]] = []
        for seed in SEEDS:
            rows = seed_histories[seed]
            epochs = [int(row["epoch"]) for row in rows]
            values = [float(row["val_bands_2_10_mre_pct"]) for row in rows]
            axis.plot(epochs, values, alpha=0.45, label=f"seed {seed}")
            common_values.append(values[:common_epochs])
        matrix = np.asarray(common_values, dtype=float)
        epoch_axis = np.arange(1, common_epochs + 1)
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0, ddof=1)
        axis.plot(epoch_axis, mean, color="black", linewidth=1.8, label="mean")
        axis.fill_between(epoch_axis, mean - std, mean + std, color="black", alpha=0.15)
        axis.set_title(model)
        axis.set_xlabel("epoch")
        axis.set_ylabel("validation bands 2-10 MRE (%)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / "figures" / f"training_curves_by_model.{suffix}", dpi=180)
    plt.close(figure)


def _plot_validation(output_dir: Path, summary_rows: list[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    rows = sorted(summary_rows, key=lambda row: float(row["validation_mre_mean_pct"]))
    labels = [str(row["model"]) for row in rows]
    means = [float(row["validation_mre_mean_pct"]) for row in rows]
    stds = [float(row["validation_mre_std_pct"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(labels, means, yerr=stds, capsize=4)
    axis.set_ylabel("best validation bands 2-10 MRE (%)")
    axis.tick_params(axis="x", rotation=18)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / "figures" / f"validation_comparison.{suffix}", dpi=180)
    plt.close(figure)


def aggregate_full_cell_forward_wave(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    repository_root: str | Path = ".",
) -> Path:
    """Validate all 12 terminal runs and write the frozen validation-only summary."""

    root = Path(repository_root).resolve()
    manifest = Path(manifest_path)
    if not manifest.is_absolute():
        manifest = root / manifest
    rows = _read_csv(manifest)
    expected = {(model, seed) for model in MODELS for seed in SEEDS}
    actual = {(str(row["model"]), int(row["seed"])) for row in rows}
    if len(rows) != 12 or actual != expected:
        raise ValueError(f"wave manifest must contain exactly 4 models x 3 seeds; got {sorted(actual)}")

    out = Path(output_dir)
    if not out.is_absolute():
        out = root / out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    histories: dict[str, dict[int, list[dict[str, str]]]] = {model: {} for model in MODELS}
    grouped: dict[str, list[float]] = {model: [] for model in MODELS}
    dataset_hashes: set[str] = set()
    split_hashes: set[tuple[str, str, str]] = set()
    config_paths: set[str] = set()

    for row in rows:
        model = str(row["model"])
        seed = int(row["seed"])
        run_dir = Path(row["run_dir"])
        if not run_dir.is_absolute():
            run_dir = root / run_dir
        missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"run {model}/seed{seed} is missing artifacts: {missing}")
        results = _read_json(run_dir / "results.json")
        metrics = _read_json(run_dir / "metrics.json")
        meta = _read_json(run_dir / "run.meta.json")
        if results.get("status") != "formal_training_success":
            raise RuntimeError(f"run {model}/seed{seed} status is {results.get('status')!r}")
        if bool(results.get("test_executed")):
            raise RuntimeError(f"run {model}/seed{seed} accessed test before validation freeze")
        value = float(metrics["best_validation_bands_2_10_mre_pct"])
        if not math.isfinite(value):
            raise RuntimeError(f"run {model}/seed{seed} has non-finite validation metric")
        checkpoint_hash = str(results.get("checkpoint_sha256", ""))
        if len(checkpoint_hash) != 64:
            raise RuntimeError(f"run {model}/seed{seed} lacks checkpoint SHA-256")
        histories[model][seed] = _read_csv(run_dir / "history.csv")
        grouped[model].append(value)
        dataset_hashes.add(str(meta.get("dataset_sha256")))
        split_hashes.add(
            (
                str(meta.get("idx_train_hash")),
                str(meta.get("idx_val_hash")),
                str(meta.get("idx_test_hash")),
            )
        )
        config_paths.add(str(meta.get("config_path")))
        run_rows.append(
            {
                "model": model,
                "seed": seed,
                "job_id": row["job_id"],
                "status": results["status"],
                "run_dir": str(run_dir.relative_to(root)),
                "checkpoint_path": results["checkpoint_path"],
                "checkpoint_sha256": checkpoint_hash,
                "best_epoch": metrics["best_epoch"],
                "stop_epoch": metrics["stop_epoch"],
                "best_validation_bands_2_10_mre_pct": value,
                "train_time_s": metrics["train_time_s"],
                "wall_time_total_s": metrics["wall_time_total_s"],
                "peak_gpu_memory_mb": metrics["peak_gpu_memory_mb"],
                "test_executed": False,
            }
        )
    if len(dataset_hashes) != 1 or len(split_hashes) != 1 or len(config_paths) != 1:
        raise RuntimeError("wave runs do not share one dataset, split contract, and config")

    summary_rows: list[dict[str, Any]] = []
    for model in MODELS:
        values = grouped[model]
        summary_rows.append(
            {
                "model": model,
                "seed0_validation_mre_pct": values[0],
                "seed1_validation_mre_pct": values[1],
                "seed2_validation_mre_pct": values[2],
                "validation_mre_mean_pct": statistics.mean(values),
                "validation_mre_std_pct": statistics.stdev(values),
                "success_count": 3,
            }
        )
    leaderboard = sorted(summary_rows, key=lambda row: float(row["validation_mre_mean_pct"]))
    leaderboard_rows = [
        {"rank": index, **row, "selection_split": "validation", "test_driven_selection": False}
        for index, row in enumerate(leaderboard, start=1)
    ]
    _write_csv(out / "run_manifest.csv", sorted(run_rows, key=lambda row: (MODELS.index(str(row["model"])), int(row["seed"]))))
    _write_csv(out / "seed_summary.csv", summary_rows)
    _write_csv(out / "leaderboard.csv", leaderboard_rows)
    _plot_training_curves(out, histories)
    _plot_validation(out, summary_rows)
    train_hash, val_hash, test_hash = next(iter(split_hashes))
    write_json(
        out / "wave.meta.json",
        {
            "status": "formal_wave_aggregation_success",
            "source_manifest": str(manifest.relative_to(root)),
            "planned_runs": 12,
            "completed_runs": 12,
            "failed_runs": 0,
            "missing_runs": 0,
            "dataset_sha256": next(iter(dataset_hashes)),
            "idx_train_hash": train_hash,
            "idx_val_hash": val_hash,
            "idx_test_hash": test_hash,
            "config_path": next(iter(config_paths)),
            "selection_split": "validation",
            "test_executed": False,
            "test_driven_selection": False,
            "validation_winner": leaderboard_rows[0]["model"],
            "validation_winner_mean_mre_pct": leaderboard_rows[0]["validation_mre_mean_pct"],
            "next_gate": "next_representation_wave",
        },
    )
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    print(
        aggregate_full_cell_forward_wave(
            args.manifest,
            output_dir=args.output_dir,
            repository_root=args.repository_root,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["aggregate_full_cell_forward_wave"]
