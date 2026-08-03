"""Build publication-ready learning curves from frozen phc2d run histories.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FORMATS = ("png", "svg", "pdf")
COLORS = ("#0F4D92", "#3775BA", "#B64342", "#42949E", "#9A4D8E", "#8BCF8B")
METHOD_LABELS = {
    "direct_inverse": "Direct inverse",
    "single_mode_generator": "Single-mode generator",
    "mixture_generator": "Mixture generator",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_rows(rows: Sequence[Mapping[str, str]], keys: Sequence[str]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in keys
    }


def _mean_series(
    histories: Sequence[Sequence[Mapping[str, str]]],
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    by_epoch: dict[int, list[float]] = defaultdict(list)
    for history in histories:
        for row in history:
            value = row.get(metric, "")
            if value in {"", None}:
                continue
            numeric = float(value)
            if np.isfinite(numeric):
                by_epoch[int(float(row["epoch"]))].append(numeric)
    epochs = np.asarray(sorted(by_epoch), dtype=int)
    means = np.asarray([np.mean(by_epoch[int(epoch)]) for epoch in epochs], dtype=float)
    lows = np.asarray([np.min(by_epoch[int(epoch)]) for epoch in epochs], dtype=float)
    highs = np.asarray([np.max(by_epoch[int(epoch)]) for epoch in epochs], dtype=float)
    return epochs, means, lows, highs


def _style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.linewidth": 1.25,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _finish_axis(axis: Any, *, ylabel: str, log_like: bool = True) -> None:
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.grid(color="#CFCECE", linewidth=0.55, alpha=0.35)
    if log_like:
        axis.set_yscale("symlog", linthresh=1.0e-6)


def _save_figure(
    figure: Any,
    output_base: Path,
    formats: Sequence[str],
    *,
    dpi: int = 300,
) -> list[Path]:
    import matplotlib.pyplot as plt

    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in formats:
        path = output_base.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
        paths.append(path)
    plt.close(figure)
    return paths


def _plot_a1_forward(
    run_dirs: Sequence[Path],
    output_base: Path,
    formats: Sequence[str],
    coverage: list[dict[str, Any]],
) -> list[Path]:
    import matplotlib.pyplot as plt

    grouped: dict[str, dict[float, list[list[dict[str, str]]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run_dir in run_dirs:
        validation_path = run_dir / "validation_history.csv"
        rows = _read_csv(validation_path)
        for row in rows:
            case_id = row["case_id"]
            history_path = run_dir / "cases" / case_id / "train" / "history.csv"
            history = _read_csv(history_path)
            grouped[row["model"]][float(row["learning_rate"])].append(history)
            coverage.append(
                {
                    "family": "A1 forward",
                    "condition": case_id,
                    "history_path": history_path.as_posix(),
                    "epochs_saved": len(history),
                    "status": "complete" if len(history) == 200 else "partial",
                    "metrics": "train_loss;val_mse;val_mae;val_rmse",
                }
            )
    models = sorted(grouped)
    figure, axes = plt.subplots(len(models), 2, figsize=(12.5, 3.4 * len(models)), squeeze=False)
    for model_position, model in enumerate(models):
        learning_rates = sorted(grouped[model], reverse=True)
        for lr_position, learning_rate in enumerate(learning_rates):
            color = COLORS[lr_position % len(COLORS)]
            histories = grouped[model][learning_rate]
            for metric_position, metric in enumerate(("train_loss", "val_rmse")):
                axis = axes[model_position, metric_position]
                epochs, mean, low, high = _mean_series(histories, metric)
                label = f"LR={learning_rate:g}"
                axis.plot(epochs, mean, color=color, linewidth=1.8, label=label)
                axis.fill_between(epochs, low, high, color=color, alpha=0.12, linewidth=0)
                axis.set_title(f"{model} | {'training loss' if metric_position == 0 else 'validation RMSE'}")
                _finish_axis(axis, ylabel=metric.replace("_", " ").title())
        axes[model_position, 1].legend(fontsize=8, ncol=min(3, len(learning_rates)))
    figure.suptitle(
        "A1 forward learning curves | mean and seed range",
        fontsize=14,
        fontweight="bold",
        y=1.005,
    )
    figure.tight_layout(pad=1.2)
    return _save_figure(figure, output_base, formats)


def _plot_b3_field(
    run_dir: Path,
    output_base: Path,
    formats: Sequence[str],
    coverage: list[dict[str, Any]],
) -> list[Path]:
    import matplotlib.pyplot as plt

    models = ("unet_field_v2", "fno_field_v2")
    groups = _field_history_groups(run_dir, models)
    figure, axes = plt.subplots(1, len(models), figsize=(12.5, 4.2), squeeze=False)
    for model_position, model in enumerate(models):
        history_entries = groups.get(model, [])
        if not history_entries:
            raise FileNotFoundError(f"No B3 histories found for {model} under {run_dir}")
        histories = [history for _, history in history_entries]
        axis = axes[0, model_position]
        for color_position, metric in enumerate(("train_data_loss", "val_data_loss")):
            epochs, mean, low, high = _mean_series(histories, metric)
            axis.plot(
                epochs,
                mean,
                color=COLORS[color_position],
                linewidth=1.9,
                label=metric.replace("_", " "),
            )
            axis.fill_between(
                epochs,
                low,
                high,
                color=COLORS[color_position],
                alpha=0.12,
                linewidth=0,
            )
        axis.set_title(f"{model} | n={len(histories)} seeds")
        _finish_axis(axis, ylabel="Data loss")
        axis.legend(fontsize=8)
        for history_path, history in history_entries:
            coverage.append(
                {
                    "family": "B3 field baseline",
                    "condition": history_path.parent.name,
                    "history_path": history_path.as_posix(),
                    "epochs_saved": len(history),
                    "status": "complete" if len(history) == 200 else "partial",
                    "metrics": "train_data_loss;val_data_loss",
                }
            )
    figure.suptitle(
        "B3 supervised field learning curves | mean and seed range",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(pad=1.2)
    return _save_figure(figure, output_base, formats)


def _plot_a2_original(
    run_dir: Path,
    output_base: Path,
    formats: Sequence[str],
    coverage: list[dict[str, Any]],
) -> list[Path]:
    import matplotlib.pyplot as plt

    methods = tuple(METHOD_LABELS)
    figure, axes = plt.subplots(1, len(methods), figsize=(15.0, 4.2), squeeze=False)
    for method_position, method in enumerate(methods):
        history_path = run_dir / "train" / method / "history.csv"
        rows = _read_csv(history_path)
        by_lr: dict[float, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_lr[float(row["learning_rate"])].append(row)
        for lr_position, learning_rate in enumerate(sorted(by_lr, reverse=True)):
            values = _float_rows(by_lr[learning_rate], ("epoch", "train_loss"))
            axes[0, method_position].plot(
                values["epoch"],
                values["train_loss"],
                color=COLORS[lr_position],
                linewidth=1.8,
                label=f"LR={learning_rate:g}",
            )
        axes[0, method_position].set_title(METHOD_LABELS[method])
        _finish_axis(axes[0, method_position], ylabel="Training loss")
        axes[0, method_position].legend(fontsize=8)
        coverage.append(
            {
                "family": "A2 original",
                "condition": method,
                "history_path": history_path.as_posix(),
                "epochs_saved": len(rows),
                "status": "complete",
                "metrics": "train_loss",
            }
        )
    figure.suptitle("Original A2 inverse learning curves", fontsize=14, fontweight="bold")
    figure.tight_layout(pad=1.2)
    return _save_figure(figure, output_base, formats)


def _plot_a2_bridge(
    run_dir: Path,
    output_base: Path,
    formats: Sequence[str],
    coverage: list[dict[str, Any]],
) -> list[Path]:
    import matplotlib.pyplot as plt

    methods = ("single_mode_generator", "mixture_generator")
    primary = ("objective_total_loss", "surrogate_objective_loss", "reconstruction_loss")
    regularizers = (
        "diversity_loss",
        "binary_diversity_loss",
        "latent_response_loss",
        "quality_weighted_collision_loss",
        "cross_surrogate_disagreement_bands_2_10",
        "gap_descriptor_error",
    )
    figure, axes = plt.subplots(len(methods), 2, figsize=(13.5, 7.4), squeeze=False)
    for method_position, method in enumerate(methods):
        history_path = run_dir / "train" / method / "history.csv"
        rows = _read_csv(history_path)
        for metric_position, metrics in enumerate((primary, regularizers)):
            axis = axes[method_position, metric_position]
            values = _float_rows(rows, ("epoch", *metrics))
            for color_position, metric in enumerate(metrics):
                axis.plot(
                    values["epoch"],
                    values[metric],
                    color=COLORS[color_position],
                    linewidth=1.6,
                    label=metric.replace("_", " "),
                )
            axis.set_title(
                f"{METHOD_LABELS[method]} | {'primary objectives' if metric_position == 0 else 'regularizers'}"
            )
            _finish_axis(axis, ylabel="Loss / diagnostic")
            axis.legend(fontsize=7.2, ncol=2)
        coverage.append(
            {
                "family": "A2 objective bridge",
                "condition": method,
                "history_path": history_path.as_posix(),
                "epochs_saved": len(rows),
                "status": "complete" if len(rows) == 50 else "partial",
                "metrics": ";".join(primary + regularizers),
            }
        )
    figure.suptitle("A2 objective-bridge learning curves", fontsize=14, fontweight="bold")
    figure.tight_layout(pad=1.2)
    return _save_figure(figure, output_base, formats)


def _field_history_groups(
    run_dir: Path,
    prefixes: Sequence[str],
) -> dict[str, list[tuple[Path, list[dict[str, str]]]]]:
    groups: dict[str, list[tuple[Path, list[dict[str, str]]]]] = defaultdict(list)
    for case_dir in sorted((run_dir / "cases").iterdir()):
        if not case_dir.is_dir():
            continue
        prefix = next((value for value in prefixes if case_dir.name.startswith(value)), None)
        if prefix is None:
            continue
        history_path = case_dir / "history.csv"
        if history_path.exists():
            groups[prefix].append((history_path, _read_csv(history_path)))
    return groups


def _plot_field_regimes(
    run_dir: Path,
    prefixes: Sequence[str],
    title: str,
    family: str,
    output_base: Path,
    formats: Sequence[str],
    coverage: list[dict[str, Any]],
) -> list[Path]:
    import matplotlib.pyplot as plt

    groups = _field_history_groups(run_dir, prefixes)
    data_metrics = ("train_data_loss", "train_dense_frequency_loss", "val_data_loss")
    physics_metrics = (
        "train_physics_loss",
        "train_normalization_loss",
        "train_rayleigh_consistency_loss",
    )
    figure, axes = plt.subplots(len(prefixes), 2, figsize=(13.5, 3.8 * len(prefixes)), squeeze=False)
    for regime_position, regime in enumerate(prefixes):
        history_entries = groups.get(regime, [])
        if not history_entries:
            axes[regime_position, 0].text(0.5, 0.5, "No history", ha="center", va="center")
            axes[regime_position, 1].text(0.5, 0.5, "No history", ha="center", va="center")
            continue
        histories = [history for _, history in history_entries]
        epoch_lengths = [len(history) for history in histories]
        availability = (
            f"n={len(histories)}, epochs={min(epoch_lengths)}"
            if len(set(epoch_lengths)) == 1
            else f"n={len(histories)}, epochs={min(epoch_lengths)}–{max(epoch_lengths)}"
        )
        for metric_position, metrics in enumerate((data_metrics, physics_metrics)):
            axis = axes[regime_position, metric_position]
            for color_position, metric in enumerate(metrics):
                epochs, mean, low, high = _mean_series(histories, metric)
                axis.plot(
                    epochs,
                    mean,
                    color=COLORS[color_position],
                    linewidth=1.8,
                    label=metric.replace("train_", "").replace("_", " "),
                )
                axis.fill_between(epochs, low, high, color=COLORS[color_position], alpha=0.12)
            axis.set_title(
                f"{regime} | {'data/frequency' if metric_position == 0 else 'physics terms'} | {availability}"
            )
            _finish_axis(axis, ylabel="Loss")
            axis.legend(fontsize=7.5)
        for history_path, history in history_entries:
            coverage.append(
                {
                    "family": family,
                    "condition": history_path.parent.name,
                    "history_path": history_path.as_posix(),
                    "epochs_saved": len(history),
                    "status": "complete" if len(history) == 200 else "partial",
                    "metrics": ";".join(data_metrics + physics_metrics),
                }
            )
    figure.suptitle(title, fontsize=14, fontweight="bold")
    figure.tight_layout(pad=1.2)
    return _save_figure(figure, output_base, formats)


def build_learning_curve_package(
    *,
    a1_forward_run: Path,
    a1_mlp_run: Path | None = None,
    a1_query_run: Path,
    a2_original_run: Path,
    a2_bridge_run: Path,
    p45_run: Path,
    p6_run: Path,
    field_b3_run: Path | None = None,
    output_dir: Path,
    formats: Sequence[str] = FORMATS,
) -> dict[str, Any]:
    """Generate the A1/A2/B3/P4/P5/P6 learning-curve package."""

    _style()
    sources = {
        "a1_forward": Path(a1_forward_run),
        "a1_query": Path(a1_query_run),
        "a2_original": Path(a2_original_run),
        "a2_bridge": Path(a2_bridge_run),
        "p45": Path(p45_run),
        "p6": Path(p6_run),
    }
    if a1_mlp_run is not None:
        sources["a1_mlp"] = Path(a1_mlp_run)
    if field_b3_run is not None:
        sources["field_b3"] = Path(field_b3_run)
    missing = [f"{name}: {path}" for name, path in sources.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing learning-curve sources: {missing}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    coverage: list[dict[str, Any]] = []
    generated: list[Path] = []
    generated.extend(
        _plot_a1_forward(
            [
                sources["a1_forward"],
                *([sources["a1_mlp"]] if "a1_mlp" in sources else []),
            ],
            figure_dir / "a1_forward_learning_curves",
            formats,
            coverage,
        )
    )
    coverage.append(
        {
            "family": "A1 query",
            "condition": "all query conditions",
            "history_path": (sources["a1_query"] / "history.csv").as_posix(),
            "epochs_saved": 0,
            "status": "missing_epoch_history",
            "metrics": "final validation/test metrics only",
        }
    )
    generated.extend(
        _plot_b3_field(
            sources["field_b3"],
            figure_dir / "b3_field_learning_curves",
            formats,
            coverage,
        )
        if "field_b3" in sources
        else []
    )
    generated.extend(
        _plot_a2_original(
            sources["a2_original"],
            figure_dir / "a2_original_learning_curves",
            formats,
            coverage,
        )
    )
    generated.extend(
        _plot_a2_bridge(
            sources["a2_bridge"],
            figure_dir / "a2_objective_bridge_learning_curves",
            formats,
            coverage,
        )
    )
    generated.extend(
        _plot_field_regimes(
            sources["p45"],
            ("P4_dense_k_multires", "P5_rayleigh_consistent"),
            "P4/P5 learning curves | mean and seed range",
            "P4/P5",
            figure_dir / "p4_p5_learning_curves",
            formats,
            coverage,
        )
    )
    generated.extend(
        _plot_field_regimes(
            sources["p6"],
            ("P6A_target_pde", "P6B_target_pde_rayleigh"),
            "P6 learning curves | available-seed mean and range",
            "P6",
            figure_dir / "p6_learning_curves",
            formats,
            coverage,
        )
    )
    coverage_path = output_dir / "curve_coverage.csv"
    _write_csv(coverage_path, coverage)
    history_path = output_dir / "history.csv"
    _write_csv(history_path, coverage)
    notes_path = output_dir / "README.md"
    _write_text(
        notes_path,
        """<!--
Purpose: Index and evidence boundary for A1/A2/B3/P4/P5/P6 learning curves.
Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
-->

# A1/A2/B3/P4/P5/P6 learning curves

- Curves are reconstructed from frozen per-epoch `history.csv` artifacts.
- Seed envelopes show the observed minimum and maximum, not confidence intervals.
- The A1 forward figure includes the separate A1.1 `mlp` run when supplied.
- A1 query did not preserve per-epoch loss and is therefore excluded from curve panels.
- The optional B3 panel compares supervised U-Net and FNO field histories only; PINO regimes remain in P4/P5/P6.
- Partial P6 histories are plotted only over epochs actually present and remain labelled partial in `curve_coverage.csv`.
- These figures are retrospective diagnostics and do not change checkpoint selection or formal results.
""",
    )
    created_at = datetime.now().isoformat(timespec="seconds")
    resolved_config_path = output_dir / "config.resolved.yaml"
    _write_text(
        resolved_config_path,
        "experiment:\n"
        "  experiment_id: training_learning_curves_a1_a2_b3_p4_p5_p6\n"
        "  stage: plot\n"
        "formats:\n"
        + "".join(f"  - {value}\n" for value in formats)
        + "sources:\n"
        + "".join(f"  {name}: {json.dumps(path.as_posix())}\n" for name, path in sources.items()),
    )
    run_meta_path = output_dir / "run.meta.json"
    _write_json(
        run_meta_path,
        {
            "status": "learning_curve_package_completed",
            "created_at": created_at,
            "training_performed": False,
            "source_histories_mutated": False,
        },
    )
    results_path = output_dir / "results.json"
    results = {
        "status": "learning_curve_package_completed",
        "a1_query_epoch_history": "missing",
        "p6_partial_histories": [
            row["condition"]
            for row in coverage
            if row["family"] == "P6" and row["status"] == "partial"
        ],
    }
    _write_json(results_path, results)
    metrics_path = output_dir / "metrics.json"
    metrics = {
        "figure_count": len(generated) // max(len(formats), 1),
        "figure_file_count": len(generated),
        "coverage_row_count": len(coverage),
        "partial_history_count": sum(row["status"] == "partial" for row in coverage),
        "missing_epoch_history_count": sum(
            row["status"] == "missing_epoch_history" for row in coverage
        ),
    }
    _write_json(metrics_path, metrics)
    run_log_path = output_dir / "run.log"
    _write_text(
        run_log_path,
        f"{created_at} learning_curve_package_completed figure_file_count={len(generated)}\n",
    )
    manifest_path = output_dir / "artifact_manifest.json"
    artifact_paths = generated + [
        coverage_path,
        history_path,
        notes_path,
        resolved_config_path,
        run_meta_path,
        results_path,
        metrics_path,
        run_log_path,
    ]
    manifest = {
        "version": 1,
        "created_at": created_at,
        "status": "learning_curve_package_completed",
        "sources": {
            name: {"path": path.as_posix(), "history_sha256": _source_history_hash(path)}
            for name, path in sources.items()
        },
        "coverage": {
            "row_count": metrics["coverage_row_count"],
            "missing_epoch_history_count": metrics["missing_epoch_history_count"],
            "partial_history_count": metrics["partial_history_count"],
        },
        "artifacts": [
            {
                "path": path.relative_to(output_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(artifact_paths)
        ],
    }
    _write_json(manifest_path, manifest)
    return {
        "status": manifest["status"],
        "output_dir": output_dir.as_posix(),
        "figure_files": len(generated),
        "coverage_rows": len(coverage),
        "partial_histories": manifest["coverage"]["partial_history_count"],
        "missing_epoch_histories": manifest["coverage"]["missing_epoch_history_count"],
        "artifact_manifest": manifest_path.as_posix(),
    }


def _source_history_hash(run_dir: Path) -> str:
    history_paths = sorted(path for path in run_dir.rglob("*.csv") if "history" in path.name)
    digest = hashlib.sha256()
    for path in history_paths:
        digest.update(path.relative_to(run_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()
