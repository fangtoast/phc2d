"""Audit cloud versus WSL replay outputs for frozen full-cell checkpoints.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def audit_full_cell_replay(
    cluster_dir: str | Path,
    *,
    wsl_dir: str | Path,
    output_dir: str | Path,
    metric_rtol: float = 1.0e-5,
    metric_atol: float = 1.0e-7,
    prediction_rtol: float = 1.0e-5,
    prediction_atol: float = 1.0e-6,
) -> Path:
    """Require the WSL replay to reproduce cloud metrics and predictions."""

    cluster = Path(cluster_dir).resolve()
    wsl = Path(wsl_dir).resolve()
    out = make_run_dir("full_cell_forward_wsl_replay_audit_v1", run_dir=output_dir)
    ensure_figures_dir(out)
    logger = setup_logging(log_file=out / RUN_LOG_NAME, run_meta={"script": "python -m phc2d.forward.full_cell_replay_audit"})
    write_resolved_config(
        out,
        yaml.safe_dump(
            {
                "cluster_dir": str(cluster),
                "wsl_dir": str(wsl),
                "metric_rtol": metric_rtol,
                "metric_atol": metric_atol,
                "prediction_rtol": prediction_rtol,
                "prediction_atol": prediction_atol,
            },
            sort_keys=False,
        ),
    )
    for source in (cluster, wsl):
        payload = json.loads((source / "results.json").read_text(encoding="utf-8"))
        if payload.get("status") != "final_test_evaluation_success":
            raise RuntimeError(f"not a successful final-test directory: {source}")

    cluster_rows = _read_csv(cluster / "test_seed_metrics.csv")
    wsl_rows = _read_csv(wsl / "test_seed_metrics.csv")
    key_fields = ("lane", "representation", "model", "resolution", "seed")

    def keyed(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, ...], Mapping[str, str]]:
        result = {tuple(str(row[field]) for field in key_fields): row for row in rows}
        if len(result) != len(rows):
            raise RuntimeError("duplicate final-test metric key")
        return result

    cluster_by_key = keyed(cluster_rows)
    wsl_by_key = keyed(wsl_rows)
    if cluster_by_key.keys() != wsl_by_key.keys() or len(cluster_by_key) != 9:
        raise RuntimeError("cloud and WSL final-test metric keys differ or are incomplete")

    comparison_rows: list[dict[str, Any]] = []
    max_metric_abs_diff = 0.0
    for key in sorted(cluster_by_key):
        row: dict[str, Any] = dict(zip(key_fields, key))
        for metric in ("mre_pct", "mae", "rmse"):
            cloud_value = float(cluster_by_key[key][metric])
            wsl_value = float(wsl_by_key[key][metric])
            difference = abs(cloud_value - wsl_value)
            if not math.isfinite(difference) or not np.isclose(
                cloud_value,
                wsl_value,
                rtol=metric_rtol,
                atol=metric_atol,
            ):
                raise RuntimeError(f"metric replay mismatch for {key}/{metric}: {cloud_value} vs {wsl_value}")
            row[f"cloud_{metric}"] = cloud_value
            row[f"wsl_{metric}"] = wsl_value
            row[f"abs_diff_{metric}"] = difference
            max_metric_abs_diff = max(max_metric_abs_diff, difference)
        comparison_rows.append(row)

    prediction_files = sorted(path.name for path in cluster.glob("predictions_test_*.npz"))
    if len(prediction_files) != 3 or prediction_files != sorted(path.name for path in wsl.glob("predictions_test_*.npz")):
        raise RuntimeError("cloud and WSL prediction artifact sets differ or are incomplete")
    max_prediction_abs_diff = 0.0
    prediction_array_count = 0
    for name in prediction_files:
        with np.load(cluster / name) as cloud_npz, np.load(wsl / name) as wsl_npz:
            if cloud_npz.files != wsl_npz.files:
                raise RuntimeError(f"prediction array keys differ for {name}")
            for key in cloud_npz.files:
                cloud_array = cloud_npz[key]
                wsl_array = wsl_npz[key]
                if key in {"indices", "sample_id"}:
                    if not np.array_equal(cloud_array, wsl_array):
                        raise RuntimeError(f"prediction identity array mismatch for {name}/{key}")
                    continue
                difference = float(np.max(np.abs(cloud_array - wsl_array))) if cloud_array.size else 0.0
                if not math.isfinite(difference) or not np.allclose(
                    cloud_array,
                    wsl_array,
                    rtol=prediction_rtol,
                    atol=prediction_atol,
                ):
                    raise RuntimeError(f"prediction replay mismatch for {name}/{key}; max_abs_diff={difference}")
                max_prediction_abs_diff = max(max_prediction_abs_diff, difference)
                prediction_array_count += 1

    _write_csv(out / "replay_comparison.csv", comparison_rows)
    _write_csv(out / "history.csv", comparison_rows)

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.8))
    labels = [f"{row['lane']}:{row['representation']}:s{row['seed']}" for row in comparison_rows]
    axis.bar(labels, [float(row["abs_diff_mre_pct"]) for row in comparison_rows])
    axis.set_ylabel("absolute MRE difference (percentage points)")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(out / "figures" / f"wsl_replay_metric_difference.{suffix}", dpi=180)
    plt.close(figure)

    metrics = {
        "metric_row_count": len(comparison_rows),
        "prediction_artifact_count": len(prediction_files),
        "prediction_array_count": prediction_array_count,
        "max_metric_abs_diff": max_metric_abs_diff,
        "max_prediction_abs_diff": max_prediction_abs_diff,
    }
    write_metrics(out, metrics)
    write_results(
        out,
        {
            "status": "wsl_checkpoint_replay_acceptance_success",
            **metrics,
            "next_gate": "forward_campaign_final_acceptance",
        },
    )
    write_run_meta(out, {"replay_environment": "WSL Ubuntu meep_env", **metrics})
    missing = validate_run_artifacts(out)
    if missing:
        raise RuntimeError(f"WSL replay audit artifact contract incomplete: {missing}")
    logger.info("WSL replay accepted: max metric diff=%g, max prediction diff=%g", max_metric_abs_diff, max_prediction_abs_diff)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-dir", type=Path, required=True)
    parser.add_argument("--wsl-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric-rtol", type=float, default=1.0e-5)
    parser.add_argument("--metric-atol", type=float, default=1.0e-7)
    parser.add_argument("--prediction-rtol", type=float, default=1.0e-5)
    parser.add_argument("--prediction-atol", type=float, default=1.0e-6)
    args = parser.parse_args(argv)
    print(
        audit_full_cell_replay(
            args.cluster_dir,
            wsl_dir=args.wsl_dir,
            output_dir=args.output_dir,
            metric_rtol=args.metric_rtol,
            metric_atol=args.metric_atol,
            prediction_rtol=args.prediction_rtol,
            prediction_atol=args.prediction_atol,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["audit_full_cell_replay"]
