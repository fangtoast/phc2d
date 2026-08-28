"""Summarize paired validation-only gap-decision screening runs.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from phc2d.forward.gap_aware_loss import BASE_LOSS_NAME, GAP_AWARE_LOSS_NAME
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


LANES = ("binary", "graded")
LOWER_IS_BETTER = (
    "selected_frequency_mre_pct",
    "gap_width_mae",
    "gap_center_mae",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be a mapping: {path}")
    return payload


def _read_run(path: Path, expected_loss: str) -> dict[str, Any]:
    metrics = _read_json(path / "metrics.json")
    results = _read_json(path / "results.json")
    meta = _read_json(path / "run.meta.json")
    if bool(results.get("test_executed", True)):
        raise ValueError(f"screen run accessed held-out test: {path}")
    if str(results.get("loss_contract", {}).get("name")) != expected_loss:
        raise ValueError(f"unexpected loss identity in {path}")
    if not isinstance(metrics.get("gap_validation"), Mapping):
        raise ValueError(f"gap validation metrics are missing from {path}")
    return {"path": str(path), "metrics": metrics, "results": results, "meta": meta}


def _relative_reduction(baseline: float, candidate: float) -> float:
    if baseline <= 0.0:
        return 0.0 if candidate == baseline else float("-inf")
    return (baseline - candidate) / baseline


def summarize_gap_screen(
    baseline_run: str | Path,
    gap_aware_run: str | Path,
    output_dir: str | Path,
    *,
    minimum_relative_error_reduction: float = 0.05,
    minimum_joint_accuracy_gain: float = 0.02,
    maximum_spectrum_mre_relative_increase: float = 0.05,
) -> Path:
    """Apply the preregistered EXP-FWD-014 validation gates."""

    baseline = _read_run(Path(baseline_run), BASE_LOSS_NAME)
    candidate = _read_run(Path(gap_aware_run), GAP_AWARE_LOSS_NAME)
    for key in ("dataset_sha256", "idx_train_hash", "idx_val_hash", "model", "seed"):
        if baseline["meta"].get(key) != candidate["meta"].get(key):
            raise ValueError(f"paired run identity mismatch: {key}")
    thresholds = {
        "minimum_relative_error_reduction": minimum_relative_error_reduction,
        "minimum_joint_accuracy_gain": minimum_joint_accuracy_gain,
        "maximum_spectrum_mre_relative_increase": maximum_spectrum_mre_relative_increase,
    }
    lane_rows: dict[str, Any] = {}
    failures: list[str] = []
    for lane in LANES:
        base_gap = baseline["metrics"]["gap_validation"][lane]
        candidate_gap = candidate["metrics"]["gap_validation"][lane]
        reductions = {
            metric: _relative_reduction(float(base_gap[metric]), float(candidate_gap[metric]))
            for metric in LOWER_IS_BETTER
        }
        joint_gain = float(candidate_gap["joint_band_k_accuracy"]) - float(
            base_gap["joint_band_k_accuracy"]
        )
        base_spectrum = float(baseline["metrics"][f"{lane}_validation_bands_2_10_mre_pct"])
        candidate_spectrum = float(
            candidate["metrics"][f"{lane}_validation_bands_2_10_mre_pct"]
        )
        spectrum_increase = (candidate_spectrum - base_spectrum) / max(base_spectrum, 1.0e-12)
        improvement_flags = {
            **{
                metric: value >= minimum_relative_error_reduction
                for metric, value in reductions.items()
            },
            "joint_band_k_accuracy": joint_gain >= minimum_joint_accuracy_gain,
        }
        lane_pass = (
            spectrum_increase <= maximum_spectrum_mre_relative_increase
            and improvement_flags["joint_band_k_accuracy"]
            and sum(improvement_flags.values()) >= 3
        )
        if not lane_pass:
            failures.append(lane)
        lane_rows[lane] = {
            "baseline": {"bands_2_10_mre_pct": base_spectrum, **base_gap},
            "gap_aware": {"bands_2_10_mre_pct": candidate_spectrum, **candidate_gap},
            "relative_error_reductions": reductions,
            "joint_band_k_accuracy_gain": joint_gain,
            "spectrum_mre_relative_increase": spectrum_increase,
            "improvement_flags": improvement_flags,
            "passed": lane_pass,
        }
    status = "gap_screen_pass" if not failures else "gap_screen_fail"
    out = make_run_dir("EXP-FWD-014-comparison", run_dir=Path(output_dir))
    logger = setup_logging(log_file=out / RUN_LOG_NAME)
    ensure_figures_dir(out)
    resolved = {
        "experiment_id": "EXP-FWD-014",
        "baseline_run": baseline["path"],
        "gap_aware_run": candidate["path"],
        "selection_split": "validation",
        "test_executed": False,
        "thresholds": thresholds,
    }
    write_resolved_config(out, yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))
    (out / "history.csv").write_text("stage,status\ncomparison,complete\n", encoding="utf-8")
    write_metrics(out, {"status": status, "lane_comparison": lane_rows, **thresholds})
    write_results(
        out,
        {
            "status": status,
            "failed_lanes": failures,
            "test_executed": False,
            "formal_multi_seed_training_started": False,
            "mpb_meep_used": False,
            "evidence_boundary": "preliminary_single_seed_validation_screen_no_test",
            "next_gate": (
                "human_authorization_for_three_seed_formal_training"
                if not failures
                else "stop_or_reformulate_gap_loss"
            ),
        },
    )
    write_run_meta(
        out,
        {
            "script": "python -m phc2d.forward.gap_screen",
            "baseline_run": baseline["path"],
            "gap_aware_run": candidate["path"],
            "dataset_sha256": baseline["meta"]["dataset_sha256"],
            "idx_train_hash": baseline["meta"]["idx_train_hash"],
            "idx_val_hash": baseline["meta"]["idx_val_hash"],
            "test_targets_loaded": False,
        },
    )
    missing = validate_run_artifacts(out)
    if missing:
        raise RuntimeError(f"incomplete gap-screen artifacts: {missing}")
    logger.info("terminal_status=%s failed_lanes=%s", status, failures)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--gap-aware-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(summarize_gap_screen(args.baseline_run, args.gap_aware_run, args.output_dir))


if __name__ == "__main__":
    main()


__all__ = ["summarize_gap_screen"]
