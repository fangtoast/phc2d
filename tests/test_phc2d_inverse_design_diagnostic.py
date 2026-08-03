from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml

from phc2d.inverse import run_inverse_design_diagnostic


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def test_inverse_design_diagnostic_writes_candidate_trace_and_multistart(tmp_path):
    dataset_path = tmp_path / "tiny.npz"
    rho = np.asarray(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    wh = np.ones((4, 2, 2), dtype=np.float32)
    np.savez(
        dataset_path,
        rho=rho,
        Wh=wh,
        idx_train=np.asarray([0, 1], dtype=np.int64),
        idx_val=np.asarray([2], dtype=np.int64),
        idx_test=np.asarray([3], dtype=np.int64),
    )

    objective_run = tmp_path / "objective"
    sample_run = tmp_path / "sample"
    replay_run = tmp_path / "replay"
    objective_run.mkdir()
    sample_run.mkdir()
    replay_run.mkdir()
    _write_csv(
        objective_run / "history.csv",
        [
            {"stage": "objective_train", "seed": 0, "epoch": 1, "objective_total_loss": 0.3},
            {"stage": "objective_train", "seed": 1, "epoch": 1, "objective_total_loss": 0.2},
        ],
    )
    (sample_run / "config.resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "dataset": {"path": str(dataset_path), "condition_target": "Wh"},
                "inverse": {"candidate_value_range": [0.0, 1.0], "threshold": 0.5},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_json(
        sample_run / "candidates.json",
        {
            "version": 1,
            "targets": [
                {
                    "target_id": "unit_seed0_Wh_00003_000",
                    "target_index": 3,
                    "condition": {"Wh": [[1.0, 1.0], [1.0, 1.0]]},
                    "candidates": [[0.0, 1.0, 0.0, 1.0], [0.2, 0.8, 0.2, 0.8]],
                    "selected_indices": [0, 1],
                    "filtered_metrics": {
                        "unique_rate": 1.0,
                        "nearest_neighbor_hamming_p10": 0.5,
                        "nearest_neighbor_hamming_mean": 0.5,
                    },
                    "surrogate_metrics": {
                        "best_of_k_surrogate_mre": 0.1,
                        "best_of_k_surrogate_mre_bands_2_10": 0.1,
                        "selected_reconstruction_errors": [0.1, 0.2],
                    },
                    "selected_surrogate_predictions": [
                        [[0.9, 1.0], [1.1, 1.0]],
                        [[1.2, 1.0], [1.0, 0.8]],
                    ],
                },
                {
                    "target_id": "unit_seed1_Wh_00003_000",
                    "target_index": 3,
                    "condition": {"Wh": [[1.0, 1.0], [1.0, 1.0]]},
                    "candidates": [[0.0, 0.0, 1.0, 1.0]],
                    "selected_indices": [0],
                    "filtered_metrics": {"unique_rate": 1.0, "nearest_neighbor_hamming_p10": 0.0},
                    "surrogate_metrics": {"selected_reconstruction_errors": [0.05]},
                },
            ],
        },
    )
    replay_cfg = tmp_path / "replay_config.yaml"
    replay_cfg.write_text(
        yaml.safe_dump({"replay": {"diagnostic_only": True, "preflight_override": True}}),
        encoding="utf-8",
    )
    _write_json(replay_run / "run.meta.json", {"config_path": str(replay_cfg)})
    _write_json(
        replay_run / "metrics.json",
        {
            "replay_backend": "real_mpb",
            "best_of_k_replay_mre_bands_2_10_pct": 12.0,
            "surrogate_replay_gap_bands_2_10_pct": 10.0,
            "gap_f1": 0.0,
        },
    )
    config_path = tmp_path / "diagnostic.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"experiment_id": "unit_inverse_design_diagnostic"},
                "dataset": {"path": str(dataset_path), "condition_target": "Wh"},
                "objective_train": {"run_dir": str(objective_run)},
                "sample": {"run_dir": str(sample_run)},
                "replay": {"run_dir": str(replay_run)},
                "diagnostics": {
                    "design_bounds": [0.0, 1.0],
                    "threshold": 0.5,
                    "check_training_distribution": True,
                    "include_design_variables": True,
                    "include_target_output": True,
                    "include_prediction_output": True,
                },
                "output": {"runs_dir": str(tmp_path / "out")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    run_dir = run_inverse_design_diagnostic(config_path)

    assert (run_dir / "selected_candidate_trace.csv").exists()
    assert (run_dir / "target_summary.csv").exists()
    assert (run_dir / "multi_start_summary.csv").exists()
    assert (run_dir / "optimization_history.csv").exists()
    assert (run_dir / "inverse_design_report.md").exists()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["targets"] == 2
    assert metrics["selected_candidate_count"] == 3
    assert metrics["n_seeds_detected"] == 2
    assert metrics["multi_start_target_count"] == 1
    assert metrics["design_bound_violation_rate"] == 0.0
    assert metrics["prediction_available_rate"] == 2 / 3
    assert metrics["objective_history_rows"] == 2
    assert metrics["real_replay_available"] == 1.0
    assert metrics["diagnostic_only"] == 1.0
