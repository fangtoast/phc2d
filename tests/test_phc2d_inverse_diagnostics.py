from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from phc2d.inverse import run_inverse_diagnose


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


def test_inverse_diagnostic_marks_preflight_failure_and_diagnostic_replay(tmp_path):
    sample_run = tmp_path / "runs" / "sample"
    replay_run = tmp_path / "runs" / "replay"
    sample_run.mkdir(parents=True)
    replay_run.mkdir(parents=True)

    _write_json(
        sample_run / "metrics.json",
        {
            "targets": 625,
            "target_seed_evaluations": 1875,
            "candidate_dim": 36,
            "validity_rate": 1.0,
            "unique_rate": 0.3966,
            "nearest_neighbor_hamming_p10": 0.0,
            "nearest_neighbor_hamming_mean": 0.0069,
            "duplicate_cluster_count": 13.032,
            "largest_duplicate_cluster_fraction": 0.1575,
            "best_of_k_surrogate_mre_bands_2_10": 0.0238897,
        },
    )
    _write_json(sample_run / "results.json", {"experiment_id": "sample_unit"})
    _write_csv(
        sample_run / "ranked_candidates.csv",
        [
            {
                "target_id": "t0",
                "target_index": 0,
                "sample_id": "s0",
                "unique_rate": 0.39,
                "nearest_neighbor_hamming_p10": 0.0,
                "best_of_k_surrogate_mre_bands_2_10": 0.02,
                "duplicate_cluster_count": 12,
                "largest_duplicate_cluster_fraction": 0.2,
            },
            {
                "target_id": "t1",
                "target_index": 1,
                "sample_id": "s1",
                "unique_rate": 0.2,
                "nearest_neighbor_hamming_p10": 0.0,
                "best_of_k_surrogate_mre_bands_2_10": 0.04,
                "duplicate_cluster_count": 18,
                "largest_duplicate_cluster_fraction": 0.4,
            },
        ],
    )

    replay_source_config = tmp_path / "diagnostic_replay.yaml"
    replay_source_config.write_text(
        yaml.safe_dump({"replay": {"backend": "real_mpb", "diagnostic_only": True, "preflight_override": True}}),
        encoding="utf-8",
    )
    _write_json(replay_run / "run.meta.json", {"config_path": str(replay_source_config)})
    _write_json(
        replay_run / "metrics.json",
        {
            "targets": 24,
            "replay_backend": "real_mpb",
            "validation_boundary": "formal_inverse_candidate_mpb_replay",
            "mpb_replay_success_rate": 1.0,
            "best_of_k_replay_mre_bands_2_10_pct": 15.4627879156261,
            "best_of_k_surrogate_mre_bands_2_10_pct": 2.6719472254626453,
            "surrogate_replay_gap_bands_2_10_pct": 12.790840690163455,
            "gap_f1": 0.0,
            "gap_precision": 0.0,
            "gap_recall": 0.0,
        },
    )
    _write_json(replay_run / "results.json", {"experiment_id": "replay_unit"})
    _write_csv(
        replay_run / "per_target_replay_metrics.csv",
        [
            {
                "target_id": "t0",
                "target_index": 0,
                "sample_id": "s0",
                "candidate_count_selected": 3,
                "best_of_k_replay_mre_bands_2_10_pct": 20.0,
                "best_of_k_surrogate_mre_bands_2_10_pct": 2.0,
                "surrogate_replay_gap_bands_2_10_pct": 18.0,
                "selected_indices": "[1, 2, 3]",
            }
        ],
    )

    config_path = tmp_path / "diagnose.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"experiment_id": "diagnostic_unit"},
                "sample": {"run_dir": str(sample_run)},
                "replay": {"run_dir": str(replay_run)},
                "gates": {
                    "candidate": {
                        "validity_rate_min": 1.0,
                        "unique_rate_min": 0.25,
                        "nearest_neighbor_hamming_p10_min": 1.0 / 36.0,
                        "best_of_k_surrogate_mre_bands_2_10_max": 0.03,
                    },
                    "replay": {
                        "require_real_mpb": True,
                        "require_not_diagnostic_for_pass": True,
                        "mpb_replay_success_rate_min": 1.0,
                    },
                },
                "failure_cases": {"candidate_max_rows": 5, "replay_max_rows": 5},
                "output": {"runs_dir": str(tmp_path / "out")},
            }
        ),
        encoding="utf-8",
    )

    run_dir = run_inverse_diagnose(config_path)

    assert (run_dir / "run.log").exists()
    assert (run_dir / "run.meta.json").exists()
    assert (run_dir / "config.resolved.yaml").exists()
    assert (run_dir / "results.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "history.csv").exists()
    assert (run_dir / "inverse_gate_ledger.csv").exists()
    assert (run_dir / "inverse_gate_ledger.json").exists()
    assert (run_dir / "failure_cases.csv").exists()
    assert (run_dir / "figures").is_dir()

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["overall_gate_pass"] == 0.0
    assert metrics["candidate_gate_pass"] == 0.0
    assert metrics["replay_gate_pass"] == 0.0
    assert metrics["diagnostic_only"] == 1.0
    assert metrics["preflight_override"] == 1.0
    assert metrics["nearest_neighbor_hamming_p10"] == 0.0
    assert metrics["best_of_k_replay_mre_bands_2_10_pct"] == 15.4627879156261
    assert metrics["surrogate_replay_gap_bands_2_10_pct"] == 12.790840690163455
    assert metrics["gap_f1"] == 0.0

    ledger = json.loads((run_dir / "inverse_gate_ledger.json").read_text(encoding="utf-8"))
    overall = next(row for row in ledger if row["row_id"] == "overall_stage_gate")
    replay = next(row for row in ledger if row["row_id"] == "real_mpb_replay")
    assert overall["gate_status"] == "fail"
    assert replay["gate_status"] == "diagnostic_only"
    assert "preflight_override replay cannot promote stage gate" in overall["failure_reasons"]
