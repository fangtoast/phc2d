from __future__ import annotations

import csv
import json

import numpy as np
import yaml

from phc2d.artifacts import MPB_REPLAY_BANDS_NAME, SURROGATE_PRED_BANDS_NAME, TARGET_BANDS_NAME
from phc2d.replay import run_replay_mpb
from phc2d.replay.runner import _select_candidates, _select_target_records


def test_replay_runner_writes_band1_to_band10_artifacts(tmp_path):
    dataset_path = tmp_path / "tiny.npz"
    target = np.linspace(0.1, 0.5, 40, dtype=np.float32).reshape(4, 10)
    np.savez(
        dataset_path,
        rho=np.zeros((1, 36), dtype=np.float32),
        Wh=target.reshape(1, 4, 10),
    )

    candidates_path = tmp_path / "candidates.json"
    candidates_payload = {
        "version": 1,
        "targets": [
            {
                "target_id": "replay_unit_Wh_00000_000",
                "target_index": 0,
                "sample_id": "tiny_000000",
                "dataset_id": "tiny",
                "split": "test",
                "condition": {
                    "Wh": target.reshape(-1).tolist(),
                },
                "candidates": [
                    [0.2] * 36,
                    [0.4] * 36,
                    [0.6] * 36,
                ],
                "selected_indices": [0, 1],
                "selected_surrogate_predictions": [
                    (target * 1.01).tolist(),
                    (target * 1.02).tolist(),
                ],
                "surrogate_metrics": {
                    "best_of_k_surrogate_mre_bands_2_10": 0.01,
                    "best_of_k_surrogate_mre_full": 0.02,
                },
            }
        ],
    }
    candidates_path.write_text(yaml.safe_dump(candidates_payload), encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config = {
        "experiment": {
            "experiment_id": "replay_unit",
            "family": "test",
            "stage": "inverse_sample",
        },
        "dataset": {
            "path": str(dataset_path),
            "condition_target": "Wh",
            "output_repr": "rho_wedge",
            "target_split": "test",
        },
        "model": {
            "name": "conditional_vae_rho",
        },
        "inverse": {
            "n_candidates": 4,
            "surrogate_top_m": 2,
        },
        "replay": {
            "enabled": True,
            "backend": "fake_mpb",
            "candidates_path": str(candidates_path),
            "replay_top_m": 2,
            "n_targets": 1,
            "timeout_seconds": 10,
            "target_strata": ["test"],
        },
        "k_path": {
            "family": "square_gxm",
            "n_total": 4,
        },
        "metrics": {
            "band_ranges": {
                "bands_1_10": [1, 10],
                "bands_2_10": [2, 10],
            },
            "gap": {
                "threshold_midgap_ratio": 0.0,
            },
        },
        "output": {
            "runs_dir": str(tmp_path / "runs"),
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    run_dir = run_replay_mpb(config_path)
    assert run_dir.is_dir()
    assert (run_dir / "run.log").exists()
    assert (run_dir / "run.meta.json").exists()
    assert (run_dir / "config.resolved.yaml").exists()
    assert (run_dir / "results.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "history.csv").exists()
    assert (run_dir / "per_target_replay_metrics.csv").exists()
    assert (run_dir / "per_band_replay_metrics.csv").exists()
    assert (run_dir / "gap_metrics.csv").exists()
    assert (run_dir / "replay_candidates.jsonl").exists()
    assert (run_dir / "figures").is_dir()
    assert (run_dir / TARGET_BANDS_NAME).is_file()
    assert (run_dir / SURROGATE_PRED_BANDS_NAME).is_file()
    assert (run_dir / MPB_REPLAY_BANDS_NAME).is_file()

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "best_of_k_replay_mre_bands_2_10_pct" in metrics
    assert "best_of_k_replay_mre_bands_1_10_pct" in metrics
    assert "best_of_k_surrogate_mre_bands_2_10_pct" in metrics
    assert "best_of_k_surrogate_mre_bands_1_10_pct" in metrics
    assert "surrogate_replay_gap_bands_2_10_pct" in metrics
    assert "surrogate_replay_gap_bands_1_10_pct" in metrics
    assert "gap_f1" in metrics
    assert metrics["validation_boundary"] == "test_fake_mpb_replay_not_physical"
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert results["band_array_artifacts"]["surrogate_pred_available_count"] == 1
    target_bands = np.load(run_dir / TARGET_BANDS_NAME)
    surrogate_bands = np.load(run_dir / SURROGATE_PRED_BANDS_NAME)
    replay_bands = np.load(run_dir / MPB_REPLAY_BANDS_NAME)
    assert target_bands.shape == surrogate_bands.shape == replay_bands.shape == (1, 4, 10)
    assert np.isfinite(surrogate_bands).all()

    with (run_dir / "per_band_replay_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["band_one_based"]) for row in rows] == list(range(1, 11))
    for row in rows:
        assert row["mpb_replay_mre_pct"] != ""
        assert row["surrogate_mre_pct"] != ""
        assert row["surrogate_replay_gap_pct"] != ""
        assert row["mae"] != ""
        assert row["rmse"] != ""
        assert int(row["sample_count"]) == 1


def test_replay_proxy_boundary_is_not_formal(tmp_path):
    dataset_path = tmp_path / "tiny_proxy.npz"
    target = np.ones((4, 3), dtype=np.float32)
    np.savez(dataset_path, rho=np.zeros((1, 36), dtype=np.float32), Wh=target.reshape(1, 4, 3))

    candidates_path = tmp_path / "candidates.json"
    candidates_payload = {
        "version": 1,
        "targets": [
            {
                "target_id": "proxy_000",
                "target_index": 0,
                "condition": {"Wh": target.reshape(-1).tolist()},
                "candidates": [[0.0] * 36],
                "selected_indices": [0],
                "surrogate_metrics": {
                    "best_of_k_surrogate_mre_bands_2_10": 0.1,
                    "best_of_k_surrogate_mre_full": 0.2,
                },
            }
        ],
    }
    candidates_path.write_text(yaml.safe_dump(candidates_payload), encoding="utf-8")

    config_path = tmp_path / "config_proxy.yaml"
    config = {
        "experiment": {"experiment_id": "proxy_replay_unit"},
        "dataset": {"path": str(dataset_path), "condition_target": "Wh", "output_repr": "rho_wedge"},
        "model": {"name": "conditional_vae_rho"},
        "inverse": {"n_candidates": 1, "surrogate_top_m": 1},
        "replay": {"enabled": True, "backend": "proxy", "candidates_path": str(candidates_path), "n_targets": 1},
        "k_path": {"family": "square_gxm", "n_total": 4},
        "output": {"runs_dir": str(tmp_path / "runs")},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    run_dir = run_replay_mpb(config_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["validation_boundary"] == "pilot_mpb_replay_proxy_not_physical"
    assert "formal" not in metrics["validation_boundary"]
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert results["band_array_artifacts"]["surrogate_pred_available_count"] == 0
    assert np.isnan(np.load(run_dir / SURROGATE_PRED_BANDS_NAME)).all()
    assert np.load(run_dir / TARGET_BANDS_NAME).shape == np.load(run_dir / MPB_REPLAY_BANDS_NAME).shape
    with (run_dir / "per_band_replay_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["surrogate_mre_pct"] != ""
    assert rows[0]["surrogate_mre_source"] == "target_scalar_bands_1_10_broadcast"
    assert rows[1]["surrogate_mre_source"] == "target_scalar_bands_2_10_broadcast"


def test_replay_selection_backfills_to_top_m_from_ranked_indices():
    target_record = {
        "candidates": [[1.0]],
        "raw_candidates": [[0.0], [1.0], [2.0], [3.0]],
        "selected_indices": [1],
        "ranked_indices": [1, 3, 0, 2],
    }

    candidates, selected = _select_candidates(target_record, top_m=3)

    assert selected == [1, 3, 0]
    assert candidates.reshape(-1).tolist() == [1.0, 3.0, 0.0]


def test_replay_unique_target_selection_dedupes_target_index() -> None:
    targets = [
        {"target_index": 7, "target_id": "a", "surrogate_metrics": {"best_of_k_surrogate_mre_bands_2_10": 0.4}},
        {"target_index": 7, "target_id": "b", "surrogate_metrics": {"best_of_k_surrogate_mre_bands_2_10": 0.1}},
        {"target_index": 8, "target_id": "c", "surrogate_metrics": {"best_of_k_surrogate_mre_bands_2_10": 0.2}},
    ]

    selected = _select_target_records(
        targets,
        n_targets=2,
        seed=0,
        selection_unit="target_index",
        within_target_record_policy="best_surrogate_mre_bands_2_10",
    )

    target_indices = [int(record["target_index"]) for _source_index, _stratum, record in selected]
    target_ids = {str(record["target_id"]) for _source_index, _stratum, record in selected}
    assert len(target_indices) == len(set(target_indices)) == 2
    assert "b" in target_ids
    assert "a" not in target_ids


def test_replay_target_indices_path_reuses_exact_targets(tmp_path) -> None:
    dataset_path = tmp_path / "tiny_unique.npz"
    target = np.linspace(0.1, 0.5, 40, dtype=np.float32).reshape(4, 10)
    np.savez(
        dataset_path,
        rho=np.zeros((2, 36), dtype=np.float32),
        Wh=np.stack([target, target * 1.1], axis=0),
    )
    candidates_path = tmp_path / "candidates.json"
    candidates_payload = {
        "version": 1,
        "targets": [
            {
                "target_id": "dup_low",
                "target_index": 11,
                "condition": {"Wh": target.reshape(-1).tolist()},
                "candidates": [[0.2] * 36, [0.4] * 36],
                "selected_indices": [0, 1],
                "surrogate_metrics": {"best_of_k_surrogate_mre_bands_2_10": 0.5},
            },
            {
                "target_id": "dup_best",
                "target_index": 11,
                "condition": {"Wh": target.reshape(-1).tolist()},
                "candidates": [[0.3] * 36, [0.5] * 36],
                "selected_indices": [0, 1],
                "surrogate_metrics": {"best_of_k_surrogate_mre_bands_2_10": 0.1},
            },
            {
                "target_id": "unique",
                "target_index": 12,
                "condition": {"Wh": (target * 1.1).reshape(-1).tolist()},
                "candidates": [[0.6] * 36, [0.7] * 36],
                "selected_indices": [0, 1],
                "surrogate_metrics": {"best_of_k_surrogate_mre_bands_2_10": 0.2},
            },
        ],
    }
    candidates_path.write_text(yaml.safe_dump(candidates_payload), encoding="utf-8")
    target_indices_path = tmp_path / "target_indices.json"
    written_indices_path = tmp_path / "written_target_indices.json"
    target_indices_path.write_text(json.dumps({"target_indices": [12, 11]}), encoding="utf-8")

    config_path = tmp_path / "unique_replay.yaml"
    config = {
        "experiment": {"experiment_id": "unique_replay_unit"},
        "dataset": {"path": str(dataset_path), "condition_target": "Wh", "output_repr": "rho_wedge"},
        "model": {"name": "conditional_vae_rho"},
        "inverse": {"n_candidates": 2, "surrogate_top_m": 2},
        "replay": {
            "enabled": True,
            "backend": "fake_mpb",
            "candidates_path": str(candidates_path),
            "selection_unit": "target_index",
            "within_target_record_policy": "best_surrogate_mre_bands_2_10",
            "target_indices_path": str(target_indices_path),
            "write_target_indices_path": str(written_indices_path),
            "replay_top_m": 2,
            "n_targets": 2,
        },
        "k_path": {"family": "square_gxm", "n_total": 4},
        "metrics": {"band_ranges": {"bands_1_10": [1, 10], "bands_2_10": [2, 10]}},
        "output": {"runs_dir": str(tmp_path / "runs")},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    run_dir = run_replay_mpb(config_path)

    with (run_dir / "per_target_replay_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["target_index"]) for row in rows] == [12, 11]
    assert json.loads(written_indices_path.read_text(encoding="utf-8"))["target_indices"] == [12, 11]
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["selection_unit"] == "target_index"
    assert metrics["unique_target_index_count"] == 2


def test_strategy_replay_dedupes_candidates_and_reports_per_strategy(tmp_path):
    dataset_path = tmp_path / "tiny_strategy.npz"
    target = np.linspace(0.1, 0.5, 40, dtype=np.float32).reshape(4, 10)
    np.savez(
        dataset_path,
        rho=np.zeros((1, 36), dtype=np.float32),
        Wh=target.reshape(1, 4, 10),
    )

    rerank_run = tmp_path / "runs" / "rerank_unit_202606301200"
    rerank_run.mkdir(parents=True)
    strategy_candidates_path = rerank_run / "strategy_candidates.json"
    strategy_candidates_payload = {
        "version": 2,
        "kind": "inverse_surrogate_rerank_candidates",
        "validation_boundary": "inverse_surrogate_rerank_no_mpb_meep",
        "target_shape": [4, 10],
        "top_m": 2,
        "primary_strategy": "strategy_a",
        "strategies": [
            {"strategy_id": "strategy_a"},
            {"strategy_id": "strategy_b"},
        ],
        "targets": [
            {
                "target_id": "strategy_replay_000",
                "target_index": 0,
                "sample_id": "tiny_000000",
                "dataset_id": "tiny_strategy",
                "split": "test",
                "condition": {"Wh": target.reshape(-1).tolist()},
                "candidates": [
                    [0.2] * 36,
                    [0.4] * 36,
                    [0.6] * 36,
                ],
                "surrogate_metrics": {
                    "best_of_k_surrogate_mre_bands_2_10": 0.01,
                    "best_of_k_surrogate_mre_full": 0.02,
                },
                "strategy_rankings": {
                    "strategy_a": {
                        "selected_indices": [0, 1],
                        "ranked_indices": [0, 1, 2],
                        "selected_surrogate_predictions": [
                            (target * 1.01).tolist(),
                            (target * 1.02).tolist(),
                        ],
                        "surrogate_metrics": {
                            "best_of_k_surrogate_mre_bands_2_10": 0.01,
                            "best_of_k_surrogate_mre_bands_1_10": 0.01,
                        },
                    },
                    "strategy_b": {
                        "selected_indices": [1, 2],
                        "ranked_indices": [1, 2, 0],
                        "selected_surrogate_predictions": [
                            (target * 1.03).tolist(),
                            (target * 1.04).tolist(),
                        ],
                        "surrogate_metrics": {
                            "best_of_k_surrogate_mre_bands_2_10": 0.03,
                            "best_of_k_surrogate_mre_bands_1_10": 0.03,
                        },
                    },
                },
            }
        ],
    }
    strategy_candidates_path.write_text(yaml.safe_dump(strategy_candidates_payload), encoding="utf-8")

    config_path = tmp_path / "strategy_replay.yaml"
    config = {
        "experiment": {"experiment_id": "strategy_replay_unit"},
        "dataset": {
            "path": str(dataset_path),
            "condition_target": "Wh",
            "output_repr": "rho_wedge",
            "target_split": "test",
        },
        "model": {"name": "conditional_vae_rho"},
        "inverse": {"n_candidates": 3, "surrogate_top_m": 2},
        "replay": {
            "enabled": True,
            "backend": "fake_mpb",
            "source_experiment_id": "rerank_unit",
            "replay_top_m": 2,
            "n_targets": 1,
            "timeout_seconds": 10,
        },
        "k_path": {"family": "square_gxm", "n_total": 4},
        "metrics": {
            "band_ranges": {
                "bands_1_10": [1, 10],
                "bands_2_10": [2, 10],
            },
            "gap": {"threshold_midgap_ratio": 0.0},
        },
        "output": {"runs_dir": str(tmp_path / "runs")},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    calls = []

    def solver(candidate, k_points, solver_config):
        calls.append(np.asarray(candidate, dtype=np.float32).copy())
        fill = float(np.mean(candidate))
        bands = target + fill * 0.01
        return bands.astype(np.float32), {"solver_call": len(calls)}

    run_dir = run_replay_mpb(config_path, solver_fn=solver)

    assert len(calls) == 3
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["strategy_replay"] is True
    assert metrics["strategy_count"] == 2
    assert metrics["strategy_candidate_selection_count"] == 4
    assert metrics["unique_candidate_solve_count"] == 3
    assert metrics["candidate_dedupe_savings_count"] == 1
    with (run_dir / "per_strategy_replay_metrics.csv").open(newline="", encoding="utf-8") as handle:
        strategy_rows = list(csv.DictReader(handle))
    assert {row["strategy_id"] for row in strategy_rows} == {"strategy_a", "strategy_b"}
    with (run_dir / "per_strategy_band_replay_metrics.csv").open(newline="", encoding="utf-8") as handle:
        band_rows = list(csv.DictReader(handle))
    assert len(band_rows) == 20
    for strategy_id in {"strategy_a", "strategy_b"}:
        bands = [int(row["band_one_based"]) for row in band_rows if row["strategy_id"] == strategy_id]
        assert bands == list(range(1, 11))
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert results["band_array_artifacts"]["surrogate_pred_available_count"] == 2
    target_bands = np.load(run_dir / TARGET_BANDS_NAME)
    surrogate_bands = np.load(run_dir / SURROGATE_PRED_BANDS_NAME)
    replay_bands = np.load(run_dir / MPB_REPLAY_BANDS_NAME)
    assert target_bands.shape == surrogate_bands.shape == replay_bands.shape == (2, 4, 10)
    assert np.isfinite(surrogate_bands).all()
