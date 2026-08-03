"""Tests for the resumable A2 single-seed inverse benchmark.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from phc2d.inverse.benchmark import (
    _binary_unique_separated,
    _bridge_gate_failures,
    _candidate_manifest_arrays,
    _continuous_candidate_diagnostics,
    _holm_adjust,
    _is_objective_bridge_test625,
    _run_a2_objective_bridge_test625_aggregate,
    _sign_test_paired,
    _target_indices,
)
from phc2d.inverse.runner import (
    _MixtureConditionalGenerator,
    _latent_input_gradient_norm,
    _objective_training_candidates,
)


def test_binary_candidates_are_deduplicated_and_hamming_separated() -> None:
    raw = np.asarray(
        [
            [0.9, 0.1, 0.9, 0.1],
            [0.8, 0.2, 0.8, 0.2],
            [0.9, 0.9, 0.1, 0.1],
            [0.1, 0.9, 0.1, 0.9],
        ],
        dtype=np.float32,
    )

    selected, metrics = _binary_unique_separated(
        raw,
        threshold=0.5,
        minimum_hamming=0.25,
    )

    assert selected.shape == (3, 4)
    assert metrics["raw_k"] == 4
    assert metrics["valid_k"] == 4
    assert metrics["effective_unique_k"] == 3
    assert metrics["hamming_min"] >= 0.25


def test_direct_repeated_output_keeps_one_effective_candidate() -> None:
    raw = np.repeat(np.asarray([[0.9, 0.1, 0.9]], dtype=np.float32), 128, axis=0)

    selected, metrics = _binary_unique_separated(
        raw,
        threshold=0.5,
        minimum_hamming=1.0 / 3.0,
    )

    assert selected.shape == (1, 3)
    assert metrics["raw_k"] == 128
    assert metrics["effective_unique_k"] == 1


def test_sign_test_and_holm_adjustment_are_deterministic() -> None:
    p_value = _sign_test_paired([1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0])
    adjusted = _holm_adjust({"retrieval": p_value, "direct": 0.04, "single": 0.5})

    assert p_value == 0.125
    assert adjusted["direct"] == 0.12
    assert adjusted["retrieval"] == 0.25
    assert adjusted["single"] == 0.5


def test_candidate_health_gate_rejects_collapsed_checkpoint() -> None:
    training = {
        "baseline_validation_mre": {"mixture_generator": 0.10},
        "candidate_health_gates": {
            "validity_rate_min": 1.0,
            "effective_k_mean_min": 64,
            "effective_k_p10_min": 32,
            "continuous_candidate_std_min": 0.04,
            "threshold_margin_fraction_max": 0.15,
            "latent_weight_change_min": 1.0e-6,
            "latent_gradient_norm_min": 1.0e-12,
            "mode_coverage_fraction_min": 1.0,
            "cross_surrogate_gap_max": 0.02,
            "validation_mre_relative_improvement_min": 0.30,
        },
    }
    collapsed = {
        "validity_rate": 1.0,
        "effective_unique_k": 20.0,
        "effective_unique_k_p10": 4.0,
        "continuous_candidate_std": 0.015,
        "threshold_margin_fraction": 0.30,
        "latent_weight_change_max": 0.0,
        "latent_input_gradient_norm": 0.0,
        "mode_coverage_fraction": 0.25,
        "cross_surrogate_gap_mean": 0.01,
        "validation_best_of_k_surrogate_mre": 0.09,
    }

    failures = _bridge_gate_failures(
        collapsed,
        method="mixture_generator",
        training=training,
    )

    assert any(item.startswith("effective_unique_k:") for item in failures)
    assert any(item.startswith("continuous_candidate_std:") for item in failures)
    assert any(item.startswith("latent_weight_change_max:") for item in failures)
    assert any(item.startswith("validation_best_of_k_surrogate_mre:") for item in failures)


def test_continuous_candidate_health_detects_diverse_outputs() -> None:
    collapsed = np.full((16, 8), 0.5, dtype=np.float32)
    diverse = np.vstack(
        [
            np.linspace(0.05, 0.45, 8, dtype=np.float32),
            np.linspace(0.55, 0.95, 8, dtype=np.float32),
        ]
        * 8
    )

    collapsed_metrics = _continuous_candidate_diagnostics(
        collapsed,
        threshold=0.5,
        mode_count=4,
    )
    diverse_metrics = _continuous_candidate_diagnostics(
        diverse,
        threshold=0.5,
        mode_count=4,
    )

    assert collapsed_metrics["continuous_candidate_std"] == 0.0
    assert collapsed_metrics["threshold_margin_fraction"] == 1.0
    assert diverse_metrics["continuous_candidate_std"] > 0.20
    assert diverse_metrics["threshold_margin_fraction"] == 0.0


def test_objective_candidates_cover_every_mode_for_each_condition() -> None:
    torch = pytest.importorskip("torch")

    class RecordingGenerator:
        latent_dim = 3
        mode_count = 4

        def __init__(self) -> None:
            self.mode_ids = None

        def _decode(self, condition, noise, mode_ids, torch_module):
            del torch_module
            self.mode_ids = mode_ids.detach().cpu()
            return noise[:, :1].repeat(1, 6).sigmoid()

    net = RecordingGenerator()
    xb = torch.zeros((3, 5), dtype=torch.float32)
    yb = torch.zeros((3, 6), dtype=torch.float32)

    candidates, _, _, group_size = _objective_training_candidates(
        net,
        xb,
        yb,
        objective={"n_objective_candidates": 16},
        seed=0,
        epoch=0,
        batch_start=0,
        torch=torch,
        device=torch.device("cpu"),
    )

    assert candidates.shape == (48, 6)
    assert group_size == 16
    assert net.mode_ids is not None
    for condition_index in range(3):
        start = condition_index * group_size
        modes = set(net.mode_ids[start : start + group_size].tolist())
        assert modes == {0, 1, 2, 3}


def test_objective_latent_gradient_is_nonzero_and_seed_is_reproducible() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    net = _MixtureConditionalGenerator(
        condition_dim=5,
        candidate_dim=6,
        latent_dim=3,
        hidden_dim=8,
        depth=2,
        mode_count=4,
        torch=torch,
    )
    xb = torch.zeros((2, 5), dtype=torch.float32)
    yb = torch.zeros((2, 6), dtype=torch.float32)
    kwargs = {
        "objective": {"n_objective_candidates": 16},
        "seed": 0,
        "epoch": 2,
        "batch_start": 4,
        "torch": torch,
        "device": torch.device("cpu"),
    }

    first, _, _, _ = _objective_training_candidates(net, xb, yb, **kwargs)
    second, _, _, _ = _objective_training_candidates(net, xb, yb, **kwargs)
    first.mean().backward()

    assert torch.equal(first, second)
    assert float(_latent_input_gradient_norm(net, torch=torch).item()) > 0.0


def test_objective_bridge_pilot_positions_cover_four_strata() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs/research/a2_inverse_objective_bridge_seed0_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    positions = [int(item) for item in config["pilot"]["target_positions"]]

    assert positions == [
        *range(0, 5),
        *range(25, 30),
        *range(50, 55),
        *range(75, 80),
    ]


def test_test625_manifest_expands_complete_frozen_test_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    config = yaml.safe_load(
        (
            root / "configs/research/a2_inverse_objective_bridge_seed0_test625_v1.yaml"
        ).read_text(encoding="utf-8")
    )

    indices = _target_indices(config)

    assert _is_objective_bridge_test625(config)
    assert len(indices) == len(set(indices)) == 625
    assert indices == list(range(9375, 10000))
    assert config["training"]["evaluation_only"] is True
    assert config["sampling"]["methods"] == [
        "single_mode_generator",
        "mixture_generator",
    ]


def test_top_m_only_manifest_storage_keeps_replay_lineage() -> None:
    candidates = np.arange(24, dtype=np.float32).reshape(6, 4)
    predictions = np.arange(36, dtype=np.float32).reshape(6, 3, 2)
    diagnostics = predictions + 1.0
    errors = np.asarray([0.6, 0.1, 0.4, 0.2, 0.3, 0.5], dtype=np.float64)
    diagnostic_errors = errors + 0.01
    cross_gap = errors + 0.02
    ranking = np.argsort(errors, kind="stable")
    top = ranking[:3]

    stored = _candidate_manifest_arrays(
        candidates=candidates,
        predictions=predictions,
        diagnostic_predictions=diagnostics,
        errors=errors,
        diagnostic_errors=diagnostic_errors,
        cross_surrogate_gap=cross_gap,
        ranking=ranking,
        top=top,
        storage="top_m_only",
    )

    assert stored["candidates"].shape == (3, 4)
    assert stored["selected_indices"].tolist() == [0, 1, 2]
    assert stored["selected_source_indices"].tolist() == top.tolist()
    assert np.array_equal(stored["candidates"], candidates[top])


def test_test625_aggregate_compares_frozen_generators_with_historical_baseline(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    selected_run = tmp_path / "selected"
    (run / "candidates").mkdir(parents=True)
    (run / "replay").mkdir()
    for method in ("single_mode_generator", "mixture_generator"):
        selected_dir = selected_run / "train" / method
        selected_dir.mkdir(parents=True)
        (selected_dir / "selected_checkpoint.json").write_text(
            json.dumps(
                {
                    "learning_rate": 1.0e-4,
                    "epoch": 50,
                    "parameter_count": 100,
                }
            ),
            encoding="utf-8",
        )

    records = []
    for method in ("single_mode_generator", "mixture_generator"):
        for target_index in (10, 11):
            records.append(
                {
                    "record_id": f"{method}_target_{target_index}",
                    "method": method,
                    "target_index": target_index,
                    "best_of_k_surrogate_mre": 0.02,
                    "best_of_k_diagnostic_query_mre": 0.021,
                    "best_of_k_cross_surrogate_gap": 0.001,
                    "effective_unique_k": 100,
                    "continuous_candidate_std": 0.1,
                    "threshold_margin_fraction": 0.05,
                    "sampling_time_s": 0.1,
                    "ranking_time_s": 0.2,
                }
            )
    (run / "candidates" / "candidate_manifest.frozen.json").write_text(
        json.dumps({"records": records}),
        encoding="utf-8",
    )
    with (run / "replay" / "replay_candidates.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "target_index", "status", "mpb_replay_mre"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "method": "single_mode_generator",
                    "target_index": 10,
                    "status": "computed",
                    "mpb_replay_mre": 0.018,
                },
                {
                    "method": "single_mode_generator",
                    "target_index": 11,
                    "status": "computed",
                    "mpb_replay_mre": 0.022,
                },
                {
                    "method": "mixture_generator",
                    "target_index": 10,
                    "status": "computed",
                    "mpb_replay_mre": 0.019,
                },
                {
                    "method": "mixture_generator",
                    "target_index": 11,
                    "status": "computed",
                    "mpb_replay_mre": 0.021,
                },
            ]
        )
    historical = tmp_path / "historical.csv"
    with historical.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_index",
                "best_of_k_replay_mre_bands_2_10_pct",
                "best_of_k_surrogate_mre_bands_2_10_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "target_index": 10,
                    "best_of_k_replay_mre_bands_2_10_pct": 3.0,
                    "best_of_k_surrogate_mre_bands_2_10_pct": 2.0,
                },
                {
                    "target_index": 11,
                    "best_of_k_replay_mre_bands_2_10_pct": 4.0,
                    "best_of_k_surrogate_mre_bands_2_10_pct": 2.5,
                },
            ]
        )
    historical_hash = hashlib.sha256(historical.read_bytes()).hexdigest()
    cfg = {
        "protocol": {
            "evidence_status": "single_seed_adjustment_test625_replay",
            "replay_top_m": 5,
        },
        "dataset": {"expected_target_count": 2},
        "training": {"mode": "objective_bridge"},
        "evaluation": {
            "scope": "full_test625_extension",
            "selected_training_run": str(selected_run),
            "historical_baseline_per_target_path": str(historical),
            "historical_baseline_per_target_sha256": historical_hash,
            "historical_effective_k_mean": 90.0,
        },
        "replay": {
            "methods": ["single_mode_generator", "mixture_generator"],
        },
    }

    output = _run_a2_objective_bridge_test625_aggregate(cfg, run=run)
    results = json.loads((output / "results.json").read_text(encoding="utf-8"))

    assert results["status"] == "adjustment_test625_completed"
    assert results["target_count"] == 2
    assert results["training_reused_without_update"] is True
    assert len(results["comparisons"]) == 3
    assert all(
        row["relative_mean_improvement"] > 0
        for row in results["comparisons"][:2]
    )
