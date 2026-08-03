"""Tests for generative inverse candidate protocol and ranking.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
import yaml

from phc2d.inverse import (
    CandidateBatch,
    GenerativeInverseConfig,
    binary_morphology_metrics,
    candidate_batch_metrics,
    candidate_morphology_metrics,
    diversity_hamming,
    nearest_neighbor_hamming_stats,
    rank_candidates,
    separated_candidate_indices,
    unique_rate,
    validity_rate,
)
from phc2d.inverse.runner import (
    _build_surrogate,
    _find_latest_generator_checkpoint,
    _filter_candidate_tensor,
    _load_targets,
    _resolve_forward_surrogate_checkpoint,
    _resolve_query_surrogate_checkpoint,
    _surrogate_error_tensor,
    run_inverse_eval,
    run_inverse_objective_train,
    run_inverse_rerank,
    run_inverse_train,
)
from phc2d.inverse.config import load_inverse_run_config
from phc2d.inverse.objective import (
    band_relative_error_loss,
    binary_candidate_diversity_loss,
    candidate_diversity_loss,
    cross_surrogate_consistency_loss,
    gap_descriptor_loss,
    latent_response_loss,
    objective_loss_components,
    quality_weighted_collision_loss,
    soft_gap_descriptor,
)


def test_candidate_batch_validates_rho_or_grid_candidates() -> None:
    rho = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
    batch = CandidateBatch(
        target_id="target-001",
        condition={"target": [0.1, 0.2]},
        rho_candidates=rho,
        input_repr="rho_wedge",
    )
    assert batch.n_candidates == 2
    assert batch.candidate_kind == "rho_candidates"
    assert batch.metadata[1]["candidate_index"] == 1

    grid = np.zeros((2, 3, 3), dtype=np.float32)
    grid_batch = CandidateBatch(
        target_id="target-002",
        condition={"target": [0.3]},
        grid_candidates=grid,
        input_repr="rho_grid",
        metadata=[{"rank": 0}, {"rank": 1}],
    )
    assert grid_batch.candidates.shape == (2, 3, 3)
    assert grid_batch.candidate_kind == "grid_candidates"

    with pytest.raises(ValueError, match="exactly one"):
        CandidateBatch(
            target_id="bad",
            condition={},
            rho_candidates=rho,
            grid_candidates=grid,
            input_repr="rho_grid",
        )
    with pytest.raises(ValueError, match="metadata length"):
        CandidateBatch(
            target_id="bad",
            condition={},
            rho_candidates=rho,
            input_repr="rho_wedge",
            metadata=[{}],
        )


def test_candidate_metrics_unique_diversity_validity_and_fill() -> None:
    candidates = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    batch = CandidateBatch("target", {"Wh": [1.0]}, candidates, "rho_wedge")
    metrics = candidate_batch_metrics(batch)

    assert validity_rate(candidates) == 1.0
    assert unique_rate(candidates) == pytest.approx(2 / 3)
    assert diversity_hamming(candidates) > 0.0
    assert metrics["fill_fraction"] == pytest.approx(float(np.mean(candidates >= 0.5)))

    invalid = np.array([[0.0, 1.0], [-0.1, 0.5], [1.2, 0.0]], dtype=np.float32)
    assert validity_rate(invalid) == pytest.approx(1 / 3)


def test_hard_duplicate_metrics_cover_clusters_unique_and_singleton() -> None:
    duplicate_cluster = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    stats = nearest_neighbor_hamming_stats(duplicate_cluster)
    assert stats["nearest_neighbor_hamming_min"] == pytest.approx(0.0)
    assert stats["nearest_neighbor_hamming_p10"] == pytest.approx(0.0)
    assert stats["nearest_neighbor_hamming_mean"] > 0.0
    assert stats["duplicate_cluster_count"] == pytest.approx(1.0)
    assert stats["largest_duplicate_cluster_fraction"] == pytest.approx(0.5)

    all_unique = np.eye(4, dtype=np.float32)
    unique_stats = nearest_neighbor_hamming_stats(all_unique)
    assert unique_stats["nearest_neighbor_hamming_min"] > 0.0
    assert unique_stats["duplicate_cluster_count"] == pytest.approx(0.0)
    assert unique_stats["largest_duplicate_cluster_fraction"] == pytest.approx(0.0)

    single = nearest_neighbor_hamming_stats(np.array([[0.0, 1.0, 0.0]], dtype=np.float32))
    assert single["nearest_neighbor_hamming_mean"] == pytest.approx(0.0)
    assert single["duplicate_cluster_count"] == pytest.approx(0.0)


def test_separated_candidate_indices_remove_binary_duplicates_and_raise_p10() -> None:
    candidates = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    idx = separated_candidate_indices(
        candidates,
        deduplicate_binary=True,
        min_hamming=1.0 / candidates.shape[1],
    )
    separated = candidates[idx]
    stats = nearest_neighbor_hamming_stats(separated)

    assert idx.tolist() == [0, 2, 3]
    assert stats["duplicate_cluster_count"] == pytest.approx(0.0)
    assert stats["nearest_neighbor_hamming_p10"] >= (1.0 / candidates.shape[1]) - 1.0e-6


def test_morphology_metrics_cover_components_isolated_pixels_and_tv() -> None:
    grid = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    metrics = binary_morphology_metrics(grid)
    assert metrics["fill_fraction"] == pytest.approx(3 / 9)
    assert metrics["num_connected_components"] == 2.0
    assert metrics["isolated_pixel_count"] == 1.0
    assert metrics["total_variation"] == 6.0

    aggregate = candidate_morphology_metrics(np.stack([grid, np.ones((3, 3), dtype=np.float32)]))
    assert aggregate["mean_connected_components"] == pytest.approx(1.5)
    assert 0.0 <= aggregate["mean_morphology_score"] <= 1.0


def test_mock_surrogate_ranking_and_diversity_aware_top_m() -> None:
    candidates = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    batch = CandidateBatch("target", {"Wh": [0.5]}, candidates, "rho_wedge")

    class MockSurrogate:
        def predict(self, values: np.ndarray) -> np.ndarray:
            return values.mean(axis=1, keepdims=True)

    result = rank_candidates(
        batch,
        MockSurrogate(),
        target=np.array([0.5], dtype=np.float32),
        top_m=2,
        diversity_weight=0.5,
    )

    assert result.ranked_indices[0] == 2
    assert set(result.selected_indices.tolist()) == {0, 2}
    assert result.metrics["best_of_k_surrogate_mre"] == pytest.approx(0.0)
    assert result.metrics["surrogate_ranking_time_s"] >= 0.0
    assert result.metrics["selected_diversity_hamming"] == pytest.approx(0.5)


def test_generative_inverse_config_validates_counts_and_range(tmp_path) -> None:
    config = GenerativeInverseConfig(
        dataset_path=tmp_path / "tiny.npz",
        output_dir=tmp_path / "runs",
        n_candidates=8,
        surrogate_top_m=3,
    )
    assert config.model_name == "conditional_vae_rho"
    assert "best_of_k_surrogate_mre" in config.metrics_include

    with pytest.raises(ValueError, match="n_candidates"):
        GenerativeInverseConfig(dataset_path=tmp_path / "tiny.npz", output_dir=tmp_path, n_candidates=0)
    with pytest.raises(ValueError, match="increasing"):
        GenerativeInverseConfig(
            dataset_path=tmp_path / "tiny.npz",
            output_dir=tmp_path,
            candidate_value_range=(1.0, 0.0),
        )


def test_inverse_config_parses_learning_rate_grid_and_optimizer(tmp_path) -> None:
    config_path = tmp_path / "inverse_grid.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"experiment_id": "unit_inverse_grid"},
                "dataset": {"path": str(tmp_path / "tiny.npz")},
                "inverse": {
                    "learning_rates": [0.001, 0.0003, 0.0001],
                    "optimizer": "adamw",
                    "weight_decay": 1.0e-5,
                    "target_limit": 40,
                    "target_selection_seed": 20260702,
                    "candidate_separation": {
                        "enabled": True,
                        "deduplicate_binary": True,
                        "min_hamming": "auto_1_over_dim",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_inverse_run_config(config_path)

    assert config.learning_rates == (0.001, 0.0003, 0.0001)
    assert config.optimizer == "adamw"
    assert config.weight_decay == pytest.approx(1.0e-5)
    assert config.target_limit == 40
    assert config.target_selection_seed == 20260702
    assert config.candidate_separation["enabled"] is True
    assert config.candidate_separation["min_hamming"] == "auto_1_over_dim"


def test_filter_candidate_tensor_applies_candidate_separation(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    cfg = GenerativeInverseConfig(
        dataset_path=tmp_path / "tiny.npz",
        output_dir=tmp_path / "runs",
        candidate_separation={
            "enabled": True,
            "deduplicate_binary": True,
            "min_hamming": "auto_1_over_dim",
        },
    )
    raw = torch.as_tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    filtered, metrics, indices = _filter_candidate_tensor(raw, cfg, torch)
    stats = nearest_neighbor_hamming_stats(filtered.detach().cpu().numpy())

    assert indices.tolist() == [0, 2, 3]
    assert int(filtered.shape[0]) == 3
    assert metrics["candidate_separation_enabled"] == pytest.approx(1.0)
    assert metrics["candidate_count_filtered"] == pytest.approx(3.0)
    assert stats["nearest_neighbor_hamming_p10"] >= (1.0 / 4.0) - 1.0e-6


def test_load_targets_honors_seeded_target_limit(tmp_path) -> None:
    dataset_path = tmp_path / "targets.npz"
    rho = np.arange(10 * 4, dtype=np.float32).reshape(10, 4)
    wh = np.arange(10 * 2 * 3, dtype=np.float32).reshape(10, 2, 3)
    idx_test = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    np.savez(
        dataset_path,
        rho=rho,
        Wh=wh,
        idx_train=np.array([0], dtype=np.int64),
        idx_val=np.array([7], dtype=np.int64),
        idx_test=idx_test,
    )
    cfg = GenerativeInverseConfig(
        dataset_path=dataset_path,
        output_dir=tmp_path / "runs",
        target_split="test",
        target_limit=3,
        target_selection_seed=123,
    )

    indices, target_pool, candidate_dim, metadata = _load_targets(cfg)

    expected = np.asarray(sorted(np.random.default_rng(123).choice(idx_test, size=3, replace=False)), dtype=np.int64)
    assert indices.tolist() == expected.tolist()
    assert target_pool.shape == wh.shape
    assert candidate_dim == 4
    assert len(metadata) == 3


def test_load_targets_without_limit_keeps_full_split(tmp_path) -> None:
    dataset_path = tmp_path / "targets.npz"
    np.savez(
        dataset_path,
        rho=np.zeros((5, 4), dtype=np.float32),
        Wh=np.zeros((5, 2, 3), dtype=np.float32),
        idx_train=np.array([0], dtype=np.int64),
        idx_val=np.array([1], dtype=np.int64),
        idx_test=np.array([2, 3, 4], dtype=np.int64),
    )
    cfg = GenerativeInverseConfig(dataset_path=dataset_path, output_dir=tmp_path / "runs", target_split="test")

    indices, _, _, metadata = _load_targets(cfg)

    assert indices.tolist() == [2, 3, 4]
    assert len(metadata) == 3


def test_formal_inverse_config_rejects_mock_surrogate(tmp_path) -> None:
    config = GenerativeInverseConfig(
        dataset_path=tmp_path / "tiny.npz",
        output_dir=tmp_path / "runs",
        allow_mock_surrogate=False,
        surrogate={},
    )

    with pytest.raises(ValueError, match="Formal inverse ranking requires"):
        _build_surrogate(
            config,
            input_dim=4,
            target_shape=(2,),
            torch=None,
            device=None,
        )


def test_run_dir_generator_checkpoint_takes_precedence_over_init_checkpoint(tmp_path) -> None:
    init_checkpoint = tmp_path / "old" / "model.pt"
    init_checkpoint.parent.mkdir(parents=True)
    init_checkpoint.write_bytes(b"old")
    run_checkpoint = tmp_path / "runs" / "objective" / "model.pt"
    run_checkpoint.parent.mkdir(parents=True)
    run_checkpoint.write_bytes(b"fine-tuned")
    config = GenerativeInverseConfig(
        dataset_path=tmp_path / "tiny.npz",
        output_dir=tmp_path / "runs",
        run_dir=run_checkpoint.parent,
        generator_checkpoint_path=init_checkpoint,
    )

    assert _find_latest_generator_checkpoint(config) == run_checkpoint


def test_forward_surrogate_checkpoint_resolves_from_leaderboard(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "forward_band_benchmark_legacy_50k_202606211200"
    checkpoint = run_dir / "cases" / "epsilon_grid_fno_lite_4_seed1" / "train" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    leaderboard = run_dir / "leaderboard.csv"
    leaderboard.write_text(
        "\n".join(
            [
                "status,model,input_repr,seed,avg_MRE_bands_2_10_pct,checkpoint_path,train_run",
                "ok,res_mlp,rho_wedge,0,12.0,,",
                f"ok,fno_lite,epsilon_grid,1,8.0,{checkpoint.as_posix()},{checkpoint.parent.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    config = GenerativeInverseConfig(
        dataset_path=tmp_path / "tiny.npz",
        output_dir=tmp_path / "runs",
        surrogate={
            "kind": "forward",
            "forward_experiment_id": "forward_band_benchmark_legacy_50k",
            "model_name": "fno_lite",
            "input_repr": "epsilon_grid",
            "seed": 1,
        },
    )

    assert _resolve_forward_surrogate_checkpoint(config) == checkpoint


def test_query_surrogate_checkpoint_resolves_configured_scale(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "query_data_scaling_v2_10000_full_N10000_202606291200"
    checkpoint = run_dir / "cases" / "unseen_structure_all_k_fno2d_query_seed0.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (run_dir / "leaderboard.csv").write_text(
        "\n".join(
            [
                "protocol,model,query_mre_bands_2_10_pct_mean",
                "unseen_structure_all_k,fno2d_query,3.0",
            ]
        ),
        encoding="utf-8",
    )
    config = GenerativeInverseConfig(
        dataset_path=tmp_path / "tiny.npz",
        output_dir=tmp_path / "runs",
        surrogate={
            "kind": "query",
            "scaling_experiment_id": "query_data_scaling_v2_10000_full",
            "scale_n": 10000,
            "seed": 0,
        },
    )

    assert _resolve_query_surrogate_checkpoint(config) == checkpoint


def test_formal_forward_inverse_config_requires_checkpoint(tmp_path) -> None:
    config = GenerativeInverseConfig(
        dataset_path=tmp_path / "tiny.npz",
        output_dir=tmp_path / "runs",
        allow_mock_surrogate=False,
        surrogate={"kind": "forward"},
    )

    with pytest.raises(ValueError, match="surrogate.kind=forward"):
        _build_surrogate(
            config,
            input_dim=4,
            target_shape=(2,),
            torch=None,
            device=None,
        )


def test_inverse_rerank_writes_strategy_and_band_artifacts(tmp_path) -> None:
    dataset_path = tmp_path / "tiny_rerank.npz"
    wh = np.linspace(0.1, 0.5, 80, dtype=np.float32).reshape(2, 4, 10)
    np.savez(
        dataset_path,
        rho=np.zeros((2, 4), dtype=np.float32),
        Wh=wh,
    )

    candidates_path = tmp_path / "candidates.json"
    candidates_payload = {
        "version": 1,
        "targets": [
            {
                "target_id": "rerank_target_000",
                "target_index": 0,
                "sample_id": "tiny_000",
                "dataset_id": "tiny_rerank",
                "split": "test",
                "condition": {"Wh": wh[0].reshape(-1).tolist()},
                "candidates": [
                    [0.0, 0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6, 0.7],
                    [0.8, 0.7, 0.6, 0.5],
                ],
            },
            {
                "target_id": "rerank_target_001",
                "target_index": 1,
                "sample_id": "tiny_001",
                "dataset_id": "tiny_rerank",
                "split": "test",
                "condition": {"Wh": wh[1].reshape(-1).tolist()},
                "candidates": [
                    [0.3, 0.2, 0.1, 0.0],
                    [0.7, 0.6, 0.5, 0.4],
                    [0.5, 0.6, 0.7, 0.8],
                ],
            },
        ],
    }
    candidates_path.write_text(yaml.safe_dump(candidates_payload), encoding="utf-8")

    config_path = tmp_path / "rerank.yaml"
    config = {
        "experiment": {"experiment_id": "unit_inverse_rerank"},
        "dataset": {
            "path": str(dataset_path),
            "condition_target": "Wh",
            "output_repr": "rho_wedge",
            "target_split": "test",
        },
        "model": {"name": "conditional_vae_rho", "seed": 3},
        "runtime": {"device": "cpu", "require_accelerator": False},
        "inverse": {
            "n_candidates": 3,
            "surrogate_top_m": 2,
            "allow_mock_surrogate": True,
            "ranking": {"error_metric": "bands_2_10_mre"},
        },
        "rerank": {
            "candidates_path": str(candidates_path),
            "top_m": 2,
            "primary_strategy": "mock_a",
            "strategies": [
                {"id": "mock_a", "kind": "mock"},
                {"id": "mock_b", "kind": "mock"},
                {
                    "id": "hybrid_mock",
                    "kind": "hybrid",
                    "components": [
                        {"kind": "mock", "weight": 0.25},
                        {"kind": "mock", "weight": 0.75},
                    ],
                },
            ],
        },
        "output": {"runs_dir": str(tmp_path / "runs")},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    run_dir = run_inverse_rerank(config_path)

    for name in (
        "strategy_candidates.json",
        "rerank_leaderboard.csv",
        "per_strategy_metrics.csv",
        "per_target_strategy_metrics.csv",
        "per_band_surrogate_metrics.csv",
        "per_target_band_surrogate_metrics.csv",
        "component_metrics.csv",
        "history.csv",
        "metrics.json",
        "results.json",
    ):
        assert (run_dir / name).exists()
    payload = yaml.safe_load((run_dir / "strategy_candidates.json").read_text(encoding="utf-8"))
    assert payload["validation_boundary"] == "inverse_surrogate_rerank_no_mpb_meep"
    assert set(payload["targets"][0]["strategy_rankings"]) == {"mock_a", "mock_b", "hybrid_mock"}
    with (run_dir / "per_band_surrogate_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 30
    for strategy_id in {"mock_a", "mock_b", "hybrid_mock"}:
        bands = [int(row["band_one_based"]) for row in rows if row["strategy_id"] == strategy_id]
        assert bands == list(range(1, 11))


def test_inverse_eval_requires_replay_enabled(tmp_path) -> None:
    config_path = tmp_path / "inverse.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"experiment_id": "unit_inverse_eval_replay_gate"},
                "dataset": {"path": str(tmp_path / "tiny.npz")},
                "replay": {"enabled": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires replay.enabled=true"):
        run_inverse_eval(config_path)


def test_inverse_ranking_can_use_bands_2_10_metric(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    dataset_path = tmp_path / "tiny.npz"
    np.savez(
        dataset_path,
        rho=np.zeros((1, 36), dtype=np.float32),
        Wh=np.ones((1, 2, 3), dtype=np.float32),
        idx_train=np.array([0], dtype=np.int64),
        idx_val=np.array([], dtype=np.int64),
        idx_test=np.array([0], dtype=np.int64),
    )
    cfg = GenerativeInverseConfig(
        dataset_path=dataset_path,
        output_dir=tmp_path / "runs",
        ranking={"error_metric": "bands_2_10_mre"},
    )
    target = torch.tensor([1.0e-8, 1.0, 1.0, 1.0e-8, 1.0, 1.0], dtype=torch.float32)
    predictions = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0e-8, 2.0, 2.0, 1.0e-8, 2.0, 2.0],
        ],
        dtype=torch.float32,
    )

    errors, metrics = _surrogate_error_tensor(predictions, target, cfg, torch)

    assert int(torch.argmin(errors).item()) == 0
    assert metrics["best_of_k_surrogate_mre_bands_2_10"] == pytest.approx(0.0)
    assert metrics["best_of_k_surrogate_mre_full"] > 0.0


def test_objective_loss_uses_bands_2_10_and_keeps_gradients() -> None:
    torch = pytest.importorskip("torch")
    target = torch.tensor([[1.0e-8, 1.0, 2.0, 1.0e-8, 1.0, 2.0]], dtype=torch.float32)
    predictions = torch.tensor(
        [[10.0, 1.0, 2.0, 10.0, 1.5, 2.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    candidates = torch.full((1, 4), 0.5, dtype=torch.float32, requires_grad=True)

    loss = band_relative_error_loss(predictions, target, torch=torch, n_bands=3, band_start_one_based=2)
    assert loss.item() == pytest.approx(0.125)

    total, metrics = objective_loss_components(
        predictions=predictions,
        target=target,
        candidates=candidates,
        reconstruction=candidates,
        reference=torch.zeros_like(candidates),
        torch=torch,
        n_bands=3,
        objective={"surrogate_weight": 1.0, "reconstruction_weight": 0.1, "binarization_weight": 0.01},
    )
    total.backward()
    assert predictions.grad is not None
    assert candidates.grad is not None
    assert metrics["surrogate_objective_loss"] == pytest.approx(0.125)
    assert metrics["objective_total_loss"] > metrics["surrogate_objective_loss"]


def test_cross_surrogate_consistency_loss_uses_band_window() -> None:
    torch = pytest.importorskip("torch")
    primary = torch.tensor([[100.0, 1.0, 2.0, 100.0, 2.0, 4.0]], dtype=torch.float32)
    secondary = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 2.0]], dtype=torch.float32)

    loss = cross_surrogate_consistency_loss(
        primary,
        secondary,
        torch=torch,
        n_bands=3,
        band_start_one_based=2,
    )

    assert loss.item() == pytest.approx(0.75)


def test_gap_descriptor_loss_and_soft_descriptor_are_differentiable() -> None:
    torch = pytest.importorskip("torch")
    target = torch.tensor(
        [[0.1, 0.2, 0.9, 0.1, 0.3, 1.0, 0.1, 0.4, 1.1]],
        dtype=torch.float32,
    )
    predictions = target.clone().detach().requires_grad_(True)

    descriptor = soft_gap_descriptor(predictions, torch=torch, n_bands=3, band_start_one_based=2)
    loss = gap_descriptor_loss(predictions, target, torch=torch, n_bands=3, band_start_one_based=2)
    loss.backward()

    assert descriptor.shape == (1, 1)
    assert loss.item() == pytest.approx(0.0)
    assert predictions.grad is not None


def test_candidate_diversity_loss_penalizes_collapsed_groups() -> None:
    torch = pytest.importorskip("torch")
    collapsed = torch.full((4, 3), 0.5, dtype=torch.float32, requires_grad=True)
    diverse = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    collapsed_loss, collapsed_value = candidate_diversity_loss(
        collapsed,
        torch=torch,
        target_diversity=0.2,
        group_size=4,
    )
    diverse_loss, diverse_value = candidate_diversity_loss(
        diverse,
        torch=torch,
        target_diversity=0.2,
        group_size=4,
    )

    assert collapsed_value.item() == pytest.approx(0.0)
    assert collapsed_loss.item() == pytest.approx(0.2)
    assert diverse_value.item() > 0.2
    assert diverse_loss.item() == pytest.approx(0.0)


def test_binary_diversity_loss_penalizes_threshold_collapsed_candidates() -> None:
    torch = pytest.importorskip("torch")
    collapsed = torch.tensor(
        [[0.51, 0.52, 0.53], [0.54, 0.55, 0.56], [0.57, 0.58, 0.59], [0.60, 0.61, 0.62]],
        dtype=torch.float32,
        requires_grad=True,
    )
    diverse = torch.tensor(
        [[0.1, 0.1, 0.1], [0.9, 0.9, 0.9], [0.1, 0.9, 0.1], [0.9, 0.1, 0.9]],
        dtype=torch.float32,
    )

    collapsed_loss, collapsed_value = binary_candidate_diversity_loss(
        collapsed,
        torch=torch,
        target_diversity=0.2,
        temperature=0.02,
        group_size=4,
    )
    diverse_loss, diverse_value = binary_candidate_diversity_loss(
        diverse,
        torch=torch,
        target_diversity=0.2,
        temperature=0.02,
        group_size=4,
    )
    collapsed_loss.backward()

    assert collapsed_value.item() < 0.2
    assert collapsed_loss.item() > 0.0
    assert diverse_value.item() > 0.2
    assert diverse_loss.item() == pytest.approx(0.0)
    assert collapsed.grad is not None


def test_latent_response_loss_penalizes_noise_insensitive_groups() -> None:
    torch = pytest.importorskip("torch")
    collapsed = torch.full((4, 3), 0.5, dtype=torch.float32, requires_grad=True)
    responsive = torch.tensor(
        [[0.1, 0.2, 0.3], [0.9, 0.8, 0.7], [0.2, 0.8, 0.2], [0.8, 0.2, 0.8]],
        dtype=torch.float32,
    )

    collapsed_loss, collapsed_std = latent_response_loss(
        collapsed,
        torch=torch,
        target_std=0.05,
        group_size=4,
    )
    responsive_loss, responsive_std = latent_response_loss(
        responsive,
        torch=torch,
        target_std=0.05,
        group_size=4,
    )
    collapsed_loss.backward()

    assert collapsed_std.item() == pytest.approx(0.0)
    assert collapsed_loss.item() == pytest.approx(0.05)
    assert responsive_std.item() > 0.05
    assert responsive_loss.item() == pytest.approx(0.0)
    assert collapsed.grad is not None


def test_quality_weighted_collision_loss_prioritizes_high_quality_duplicates() -> None:
    torch = pytest.importorskip("torch")
    candidates = torch.tensor(
        [
            [0.9, 0.9, 0.1, 0.1],
            [0.9, 0.9, 0.1, 0.1],
            [0.1, 0.1, 0.9, 0.9],
            [0.1, 0.9, 0.1, 0.9],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.ones((4, 6), dtype=torch.float32)
    high_quality_predictions = torch.ones((4, 6), dtype=torch.float32)
    low_quality_predictions = torch.full((4, 6), 2.0, dtype=torch.float32)
    low_quality_predictions[:2] = 1.0

    loss, nn_mean, nn_min, nn_p10, quality_mean = quality_weighted_collision_loss(
        candidates,
        predictions=low_quality_predictions,
        target=target,
        torch=torch,
        n_bands=3,
        target_nn_hamming=0.2,
        group_size=4,
        quality_temperature=0.03,
    )
    uniform_loss, *_ = quality_weighted_collision_loss(
        candidates,
        predictions=high_quality_predictions,
        target=target,
        torch=torch,
        n_bands=3,
        target_nn_hamming=0.2,
        group_size=4,
        quality_temperature=0.03,
    )
    loss.backward()

    assert loss.item() > uniform_loss.item()
    assert nn_min.item() == pytest.approx(0.0)
    assert nn_p10.item() == pytest.approx(0.0)
    assert nn_mean.item() > 0.0
    assert quality_mean.item() == pytest.approx(1.0)
    assert candidates.grad is not None


def test_objective_loss_combines_cross_gap_and_diversity_terms() -> None:
    torch = pytest.importorskip("torch")
    target = torch.tensor(
        [
            [0.1, 0.2, 0.9, 0.1, 0.3, 1.0, 0.1, 0.4, 1.1],
            [0.1, 0.2, 0.9, 0.1, 0.3, 1.0, 0.1, 0.4, 1.1],
        ],
        dtype=torch.float32,
    )
    predictions = target.clone().detach().requires_grad_(True)
    secondary = (target * 1.1).clone().detach().requires_grad_(True)
    candidates = torch.tensor([[0.2, 0.2], [0.8, 0.8]], dtype=torch.float32, requires_grad=True)

    total, metrics = objective_loss_components(
        predictions=predictions,
        secondary_predictions=secondary,
        target=target,
        candidates=candidates,
        reconstruction=candidates,
        reference=torch.zeros_like(candidates),
        torch=torch,
        n_bands=3,
        objective={
            "surrogate_weight": 1.0,
            "cross_surrogate_weight": 0.5,
            "gap_weight": 0.5,
            "diversity_weight": 0.5,
            "target_diversity_hamming": 0.2,
            "candidate_group_size": 2,
            "binary_diversity_weight": 0.5,
            "target_binary_diversity_hamming": 1.2,
            "soft_binary_temperature": 0.05,
            "latent_response_weight": 0.5,
            "target_latent_response_std": 0.7,
            "reconstruction_weight": 0.01,
        },
    )
    total.backward()

    assert metrics["cross_surrogate_disagreement_bands_2_10"] > 0.0
    assert "gap_descriptor_error" in metrics
    assert metrics["diversity_hamming"] > 0.0
    assert metrics["binary_diversity_hamming"] > 0.0
    assert metrics["binary_diversity_loss"] > 0.0
    assert metrics["latent_response_std"] > 0.0
    assert metrics["latent_response_loss"] > 0.0
    assert predictions.grad is not None
    assert candidates.grad is not None


def test_direct_inverse_train_writes_direct_generator_checkpoint(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    _ = torch
    dataset_path = tmp_path / "tiny_direct.npz"
    np.savez(
        dataset_path,
        rho=np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4),
        Wh=np.linspace(0.1, 1.0, 36, dtype=np.float32).reshape(3, 4, 3),
        idx_train=np.array([0, 1], dtype=np.int64),
        idx_val=np.array([], dtype=np.int64),
        idx_test=np.array([2], dtype=np.int64),
    )
    config_path = tmp_path / "direct.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"experiment_id": "unit_direct_inverse"},
                "dataset": {"path": str(dataset_path), "output_repr": "rho_wedge"},
                "model": {"name": "direct_inverse_mlp", "seed": 0, "hidden_dim": 8, "latent_dim": 1},
                "runtime": {"device": "cpu", "require_accelerator": False},
                "inverse": {"train_epochs": 2, "train_batch_size": 2, "n_candidates": 1},
                "output": {"runs_dir": str(tmp_path / "runs")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    run_dir = run_inverse_train(config_path)

    assert (run_dir / "model.pt").exists()
    payload = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    assert payload["generator_kind"] == "direct_inverse_mlp"
    assert (run_dir / "history.csv").exists()
    assert (run_dir / "metrics.json").exists()


def test_mixture_conditional_generator_train_writes_mode_payload(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    dataset_path = tmp_path / "tiny_mixture.npz"
    np.savez(
        dataset_path,
        rho=np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4),
        Wh=np.linspace(0.1, 1.0, 36, dtype=np.float32).reshape(3, 4, 3),
        idx_train=np.array([0, 1], dtype=np.int64),
        idx_val=np.array([], dtype=np.int64),
        idx_test=np.array([2], dtype=np.int64),
    )
    config_path = tmp_path / "mixture.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"experiment_id": "unit_mixture_inverse"},
                "dataset": {"path": str(dataset_path), "output_repr": "rho_wedge"},
                "model": {
                    "name": "mixture_conditional_generator",
                    "seed": 0,
                    "hidden_dim": 8,
                    "latent_dim": 2,
                    "mode_count": 3,
                    "depth": 2,
                },
                "runtime": {"device": "cpu", "require_accelerator": False},
                "inverse": {
                    "train_epochs": 1,
                    "checkpoint_epochs": [1],
                    "learning_rates": [0.001, 0.0005],
                    "train_batch_size": 2,
                    "n_candidates": 6,
                },
                "output": {"runs_dir": str(tmp_path / "runs")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    run_dir = run_inverse_train(config_path)

    payload = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    assert payload["generator_kind"] == "mixture_conditional_generator"
    assert payload["mode_count"] == 3
    assert payload["depth"] == 2
    assert payload["learning_rate"] == pytest.approx(0.001)
    assert {"condition_encoder", "mode_embedding", "decoder"}.issubset(payload["state_dict"])
    assert (run_dir / "checkpoint_epoch_1_seed0_lr1em03.pt").exists()
    assert (run_dir / "checkpoint_epoch_1_seed0_lr5em04.pt").exists()
    assert (run_dir / "model_seed0_lr1em03.pt").exists()
    assert (run_dir / "model_seed0_lr5em04.pt").exists()


def test_inverse_objective_train_writes_surrogate_side_boundary(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    _ = torch
    dataset_path = tmp_path / "tiny_objective.npz"
    np.savez(
        dataset_path,
        rho=np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4),
        Wh=np.linspace(0.1, 1.0, 36, dtype=np.float32).reshape(3, 4, 3),
        idx_train=np.array([0, 1], dtype=np.int64),
        idx_val=np.array([], dtype=np.int64),
        idx_test=np.array([2], dtype=np.int64),
    )
    config_path = tmp_path / "objective.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"experiment_id": "unit_objective_inverse"},
                "dataset": {"path": str(dataset_path), "output_repr": "rho_wedge"},
                "model": {"name": "conditional_vae_rho", "seed": 0, "hidden_dim": 8, "latent_dim": 2},
                "runtime": {"device": "cpu", "require_accelerator": False},
                "inverse": {
                    "train_epochs": 2,
                    "train_batch_size": 2,
                    "n_candidates": 2,
                    "allow_mock_surrogate": True,
                    "ranking": {
                        "error_metric": "bands_2_10_mre",
                        "objective": {"surrogate_weight": 1.0, "reconstruction_weight": 0.01},
                    },
                },
                "output": {"runs_dir": str(tmp_path / "runs")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    run_dir = run_inverse_objective_train(config_path)

    metrics = yaml.safe_load((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["validation_boundary"] == "inverse_objective_training_surrogate_side_no_mpb_meep"
    assert metrics["surrogate_side_no_mpb_meep"] is True
    assert metrics["final_objective_total_loss_mean"] > 0.0
    assert (run_dir / "objective_model_seed0.pt").exists()


def test_inverse_objective_train_requires_secondary_surrogate_for_cross_loss(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    _ = torch
    dataset_path = tmp_path / "tiny_missing_secondary.npz"
    np.savez(
        dataset_path,
        rho=np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4),
        Wh=np.linspace(0.1, 1.0, 36, dtype=np.float32).reshape(3, 4, 3),
        idx_train=np.array([0, 1], dtype=np.int64),
        idx_val=np.array([], dtype=np.int64),
        idx_test=np.array([2], dtype=np.int64),
    )
    config_path = tmp_path / "objective_missing_secondary.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"experiment_id": "unit_objective_missing_secondary"},
                "dataset": {"path": str(dataset_path), "output_repr": "rho_wedge"},
                "model": {"name": "conditional_vae_rho", "seed": 0, "hidden_dim": 8, "latent_dim": 2},
                "runtime": {"device": "cpu", "require_accelerator": False},
                "inverse": {
                    "train_epochs": 1,
                    "train_batch_size": 2,
                    "n_candidates": 2,
                    "allow_mock_surrogate": True,
                    "ranking": {
                        "error_metric": "bands_2_10_mre",
                        "objective": {"surrogate_weight": 1.0, "cross_surrogate_weight": 0.5},
                    },
                },
                "output": {"runs_dir": str(tmp_path / "runs")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cross-surrogate objective requires"):
        run_inverse_objective_train(config_path)
