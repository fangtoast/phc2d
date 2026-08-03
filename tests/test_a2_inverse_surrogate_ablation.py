"""Tests for the A2 seven-surrogate inverse-design ablation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from phc2d.data import load_forward_dataset
from phc2d.inverse.benchmark import (
    SURROGATE_ABLATION_MODELS,
    _promotion_choice,
    _spearman_permutation,
    _surrogate_ablation_rerank_config,
    _surrogate_ablation_sample_config,
    _surrogate_objective_arm_config,
    _surrogate_variants,
    run_a2_surrogate_derive_objective,
    run_a2_surrogate_preflight,
    run_a2_surrogate_promote,
)
from phc2d.inverse.config import GenerativeInverseConfig, load_inverse_run_config
from phc2d.inverse.runner import _build_surrogate, _load_targets


REPO_ROOT = Path(__file__).resolve().parents[1]
SCREEN_CONFIG = REPO_ROOT / "configs" / "research" / "a2_inverse_surrogate_screen_v1.yaml"
OBJECTIVE_CONFIG = REPO_ROOT / "configs" / "research" / "a2_inverse_surrogate_objective_v1.yaml"


def _read_yaml(path: Path) -> dict:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_surrogate_matrix_has_exactly_seven_approved_seed0_variants() -> None:
    config = _read_yaml(SCREEN_CONFIG)
    variants = _surrogate_variants(config)

    assert tuple(item["id"] for item in variants) == SURROGATE_ABLATION_MODELS
    assert [item["kind"] for item in variants].count("forward") == 4
    assert [item["kind"] for item in variants].count("query") == 3
    assert all(int(item["seed"]) == 0 for item in variants)
    assert all(item.get("sha256") for item in variants)
    for variant in variants:
        if variant["kind"] == "query":
            assert variant["protocol"] == "unseen_structure_all_k"
            assert variant["k_path_sha256"] == config["dataset"]["k_path_sha256"]


def test_derived_sampling_and_rerank_configs_share_one_candidate_pool(tmp_path) -> None:
    config = _read_yaml(SCREEN_CONFIG)
    candidate_dir = tmp_path / "candidates"
    sample = _surrogate_ablation_sample_config(config, run_dir=candidate_dir)
    rerank = _surrogate_ablation_rerank_config(
        config,
        candidates_path=candidate_dir / "candidates.json",
        run_dir=tmp_path / "screen",
    )

    assert sample["dataset"]["target_split"] == "val"
    assert sample["dataset"]["target_manifest_path"] == config["dataset"]["target_manifest_path"]
    assert sample["inverse"]["n_candidates"] == 128
    assert sample["inverse"]["generator_checkpoint_path"] == (
        config["candidate_generation"]["generator_checkpoint"]["path"]
    )
    assert rerank["rerank"]["candidates_path"] == str(candidate_dir / "candidates.json")
    assert [item["id"] for item in rerank["rerank"]["strategies"]] == list(
        SURROGATE_ABLATION_MODELS
    )
    assert rerank["rerank"]["top_m"] == 5


def test_target_manifest_is_exact_and_cannot_be_truncated(tmp_path) -> None:
    dataset_path = tmp_path / "tiny.npz"
    np.savez(
        dataset_path,
        rho=np.zeros((6, 3), dtype=np.float32),
        Wh=np.ones((6, 2, 2), dtype=np.float32),
        idx_train=np.array([0, 1], dtype=np.int64),
        idx_val=np.array([2, 3, 4], dtype=np.int64),
        idx_test=np.array([5], dtype=np.int64),
    )
    manifest = tmp_path / "val.json"
    manifest.write_text(
        json.dumps({"dataset_split": "val", "target_indices": [4, 2]}),
        encoding="utf-8",
    )
    config_path = tmp_path / "inverse.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "path": str(dataset_path),
                    "target_split": "val",
                    "target_manifest_path": str(manifest),
                },
                "inverse": {"target_limit": 2},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_inverse_run_config(config_path)
    indices, _targets, _candidate_dim, _metadata = _load_targets(cfg)
    assert indices.tolist() == [4, 2]

    limited = GenerativeInverseConfig(
        dataset_path=dataset_path,
        output_dir=tmp_path,
        target_split="val",
        target_manifest_path=manifest,
        target_limit=1,
    )
    with pytest.raises(ValueError, match="cannot truncate an explicit target manifest"):
        _load_targets(limited)


def test_ranking_statistics_and_tie_break_are_deterministic() -> None:
    assert _spearman_permutation([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)
    assert _spearman_permutation([0, 1, 2, 3], [3, 2, 1, 0]) == pytest.approx(-1.0)

    choice = _promotion_choice(
        [
            {
                "strategy_id": "lower_mre_larger_gap",
                "best_of_k_replay_mre_bands_2_10_pct": 2.00,
                "ci_low_pct": 1.8,
                "ci_high_pct": 2.2,
                "surrogate_replay_gap_bands_2_10_pct": 0.50,
                "inference_time_per_sample_ms": 1.0,
            },
            {
                "strategy_id": "overlap_smaller_gap",
                "best_of_k_replay_mre_bands_2_10_pct": 2.05,
                "ci_low_pct": 1.9,
                "ci_high_pct": 2.3,
                "surrogate_replay_gap_bands_2_10_pct": 0.10,
                "inference_time_per_sample_ms": 2.0,
            },
        ]
    )
    assert choice["strategy_id"] == "overlap_smaller_gap"


def test_objective_arm_is_primary_only_and_keeps_common_warm_start(tmp_path) -> None:
    config = _read_yaml(OBJECTIVE_CONFIG)
    variants = {item["id"]: item for item in _surrogate_variants(config)}
    arm = _surrogate_objective_arm_config(
        config,
        variant=variants["mlp_query"],
        run_dir=tmp_path / "mlp_query",
    )
    objective = arm["inverse"]["ranking"]["objective"]

    assert arm["model"]["seeds"] == [0]
    assert arm["inverse"]["train_epochs"] == 50
    assert arm["inverse"]["learning_rate"] == pytest.approx(1.0e-4)
    assert objective["n_objective_candidates"] == 16
    assert objective["cross_surrogate_weight"] == 0.0
    assert "secondary_surrogate" not in objective
    assert arm["inverse"]["generator_checkpoint_path"] == config["training"]["warm_start"]["path"]


def test_all_frequency_objective_derivation_writes_seven_arms(tmp_path) -> None:
    objective_root = run_a2_surrogate_derive_objective(OBJECTIVE_CONFIG, run_dir=tmp_path)
    manifest = json.loads(
        (objective_root / "objective_arm_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["selection_mode"] == "all_frequency_surrogates"
    assert manifest["promotion_decision_path"] is None
    assert [arm["surrogate_id"] for arm in manifest["arms"]] == list(
        SURROGATE_ABLATION_MODELS
    )
    assert all(arm["cross_surrogate_weight"] == 0.0 for arm in manifest["arms"])


def test_preflight_records_physical_k_sweep_and_repeated_gamma(tmp_path) -> None:
    required = [
        REPO_ROOT / item["checkpoint_path"]
        for item in _read_yaml(SCREEN_CONFIG)["surrogate_matrix"]["variants"]
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("validation-selected A1 checkpoints are not present on this workstation")

    run_a2_surrogate_preflight(SCREEN_CONFIG, run_dir=tmp_path)
    lineage = json.loads((tmp_path / "lineage.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "surrogate_manifest.json").read_text(encoding="utf-8"))

    assert lineage["k_query_mode"] == "physical_coordinates_only_no_index_fallback"
    assert lineage["k_point_count"] == 31
    assert lineage["repeated_endpoint_indices"] == [30]
    assert manifest["variant_count"] == 7


@pytest.mark.parametrize("variant_id", SURROGATE_ABLATION_MODELS)
def test_frozen_surrogate_restores_full_wh_and_has_finite_nonzero_gradient(
    variant_id,
    tmp_path,
) -> None:
    torch = pytest.importorskip("torch")
    config = _read_yaml(SCREEN_CONFIG)
    variant = next(item for item in _surrogate_variants(config) if item["id"] == variant_id)
    checkpoint = REPO_ROOT / variant["checkpoint_path"]
    if not checkpoint.is_file():
        pytest.skip(f"validation-selected checkpoint is unavailable: {checkpoint}")

    dataset_path = REPO_ROOT / config["dataset"]["path"]
    dataset = load_forward_dataset(dataset_path)
    surrogate_cfg = {
        "kind": variant["kind"],
        "model_name": variant["model_name"],
        "input_repr": variant["input_repr"],
        "checkpoint_path": str(checkpoint),
        "protocol": variant.get("protocol"),
    }
    inverse_cfg = GenerativeInverseConfig(
        dataset_path=dataset_path,
        output_dir=tmp_path,
        output_repr="rho_wedge",
        allow_mock_surrogate=False,
        surrogate=surrogate_cfg,
    )
    surrogate = _build_surrogate(
        inverse_cfg,
        input_dim=int(dataset.rho.shape[1]),
        target_shape=(int(dataset.nk), int(dataset.nb)),
        torch=torch,
        device=torch.device("cpu"),
    )
    candidates = torch.as_tensor(
        np.asarray(dataset.rho[:1], dtype=np.float32),
        dtype=torch.float32,
    ).clone().requires_grad_(True)
    prediction = surrogate.predict_tensor(candidates)

    assert tuple(prediction.shape) == (1, int(dataset.nk), int(dataset.nb))
    if variant["kind"] == "query":
        assert surrogate.k_path.is_physical is True
        assert surrogate.k_path.source.startswith("sidecar:")
        assert surrogate.k_path.repeated_endpoint_indices == (int(dataset.nk) - 1,)
    prediction[:, :, 1:].mean().backward()
    assert candidates.grad is not None
    assert bool(torch.isfinite(candidates.grad).all())
    assert float(torch.linalg.vector_norm(candidates.grad).item()) > 0.0


def test_promotion_requires_real_mpb_and_selects_three_training_arms(tmp_path) -> None:
    config = _read_yaml(SCREEN_CONFIG)
    strategy_ids = [item["id"] for item in _surrogate_variants(config)]
    replay_values = {
        "mlp": 4.0,
        "res_mlp_wide": 3.0,
        "lifted_cnn": 3.7,
        "fno2d_band_v2": 2.9,
        "mlp_query": 2.8,
        "deep_onet_query_v2": 3.3,
        "fno2d_query_v2": 3.1,
    }
    replay_dir = tmp_path / "pilot_replay"
    replay_dir.mkdir(parents=True)
    (replay_dir / "metrics.json").write_text(
        json.dumps(
            {
                "replay_backend": "real_mpb",
                "validation_boundary": "formal_inverse_candidate_mpb_replay",
                "targets": 20,
                "strategy_count": 7,
                "mpb_replay_success_rate": 1.0,
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        replay_dir / "per_strategy_replay_metrics.csv",
        [
            {
                "strategy_id": strategy_id,
                "target_count": 20,
                "best_of_k_replay_mre_bands_2_10_pct": replay_values[strategy_id],
                "best_of_k_surrogate_mre_bands_2_10_pct": replay_values[strategy_id] - 0.1,
                "surrogate_replay_gap_bands_2_10_pct": 0.1,
            }
            for strategy_id in strategy_ids
        ],
    )
    _write_csv(
        replay_dir / "per_target_replay_metrics.csv",
        [
            {
                "strategy_id": strategy_id,
                "target_index": target_index,
                "best_of_k_replay_mre_bands_2_10_pct": replay_values[strategy_id]
                + 0.01 * target_index,
            }
            for strategy_id in strategy_ids
            for target_index in range(20)
        ],
    )
    _write_csv(
        tmp_path / "screen" / "strategy_metrics.csv",
        [
            {
                "strategy_id": strategy_id,
                "inference_time_per_sample_ms": 1.0,
                "surrogate_ranking_time_s": 0.1,
            }
            for strategy_id in strategy_ids
        ],
    )

    run_a2_surrogate_promote(SCREEN_CONFIG, run_dir=tmp_path)
    decision = json.loads((tmp_path / "promotion_decision.json").read_text(encoding="utf-8"))
    assert decision["promoted_surrogate_ids"] == [
        "fno2d_band_v2",
        "res_mlp_wide",
        "mlp_query",
    ]
    assert decision["formal_claims_authorized"] is False
    assert (tmp_path / "pilot_replay_metrics.csv").is_file()
