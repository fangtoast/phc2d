"""Tests for full-cell inverse Stage-A data and preflight contracts.

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

from phc2d.cli import build_parser
from phc2d.data import build_full_bz_grid, write_full_cell_hdf5
from phc2d.generators import (
    generate_binary_polar_holes,
    generate_binary_spectral_threshold,
    generate_graded_rbf_fourier,
)
from phc2d.inverse.full_cell_training import (
    FULL_CELL_STAGE_A_METHODS,
    load_full_cell_stage_a_preflight_config,
    load_full_cell_stage_a_train_config,
    load_full_cell_inverse_variant_datasets,
    run_full_cell_stage_a_preflight,
    run_full_cell_stage_a_train,
)
from phc2d.runs import validate_run_artifacts


def _generator_config(cutoff: int) -> dict[str, object]:
    return {
        "epsilon_air": 1.0,
        "epsilon_material": 8.9,
        "audit_resolution": 64,
        "fourier_cutoff": cutoff,
        "max_attempts": 512,
        "spectral_cutoff_scale_range": [2.5, 4.5],
    }


@pytest.fixture(scope="module")
def tiny_full_cell_routes(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    pytest.importorskip("h5py")
    root = tmp_path_factory.mktemp("full_cell_inverse_stage_a")
    points, grid_indices = build_full_bz_grid(3)
    routes: dict[str, Path] = {}
    for route in ("binary", "graded"):
        structures = []
        for repetition in range(4):
            if route == "binary":
                structures.extend(
                    [
                        generate_binary_polar_holes(
                            f"polar1_{repetition}",
                            master_seed=100 + repetition,
                            hole_count=1,
                            config=_generator_config(8),
                        ),
                        generate_binary_polar_holes(
                            f"polar2_{repetition}",
                            master_seed=200 + repetition,
                            hole_count=2,
                            config=_generator_config(8),
                        ),
                        generate_binary_spectral_threshold(
                            f"spectral_{repetition}",
                            master_seed=300 + repetition,
                            config=_generator_config(8),
                        ),
                    ]
                )
            else:
                structures.extend(
                    [
                        generate_graded_rbf_fourier(
                            f"graded1_{repetition}",
                            master_seed=400 + repetition,
                            core_count=1,
                            config=_generator_config(3),
                        ),
                        generate_graded_rbf_fourier(
                            f"graded2_{repetition}",
                            master_seed=500 + repetition,
                            core_count=2,
                            config=_generator_config(3),
                        ),
                    ]
                )
        stride = 3 if route == "binary" else 2
        train = np.asarray([index for index in range(len(structures)) if index // stride < 2])
        val = np.asarray([index for index in range(len(structures)) if index // stride == 2])
        test = np.asarray([index for index in range(len(structures)) if index // stride == 3])
        wh = np.arange(len(structures) * 9 * 3, dtype=np.float32).reshape(len(structures), 9, 3)
        wh[val] += 1.0e6
        wh[test] += 2.0e6
        path = root / f"{route}.h5"
        write_full_cell_hdf5(
            path,
            structures=structures,
            Wh=wh,
            k_points=points,
            k_grid_index=grid_indices,
            idx_train=train,
            idx_val=val,
            idx_test=test,
            raster_resolutions=[8],
            metadata={"route": route},
        )
        routes[route] = path
    return routes


def test_variant_data_adapter_uses_route_train_only_normalization(tiny_full_cell_routes) -> None:
    datasets = load_full_cell_inverse_variant_datasets(
        tiny_full_cell_routes["binary"],
        variants=(
            "binary_polar_holes_1",
            "binary_polar_holes_2",
            "binary_spectral_threshold",
        ),
    )

    assert tuple(datasets) == (
        "binary_polar_holes_1",
        "binary_polar_holes_2",
        "binary_spectral_threshold",
    )
    first = datasets["binary_polar_holes_1"]
    assert first.idx_train.size == 2
    assert first.idx_val.size == 1
    assert first.idx_test.size == 1
    assert first.Wh.shape == (4, 9, 3)
    assert first.encoded_parameters.shape == (4, 13)
    assert float(np.max(first.condition_mean)) < 1.0e6
    normalized = first.normalized_conditions(first.idx_train)
    assert normalized.shape == (2, 27)
    assert np.all(np.isfinite(normalized))


def test_stage_a_cpu_preflight_writes_15_complete_cases(tiny_full_cell_routes, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    config = {
        "experiment": {"experiment_id": "full_cell_inverse_stage_a_unit"},
        "representation": {
            "condition_target": "Wh",
            "output_repr": "full_cell_analytic_params",
            "condition_normalization": "route_train_per_k_band",
        },
        "datasets": {
            "binary": {"path": str(tiny_full_cell_routes["binary"])},
            "graded": {"path": str(tiny_full_cell_routes["graded"])},
        },
        "models": {
            "methods": list(FULL_CELL_STAGE_A_METHODS),
            "overrides": {
                "direct_inverse_mlp": {"hidden_dim": 8, "depth": 2, "latent_dim": 0, "mode_count": 1},
                "single_mode_generator": {"hidden_dim": 8, "depth": 2, "latent_dim": 4, "mode_count": 1},
                "mixture_conditional_generator": {"hidden_dim": 8, "depth": 2, "latent_dim": 4, "mode_count": 4},
            },
        },
        "training": {
            "seed": 0,
            "batch_size": 2,
            "raster_shape": [8, 8],
            "binary_temperature": 0.01,
            "learning_rate": 1.0e-4,
            "weight_decay": 1.0e-5,
            "raster_weight": 0.1,
            "unit_circle_weight": 0.05,
        },
        "runtime": {
            "device": "cpu",
            "require_accelerator": False,
            "precision": "fp32",
            "amp_enabled": False,
            "tf32_enabled": False,
            "cpu_threads": 2,
        },
        "output": {"runs_dir": str(tmp_path)},
    }
    config_path = tmp_path / "preflight.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    run_dir = run_full_cell_stage_a_preflight(config_path, run_dir=tmp_path / "run")

    assert validate_run_artifacts(run_dir) == []
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    with (run_dir / "history.csv").open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    assert results["status"] == "cpu_contract_preflight_pass"
    assert results["formal_training_started"] is False
    assert results["mpb_meep_used"] is False
    assert len(results["case_results"]) == 15
    assert len(history) == 15
    assert metrics["all_gradients_finite_nonzero"] is True
    assert metrics["all_checkpoint_reloads_exact"] is True
    assert (run_dir / "codec.resolved.json").is_file()
    assert (run_dir / "condition_normalization_binary.npz").is_file()
    assert (run_dir / "condition_normalization_graded.npz").is_file()
    assert len(list((run_dir / "cases").glob("*/*/preflight.pt"))) == 15


def test_full_cell_preflight_cli_help_is_lazy(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["inverse", "full-cell-preflight", "--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "15-case full-cell analytic Stage-A" in output
    assert "does not start formal training" in output


def _stage_a_train_config(tiny_full_cell_routes: dict[str, Path], tmp_path: Path) -> Path:
    config = {
        "experiment": {
            "experiment_id": "full_cell_inverse_stage_a_train_unit",
            "stage": "inverse_full_cell_stage_a_train",
        },
        "representation": {
            "condition_target": "Wh",
            "output_repr": "full_cell_analytic_params",
            "condition_normalization": "route_train_per_k_band",
        },
        "datasets": {
            "binary": {"path": str(tiny_full_cell_routes["binary"])},
            "graded": {"path": str(tiny_full_cell_routes["graded"])},
        },
        "models": {
            "methods": list(FULL_CELL_STAGE_A_METHODS),
            "overrides": {
                "direct_inverse_mlp": {
                    "hidden_dim": 8,
                    "depth": 2,
                    "latent_dim": 0,
                    "mode_count": 1,
                },
                "single_mode_generator": {
                    "hidden_dim": 8,
                    "depth": 2,
                    "latent_dim": 4,
                    "mode_count": 1,
                },
                "mixture_conditional_generator": {
                    "hidden_dim": 8,
                    "depth": 2,
                    "latent_dim": 4,
                    "mode_count": 4,
                },
            },
        },
        "training": {
            "epochs": 3,
            "checkpoint_epochs": [1, 2, 3],
            "early_stopping_patience": 3,
            "batch_size": 2,
            "val_batch_size": 2,
            "raster_shape": [8, 8],
            "binary_temperature": 0.01,
            "learning_rates": {
                "direct_inverse_mlp": 3.0e-4,
                "single_mode_generator": 1.0e-4,
                "mixture_conditional_generator": 1.0e-4,
            },
            "weight_decay": 1.0e-5,
            "raster_weight": 0.1,
            "unit_circle_weight": 0.05,
        },
        "gates": {"forward_surrogate_status": "not_trained"},
        "runtime": {
            "device": "cpu",
            "require_accelerator": False,
            "precision": "fp32",
            "amp_enabled": False,
            "tf32_enabled": False,
            "cpu_threads": 2,
        },
        "output": {"runs_dir": str(tmp_path)},
    }
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_stage_a_epoch_runner_resumes_with_optimizer_and_writes_curves(
    tiny_full_cell_routes: dict[str, Path], tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    config_path = _stage_a_train_config(tiny_full_cell_routes, tmp_path)
    run_dir = tmp_path / "train_run"

    run_full_cell_stage_a_train(
        config_path,
        variant="binary_polar_holes_1",
        method="direct_inverse_mlp",
        seed=0,
        run_dir=run_dir,
        stop_after_epoch=1,
    )
    partial = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert partial["status"] == "stage_a_train_partial"
    assert partial["next_gate"] == "resume_same_run_dir"

    resumed_dir = run_full_cell_stage_a_train(
        config_path,
        variant="binary_polar_holes_1",
        method="direct_inverse_mlp",
        seed=0,
        run_dir=run_dir,
    )

    assert resumed_dir == run_dir
    assert validate_run_artifacts(run_dir) == []
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    with (run_dir / "history.csv").open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    assert [int(row["epoch"]) for row in history] == [1, 2, 3]
    assert results["status"] == "stage_a_train_complete"
    assert results["formal_training_started"] is False
    assert results["forward_surrogate_status"] == "not_trained"
    assert results["stage_b_admitted"] is False
    assert results["candidate_ranking_admitted"] is False
    assert results["inverse_mpb_replay_admitted"] is False
    assert results["test_split_touched"] is False
    assert metrics["resumed_from_checkpoint"] is True
    assert metrics["checkpoint_reload_exact_match"] is True
    assert metrics["train_time_s"] > 0.0
    assert metrics["eval_time_s"] > 0.0
    assert (run_dir / "last.pt").is_file()
    assert (run_dir / "best.pt").is_file()
    assert (run_dir / "model.pt").is_file()
    assert len(list(run_dir.glob("checkpoint_epoch_*.pt"))) == 3
    assert (run_dir / "figures" / "training_curve.png").is_file()
    assert (run_dir / "figures" / "training_curve.pdf").is_file()


def test_full_cell_train_cli_help_is_lazy(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["inverse", "full-cell-train", "--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "one supervised full-cell analytic Stage-A job" in output
    assert "does not use a forward surrogate" in output


def test_repository_cpu_and_n40_preflight_configs_are_fail_closed() -> None:
    root = Path(__file__).parents[1]
    cpu = load_full_cell_stage_a_preflight_config(
        root / "configs" / "smoke" / "full_cell_inverse_stage_a_cpu_v1.yaml"
    )
    n40 = load_full_cell_stage_a_preflight_config(
        root / "configs" / "research" / "full_cell_inverse_stage_a_n40_preflight_v1.yaml"
    )

    assert cpu["runtime"]["device"] == "cpu"
    assert cpu["training"]["batch_size"] == 4
    assert n40["runtime"]["device"] == "cuda:0"
    assert n40["runtime"]["require_accelerator"] is True
    assert n40["training"]["batch_size"] == 64
    assert n40["training"]["raster_shape"] == [64, 64]


def test_repository_stage_a_train_configs_are_fail_closed() -> None:
    root = Path(__file__).parents[1]
    cpu = load_full_cell_stage_a_train_config(
        root / "configs" / "smoke" / "full_cell_inverse_stage_a_train_cpu_v1.yaml"
    )
    n40 = load_full_cell_stage_a_train_config(
        root / "configs" / "research" / "full_cell_inverse_stage_a_train_n40_v1.yaml"
    )

    assert cpu["runtime"]["device"] == "cpu"
    assert cpu["training"]["epochs"] == 3
    assert n40["runtime"]["device"] == "cuda:0"
    assert n40["runtime"]["require_accelerator"] is True
    assert n40["training"]["epochs"] == 200
    assert n40["training"]["checkpoint_epochs"] == [50, 100, 150, 200]
    assert n40["gates"]["forward_surrogate_status"] == "not_trained"
