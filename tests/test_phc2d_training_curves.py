"""Tests for publication learning-curve reconstruction.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from phc2d.plot import training_curves


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _field_rows(count: int) -> list[dict[str, object]]:
    return [
        {
            "epoch": epoch,
            "train_data_loss": 1.0 / epoch,
            "train_dense_frequency_loss": 0.8 / epoch,
            "train_physics_loss": 0.6 / epoch,
            "train_normalization_loss": 0.4 / epoch,
            "train_rayleigh_consistency_loss": 0.2 / epoch,
            "val_data_loss": 1.2 / epoch,
            "lambda_pde_active": 0.1,
            "learning_rate": 1.0e-4,
        }
        for epoch in range(1, count + 1)
    ]


def test_build_learning_curve_package(monkeypatch, tmp_path: Path) -> None:
    def fake_save(figure, output_base, formats, dpi=300):
        del figure, dpi
        paths = []
        output_base.parent.mkdir(parents=True, exist_ok=True)
        for suffix in formats:
            path = output_base.with_suffix(f".{suffix}")
            path.write_bytes(b"figure")
            paths.append(path)
        return paths

    monkeypatch.setattr(training_curves, "_save_figure", fake_save)
    a1_forward = tmp_path / "a1_forward"
    a1_mlp = tmp_path / "a1_mlp"
    a1_query = tmp_path / "a1_query"
    a2_original = tmp_path / "a2_original"
    a2_bridge = tmp_path / "a2_bridge"
    field_b3 = tmp_path / "field_b3"
    p45 = tmp_path / "p45"
    p6 = tmp_path / "p6"
    for path in (
        a1_forward,
        a1_mlp,
        a1_query,
        a2_original,
        a2_bridge,
        field_b3,
        p45,
        p6,
    ):
        path.mkdir(parents=True)

    _write_csv(
        a1_forward / "validation_history.csv",
        [
            {
                "case_id": f"case_seed{seed}",
                "model": "fno2d_band_v2",
                "learning_rate": 1.0e-4,
                "seed": seed,
            }
            for seed in range(2)
        ],
    )
    for seed in range(2):
        _write_csv(
            a1_forward / "cases" / f"case_seed{seed}" / "train" / "history.csv",
            [
                {
                    "epoch": epoch,
                    "train_loss": 1.0 / (epoch + seed),
                    "val_mse": 0.5 / epoch,
                    "val_mae": 0.4 / epoch,
                    "val_rmse": 0.3 / epoch,
                }
                for epoch in range(1, 3)
            ],
        )
    _write_csv(
        a1_mlp / "validation_history.csv",
        [
            {
                "case_id": f"mlp_seed{seed}",
                "model": "mlp",
                "learning_rate": 1.0e-4,
                "seed": seed,
            }
            for seed in range(2)
        ],
    )
    for seed in range(2):
        _write_csv(
            a1_mlp / "cases" / f"mlp_seed{seed}" / "train" / "history.csv",
            [
                {
                    "epoch": epoch,
                    "train_loss": 1.1 / (epoch + seed),
                    "val_mse": 0.6 / epoch,
                    "val_mae": 0.5 / epoch,
                    "val_rmse": 0.4 / epoch,
                }
                for epoch in range(1, 3)
            ],
        )
    _write_csv(a1_query / "history.csv", [{"case_id": "query", "query_rmse": 0.1}])

    for method in training_curves.METHOD_LABELS:
        _write_csv(
            a2_original / "train" / method / "history.csv",
            [
                {
                    "stage": "train",
                    "seed": 0,
                    "learning_rate": lr,
                    "epoch": epoch,
                    "train_loss": 1.0 / epoch,
                }
                for lr in (1.0e-3, 1.0e-4)
                for epoch in range(1, 3)
            ],
        )

    bridge_metrics = (
        "objective_total_loss",
        "surrogate_objective_loss",
        "reconstruction_loss",
        "diversity_loss",
        "binary_diversity_loss",
        "latent_response_loss",
        "quality_weighted_collision_loss",
        "cross_surrogate_disagreement_bands_2_10",
        "gap_descriptor_error",
    )
    for method in ("single_mode_generator", "mixture_generator"):
        _write_csv(
            a2_bridge / "train" / method / "history.csv",
            [
                {
                    "epoch": epoch,
                    **{metric: (position + 1) / epoch for position, metric in enumerate(bridge_metrics)},
                }
                for epoch in range(1, 3)
            ],
        )

    for prefix in ("P4_dense_k_multires", "P5_rayleigh_consistent"):
        for seed in range(2):
            _write_csv(p45 / "cases" / f"{prefix}_seed{seed}" / "history.csv", _field_rows(2))
    for prefix in ("unet_field_v2", "fno_field_v2"):
        for seed in range(2):
            _write_csv(
                field_b3 / "cases" / f"{prefix}_seed{seed}" / "history.csv",
                _field_rows(2),
            )
    _write_csv(p6 / "cases" / "P6A_target_pde_seed0" / "history.csv", _field_rows(2))
    _write_csv(p6 / "cases" / "P6B_target_pde_rayleigh_seed0" / "history.csv", _field_rows(1))

    output = tmp_path / "output"
    result = training_curves.build_learning_curve_package(
        a1_forward_run=a1_forward,
        a1_mlp_run=a1_mlp,
        a1_query_run=a1_query,
        a2_original_run=a2_original,
        a2_bridge_run=a2_bridge,
        p45_run=p45,
        p6_run=p6,
        field_b3_run=field_b3,
        output_dir=output,
        formats=("png",),
    )
    assert result["status"] == "learning_curve_package_completed"
    assert result["figure_files"] == 6
    assert result["missing_epoch_histories"] == 1
    assert result["partial_histories"] >= 1
    for stem in (
        "a1_forward_learning_curves",
        "b3_field_learning_curves",
        "a2_original_learning_curves",
        "a2_objective_bridge_learning_curves",
        "p4_p5_learning_curves",
        "p6_learning_curves",
    ):
        assert (output / "figures" / f"{stem}.png").exists()
    manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["coverage"]["missing_epoch_history_count"] == 1
    coverage = (output / "curve_coverage.csv").read_text(encoding="utf-8")
    assert "mlp_seed0" in coverage
    assert "unet_field_v2_seed0" in coverage
    assert "fno_field_v2_seed0" in coverage
    assert (output / "curve_coverage.csv").exists()
    assert (output / "README.md").exists()
    for standard_name in (
        "run.log",
        "run.meta.json",
        "config.resolved.yaml",
        "results.json",
        "metrics.json",
        "history.csv",
    ):
        assert (output / standard_name).exists()
