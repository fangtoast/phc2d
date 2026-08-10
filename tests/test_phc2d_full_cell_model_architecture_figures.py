"""Tests for the curated Full-Wh model architecture figure renderer.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from pathlib import Path

import pytest

from phc2d.plot import full_cell_model_architectures


def test_model_architecture_renderer_covers_all_ten_figures(monkeypatch, tmp_path: Path) -> None:
    def fake_save(_figure, output_base: Path, formats):
        paths = []
        for suffix in formats:
            path = output_base.with_suffix(f".{suffix}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("figure", encoding="utf-8")
            paths.append(path)
        return paths

    monkeypatch.setattr(full_cell_model_architectures, "_save_figure", fake_save)
    result = full_cell_model_architectures.build_full_cell_model_architecture_figures(
        output_dir=tmp_path,
        formats=("png", "svg"),
    )

    assert result["status"] == "full_cell_model_architecture_figures_completed"
    assert result["figure_count"] == 10
    assert result["artifact_count"] == 20
    assert {Path(path).stem for path in result["artifacts"]} == {
        "figure_2_1_data_generation_pipeline",
        "figure_4_1_baseline_mlp_residual",
        "figure_4_2_baseline_lifted_cnn",
        "figure_4_3_baseline_fno2d_band",
        "figure_5_1_full_wh_heads",
        "figure_5_2_mlp_upgrades",
        "figure_5_3_convnext_band",
        "figure_5_4_tfno_dual_lane",
        "figure_6_1_native_50k_model_comparison",
        "figure_7_1_inverse_stage_a_models",
    }


def test_model_architecture_renderer_rejects_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported Full-Wh figure formats"):
        full_cell_model_architectures.build_full_cell_model_architecture_figures(
            output_dir=tmp_path,
            formats=("pdf",),
        )
