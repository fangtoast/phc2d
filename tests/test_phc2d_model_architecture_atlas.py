"""Tests for the registry-complete phc2d model architecture atlas.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from pathlib import Path

import pytest

from phc2d.plot import model_architecture_atlas


def test_model_architecture_atlas_covers_live_registry() -> None:
    coverage = model_architecture_atlas.registry_coverage()
    assert coverage["missing"] == []
    assert coverage["extra"] == []
    assert len(coverage["registered"]) == 27
    assert "global_local_band_operator_v1" in coverage["registered"]


def test_model_architecture_atlas_renders_eight_figures(monkeypatch, tmp_path: Path) -> None:
    def fake_save(_figure, output_base: Path, formats):
        paths = []
        for suffix in formats:
            path = output_base.with_suffix(f".{suffix}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("figure", encoding="utf-8")
            paths.append(path)
        return paths

    monkeypatch.setattr(model_architecture_atlas, "_save_figure", fake_save)
    result = model_architecture_atlas.build_model_architecture_atlas(
        output_dir=tmp_path,
        formats=("png", "svg"),
    )

    assert result["status"] == "model_architecture_atlas_completed"
    assert result["figure_count"] == 8
    assert result["registered_model_count"] == 27
    assert result["artifact_count"] == 17
    assert (tmp_path / "atlas_manifest.json").is_file()


def test_model_architecture_atlas_rejects_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported model-atlas figure formats"):
        model_architecture_atlas.build_model_architecture_atlas(
            output_dir=tmp_path,
            formats=("eps",),
        )
