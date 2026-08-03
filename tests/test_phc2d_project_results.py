"""Tests for the frozen project-wide result figure package.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from phc2d.plot import project_results


_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")


def test_project_result_package_covers_required_models(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_save(figure, output_base, formats, dpi=300):
        del figure, dpi
        output_base.parent.mkdir(parents=True, exist_ok=True)
        paths = []
        for suffix in formats:
            path = output_base.with_suffix(f".{suffix}")
            path.write_bytes(b"figure")
            paths.append(path)
        return paths

    monkeypatch.setattr(project_results, "_save_figure", fake_save)
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "curated"
    result = project_results.build_project_result_figures(
        inventory_path=repo_root / "docs" / "reports" / "phc2d_model_inventory_v1.yaml",
        output_dir=output_dir,
        formats=("png", "svg", "pdf"),
    )

    assert result["status"] == "project_result_figure_package_completed"
    assert result["figure_count"] == 6
    assert result["figure_file_count"] == 18
    assert result["figure_series"]["a1_forward_model_comparison"] == [
        "mlp",
        "res_mlp_wide",
        "lifted_cnn",
        "fno2d_band_v2",
    ]
    assert result["figure_series"]["a1_query_model_comparison"] == [
        "mlp_query",
        "deep_onet_query_v2",
        "fno2d_query_v2",
    ]
    assert result["figure_series"]["b3_field_model_comparison"] == [
        "unet_field_v2",
        "fno_field_v2",
        "adapted_pino",
    ]
    assert result["figure_series"]["a2_100_model_comparison"] == [
        "train_only_nearest_neighbor",
        "direct_inverse_mlp",
        "single_mode_generator",
        "mixture_conditional_generator",
    ]
    inventory = project_results._entry_map(
        project_results._load_inventory(
            repo_root / "docs" / "reports" / "phc2d_model_inventory_v1.yaml"
        )
    )
    assert (
        project_results._inventory_column(inventory["direct_inverse_mlp"])
        == "historical"
    )
    assert (
        project_results._inventory_column(inventory["train_only_nearest_neighbor"])
        == "historical"
    )
    assert (
        project_results._inventory_column(inventory["single_mode_generator"])
        == "current"
    )
    manifest = json.loads(
        (output_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["figure_series"] == result["figure_series"]
    assert len(manifest["artifacts"]) == 19
    assert (output_dir / "source_index.csv").exists()


def test_committed_curated_assets_have_valid_links_and_hashes() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    asset_dir = (
        repo_root
        / "docs"
        / "reports"
        / "assets"
        / "phc2d_project_full_introduction_20260724"
    )
    index_path = asset_dir / "figure_index.md"
    for match in _MARKDOWN_LINK.finditer(index_path.read_text(encoding="utf-8")):
        raw = match.group(1).split("#", 1)[0]
        assert (index_path.parent / raw).resolve().exists(), raw

    manifest = json.loads(
        (asset_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    recorded = {row["path"]: row for row in manifest["artifacts"]}
    required_stems = (
        "results/a1_forward_model_comparison",
        "results/a1_query_model_comparison",
        "results/b3_field_model_comparison",
        "results/a2_100_model_comparison",
        "results/a2_bridge_625_comparison",
        "learning_curves/a1_forward_learning_curves",
        "learning_curves/b3_field_learning_curves",
        "inverse/paired_main4",
    )
    for stem in required_stems:
        for suffix in ("png", "svg"):
            relative = f"{stem}.{suffix}"
            assert relative in recorded
            path = asset_dir / relative
            assert hashlib.sha256(path.read_bytes()).hexdigest() == recorded[relative]["sha256"]
    assert not [path for path in recorded if path.endswith(".pdf")]
