"""Regression checks for portable native-comparison figure exports.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from pathlib import Path

from phc2d.plot.native_comparison import render_native_comparison


def test_native_comparison_writes_all_formats_from_curated_csv(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    source = root / "docs/reports/assets/global_local_native_comparison_20261012/binary_native_grouped_average_source.csv"
    paths = render_native_comparison(source, tmp_path / "comparison")
    assert {path.suffix for path in paths} == {".png", ".svg", ".pdf"}
    assert all(path.stat().st_size > 1000 for path in paths)
    assert "Bands 2" in (tmp_path / "comparison.svg").read_text(encoding="utf-8")
