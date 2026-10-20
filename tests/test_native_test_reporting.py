"""Check the full-band reporting contract and preservation of subset evidence.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import csv
import json

import pytest

from phc2d.forward.native_test_reporting import export_primary_reports


def test_primary_report_uses_all_bands_and_preserves_raw_files(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps({"status": "fixed300_native_surrogate_evaluation_complete", "test_cell_count": 1}))
    cells = "source_domain,model,seed,bands_1_10_mre_pct,bands_2_10_mre_pct\nk5_binary,model,0,1.23,9.87\n"
    summary = "source_domain,model,seed_count,bands_1_10_mre_pct_mean,bands_1_10_mre_pct_seed_sd,bands_2_10_mre_pct_mean\nk5_binary,model,3,1.23,0.05,9.87\n"
    (tmp_path / "native_test_cells.csv").write_text(cells)
    (tmp_path / "native_test_summary.csv").write_text(summary)
    record = export_primary_reports(tmp_path)
    with (tmp_path / "native_test_primary_summary.csv").open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert float(rows[0]["bands_1_10_mre_pct_mean"]) == 1.23
    assert not any("bands_2_10" in key or "band1_" in key for key in rows[0])
    assert (tmp_path / "native_test_cells.csv").read_text() == cells
    assert (tmp_path / "native_test_summary.csv").read_text() == summary
    assert record["auxiliary_bands_2_10_retained"] and not record["checkpoint_selection_changed"]


def test_reporting_waits_for_completed_tests(tmp_path):
    (tmp_path / "results.json").write_text('{"status":"training"}')
    with pytest.raises(ValueError, match="must complete"):
        export_primary_reports(tmp_path)


def test_subset_metric_cannot_substitute_for_missing_full_band_mre(tmp_path):
    (tmp_path / "results.json").write_text('{"status":"fixed300_native_surrogate_evaluation_complete","test_cell_count":1}')
    (tmp_path / "native_test_cells.csv").write_text("source_domain,model,seed,bands_2_10_mre_pct\nk5,model,0,0.1\n")
    with pytest.raises(ValueError, match="subset metrics cannot substitute"):
        export_primary_reports(tmp_path)
