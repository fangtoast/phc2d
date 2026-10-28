"""Independent raw-row recomputation for EXP-FWD-031 cells.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import csv
import json

import pytest

from phc2d.forward.official_comparison_test_audit import audit_cell, file_digest


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_audit_cell_recomputes_raw_sample_and_band_metrics(tmp_path):
    source = tmp_path / "training"
    source.mkdir()
    (source / "model.pt").write_bytes(b"checkpoint")
    digest = file_digest(source / "model.pt")
    (source / "results.json").write_text(json.dumps(dict(
        checkpoint_sha256=digest, status="formal_training_success")), encoding="utf-8")
    out = tmp_path / "test"
    out.mkdir()
    (out / "results.json").write_text(json.dumps(dict(
        status="test_complete", model="official_fno_band_v1", source_run=str(source),
        checkpoint_sha256=digest, target_sha256="a" * 64, sample_count=2)), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(dict(
        samples=2, bands_1_10_mre_pct=2.0, bands_2_10_mre_pct=2.0,
        sample_p95_mre_pct=2.9)), encoding="utf-8")
    _write_csv(out / "per_sample_metrics.csv", [
        dict(dataset_index=1, full_matrix_mre_pct=1.0, bands_2_10_mre_pct=1.0),
        dict(dataset_index=2, full_matrix_mre_pct=3.0, bands_2_10_mre_pct=3.0)])
    _write_csv(out / "per_band_metrics.csv", [
        dict(band_one_based=1, mre_pct=2.0), dict(band_one_based=2, mre_pct=2.0)])
    row = dict(run_dir=str(source), model="official_fno_band_v1", case_id="case", seed="0")
    assert audit_cell(out, row, "k5_binary", "a" * 64, 2)["bands_1_10_mre_pct"] == 2.0
    (out / "metrics.json").write_text(json.dumps(dict(
        samples=2, bands_1_10_mre_pct=1.0, bands_2_10_mre_pct=2.0,
        sample_p95_mre_pct=2.9)), encoding="utf-8")
    with pytest.raises(ValueError, match="aggregate"):
        audit_cell(out, row, "k5_binary", "a" * 64, 2)
