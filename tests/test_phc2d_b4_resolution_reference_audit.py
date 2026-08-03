"""Tests for the B4 real-MPB resolution reference audit scaffold."""

from __future__ import annotations

from scripts.run_field_eigen_te_b4_resolution_reference_audit import _selection


def test_reference_selection_is_stable_and_validation_only() -> None:
    records = []
    for source in range(12, 22):
        for k_index in (0, 4):
            for band in (2, 3):
                records.append({"source_sample_index": source, "source_split": "val", "k_index": k_index, "band_index_one_based": band, "solver_job_id": f"s{source}_k{k_index}"})
    manifest = {"contract_id": "field_eigen_te_contract", "contract_version": 4, "dataset_path": "data/v2.npz", "dataset_sha256": "a" * 64, "k_path_id": "square_gxm", "k_points_hash": "b" * 64, "selector": {"band_range_one_based": [2, 3]}, "records": records}
    first = _selection(manifest, source_count=3)
    second = _selection(manifest, source_count=3)
    assert first == second
    assert first["selector"]["source_structure_split_counts"] == {"val": 3}
    assert {row["source_split"] for row in first["records"]} == {"val"}
    assert len(first["records"]) == 3 * 2 * 2
