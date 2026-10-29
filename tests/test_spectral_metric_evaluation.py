"""Scientific contracts for frozen spectral metric sidecars.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from phc2d.forward.budget_confirmation import file_digest
from phc2d.forward.full_cell_metrics import compute_spectral_regression_metrics
from phc2d.forward.spectral_metric_evaluation import evaluate_manifest, read_cell_arrays, summarize_comparisons, METRICS
from phc2d.forward.official_comparison_test import restore_split_order
from phc2d.forward.spectral_metric_aggregation import merge_metric_evaluations


def test_exact_and_scaled_predictions_with_metadata_dimensions():
    truth = np.arange(1, 61, dtype=float).reshape(4, 5, 3)
    exact, _, bands = compute_spectral_regression_metrics(truth, truth)
    assert len(bands) == 3
    assert exact["groups"]["all_bands"]["mse"] == 0
    scaled, sample, _ = compute_spectral_regression_metrics(truth * 1.1, truth)
    metrics = scaled["groups"]["all_bands"]
    assert metrics["relative_l2_mean_pct"] == pytest.approx(10)
    assert metrics["relative_l2_global_pct"] == pytest.approx(10)
    assert metrics["rmse_global"] ** 2 == pytest.approx(metrics["mse"])
    assert metrics["rmse_global"] > metrics["rmse_sample_mean"]
    assert sample["all_bands"]["relative_l2_pct"] == pytest.approx(np.full(4, 10))


def test_norm_aggregation_and_undefined_are_not_silently_filtered():
    truth = np.array([1, 100], dtype=float).reshape(2, 1, 1)
    pred = truth + 1
    result, sample, _ = compute_spectral_regression_metrics(pred, truth)
    group = result["groups"]["all_bands"]
    assert group["relative_l2_mean_pct"] == pytest.approx(50.5)
    assert group["relative_l2_global_pct"] < 2
    assert group["relative_l2_p95_pct"] == pytest.approx(np.quantile([100, 1], .95))
    result, samples, _ = compute_spectral_regression_metrics(np.ones((2, 2, 1)), np.zeros((2, 2, 1)))
    group = result["groups"]["all_bands"]
    assert group["relative_l2_mean_pct"] is None
    assert group["undefined_relative_l2_count"] == 2
    assert group["r2_variance_weighted"] is None
    assert np.isnan(samples["all_bands"]["relative_l2_pct"]).all()
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("pred,truth", [
    (np.zeros((0, 2, 3)), np.zeros((0, 2, 3))),
    (np.ones((2, 3)), np.ones((2, 3))),
    (np.ones((2, 2, 3)), np.ones((2, 2, 4))),
    (np.full((1, 2, 3), np.nan), np.ones((1, 2, 3))),
    (np.full((1, 2, 3), np.inf), np.ones((1, 2, 3))),
])
def test_invalid_arrays(pred, truth):
    with pytest.raises(ValueError):
        compute_spectral_regression_metrics(pred, truth)


def test_sample_results_invariant_to_batch_partition():
    rng = np.random.default_rng(41)
    truth = rng.uniform(1, 2, size=(7, 4, 5))
    pred = truth + rng.normal(0, .1, size=truth.shape)
    _, whole, _ = compute_spectral_regression_metrics(pred, truth)
    parts = [compute_spectral_regression_metrics(pred[s], truth[s])[1] for s in (slice(0, 2), slice(2, 7))]
    for metric in whole["all_bands"]:
        np.testing.assert_allclose(whole["all_bands"][metric], np.concatenate([p["all_bands"][metric] for p in parts]))


def fixture_cell(tmp_path, seed):
    folder = tmp_path / str(seed)
    folder.mkdir()
    truth = np.arange(1, 25, dtype=float).reshape(4, 3, 2)
    for name, data in [("predictions", truth * 1.1), ("targets", truth), ("indices", np.arange(4, dtype=np.int64))]:
        np.save(folder / f"{name}.npy", data)
    paths = {key: str(folder / f"{name}.npy") for key, name in [("prediction_path", "predictions"), ("target_path", "targets"), ("indices_path", "indices")]}
    return dict(cell_id=f"cell_{seed}", model="model", source_domain="k5_binary",
                target_domain="k5_binary", seed=seed, comparison_role="native",
                sample_count=4, expected_mre_pct=10, checkpoint_sha256="a" * 64,
                dataset_sha256="b" * 64, k_points_sha256="c" * 64,
                split_sha256=hashlib.sha256(np.arange(4, dtype=np.int64).tobytes()).hexdigest(),
                files_sha256={p: file_digest(Path(p)) for p in paths.values()}, **paths)


def test_workflow_preserves_sources_and_requires_three_seeds(tmp_path):
    cells = [fixture_cell(tmp_path, s) for s in range(3)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(cells=cells)))
    result = evaluate_manifest(manifest, tmp_path / "complete")
    assert result["status"] == "complete"
    assert result["computed_cells"] == 3
    assert result["complete_seed_groups"] == 1
    for cell in cells:
        read_cell_arrays(cell)
    cells[-1]["expected_mre_pct"] = 11
    manifest.write_text(json.dumps(dict(cells=cells)))
    result = evaluate_manifest(manifest, tmp_path / "partial")
    assert result["status"] == "partial"
    assert result["computed_cells"] == 2
    assert result["complete_seed_groups"] == 0
    with pytest.raises(FileExistsError):
        evaluate_manifest(manifest, tmp_path / "complete")


def test_sample_order_and_file_corruption_rejected(tmp_path):
    cell = fixture_cell(tmp_path, 0)
    np.save(cell["indices_path"], np.array([1, 0, 2, 3], dtype=np.int64))
    with pytest.raises(ValueError, match="file hash"):
        read_cell_arrays(cell)
    cell["files_sha256"][cell["indices_path"]] = file_digest(Path(cell["indices_path"]))
    with pytest.raises(ValueError, match="sample order"):
        read_cell_arrays(cell)


def test_duplicate_manifest_rejected_before_output(tmp_path):
    cell = fixture_cell(tmp_path, 0)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(cells=[cell, cell])))
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_manifest(manifest, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    manifest.write_text(json.dumps(dict(cells=[cell], expected_cells=1176)))
    with pytest.raises(ValueError, match="expected_cells"):
        evaluate_manifest(manifest, tmp_path / "output")


def test_export_restores_unsorted_frozen_split_order():
    frozen = np.array([8, 3, 7, 1, 9])
    actual = np.array([3, 7, 8, 1, 9])
    order = restore_split_order(actual, frozen)
    np.testing.assert_array_equal(actual[order], frozen)
    with pytest.raises(ValueError, match="differ"):
        restore_split_order(np.array([1, 1, 2]), np.array([1, 2, 3]))


def test_macro_summary_does_not_promote_missing_domains():
    cells = [dict(model="m", comparison_role="native", source_domain=d, target_domain=d, seed=s)
             for d in ("a", "b", "c", "d") for s in (0, 1, 2)]
    rows = [dict(c, band_group="all_bands", **{m: 1.0 + c["seed"] for m in METRICS}) for c in cells]
    complete = summarize_comparisons(rows, cells)[0]
    assert complete["mre_pct_mean"] == 2
    assert complete["mre_pct_seed_sd"] == 1
    partial = summarize_comparisons(rows[:6], cells)[0]
    assert partial["status"] == "blocked_incomplete_scope"
    assert "mre_pct_mean" not in partial


def test_historical_hdf5_export_provenance_and_embedded_indices(tmp_path):
    h5py = pytest.importorskip("h5py")
    cell = fixture_cell(tmp_path, 0)
    path = tmp_path / "predictions.h5"
    with h5py.File(path, "w") as handle:
        handle["prediction"] = np.load(cell["prediction_path"])
        handle["label"] = np.load(cell["target_path"])
        handle["dataset_index"] = np.load(cell["indices_path"])
        handle.attrs["checkpoint_sha256"] = cell["checkpoint_sha256"]
        handle.attrs["target_sha256"] = cell["dataset_sha256"]
        handle.attrs["test_idx_sha256"] = cell["split_sha256"]
    cell.update(h5_path=str(path), files_sha256={str(path): file_digest(path)})
    pred, truth, indices = read_cell_arrays(cell)
    assert pred.shape == truth.shape == (4, 3, 2)
    np.testing.assert_array_equal(indices, np.arange(4))
    cell["checkpoint_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="HDF5 provenance"):
        read_cell_arrays(cell)


def test_merge_disjoint_runs_and_reject_overlap_or_changed_tables(tmp_path):
    cells = [fixture_cell(tmp_path, s) for s in range(3)]
    full = tmp_path / "full.json"
    full.write_text(json.dumps(dict(cells=cells, expected_cells=3)))
    parts = []
    for i, subset in enumerate((cells[:1], cells[1:])):
        path = tmp_path / f"part{i}.json"
        path.write_text(json.dumps(dict(cells=subset)))
        output = tmp_path / f"part{i}"
        evaluate_manifest(path, output)
        parts.append(output)
    merged = merge_metric_evaluations(full, parts, tmp_path / "merged")
    assert merged["status"] == "complete"
    assert merged["computed_cells"] == 3
    with pytest.raises(ValueError, match="overlapping"):
        merge_metric_evaluations(full, [parts[0], parts[0]], tmp_path / "overlap")
    with (parts[0] / "cell_metrics.csv").open("a") as stream:
        stream.write("changed")
    with pytest.raises(ValueError, match="changed source table"):
        merge_metric_evaluations(full, parts, tmp_path / "changed")
