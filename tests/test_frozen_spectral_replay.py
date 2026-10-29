"""Replay staging must stop on native regression before transfer evaluation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import json

import pytest

from phc2d.forward import frozen_spectral_replay as replay


def test_native_failure_blocks_transfer_and_records_failure(tmp_path, monkeypatch):
    original = tmp_path / "original"
    original.mkdir()
    result = dict(checkpoint_sha256="a"*64, target_sha256="b"*64, target_split_sha256="c"*64)
    (original / "results.json").write_text(json.dumps(result))
    (original / "metrics.json").write_text(json.dumps(dict(bands_1_10_mre_pct=1)))
    base = dict(model="m", source_domain="source", seed=0, original_result_dir=str(original),
                checkpoint_sha256="a"*64, dataset_sha256="b"*64, split_sha256="c"*64,
                expected_mre_pct=1, source_run="source", completed_marker="marker", dataset_path="dataset",
                grid_field="grid", external=False)
    cells = [dict(base, cell_id="transfer", target_domain="other", comparison_role="cross_domain"),
             *[dict(base, cell_id=f"native{s}", seed=s, target_domain="source", comparison_role="native") for s in range(3)]]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(cells=cells, expected_native_cells=3)))
    calls = []
    def fail(*args, **kwargs):
        calls.append(args[5].name)
        raise ValueError("native replay failed")
    monkeypatch.setattr(replay, "evaluate", fail)
    with pytest.raises(ValueError, match="native replay failed"):
        replay.replay_manifest(manifest, tmp_path / "output")
    assert calls == ["native0"]
    assert json.loads((tmp_path / "output" / "failure.json").read_text())["status"] == "blocked_no_automatic_retry"
    assert not (tmp_path / "output" / "native_pilot_pass.json").exists()


def test_native_coverage_checked_before_output(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(expected_native_cells=36, cells=[dict(
        model="m", source_domain="a", target_domain="a", seed=0, comparison_role="native")])))
    with pytest.raises(ValueError, match="native pilot coverage"):
        replay.replay_manifest(manifest, tmp_path / "output")
    assert not (tmp_path / "output").exists()
