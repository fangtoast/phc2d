"""Selected-checkpoint per-band validation preserves the test boundary.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import json

import numpy as np
import pytest
import yaml

from phc2d.forward.budget_confirmation import file_digest
from phc2d.forward.official_comparison_validation import per_band_mre_sum, report_per_band_validation
from phc2d.models import build_model


@pytest.mark.parametrize("experiment_id", ["EXP-FWD-031", "EXP-FWD-031_n8000_e1000"])
def test_report_per_band_validation_agrees_with_aggregates(tmp_path, experiment_id):
    torch = pytest.importorskip("torch")
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("neuralop")
    data = tmp_path / "data.h5"
    with h5py.File(data, "w") as handle:
        handle["grid"] = np.ones((5, 8, 8), dtype=np.float32)
        labels = np.ones((5, 3, 2), dtype=np.float32)
        labels[4] = np.nan
        handle["Wh"] = labels
        handle["idx_val"] = np.asarray([2, 3], dtype=np.int64)
        handle["idx_test"] = np.asarray([4], dtype=np.int64)
    run = tmp_path / "run"
    run.mkdir()
    params = dict(width=8, depth=2, modes=(2, 2))
    model = build_model("official_fno_band_v1", input_shape=(1, 8, 8), output_shape=(3, 2), **params).eval()
    checkpoint = run / "model.pt"
    torch.save(dict(state_dict=model.state_dict(), model_name="official_fno_band_v1",
                    model_params=params, input_shape=(1, 8, 8), output_shape=(3, 2),
                    target_mean=torch.zeros(3, 2), target_std=torch.ones(3, 2)), checkpoint)
    with torch.no_grad():
        prediction = model(torch.ones(2, 8, 8))
        per_band = (per_band_mre_sum(prediction, torch.ones_like(prediction), torch) * (100 / 6)).tolist()
    (run / "results.json").write_text(json.dumps(dict(status="formal_training_success",
        selection_split="validation", test_executed=False, model="official_fno_band_v1",
        checkpoint_sha256=file_digest(checkpoint))), encoding="utf-8")
    (run / "metrics.json").write_text(json.dumps(dict(
        bands_1_10_mre_pct=sum(per_band) / 2,
        bands_2_10_mre_pct=per_band[1])), encoding="utf-8")
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(dict(
        experiment=dict(experiment_id=experiment_id),
        dataset=dict(path=str(data), grid_field="grid", expected_dataset_sha256=file_digest(data)))),
        encoding="utf-8")
    record = report_per_band_validation(run, batch_size=1)
    assert record["validation_count"] == 2 and not record["test_executed"]
    assert record["experiment_id"] == experiment_id
    assert record["per_band_mre_pct"] == pytest.approx(per_band, abs=1e-4)
    assert (run / "per_band_validation.json").is_file()
