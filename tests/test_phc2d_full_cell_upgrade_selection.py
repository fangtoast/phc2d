"""Tests for validation-only Full-Wh upgrade head selection.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from phc2d.forward.full_cell_upgrade_selection import freeze_head_selection, summarize_validation


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    models = (
        "mlp_mixer_band_v1",
        "res_mlp_gated_v2",
        "convnext_band_v1",
        "tfno2d_band_v3",
    )
    for model in models:
        for lane in ("B", "G"):
            for head, value in (("dense", 1.0), ("k_conditioned", 1.2)):
                if model == "res_mlp_gated_v2" and lane == "B":
                    value = 1.0 if head == "dense" else 0.98
                for seed in (0, 1, 2):
                    rows.append(
                        {
                            "model": model,
                            "head": head,
                            "lane": lane,
                            "seed": seed,
                            "validation_bands_2_10_mre_pct": value + seed * 0.001,
                            "parameter_count": 2_500_000,
                            "peak_gpu_memory_mb": 1000.0,
                            "train_time_s": 10.0,
                            "run_dir": f"run/{model}/{head}/{lane}/{seed}",
                            "checkpoint_sha256": str(seed) * 64,
                            "results_sha256": str(seed + 1) * 64,
                        }
                    )
    for head, value in (("dense", 0.5), ("k_conditioned", 0.7)):
        for seed in (0, 1, 2):
            rows.append(
                {
                    "model": "dual_lane_band_operator_v1",
                    "head": head,
                    "lane": "J",
                    "seed": seed,
                    "validation_bands_2_10_mre_pct": value + seed * 0.001,
                    "parameter_count": 2_500_000,
                    "peak_gpu_memory_mb": 1000.0,
                    "train_time_s": 10.0,
                    "run_dir": f"run/dual/{head}/{seed}",
                    "checkpoint_sha256": str(seed) * 64,
                    "results_sha256": str(seed + 1) * 64,
                }
            )
    return rows


def test_summary_and_selection_use_validation_only_rule() -> None:
    rows = _rows()
    summary = summarize_validation(rows)
    lock = freeze_head_selection(rows, summary)
    assert len(rows) == 54
    assert len(summary) == 18
    assert lock["test_read_during_selection"] is False
    selected = {
        (item["model"], item["lane"]): item["selected_head"]
        for item in lock["selections"]
    }
    assert selected[("res_mlp_gated_v2", "B")] == "k_conditioned"
    assert all(
        head == "dense"
        for key, head in selected.items()
        if key != ("res_mlp_gated_v2", "B")
    )


def test_tie_requires_resolution_transfer_evidence() -> None:
    rows = _rows()
    for row in rows:
        if row["model"] == "tfno2d_band_v3" and row["lane"] == "B":
            row["validation_bands_2_10_mre_pct"] = 1.0
    summary = summarize_validation(rows)
    try:
        freeze_head_selection(rows, summary)
    except RuntimeError as exc:
        assert "32/128 transfer tie-break" in str(exc)
    else:
        raise AssertionError("a validation tie must not silently select a head")
