"""Tests for the EXP-FWD-007 six-domain plotting contract.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import pytest

from scripts import plot_exp_fwd_007_six_domain as plotter


def _synthetic_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source_index, source in enumerate(plotter.DOMAIN_ORDER):
        source_family, source_lane = source.split("-")
        for target_index, target in enumerate(plotter.DOMAIN_ORDER):
            for model_index, model in enumerate(plotter.MODEL_ORDER):
                rows.append(
                    {
                        "source_family": source_family,
                        "source_lane": source_lane,
                        "source_seed": "0",
                        "source_model": model,
                        "target_domain": target,
                        "bands_2_10_mre_pct": str(
                            1.0 + source_index + target_index / 10 + model_index / 100
                        ),
                    }
                )
    return rows


def test_matrix_summary_uses_eight_models_and_reports_the_true_winner() -> None:
    mean, best, winner, counts = plotter._matrix_summary(_synthetic_rows())

    assert mean.shape == (6, 6)
    assert counts.tolist() == [[8] * 6 for _ in range(6)]
    assert mean[2, 4] == pytest.approx(3.435)
    assert best[2, 4] == pytest.approx(3.4)
    assert winner[2][4] == "MLP"


def test_matrix_summary_rejects_an_incomplete_transfer_cell() -> None:
    rows = _synthetic_rows()
    rows.pop()

    with pytest.raises(ValueError, match="Expected 8 models"):
        plotter._matrix_summary(rows)
