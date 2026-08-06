"""Contract tests for full-cell forward cross evaluation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from phc2d.forward.full_cell_cross_eval import MODELS, REPRESENTATIONS, SEEDS, planned_cross_evaluation_cells


def test_cross_evaluation_plan_has_all_576_frozen_cells() -> None:
    cells = planned_cross_evaluation_cells()
    assert len(cells) == 576
    assert len(set(cells)) == 576
    assert sum(1 for source, target, _, _ in cells if source == "binary" and target in REPRESENTATIONS) == len(MODELS) * len(SEEDS) * 6
    assert sum(1 for source, target, _, _ in cells if source in REPRESENTATIONS and target == "binary") == len(MODELS) * len(SEEDS) * 6
    assert sum(1 for source, target, _, _ in cells if source in REPRESENTATIONS and target in REPRESENTATIONS) == len(MODELS) * len(SEEDS) * 36
