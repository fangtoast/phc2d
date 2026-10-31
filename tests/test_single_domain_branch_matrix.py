"""Contract checks for provisional branch-ablation identities.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from pathlib import Path
import copy

import pytest
import yaml

from phc2d.forward.single_domain_branch_matrix import build_matrix

PROFILE = Path("configs/research/full_cell_single_domain_branch_ablation_exp_fwd_032_v1.yaml")


def test_unique_runs_and_test_cells():
    matrix = build_matrix(yaml.safe_load(PROFILE.read_text(encoding="utf-8")))
    runs, cells = matrix["runs"], matrix["new_test_cells"]
    assert len(runs) == len({r["case_id"] for r in runs}) == 96
    assert sum(r["disposition"] == "candidate_reuse" for r in runs) == 24
    assert len(cells) == len({(r["case_id"], r["target"]) for r in cells}) == 432
    assert sum(r["platform"] == "n40" for r in runs) == 48
    assert sum(r["platform"] == "n56" for r in runs) == 48


def test_rejects_duplicate_target_and_unmatched_capacity():
    profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    bad = copy.deepcopy(profile)
    bad["test"]["targets"][-1] = bad["test"]["targets"][0]
    with pytest.raises(ValueError):
        build_matrix(bad)
    bad = copy.deepcopy(profile)
    bad["variants"]["g_only_cap"]["parameters"] = 1
    with pytest.raises(ValueError):
        build_matrix(bad)


def test_planned_parameter_counts_match_model_factory():
    torch = pytest.importorskip("torch")
    from phc2d.models import build_model

    profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    source = yaml.safe_load(Path(profile["source_profile"]).read_text(encoding="utf-8"))
    params = source["models"][profile["model"]]
    for variant in profile["variants"].values():
        with torch.device("meta"):
            model = build_model(profile["model"], input_shape=(1, 64, 64), output_shape=(81, 10),
                                **{**params, "enabled_branches": variant["branches"],
                                   "local_kernel": variant["local_kernel"], "width": variant["width"],
                                   "dense_head_hidden_dim": variant["dense_head_hidden_dim"]})
        assert sum(p.numel() for p in model.parameters()) == variant["parameters"]
