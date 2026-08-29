"""Regression tests for the documentation navigation assets.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_mkdocs_config() -> dict[str, object]:
    text = (REPOSITORY_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    text = text.replace(
        "!!python/name:pymdownx.superfences.fence_div_format", "fence_div_format"
    )
    return yaml.safe_load(text)


def _find_nav_group(nav: list[object], title: str) -> list[object]:
    for item in nav:
        if isinstance(item, dict) and title in item:
            value = item[title]
            assert isinstance(value, list)
            return value
    raise AssertionError(f"missing navigation group: {title}")


def _flatten_nav_paths(items: list[object]) -> list[str]:
    paths: list[str] = []
    for item in items:
        if isinstance(item, str):
            paths.append(item)
        elif isinstance(item, dict):
            for value in item.values():
                if isinstance(value, str):
                    paths.append(value)
                elif isinstance(value, list):
                    paths.extend(_flatten_nav_paths(value))
    return paths


def test_navigation_initializes_before_remote_math_assets() -> None:
    config = (REPOSITORY_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    navigation_index = config.index("  - assets/javascripts/navigation.js")
    remote_script_indices = [
        config.index("  - https://cdn.jsdelivr.net/npm/mathjax"),
        config.index("  - https://cdn.jsdelivr.net/npm/mermaid"),
    ]

    assert navigation_index < min(remote_script_indices)


def test_navigation_preserves_accessibility_and_motion_preferences() -> None:
    script = (REPOSITORY_ROOT / "docs/assets/javascripts/navigation.js").read_text(
        encoding="utf-8"
    )
    stylesheet = (REPOSITORY_ROOT / "docs/assets/stylesheets/extra.css").read_text(
        encoding="utf-8"
    )

    assert 'setAttribute("aria-expanded"' in script
    assert "prefers-reduced-motion: reduce" in script
    assert "requestAnimationFrame" in script
    assert "sessionStorage" in script
    assert "height 180ms" in stylesheet
    assert "@media (prefers-reduced-motion: reduce)" in stylesheet


def test_formal_results_navigation_is_an_explicit_whitelist() -> None:
    nav = _load_mkdocs_config()["nav"]
    assert isinstance(nav, list)
    formal_results = set(_flatten_nav_paths(_find_nav_group(nav, "实验与证据库")))

    assert formal_results == {
        "results/README.md",
        "results/V2-10k正式结果.md",
        "results/forward/a1_forward_query_fair_v1_20260726.md",
        "results/inverse/inverse_surrogate_rerank_v2_10000_20260630.md",
        "results/inverse/inverse_retrain_v2_10000_20260706.md",
        "results/inverse/inverse_mpb_replay_mixture_v2_10000_test625_20260706.md",
        "results/inverse/a2_inverse_single_seed_formal_20260727.md",
        "results/inverse/a2_inverse_objective_bridge_seed0_v1_20260727.md",
        "results/inverse/a2_inverse_surrogate_postrepair_validation_v2_20260802.md",
        "results/全胞数据正式结果.md",
        "results/data-provenance/full_cell_dual_lane_pilot_20260803.md",
        "results/data-provenance/全胞MPB分辨率收敛验证_20260817.md",
        "results/data-provenance/全胞Fourier_64加64星仪试点_20260817.md",
        "results/data-provenance/全胞Fourier高频截断配对试验_20260817.md",
        "results/forward/full_cell_dual_lane_forward_training_20260806.md",
        "results/forward/full_cell_dual_lane_50000_forward_training_20260808.md",
        "results/forward/full_cell_forward_model_upgrade_50k_evaluation_20260810.md",
        "results/forward/full_cell_k20_b_g_seed0_cross_eval_20260820.md",
        "results/forward/全胞K20与50K跨数据域正向验证_20260820.md",
        "results/forward/全胞前向50K模型R方补充评估_20260819.md",
        "results/forward/EXP-FWD-007-K5训练与六域验证.md",
        "results/forward/EXP-FWD-008-K5固定200轮训练与六域验证.md",
        "results/forward/EXP-FWD-009-K5权重热启动与六域验证.md",
        "results/forward/EXP-FWD-012-工作频率先验覆盖审计.md",
        "results/forward/EXP-FWD-014-gap决策损失短预算筛选.md",
        "results/forward/EXP-FWD-015-gap决策近简并审计.md",
        "results/field-physics/full_cell_dual_lane_meep_scattering_20260806.md",
        "results/field-physics/代理选频与真实响应差异_20260825.md",
        "results/field-physics/EXP-PHY-005-TE有效折射率与折叠频率校准.md",
        "results/field-physics/EXP-PHY-006-全BZ带隙边缘加密试点.md",
        "reports/全胞双路线正向与逆向模型报告_20260810.md",
    }


def test_original_planning_pages_each_have_one_navigation_entry() -> None:
    nav = _load_mkdocs_config()["nav"]
    assert isinstance(nav, list)
    all_paths = _flatten_nav_paths(nav)
    original_planning_pages = {
        "planning/README.md",
        "planning/fno_generative_network_adjustment_plan_20260617.md",
        "planning/research_todo_20260617.md",
        "planning/a2_inverse_single_seed_protocol_20260726.md",
        "planning/0718-0801/02_physics_model_reproduction_plan.md",
        "planning/0718-0801/09_phase1_a0_b0_protocol.md",
        "planning/0718-0801/10_b4_frequency_fidelity_dense_k_admission.md",
        "planning/full_cell_forward_inverse_training/README.md",
        "planning/full_cell_forward_inverse_training/full_cell_dual_lane_hybrid_compute_plan.md",
        "planning/full_cell_forward_inverse_training/full_cell_dual_lane_50000_training_hybrid_compute_plan.md",
        "planning/full_cell_forward_inverse_training/full_cell_forward_model_upgrade_implementation_plan.md",
        "planning/full_cell_forward_inverse_training/full_cell_forward_model_upgrade_10k_50k_training_validation_plan.md",
        "planning/full_cell_forward_inverse_training/full_cell_inverse_design_extension_plan.md",
        "planning/full_cell_forward_inverse_training/archive/full_cell_forward_single_seed_legacy.md",
        "planning/reference/README.md",
        "planning/reference/experimental_validation_plan_20260617.md",
        "planning/reference/network_literature_survey_20260617.md",
        "planning/reference/repo_data_audit_20260617.md",
        "planning/非 p4m 全布里渊区双路线数据生成方案.md",
    }

    assert all(all_paths.count(path) == 1 for path in original_planning_pages)
