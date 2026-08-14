"""Regression tests for the documentation navigation assets.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


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
