"""Tests for the Git-tracked documentation graph audit.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

from phc2d.reporting.docs_audit import audit_documentation


_META = """| Metadata | Value |
|---|---|
| Status | Canonical |
| Audience | Test |
| Owner | Tester |
| Last reviewed | 2026-08-06 |
| Reading time | 1 minute |
"""


def test_docs_audit_passes_reachable_graph_and_explicit_anchor(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text(
        f"# Root\n\n{_META}\n[Docs](docs/README.md)\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "README.md").write_text(
        f'<a id="docs-home"></a>\n# Docs\n\n{_META}\n[Guide](guide.md#guide-start)\n',
        encoding="utf-8",
    )
    (tmp_path / "docs" / "guide.md").write_text(
        '<a id="guide-start"></a>\n# Guide\n',
        encoding="utf-8",
    )

    result = audit_documentation(
        repo_root=tmp_path,
        tracked_markdown=["README.md", "docs/README.md", "docs/guide.md"],
        entry_documents=["README.md", "docs/README.md"],
    )

    assert result["status"] == "pass"
    assert result["orphan_documents"] == []
    assert result["missing_anchors"] == []


def test_docs_audit_reports_broken_orphan_metadata_and_local_path(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "# Root\n\n[Missing](docs/missing.md)\nD:\\private\\repo\n",
        encoding="utf-8",
    )
    (tmp_path / "orphan.md").write_text("# Orphan\n", encoding="utf-8")

    result = audit_documentation(
        repo_root=tmp_path,
        tracked_markdown=["README.md", "orphan.md"],
        entry_documents=["README.md"],
    )

    assert result["status"] == "fail"
    assert result["orphan_documents"] == ["orphan.md"]
    assert result["entry_absolute_path_failures"] == ["README.md"]
    assert "README.md" in result["entry_metadata_failures"]
    assert result["broken_links"] == [
        {"source": "README.md", "target": "docs/missing.md"}
    ]


def test_docs_audit_ignores_untracked_markdown_and_missing_artifacts(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        f"# Root\n\n{_META}\n[Run](runs/missing/report.md)\n",
        encoding="utf-8",
    )
    (tmp_path / "draft.md").write_text("# Untracked draft\n", encoding="utf-8")

    result = audit_documentation(
        repo_root=tmp_path,
        tracked_markdown=["README.md"],
        entry_documents=["README.md"],
    )

    assert result["status"] == "pass"
    assert result["tracked_markdown_count"] == 1
    assert result["external_artifact_links"] == [
        {"source": "README.md", "target": "runs/missing/report.md"}
    ]
