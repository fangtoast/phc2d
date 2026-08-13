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

_FRONT_MATTER = """---
title: Test page
summary: Test summary
page_type: reference
status: draft
audience: Test readers
owner: Tester
last_reviewed: 2026-08-13
reading_time: 1 minute
primary_layer: L4
---
"""


def test_docs_audit_passes_reachable_graph_and_explicit_anchor(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text(
        "# Root\n\n[Docs](docs/README.md)\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "README.md").write_text(
        f'{_FRONT_MATTER}<a id="docs-home"></a>\n# Docs\n\n{_META}\n[Guide](guide.md#guide-start)\n',
        encoding="utf-8",
    )
    (tmp_path / "docs" / "guide.md").write_text(
        f'{_FRONT_MATTER}<a id="guide-start"></a>\n# Guide\n',
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
    assert result["front_matter_failures"] == {}
    assert result["page_type_counts"] == {"reference": 2}


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
    assert "README.md" not in result["entry_metadata_failures"]
    assert result["broken_links"] == [
        {"source": "README.md", "target": "docs/missing.md"}
    ]


def test_docs_audit_ignores_untracked_markdown_and_missing_artifacts(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "# Root\n\n[Run](runs/missing/report.md)\n",
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


def test_docs_audit_exempts_only_root_readme_from_entry_metadata(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text(
        "# Root\n\n[Docs](docs/README.md)\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "README.md").write_text("# Docs\n", encoding="utf-8")

    result = audit_documentation(
        repo_root=tmp_path,
        tracked_markdown=["README.md", "docs/README.md"],
        entry_documents=["README.md", "docs/README.md"],
    )

    assert result["status"] == "fail"
    assert result["entry_metadata_failures"] == {
        "docs/README.md": ["Status", "Audience", "Owner", "Last reviewed", "Reading time"]
    }


def test_docs_audit_checks_front_matter_and_mkdocs_navigation(tmp_path: Path) -> None:
    (tmp_path / "docs" / "hidden").mkdir(parents=True)
    (tmp_path / "README.md").write_text("# Root\n", encoding="utf-8")
    (tmp_path / "docs" / "index.md").write_text(
        f"{_FRONT_MATTER}# Home\n\n[Guide](guide.md)\n[Hidden](hidden/asset.md)\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "guide.md").write_text(
        "---\ntitle: Broken\npage_type: unknown\nstatus: accepted\n---\n# Guide\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "hidden" / "asset.md").write_text(
        f"{_FRONT_MATTER}# Asset\n",
        encoding="utf-8",
    )
    (tmp_path / "mkdocs.yml").write_text(
        "docs_dir: docs\nnav:\n  - Home: index.md\nnot_in_nav: |\n  hidden/*.md\n",
        encoding="utf-8",
    )

    result = audit_documentation(
        repo_root=tmp_path,
        tracked_markdown=[
            "README.md",
            "docs/index.md",
            "docs/guide.md",
            "docs/hidden/asset.md",
        ],
        entry_documents=["README.md"],
    )

    assert result["status"] == "fail"
    assert result["unnavigated_documents"] == ["docs/guide.md"]
    assert result["intentionally_hidden_documents"] == ["docs/hidden/asset.md"]
    assert "docs/guide.md" in result["front_matter_failures"]


def test_docs_audit_requires_evidence_scope_and_date(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("# Root\n\n[Evidence](docs/evidence.md)\n", encoding="utf-8")
    metadata = _FRONT_MATTER.replace("page_type: reference", "page_type: evidence")
    (tmp_path / "docs" / "evidence.md").write_text(f"{metadata}# Evidence\n", encoding="utf-8")

    result = audit_documentation(
        repo_root=tmp_path,
        tracked_markdown=["README.md", "docs/evidence.md"],
        entry_documents=["README.md"],
    )

    assert result["front_matter_failures"]["docs/evidence.md"] == [
        "evidence_date",
        "evidence_scope",
    ]
