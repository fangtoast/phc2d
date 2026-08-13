"""Audit phc2d documentation links, reachability, anchors, and entry metadata.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
from collections import Counter
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import unquote

import yaml


_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")
_EXPLICIT_ANCHOR = re.compile(r"<a\s+(?:id|name)=[\"']([^\"']+)[\"'][^>]*>", re.IGNORECASE)
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
_LOCAL_ABSOLUTE_PATH = re.compile(r"(?:\b[A-Za-z]:[\\/]|/mnt/[A-Za-z]/)")
_EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "tel:", "data:")
_ARTIFACT_NAMESPACES = ("data/", "runs/", "checkpoints/")
_ENTRY_METADATA = ("Status", "Audience", "Owner", "Last reviewed", "Reading time")
_ENTRY_METADATA_EXEMPT = frozenset({"README.md"})
_DEFAULT_ENTRY_DOCUMENTS = (
    "README.md",
    "PROJECT_MAP.md",
    "docs/README.md",
    "docs/getting-started.md",
    "docs/reproducibility.md",
    "docs/collaboration.md",
)
_EXCLUDED_PREFIXES = ("manuscript/",)
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_DOCS_METADATA = (
    "title",
    "summary",
    "page_type",
    "status",
    "audience",
    "owner",
    "last_reviewed",
    "reading_time",
    "primary_layer",
)
_PAGE_TYPES = frozenset(
    {
        "landing",
        "tutorial",
        "how-to",
        "explanation",
        "reference",
        "evidence",
        "plan",
        "archive",
        "asset-index",
    }
)
_DOCUMENT_STATUSES = frozenset({"draft", "reviewed", "canonical", "historical", "superseded"})


def _github_slug(value: str) -> str:
    text = re.sub(r"<[^>]+>", "", value).strip().lower()
    text = re.sub(r"[`*_~]", "", text)
    text = re.sub(r"[^\w\-\s\u4e00-\u9fff]", "", text)
    return re.sub(r"[\s\-]+", "-", text).strip("-")


def _anchors(text: str) -> tuple[set[str], list[str]]:
    explicit = _EXPLICIT_ANCHOR.findall(text)
    headings = {_github_slug(value) for value in _HEADING.findall(text)}
    duplicates = sorted({value for value in explicit if explicit.count(value) > 1})
    return set(explicit) | {value for value in headings if value}, duplicates


def _link_target(raw: str) -> tuple[str, str]:
    value = raw.strip().split(maxsplit=1)[0].strip("<>")
    value = unquote(value)
    if "#" not in value:
        return value, ""
    path, fragment = value.split("#", 1)
    return path, fragment


def _tracked_markdown(repo_root: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--", "*.md"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
        values = result.stdout.decode("utf-8").split("\0")
        return sorted(value.replace("\\", "/") for value in values if value)
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        return sorted(path.relative_to(repo_root).as_posix() for path in repo_root.rglob("*.md"))


def _is_artifact_path(relative: str) -> bool:
    normalized = relative.replace("\\", "/").lstrip("./")
    return normalized.startswith(_ARTIFACT_NAMESPACES)


def _front_matter(text: str) -> dict[str, object] | None:
    match = _FRONT_MATTER.search(text)
    if match is None:
        return None
    try:
        value = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None
    return value if isinstance(value, dict) else None


def _flatten_nav(value: object) -> set[str]:
    paths: set[str] = set()
    if isinstance(value, str):
        if value.endswith(".md"):
            paths.add(value.replace("\\", "/"))
    elif isinstance(value, list):
        for item in value:
            paths.update(_flatten_nav(item))
    elif isinstance(value, dict):
        for item in value.values():
            paths.update(_flatten_nav(item))
    return paths


def _patterns(value: object) -> list[str]:
    if isinstance(value, str):
        return [
            line.strip()
            for line in value.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _matches_any(relative: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(relative, pattern.lstrip("/")) for pattern in patterns)


def audit_documentation(
    *,
    repo_root: str | Path,
    root_path: str | Path = "README.md",
    tracked_markdown: Sequence[str | Path] | None = None,
    entry_documents: Sequence[str | Path] = _DEFAULT_ENTRY_DOCUMENTS,
    mkdocs_config: str | Path = "mkdocs.yml",
) -> dict[str, object]:
    """Return a read-only documentation graph and entry-contract audit."""

    repo = Path(repo_root).resolve()
    root_rel = Path(root_path).as_posix()
    tracked = {
        Path(value).as_posix()
        for value in (tracked_markdown if tracked_markdown is not None else _tracked_markdown(repo))
    }
    active = {value for value in tracked if not value.startswith(_EXCLUDED_PREFIXES)}
    failures: list[str] = []
    broken_links: list[dict[str, str]] = []
    missing_anchors: list[dict[str, str]] = []
    duplicate_anchors: dict[str, list[str]] = {}
    external_artifact_links: list[dict[str, str]] = []
    graph: dict[str, set[str]] = {value: set() for value in active}
    anchor_cache: dict[str, set[str]] = {}

    for relative in sorted(active):
        path = repo / relative
        if not path.is_file():
            failures.append(f"tracked Markdown is missing: {relative}")
            continue
        text = path.read_text(encoding="utf-8")
        anchor_cache[relative], duplicates = _anchors(text)
        if duplicates:
            duplicate_anchors[relative] = duplicates
            failures.append(f"duplicate explicit anchors in {relative}: {duplicates}")
        for match in _MARKDOWN_LINK.finditer(text):
            raw_path, fragment = _link_target(match.group(1))
            if not raw_path and not fragment:
                continue
            if raw_path.startswith(_EXTERNAL_PREFIXES):
                continue
            target = (path.parent / raw_path).resolve() if raw_path else path.resolve()
            try:
                target_rel = target.relative_to(repo).as_posix()
            except ValueError:
                target_rel = target.as_posix()
            if _is_artifact_path(target_rel) and not target.exists():
                external_artifact_links.append({"source": relative, "target": target_rel})
                continue
            if not target.exists():
                row = {"source": relative, "target": target_rel}
                broken_links.append(row)
                failures.append(f"broken local link: {relative} -> {target_rel}")
                continue
            if target.suffix.lower() == ".md" and target_rel in active:
                graph[relative].add(target_rel)
                # Same-page heading slugs are renderer-specific in older Chinese documents.
                # Cross-file fragments must use stable anchors and are checked strictly.
                if fragment and target_rel != relative:
                    if target_rel not in anchor_cache:
                        target_text = target.read_text(encoding="utf-8")
                        anchor_cache[target_rel], _ = _anchors(target_text)
                    if fragment not in anchor_cache[target_rel]:
                        row = {"source": relative, "target": target_rel, "anchor": fragment}
                        missing_anchors.append(row)
                        failures.append(
                            f"missing Markdown anchor: {relative} -> {target_rel}#{fragment}"
                        )

    reachable: set[str] = set()
    queue: deque[str] = deque([root_rel])
    while queue:
        current = queue.popleft()
        if current in reachable or current not in graph:
            continue
        reachable.add(current)
        queue.extend(sorted(graph[current] - reachable))
    orphan_documents = sorted(active - reachable)
    for relative in orphan_documents:
        failures.append(f"unreachable tracked Markdown: {relative}")

    metadata_failures: dict[str, list[str]] = {}
    absolute_path_failures: list[str] = []
    for value in entry_documents:
        relative = Path(value).as_posix()
        path = repo / relative
        if not path.is_file():
            metadata_failures[relative] = list(_ENTRY_METADATA)
            failures.append(f"required entry document is missing: {relative}")
            continue
        text = path.read_text(encoding="utf-8")
        if relative not in _ENTRY_METADATA_EXEMPT:
            missing = [field for field in _ENTRY_METADATA if field not in text]
            if missing:
                metadata_failures[relative] = missing
                failures.append(f"entry metadata missing in {relative}: {missing}")
        if _LOCAL_ABSOLUTE_PATH.search(text):
            absolute_path_failures.append(relative)
            failures.append(f"entry document contains a local absolute path: {relative}")

    front_matter_failures: dict[str, list[str]] = {}
    page_type_counts: Counter[str] = Counter()
    docs_pages = sorted(value for value in active if value.startswith("docs/") and value.endswith(".md"))
    for relative in docs_pages:
        text = (repo / relative).read_text(encoding="utf-8")
        metadata = _front_matter(text)
        if metadata is None:
            front_matter_failures[relative] = ["valid YAML front matter"]
            failures.append(f"documentation front matter missing or invalid: {relative}")
            continue
        missing = [field for field in _DOCS_METADATA if not metadata.get(field)]
        page_type = str(metadata.get("page_type", ""))
        status = str(metadata.get("status", ""))
        if page_type and page_type not in _PAGE_TYPES:
            missing.append(f"page_type in {sorted(_PAGE_TYPES)}")
        if status and status not in _DOCUMENT_STATUSES:
            missing.append(f"status in {sorted(_DOCUMENT_STATUSES)}")
        if page_type in {"evidence", "archive"}:
            for field in ("evidence_date", "evidence_scope"):
                if not metadata.get(field):
                    missing.append(field)
        if missing:
            front_matter_failures[relative] = missing
            failures.append(f"documentation front matter invalid in {relative}: {missing}")
        if page_type:
            page_type_counts[page_type] += 1

    mkdocs_rel = Path(mkdocs_config).as_posix()
    mkdocs_path = repo / mkdocs_rel
    navigated_documents: list[str] = []
    intentionally_hidden_documents: list[str] = []
    unnavigated_documents: list[str] = []
    mkdocs_failures: list[str] = []
    if mkdocs_path.is_file():
        try:
            mkdocs_value = yaml.safe_load(mkdocs_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            mkdocs_value = {}
            mkdocs_failures.append(f"invalid YAML: {exc}")
        docs_dir = str(mkdocs_value.get("docs_dir", "docs")).strip("/\\")
        nav_paths = _flatten_nav(mkdocs_value.get("nav", []))
        hidden_patterns = _patterns(mkdocs_value.get("not_in_nav", ""))
        for relative in docs_pages:
            site_relative = Path(relative).relative_to(docs_dir).as_posix()
            if site_relative in nav_paths:
                navigated_documents.append(relative)
            elif _matches_any(site_relative, hidden_patterns):
                intentionally_hidden_documents.append(relative)
            else:
                unnavigated_documents.append(relative)
        if unnavigated_documents:
            mkdocs_failures.append(f"published pages missing from nav/not_in_nav: {unnavigated_documents}")
        for message in mkdocs_failures:
            failures.append(f"MkDocs contract failure: {message}")

    return {
        "status": "pass" if not failures else "fail",
        "root": root_rel,
        "tracked_markdown_count": len(tracked),
        "active_markdown_count": len(active),
        "reachable_markdown_count": len(reachable),
        "broken_links": broken_links,
        "missing_anchors": missing_anchors,
        "duplicate_anchors": duplicate_anchors,
        "orphan_documents": orphan_documents,
        "entry_metadata_failures": metadata_failures,
        "entry_absolute_path_failures": absolute_path_failures,
        "external_artifact_links": external_artifact_links,
        "docs_markdown_count": len(docs_pages),
        "front_matter_failures": front_matter_failures,
        "page_type_counts": dict(sorted(page_type_counts.items())),
        "mkdocs_config": mkdocs_rel if mkdocs_path.is_file() else None,
        "navigated_documents": navigated_documents,
        "intentionally_hidden_documents": intentionally_hidden_documents,
        "unnavigated_documents": unnavigated_documents,
        "mkdocs_failures": mkdocs_failures,
        "failures": failures,
    }


__all__ = ["audit_documentation"]
