"""Audit phc2d documentation links, reachability, anchors, and entry metadata.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import re
import subprocess
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import unquote


_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")
_EXPLICIT_ANCHOR = re.compile(r"<a\s+(?:id|name)=[\"']([^\"']+)[\"'][^>]*>", re.IGNORECASE)
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
_LOCAL_ABSOLUTE_PATH = re.compile(r"(?:\b[A-Za-z]:[\\/]|/mnt/[A-Za-z]/)")
_EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "tel:", "data:")
_ARTIFACT_NAMESPACES = ("data/", "runs/", "checkpoints/")
_ENTRY_METADATA = ("Status", "Audience", "Owner", "Last reviewed", "Reading time")
_DEFAULT_ENTRY_DOCUMENTS = (
    "README.md",
    "PROJECT_MAP.md",
    "docs/README.md",
    "docs/getting-started.md",
    "docs/reproducibility.md",
    "docs/collaboration.md",
)
_EXCLUDED_PREFIXES = ("manuscript/",)


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


def audit_documentation(
    *,
    repo_root: str | Path,
    root_path: str | Path = "README.md",
    tracked_markdown: Sequence[str | Path] | None = None,
    entry_documents: Sequence[str | Path] = _DEFAULT_ENTRY_DOCUMENTS,
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
        missing = [field for field in _ENTRY_METADATA if field not in text]
        if missing:
            metadata_failures[relative] = missing
            failures.append(f"entry metadata missing in {relative}: {missing}")
        if _LOCAL_ABSOLUTE_PATH.search(text):
            absolute_path_failures.append(relative)
            failures.append(f"entry document contains a local absolute path: {relative}")

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
        "failures": failures,
    }


__all__ = ["audit_documentation"]
