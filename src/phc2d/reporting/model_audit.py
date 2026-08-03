"""Audit model-inventory, report-coverage, link, and source-artifact evidence.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from phc2d.models import list_models


REQUIRED_NON_REGISTRY_ENTRIES = {
    "train_only_nearest_neighbor",
    "direct_inverse_mlp",
    "single_mode_generator",
    "mixture_conditional_generator",
    "adapted_pino",
}
AUDITED_REGISTERED_MODELS = frozenset(list_models())
_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_inventory(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Model inventory root must be a mapping.")
    if not isinstance(payload.get("entries"), list):
        raise ValueError("Model inventory must contain an entries list.")
    return payload


def _entry_map(entries: list[Any]) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    failures: list[str] = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            failures.append(f"entries[{position}] is not a mapping")
            continue
        entry_id = str(entry.get("id", "")).strip()
        if not entry_id:
            failures.append(f"entries[{position}] has no id")
            continue
        if entry_id in by_id:
            failures.append(f"duplicate inventory id: {entry_id}")
            continue
        by_id[entry_id] = entry
    return by_id, failures


def _report_tokens(entry: Mapping[str, Any]) -> list[str]:
    tokens = entry.get("report_tokens")
    if isinstance(tokens, list) and tokens:
        return [str(value) for value in tokens]
    return [str(entry.get("display_name") or entry.get("id") or "")]


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _audit_sources(
    entries: list[Mapping[str, Any]],
    *,
    repo_root: Path,
    require_source_artifacts: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for entry in entries:
        for artifact in entry.get("source_artifacts", []) or []:
            if not isinstance(artifact, Mapping):
                failures.append(f"{entry['id']}: source_artifact is not a mapping")
                continue
            raw_path = str(artifact.get("path", "")).strip()
            expected = str(artifact.get("sha256", "")).strip().lower()
            if not raw_path or len(expected) != 64:
                failures.append(f"{entry['id']}: source_artifact requires path and SHA-256")
                continue
            path = _resolve_repo_path(repo_root, raw_path)
            exists = path.is_file()
            actual = _sha256(path) if exists else None
            matches = exists and actual == expected
            rows.append(
                {
                    "entry_id": entry["id"],
                    "path": raw_path,
                    "exists": exists,
                    "sha256_matches": matches,
                }
            )
            if exists and not matches:
                failures.append(f"{entry['id']}: source SHA-256 mismatch: {raw_path}")
            elif require_source_artifacts and not exists:
                failures.append(f"{entry['id']}: required source artifact missing: {raw_path}")
    return rows, failures


def _audit_markdown_links(
    report_path: Path,
    *,
    repo_root: Path,
    require_source_artifacts: bool,
) -> tuple[list[str], list[str]]:
    missing_docs: list[str] = []
    missing_runs: list[str] = []
    text = report_path.read_text(encoding="utf-8")
    for match in _MARKDOWN_LINK.finditer(text):
        raw = match.group(1).strip().split(maxsplit=1)[0].strip("<>")
        raw = raw.split("#", 1)[0]
        if not raw or raw.startswith(("http://", "https://", "mailto:")):
            continue
        target = (report_path.parent / raw).resolve()
        if target.exists():
            continue
        try:
            relative = target.relative_to(repo_root.resolve())
        except ValueError:
            relative = target
        rendered = Path(relative).as_posix()
        if isinstance(relative, Path) and relative.parts and relative.parts[0] == "runs":
            missing_runs.append(rendered)
        else:
            missing_docs.append(rendered)
    failures = [f"missing committed report link: {path}" for path in missing_docs]
    if require_source_artifacts:
        failures.extend(f"missing run/source report link: {path}" for path in missing_runs)
    return missing_docs + missing_runs, failures


def audit_model_report(
    *,
    inventory_path: Path,
    report_path: Path,
    repo_root: Path | None = None,
    require_source_artifacts: bool = False,
) -> dict[str, Any]:
    """Return a read-only coverage and source-integrity audit payload."""

    inventory_path = Path(inventory_path)
    report_path = Path(report_path)
    repo_root = Path(repo_root or Path.cwd()).resolve()
    inventory = _load_inventory(inventory_path)
    entries_raw = inventory["entries"]
    entries_by_id, failures = _entry_map(entries_raw)
    entries = list(entries_by_id.values())

    registered_expected = set(AUDITED_REGISTERED_MODELS)
    registered_recorded = {
        entry_id
        for entry_id, entry in entries_by_id.items()
        if entry.get("registry_status") == "registered"
    }
    missing_registered = sorted(registered_expected - registered_recorded)
    unexpected_registered = sorted(registered_recorded - registered_expected)
    if missing_registered:
        failures.append(f"registered models missing from inventory: {missing_registered}")
    if unexpected_registered:
        failures.append(f"inventory marks unknown models registered: {unexpected_registered}")

    missing_non_registry = sorted(REQUIRED_NON_REGISTRY_ENTRIES - set(entries_by_id))
    if missing_non_registry:
        failures.append(f"required non-registry entries missing: {missing_non_registry}")

    report_text = report_path.read_text(encoding="utf-8")
    report_missing: list[str] = []
    for entry in entries:
        if entry.get("report_scope") not in {"main", "appendix"}:
            continue
        tokens = [token for token in _report_tokens(entry) if token]
        if tokens and not any(token in report_text for token in tokens):
            report_missing.append(str(entry["id"]))
    if report_missing:
        failures.append(f"inventory entries missing from report: {sorted(report_missing)}")

    source_rows, source_failures = _audit_sources(
        entries,
        repo_root=repo_root,
        require_source_artifacts=require_source_artifacts,
    )
    failures.extend(source_failures)
    broken_links, link_failures = _audit_markdown_links(
        report_path,
        repo_root=repo_root,
        require_source_artifacts=require_source_artifacts,
    )
    failures.extend(link_failures)

    return {
        "status": "pass" if not failures else "fail",
        "inventory_version": inventory.get("version"),
        "inventory_entry_count": len(entries),
        "registered_model_count": len(registered_expected),
        "registered_models": sorted(registered_expected),
        "required_non_registry_entries": sorted(REQUIRED_NON_REGISTRY_ENTRIES),
        "missing_registered_models": missing_registered,
        "unexpected_registered_models": unexpected_registered,
        "missing_non_registry_entries": missing_non_registry,
        "report_missing_entries": sorted(report_missing),
        "source_artifact_checks": source_rows,
        "broken_report_links": broken_links,
        "require_source_artifacts": bool(require_source_artifacts),
        "failures": failures,
    }
