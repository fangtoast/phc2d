"""Tests for project-wide model inventory and report audits.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from phc2d.reporting import audit_model_report
from phc2d.reporting.model_audit import (
    AUDITED_REGISTERED_MODELS,
    REQUIRED_NON_REGISTRY_ENTRIES,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY = REPO_ROOT / "docs" / "reports" / "phc2d_model_inventory_v1.yaml"


def test_project_inventory_matches_registered_models_and_manual_methods() -> None:
    payload = yaml.safe_load(INVENTORY.read_text(encoding="utf-8"))
    entries = payload["entries"]
    registered = {
        entry["id"] for entry in entries if entry["registry_status"] == "registered"
    }
    assert registered == set(AUDITED_REGISTERED_MODELS)
    entry_ids = {entry["id"] for entry in entries}
    assert REQUIRED_NON_REGISTRY_ENTRIES <= entry_ids
    assert len(registered) == 17

    by_id = {entry["id"]: entry for entry in entries}
    assert by_id["pod_res_mlp"]["implementation_status"] == "placeholder"
    assert by_id["conditional_diffusion_grid"]["implementation_status"] == "configured_only"
    assert by_id["adapted_pino"]["category"] == "training_regime"
    assert by_id["mixture_conditional_generator"]["family"] == "conditional_generator"


def test_model_audit_checks_report_tokens_links_and_source_hashes(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"status":"ok"}\n', encoding="utf-8")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    registered_entries = [
        {
            "id": name,
            "display_name": name,
            "registry_status": "registered",
            "report_scope": "appendix",
            "report_tokens": [name],
            "source_artifacts": [],
        }
        for name in AUDITED_REGISTERED_MODELS
    ]
    manual_entries = [
        {
            "id": name,
            "display_name": name,
            "registry_status": "not_registered",
            "report_scope": "appendix",
            "report_tokens": [name],
            "source_artifacts": (
                [{"path": "source.json", "sha256": source_hash}]
                if name == "adapted_pino"
                else []
            ),
        }
        for name in sorted(REQUIRED_NON_REGISTRY_ENTRIES)
    ]
    entries = registered_entries + manual_entries
    inventory = tmp_path / "inventory.yaml"
    inventory.write_text(
        yaml.safe_dump({"version": 1, "entries": entries}, sort_keys=False),
        encoding="utf-8",
    )
    linked = tmp_path / "figure.svg"
    linked.write_text("<svg/>", encoding="utf-8")
    report = tmp_path / "report.md"
    report.write_text(
        "\n".join(["# Report", *[entry["id"] for entry in entries], "![figure](figure.svg)"]),
        encoding="utf-8",
    )

    result = audit_model_report(
        inventory_path=inventory,
        report_path=report,
        repo_root=tmp_path,
        require_source_artifacts=True,
    )
    assert result["status"] == "pass"
    assert result["registered_model_count"] == 17
    assert result["broken_report_links"] == []
    assert result["source_artifact_checks"] == [
        {
            "entry_id": "adapted_pino",
            "path": "source.json",
            "exists": True,
            "sha256_matches": True,
        }
    ]


def test_model_audit_fails_when_registered_model_is_omitted(tmp_path: Path) -> None:
    names = sorted(AUDITED_REGISTERED_MODELS)
    entries = [
        {
            "id": name,
            "registry_status": "registered",
            "report_scope": "appendix",
            "report_tokens": [name],
            "source_artifacts": [],
        }
        for name in names[1:]
    ]
    entries.extend(
        {
            "id": name,
            "registry_status": "not_registered",
            "report_scope": "appendix",
            "report_tokens": [name],
            "source_artifacts": [],
        }
        for name in sorted(REQUIRED_NON_REGISTRY_ENTRIES)
    )
    inventory = tmp_path / "inventory.yaml"
    inventory.write_text(
        yaml.safe_dump({"version": 1, "entries": entries}, sort_keys=False),
        encoding="utf-8",
    )
    report = tmp_path / "report.md"
    report.write_text("\n".join(entry["id"] for entry in entries), encoding="utf-8")

    result = audit_model_report(
        inventory_path=inventory,
        report_path=report,
        repo_root=tmp_path,
    )
    assert result["status"] == "fail"
    assert result["missing_registered_models"] == [names[0]]
