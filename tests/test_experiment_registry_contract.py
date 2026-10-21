"""Validate experiment routing without inferring scientific admission from registration.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_experiment_registry_obeys_declared_schema_and_resolves_links():
    registry = yaml.safe_load((ROOT / "experiments/registry.yaml").read_text(encoding="utf-8"))
    schema, entries = registry["schema"], registry["experiments"]
    ids = [entry["id"] for entry in entries]
    assert len(ids) == len(set(ids))
    for entry in entries:
        assert set(schema["required_fields"]) <= entry.keys(), entry["id"]
        assert entry["status"] in schema["status_values"], entry["id"]
        assert entry["decision"]["state"] in schema["decision_state_values"], entry["id"]
        assert set(entry["related_experiments"]) <= set(ids), entry["id"]
        for field in ("spec", "config", "result", "report"):
            for relative in entry[field]:
                assert (ROOT / relative).is_file(), (entry["id"], field, relative)
        assert entry["task"] in schema["task_values"]
        assert entry["mapping_status"] in schema["mapping_status_values"]


def test_pending_acceptance_and_partial_training_are_not_completed_results():
    registry = yaml.safe_load((ROOT / "experiments/registry.yaml").read_text(encoding="utf-8"))
    entries = {entry["id"]: entry for entry in registry["experiments"]}
    for ident in ("EXP-FWD-024", "EXP-FWD-025", "EXP-DAT-007"):
        assert entries[ident]["status"] == "active"
        assert entries[ident]["result"] == []
        assert entries[ident]["decision"]["state"] == "pending"
    assert entries["EXP-FWD-016"]["status"] == "blocked"
