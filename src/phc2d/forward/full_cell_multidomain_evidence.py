"""Fail-closed evaluation audits and append-only log-correction lineage.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from phc2d.forward.full_cell_multidomain_contracts import (
    read_csv, repository_commit, sha256_array, sha256_file, write_csv,
)
from phc2d.models import get_model_factory_identity

CELL_ARTIFACTS = (
    "run.log", "run.meta.json", "config.resolved.yaml", "history.csv",
    "metrics.json", "results.json", "per_band_metrics.csv", "per_k_metrics.csv",
    "sample_metrics.npz",
)
CORRECTION_MANIFEST = "log_correction_manifest.csv"


def _check_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("non-finite evaluation value")
    if isinstance(value, dict):
        for item in value.values():
            _check_finite(item)
    elif isinstance(value, list):
        for item in value:
            _check_finite(item)


def _cell_path(campaign: Path, relative: str) -> Path:
    path = (campaign / relative).resolve()
    if not path.is_relative_to((campaign / "evaluation").resolve()):
        raise RuntimeError("evaluation cell path escapes the campaign")
    return path


def evaluation_rows(campaign_dir: str | Path) -> list[dict[str, str]]:
    """Resolve explicitly frozen derivatives without rewriting original identities."""
    campaign = Path(campaign_dir).resolve()
    rows = read_csv(campaign / "evaluation/evaluation_manifest.csv")
    if len({r["cell_id"] for r in rows}) != len(rows) or len({r["cell_index"] for r in rows}) != len(rows):
        raise RuntimeError("duplicate evaluation identity")
    correction_file = campaign / "evaluation" / CORRECTION_MANIFEST
    if not correction_file.exists():
        for row in rows:
            _cell_path(campaign, row["cell_dir"])
        if len({r["cell_dir"] for r in rows}) != len(rows):
            raise RuntimeError("evaluation output path collision")
        return rows
    corrections = read_csv(correction_file)
    by_id = {r["cell_id"]: r for r in corrections}
    if len(by_id) != len(corrections) or not set(by_id).issubset({r["cell_id"] for r in rows}):
        raise RuntimeError("invalid correction identity")
    output = []
    for row in rows:
        correction = by_id.get(row["cell_id"])
        if correction is not None:
            if correction["source_cell_dir"] != row["cell_dir"] or correction["cell_index"] != row["cell_index"]:
                raise RuntimeError("correction source identity mismatch")
            source = _cell_path(campaign, correction["source_cell_dir"])
            source_hashes = json.loads(correction["source_artifact_hashes"])
            if set(source_hashes) != set(CELL_ARTIFACTS):
                raise RuntimeError("incomplete original correction artifact hashes")
            if correction["cell_dir"] == row["cell_dir"]:
                raise RuntimeError("correction must use a distinct derivative directory")
            for name, digest in source_hashes.items():
                if sha256_file(source / name) != digest:
                    raise RuntimeError(f"original correction evidence changed: {row['cell_id']}/{name}")
            row = {**row, **correction}
        _cell_path(campaign, row["cell_dir"])
        output.append(row)
    if len({r["cell_dir"] for r in output}) != len(output):
        raise RuntimeError("evaluation output path collision")
    return output


def audit_evaluation_runs(
    campaign_dir: str | Path, *, start_index: int = 0, end_index: int | None = None,
    allow_empty_logs: bool = False,
) -> dict[str, Any]:
    """Audit a completed slice, including immutable identities and finite per-band evidence."""
    campaign = Path(campaign_dir).resolve()
    evaluation = campaign / "evaluation"
    manifest = evaluation_rows(campaign)
    end = max(int(r["cell_index"]) for r in manifest) if end_index is None else end_index
    rows = [r for r in manifest if start_index <= int(r["cell_index"]) <= end]
    if len(rows) != end - start_index + 1:
        raise RuntimeError("evaluation slice manifest is incomplete")
    caches = {r["target_domain"]: r for r in read_csv(evaluation / "cache_manifest.csv")}
    norms = {r["regime"]: r for r in read_csv(campaign / "normalizer_manifest.csv")}
    checked_files: dict[str, str] = {}
    for row in rows:
        cell = _cell_path(campaign, row["cell_dir"])
        for name in CELL_ARTIFACTS:
            file = cell / name
            if not file.is_file() or (file.stat().st_size == 0 and not (name == "run.log" and allow_empty_logs)):
                raise RuntimeError(f"evaluation artifact missing or empty: {row['cell_id']}/{name}")
        result = json.loads((cell / "results.json").read_text(encoding="utf-8"))
        metrics = json.loads((cell / "metrics.json").read_text(encoding="utf-8"))
        _check_finite(metrics)
        for key in ("band1_mre_pct", "bands_2_10_mre_pct", "bands_1_10_mre_pct"):
            if key not in metrics or not math.isfinite(float(metrics[key])):
                raise RuntimeError(f"invalid required metric: {row['cell_id']}/{key}")
        if result.get("status") != "evaluation_cell_success":
            raise RuntimeError(f"evaluation did not succeed: {row['cell_id']}")
        for key in ("cell_id", "regime", "model", "target_domain"):
            if result.get(key) != row[key]:
                raise RuntimeError(f"evaluation identity mismatch: {row['cell_id']}/{key}")
        if result.get("seed") != int(row["seed"]):
            raise RuntimeError(f"evaluation seed mismatch: {row['cell_id']}")
        cache = caches[row["target_domain"]]
        normalizer = norms[row["regime"]]
        for path, expected in (
            (Path(row["checkpoint_path"]), row["checkpoint_sha256"]),
            (evaluation / cache["cache_path"], cache["cache_sha256"]),
            (campaign / normalizer["normalizer_path"], normalizer["normalizer_file_sha256"]),
        ):
            digest = checked_files.setdefault(str(path), "")
            if not digest:
                digest = checked_files[str(path)] = sha256_file(path)
            if digest != expected:
                raise RuntimeError(f"evaluation source hash mismatch: {path}")
        for payload in (result, metrics):
            if payload.get("checkpoint_sha256") != row["checkpoint_sha256"] or payload.get("cache_sha256") != cache["cache_sha256"]:
                raise RuntimeError(f"evaluation provenance mismatch: {row['cell_id']}")
        identity = metrics["model_identity"]
        if identity.get("canonical_model_name") != row["model"] or identity.get("factory_function_identity") != get_model_factory_identity(row["model"]):
            raise RuntimeError(f"evaluation factory identity mismatch: {row['cell_id']}")
        if metrics.get("normalizer_file_sha256") != normalizer["normalizer_file_sha256"]:
            raise RuntimeError(f"evaluation pooled normalizer mismatch: {row['cell_id']}")
        if "correction_id" in row:
            lineage = result.get("correction_lineage", {})
            for key in ("correction_id", "source_cell_dir", "source_artifact_hashes", "correction_execution_commit"):
                if lineage.get(key) != row[key]:
                    raise RuntimeError(f"correction lineage mismatch: {row['cell_id']}/{key}")
        bands = read_csv(cell / "per_band_metrics.csv")
        if result.get("per_band_record_count") != 10 or len(bands) != 10 or {int(b["band_one_based"]) for b in bands} != set(range(1, 11)):
            raise RuntimeError(f"invalid per-band records: {row['cell_id']}")
        for band in bands:
            if band.get("cell_id") != row["cell_id"] or not math.isfinite(float(band["mre_pct"])):
                raise RuntimeError(f"invalid per-band identity/value: {row['cell_id']}")
        with np.load(cell / "sample_metrics.npz") as samples:
            ids = np.asarray(samples["sample_ids"]).astype("S")
            if "sample_ids_sha256" in cache and sha256_array(ids) != cache["sample_ids_sha256"]:
                raise RuntimeError(f"evaluation sample identity mismatch: {row['cell_id']}")
            for key in samples.files:
                values = samples[key]
                if np.issubdtype(values.dtype, np.number) and not np.isfinite(values).all():
                    raise RuntimeError(f"non-finite sample metrics: {row['cell_id']}/{key}")
    return {"status": "evaluation_slice_complete", "start_index": start_index, "end_index": end, "cell_count": len(rows)}


def freeze_log_correction(
    campaign_dir: str | Path, *, correction_id: str, authorization_id: str,
    repository_root: str | Path,
) -> Path:
    """Freeze only log-defective successful cells as an immutable derivative plan."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", correction_id) or not authorization_id:
        raise ValueError("a safe correction ID and explicit authorization identity are required")
    campaign = Path(campaign_dir).resolve()
    destination = campaign / "evaluation" / CORRECTION_MANIFEST
    if destination.exists():
        raise FileExistsError("a log correction is already frozen; no automatic retry")
    correction = campaign / "evaluation/corrections" / correction_id
    if correction.exists():
        raise FileExistsError("correction directory already exists; refusing reuse")
    rows = read_csv(campaign / "evaluation/evaluation_manifest.csv")
    corrections = []
    commit = repository_commit(repository_root)
    for row in rows:
        cell = _cell_path(campaign, row["cell_dir"])
        if not (cell / "results.json").exists():
            continue
        log = cell / "run.log"
        if not log.is_file() or log.stat().st_size != 0:
            continue
        audit_evaluation_runs(campaign, start_index=int(row["cell_index"]), end_index=int(row["cell_index"]), allow_empty_logs=True)
        replacement = f"evaluation/corrections/{correction_id}/cells/{row['regime']}/{row['model']}_s0/{row['target_domain']}"
        corrections.append({
            "cell_index": row["cell_index"], "cell_id": row["cell_id"],
            "source_cell_dir": row["cell_dir"], "cell_dir": replacement,
            "correction_id": correction_id, "authorization_id": authorization_id,
            "correction_execution_commit": commit,
            "source_artifact_hashes": json.dumps({name: sha256_file(cell / name) for name in CELL_ARTIFACTS}, sort_keys=True),
        })
    if not corrections:
        raise RuntimeError("no proven log-only defective cells to correct")
    write_csv(destination, corrections)
    correction.mkdir(parents=True, exist_ok=False)
    (correction / "config.resolved.yaml").write_bytes((campaign / "config.resolved.yaml").read_bytes())
    (correction / "run.log").write_text(f"Frozen {len(corrections)} log-only derivatives; original cells preserved.\n", encoding="utf-8")
    return destination
