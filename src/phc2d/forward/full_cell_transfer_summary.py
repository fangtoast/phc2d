"""Merge independently audited domain matrices without re-reading held-out data.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from phc2d.forward.full_cell_multidomain_contracts import audit_training_runs, read_csv, sha256_file, write_csv
from phc2d.forward.full_cell_multidomain_evaluation import (
    evaluation_status, finalize_evaluation, load_evaluation_campaign,
)
from phc2d.forward.full_cell_multidomain_evidence import evaluation_rows
from phc2d.forward.full_cell_transfer_contracts import checkpoint_identity_map
from phc2d.forward.global_local_bridge_evaluation import paired_bootstrap
from phc2d.runs import write_json

METRIC_KEYS = ("band1_mre_pct", "bands_2_10_mre_pct", "bands_1_10_mre_pct")


def merge_domain_matrix(
    rows: Sequence[Mapping[str, Any]], checkpoint_rows: Sequence[Mapping[str, Any]],
    targets: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Require an exact checkpoint-by-target product and equal-domain macros."""
    expected = {(r["regime"], r["model"], target) for r in checkpoint_rows for target in targets}
    observed = [(r["regime"], r["model"], r["target_domain"]) for r in rows]
    if len(expected) != len(checkpoint_rows) * len(targets) or len(set(targets)) != len(targets):
        raise RuntimeError("matrix contract identity collision")
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise RuntimeError("matrix is incomplete or contains duplicate/unexpected cells")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    normalized = []
    for row in rows:
        metrics = {key: float(row[key]) for key in METRIC_KEYS}
        if not np.isfinite(list(metrics.values())).all():
            raise RuntimeError("combined matrix contains non-finite metrics")
        normalized.append({**row, **metrics})
        grouped[(row["regime"], row["model"])].append(metrics)
    macros = [{"regime": regime, "model": model, "target_domain_count": len(targets),
               "macro_weighting": "equal_target_domain",
               **{key: float(np.mean([r[key] for r in values])) for key in METRIC_KEYS}}
              for (regime, model), values in grouped.items()]
    return normalized, macros


def finalize_transfer_summary(campaign_dir: str | Path) -> Path:
    """Audit both halves and produce a 756-cell derivative plus paired K shifts."""
    campaign, profile = load_evaluation_campaign(campaign_dir)
    contract = profile["transfer_evaluation"]
    source, source_profile = load_evaluation_campaign(contract["source_campaign"])
    audit_training_runs(source / "task_manifest.csv")
    if sha256_file(source / "evaluation/checkpoint_manifest.csv") != contract["source_checkpoint_manifest_sha256"]:
        raise RuntimeError("frozen source checkpoint manifest changed")
    if checkpoint_identity_map(source) != checkpoint_identity_map(campaign):
        raise RuntimeError("source/extension checkpoint identity mismatch")
    targets = [*source_profile["evaluation"]["targets"], *profile["evaluation"]["targets"]]
    checkpoints = read_csv(campaign / "evaluation/checkpoint_manifest.csv")
    if len(checkpoints) * len(targets) != contract["combined_cell_count"]:
        raise RuntimeError("combined cell count mismatch")
    output = campaign / "evaluation/combined"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("combined evidence already exists; no automatic overwrite")
    matrices = []
    bands = []
    bootstrap = []
    samples: dict[tuple[str, str, str], Path] = {}
    parents = []
    for parent in (source, campaign):
        status = evaluation_status(parent)
        if status["status"] != "evaluation_complete":
            raise RuntimeError(f"evaluation half is incomplete: {parent}")
        result = json.loads((parent / "evaluation/results.json").read_text(encoding="utf-8"))
        if result.get("status") != "finalized_seed0_exploratory_surrogate_evidence":
            finalize_evaluation(parent)
        for row in read_csv(parent / "evaluation/cell_matrix.csv"):
            acceptance = "source_frozen_dataset" if parent == source else contract["dataset_acceptance"]
            matrices.append({**row, "evidence_campaign": str(parent), "dataset_acceptance": acceptance})
        for row in evaluation_rows(parent):
            cell = parent / row["cell_dir"]
            bands.extend({**b, "regime": row["regime"], "model": row["model"]} for b in read_csv(cell / "per_band_metrics.csv"))
            samples[(row["regime"], row["model"], row["target_domain"])] = cell / "sample_metrics.npz"
        bootstrap.extend(read_csv(parent / "evaluation/paired_bootstrap.csv"))
        parents.append({"campaign": str(parent), "checkpoint_manifest_sha256": sha256_file(parent / "evaluation/checkpoint_manifest.csv"),
                        "cache_manifest_sha256": sha256_file(parent / "evaluation/cache_manifest.csv"),
                        "cell_matrix_sha256": sha256_file(parent / "evaluation/cell_matrix.csv")})
    matrix, macros = merge_domain_matrix(matrices, checkpoints, targets)
    band_count = int(next(iter(source_profile["datasets"].values()))["target_shape"][-1])
    if len(bands) != band_count * len(matrix):
        raise RuntimeError("combined per-band record count mismatch")
    shifts = []
    replicates = int(profile["evaluation"]["bootstrap_replicates"])
    for checkpoint in checkpoints:
        regime, model = checkpoint["regime"], checkpoint["model"]
        for lane in ("binary", "graded"):
            with np.load(samples[(regime, model, f"k8_{lane}")]) as payload:
                baseline = np.asarray(payload["sample_mre_bands_2_10_pct"], dtype=np.float64)
                ids = np.asarray(payload["sample_ids"]).astype("U")
            for cutoff in (10, 15):
                target = f"k{cutoff}_{lane}"
                with np.load(samples[(regime, model, target)]) as payload:
                    candidate = np.asarray(payload["sample_mre_bands_2_10_pct"], dtype=np.float64)
                    if not np.array_equal(ids, np.asarray(payload["sample_ids"]).astype("U")):
                        raise RuntimeError("paired K-shift sample identities do not align")
                shifts.append({"regime": regime, "model": model, "baseline_domain": f"k8_{lane}", "target_domain": target,
                               **paired_bootstrap(candidate, baseline, key=f"EXP-FWD-023:{regime}:{model}:{target}:paired-K", replicates=replicates)})
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in matrix:
        grouped[(row["regime"], row["model"], row["transfer_group"])].append(row)
    group_rows = [{"regime": regime, "model": model, "transfer_group": group, "target_domain_count": len(values),
                   **{key: float(np.mean([r[key] for r in values])) for key in METRIC_KEYS}}
                  for (regime, model, group), values in grouped.items()]
    resources = []
    for task in read_csv(source / "task_manifest.csv"):
        run = source / task["run_dir"]
        result = json.loads((run / "results.json").read_text(encoding="utf-8"))
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        resources.append({"regime": task["regime"], "model": task["model"], "seed": task["seed"], "training_status": result["status"],
                          "train_time_s": metrics["train_time_s"], "peak_gpu_memory_mb": metrics["peak_gpu_memory_mb"]})
    ranks = {key: sorted(macros, key=lambda r: (r[key], r["regime"], r["model"])) for key in METRIC_KEYS}
    conflicts = []
    for regime in profile["regimes"]:
        best = {key: next(r for r in ranks[key] if r["regime"] == regime)["model"] for key in METRIC_KEYS[1:]}
        conflicts.append({"regime": regime, **best, "ranking_conflict": best[METRIC_KEYS[1]] != best[METRIC_KEYS[2]]})
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "cell_matrix.csv", matrix)
    write_csv(output / "twelve_domain_macro.csv", macros)
    write_csv(output / "per_band_metrics.csv", bands)
    write_csv(output / "paired_model_bootstrap.csv", bootstrap)
    write_csv(output / "paired_k_shift_bootstrap.csv", shifts)
    write_csv(output / "transfer_groups.csv", group_rows)
    write_csv(output / "training_resources.csv", resources)
    write_csv(output / "ranking_conflicts.csv", conflicts)
    summary = {"status": "finalized_twelve_domain_exploratory_evidence", "checkpoint_count": len(checkpoints),
               "target_domain_count": len(targets), "evaluation_cell_count": len(matrix), "per_band_record_count": len(bands),
               "macro_weighting": "equal_target_domain", "parents": parents, "dataset_acceptance": contract["dataset_acceptance"],
               "model_promotion": False, "physical_claim": False, "seed_count": 1,
               "sample_bootstrap_is_not_cross_seed_stability": True}
    write_json(output / "results.json", summary)
    write_json(output / "metrics.json", {**summary, "training_time_s_total_unique": sum(float(r["train_time_s"]) for r in resources)})
    (output / "run.log").write_text("Merged two audited 378-cell halves; no duplicate training or source test reads.\n", encoding="utf-8")
    (output / "config.resolved.yaml").write_text((campaign / "config.resolved.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    write_json(output / "run.meta.json", {"experiment_id": "EXP-FWD-023", "source_campaign": str(source), "extension_campaign": str(campaign)})
    write_csv(output / "history.csv", [{"phase": "merge", "cell_count": len(matrix), "status": "complete"}])
    return output
