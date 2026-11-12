"""Audit adaptive training artifacts and summarize validation-only three-seed results.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from phc2d.forward.budget_confirmation import file_digest, read_tasks, write_json, write_table


def summarize_campaign(campaign: Path) -> dict:
    cells, groups, artifacts = [], defaultdict(list), []
    for row in read_tasks(campaign):
        run = Path(row["run_dir"])
        result = json.loads((run / "results.json").read_text())
        metrics = json.loads((run / "metrics.json").read_text())
        if not 400 <= result["stop_epoch"] <= 1000 or (result["stop_epoch"] < 1000 and not result["convergence"]["passed"]):
            raise ValueError("invalid adaptive stopping epoch or decision")
        with (run / "history.csv").open() as handle:
            history = list(csv.DictReader(handle))
        if [int(r["epoch"]) for r in history] != list(range(1, result["stop_epoch"] + 1)):
            raise ValueError("non-contiguous completed history")
        if result["model"] != row["model"] or int(result["seed"]) != int(row["seed"]) or result["test_executed"] or result["selection_split"] != "validation":
            raise ValueError("task identity or validation boundary changed")
        if result["checkpoint_sha256"] != file_digest(run / "model.pt") or result["terminal_checkpoint_sha256"] != file_digest(Path(result["terminal_checkpoint_path"])):
            raise ValueError("completed checkpoint hash mismatch")
        for record in history:
            if not np.isfinite([float(record[key]) for key in ("train_loss", "val_bands_1_10_mre_pct", "val_bands_2_10_mre_pct")]).all():
                raise ValueError("non-finite completed history")
        primary = float(metrics["bands_1_10_mre_pct"])
        if not np.isfinite(primary):
            raise ValueError("non-finite selected checkpoint metric")
        cell = dict(case_id=row["case_id"], group=row["case_id"].rsplit("_s", 1)[0], model=row["model"],
                    seed=int(row["seed"]), bands_1_10_mre_pct=primary,
                    best_epoch=result["best_epoch"], stop_epoch=result["stop_epoch"], converged=result["convergence"]["passed"])
        cells.append(cell)
        groups[cell["group"]].append(cell)
        artifacts.append(dict(case_id=row["case_id"], hashes={name: file_digest(run / name) for name in
                              ("model.pt", "training_state.pt", "history.csv", "metrics.json", "results.json", "config.resolved.yaml", "run.meta.json")}))
    summaries = []
    for group, records in groups.items():
        if sorted(r["seed"] for r in records) != [0, 1, 2]:
            raise ValueError("incomplete or duplicated three-seed group")
        values = [r["bands_1_10_mre_pct"] for r in records]
        summaries.append(dict(group=group, model=records[0]["model"], seed_count=3,
                              bands_1_10_mre_pct_mean=float(np.mean(values)),
                              bands_1_10_mre_pct_seed_sd=float(np.std(values, ddof=1)),
                              converged_seed_count=sum(r["converged"] for r in records)))
    write_table(campaign / "validation_primary_cells.csv", cells)
    write_table(campaign / "validation_primary_summary.csv", summaries)
    audit = dict(status="training_artifacts_complete_pending_independent_review", task_count=len(cells),
                 converged_count=sum(r["converged"] for r in cells), seed_group_count=len(summaries),
                 validation_only=True, test_executed=False, physical_claim=False, artifacts=artifacts)
    write_json(campaign / "training_audit.json", audit)
    write_json(campaign / "results.json", {key: value for key, value in audit.items() if key != "artifacts"})
    return audit


def finish_if_complete(suite: Path, campaigns: list[str]) -> None:
    if not all((suite / "train_claims" / (Path(c).name + "_" + row["case_id"]) / "completed.json").is_file()
               for c in campaigns for row in read_tasks(Path(c))):
        return
    try:
        with (suite / "aggregation_claim.json").open("x") as handle:
            json.dump(dict(status="aggregation_intent_no_automatic_retry"), handle)
    except FileExistsError:
        return
    for campaign in campaigns:
        summarize_campaign(Path(campaign))
