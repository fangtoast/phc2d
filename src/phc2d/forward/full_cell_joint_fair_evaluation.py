"""Incremental six-domain evaluation for a fair-total-data joint checkpoint.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from phc2d.forward.full_cell_cross_domain_n40 import sha256_file
from phc2d.forward.full_cell_metrics import compute_band_metrics, paired_bootstrap_mre
from phc2d.forward.full_cell_six_domain_n40 import DOMAINS, run_shard


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("fair joint evaluation config must be a mapping")
    return dict(value)


def _baseline_rows(config: Mapping[str, Any]) -> tuple[Path, dict[str, dict[str, str]]]:
    campaign = Path(str(config["baseline_campaign"]))
    result = json.loads((campaign / "results.json").read_text(encoding="utf-8"))
    if result.get("status") != "evaluation_complete_pending_report":
        raise RuntimeError("EXP-FWD-009 baseline campaign is not complete")
    identity = {key: str(value) for key, value in dict(config["baseline_identity"]).items()}
    selected = [
        row
        for row in _read_csv(campaign / "evaluation_cells.csv")
        if all(str(row[key]) == value for key, value in identity.items())
    ]
    by_domain = {row["target_domain"]: row for row in selected}
    if len(selected) != len(DOMAINS) or set(by_domain) != set(DOMAINS):
        raise RuntimeError("baseline joint identity does not cover the six frozen domains exactly")
    return campaign, {domain: by_domain[domain] for domain in DOMAINS}


def freeze_evaluation(
    config_path: str | Path,
    checkpoint_path: str | Path,
    campaign_dir: str | Path,
    *,
    repository_root: str | Path = ".",
) -> Path:
    """Freeze only the six cells affected by the replacement joint checkpoint."""

    root = Path(repository_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = _load_config(config_file)
    if str(config.get("experiment_id")) != "EXP-FWD-010" or tuple(config.get("domains", ())) != DOMAINS:
        raise ValueError("EXP-FWD-010 must freeze K5/K20/F50 x Binary/Graded exactly")
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    campaign = Path(campaign_dir)
    if not campaign.is_absolute():
        campaign = root / campaign
    if campaign.exists():
        raise FileExistsError(campaign)
    campaign.mkdir(parents=True)
    baseline_campaign, baseline_rows = _baseline_rows(config)
    cache_rows = _read_csv(baseline_campaign / "test_cache_manifest.csv")
    cache_by_domain = {row["domain"]: row for row in cache_rows}
    if tuple(cache_by_domain) != DOMAINS:
        raise RuntimeError("baseline cache does not cover the six frozen domains exactly")
    checkpoint_sha256 = sha256_file(checkpoint)
    cells = [
        {
            "evaluation_id": f"fairjoint_{index:02d}",
            "checkpoint_id": "EXP-FWD-010_dual_lane_band_operator_v1_J_s0",
            "source_family": "K5",
            "source_model": "dual_lane_band_operator_v1",
            "source_lane": "J",
            "source_seed": 0,
            "checkpoint_path": checkpoint.as_posix(),
            "checkpoint_sha256": checkpoint_sha256,
            "target_domain": domain,
            "target_family": domain.split("-")[0],
            "target_lane": domain[-1],
            "comparison_kind": "joint_source_family_fair_total_8k",
            "status": "planned",
            "shard_id": 0,
            "baseline_evaluation_id": baseline_rows[domain]["evaluation_id"],
        }
        for index, domain in enumerate(DOMAINS)
    ]
    _write_csv(campaign / "test_cache_manifest.csv", cache_rows)
    _write_csv(campaign / "evaluation_plan.csv", cells)
    _write_csv(
        campaign / "checkpoint_manifest.csv",
        [{"model": "dual_lane_band_operator_v1", "source_lane": "J", "seed": 0, "checkpoint_path": checkpoint.as_posix(), "checkpoint_sha256": checkpoint_sha256}],
    )
    baseline_config = _load_config(baseline_campaign / "config.resolved.yaml")
    resolved = {
        **config,
        "datasets": baseline_config["datasets"],
        "checkpoint_path": checkpoint.as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "baseline_evaluation_cells_sha256": sha256_file(baseline_campaign / "evaluation_cells.csv"),
        "inherited_test_cache_manifest_sha256": sha256_file(baseline_campaign / "test_cache_manifest.csv"),
    }
    (campaign / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        campaign / "campaign_status.json",
        {"status": "evaluation_plan_ready", "planned_checkpoints": 1, "planned_cells": 6, "planned_shards": 1, "next_gate": "authorized_gpu_evaluation"},
    )
    (campaign / "history.csv").write_text("evaluation_only,no_training_history\ntrue,true\n", encoding="utf-8")
    return campaign


def finalize_evaluation(campaign_dir: str | Path, *, repository_root: str | Path = ".") -> Path:
    root = Path(repository_root).resolve()
    campaign = Path(campaign_dir)
    if not campaign.is_absolute():
        campaign = root / campaign
    config = _load_config(campaign / "config.resolved.yaml")
    baseline_campaign, baseline_rows = _baseline_rows(config)
    current_rows = _read_csv(campaign / "shards" / "shard_00" / "evaluation_cells.csv")
    if len(current_rows) != 6 or {row["target_domain"] for row in current_rows} != set(DOMAINS):
        raise RuntimeError("EXP-FWD-010 incremental evaluation is not 6/6 complete")
    comparisons: list[dict[str, Any]] = []
    for current in current_rows:
        domain = current["target_domain"]
        baseline = baseline_rows[domain]
        current_npz = campaign / "shards" / "shard_00" / f"{current['evaluation_id']}.npz"
        baseline_npz = baseline_campaign / "shards" / f"shard_{int(baseline['shard_id']):02d}" / f"{baseline['evaluation_id']}.npz"
        with np.load(current_npz) as current_archive, np.load(baseline_npz) as baseline_archive:
            current_target = current_archive["target"]
            baseline_target = baseline_archive["target"]
            if not np.array_equal(current_target, baseline_target):
                raise RuntimeError(f"frozen target drift for {domain}")
            current_metrics, current_samples, _, _ = compute_band_metrics(current_archive["prediction"], current_target, bootstrap_replicates=0)
            baseline_metrics, baseline_samples, _, _ = compute_band_metrics(baseline_archive["prediction"], baseline_target, bootstrap_replicates=0)
        seed = int(hashlib.sha256(domain.encode()).hexdigest()[:8], 16)
        comparisons.append(
            {
                "target_domain": domain,
                "new_band1_mre_pct": current_metrics["band1_mre"],
                "new_bands2_10_mre_pct": current_metrics["bands2_10_mre"],
                "new_bands1_10_mre_pct": current_metrics["bands1_10_mre"],
                "baseline_band1_mre_pct": baseline_metrics["band1_mre"],
                "baseline_bands2_10_mre_pct": baseline_metrics["bands2_10_mre"],
                "baseline_bands1_10_mre_pct": baseline_metrics["bands1_10_mre"],
                **paired_bootstrap_mre(
                    current_samples,
                    baseline_samples,
                    seed=seed,
                    replicates=int(config["statistics"]["paired_bootstrap_replicates"]),
                ),
            }
        )
    _write_csv(campaign / "evaluation_cells.csv", current_rows)
    _write_csv(campaign / "comparison_to_exp_fwd_009_joint.csv", comparisons)
    _write_json(campaign / "metrics.json", {"domains": comparisons})
    _write_json(
        campaign / "results.json",
        {"status": "evaluation_complete_pending_report", "completed_checkpoints": 1, "completed_cells": 6, "completed_shards": 1, "baseline": "EXP-FWD-009 8k+8k joint", "surrogate_only": True, "real_mpb_replay": False},
    )
    _write_json(
        campaign / "run.meta.json",
        {"script": "python -m phc2d.forward.full_cell_joint_fair_evaluation", "created_at_utc": datetime.now(timezone.utc).isoformat(), "checkpoint_sha256": config["checkpoint_sha256"]},
    )
    (campaign / "run.log").write_text("six affected joint cells evaluated and compared with EXP-FWD-009.\n", encoding="utf-8")
    _write_json(campaign / "campaign_status.json", {"status": "evaluation_complete_pending_report", "planned_cells": 6, "completed_cells": 6, "next_gate": "draft_report_update"})
    return campaign


def run_evaluation(campaign_dir: str | Path, *, repository_root: str | Path = ".", device: str = "cuda:0") -> Path:
    run_shard(campaign_dir, 0, repository_root=repository_root, device=device)
    return finalize_evaluation(campaign_dir, repository_root=repository_root)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--checkpoint", required=True)
    freeze.add_argument("--campaign-dir", required=True)
    freeze.add_argument("--repository-root", default=".")
    run = sub.add_parser("run")
    run.add_argument("--campaign-dir", required=True)
    run.add_argument("--repository-root", default=".")
    run.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        print(freeze_evaluation(args.config, args.checkpoint, args.campaign_dir, repository_root=args.repository_root))
    else:
        print(run_evaluation(args.campaign_dir, repository_root=args.repository_root, device=args.device))


if __name__ == "__main__":
    main()


__all__ = ["finalize_evaluation", "freeze_evaluation", "run_evaluation"]
