"""Freeze upgraded Full-Wh output heads from validation-only evidence.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


TASK_PATTERN = re.compile(
    r"^10K-W(?P<wave>[1-7])_(?P<model>MM|RM|CX|TF|DL)_"
    r"(?P<head>D|Q)_(?P<lane>B|G|J)_s(?P<seed>[0-2])$"
)
MODEL_NAMES = {
    "MM": "mlp_mixer_band_v1",
    "RM": "res_mlp_gated_v2",
    "CX": "convnext_band_v1",
    "TF": "tfno2d_band_v3",
    "DL": "dual_lane_band_operator_v1",
}
HEAD_NAMES = {"D": "dense", "Q": "k_conditioned"}
EXPECTED_STATUSES = {"formal_training_success", "dual_lane_training_success"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in keys:
                keys.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def collect_validation_rows(
    campaign_dirs: Sequence[str | Path],
    *,
    repository_root: str | Path = ".",
    verify_checkpoint_hashes: bool = True,
) -> list[dict[str, Any]]:
    """Collect and validate the frozen 54-run validation matrix."""

    root = Path(repository_root).resolve()
    rows: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    for campaign_value in campaign_dirs:
        campaign = _resolve(root, campaign_value).resolve()
        manifest_path = campaign / "task_manifest.csv"
        for manifest_row in _read_csv(manifest_path):
            task_id = str(manifest_row["task_id"])
            match = TASK_PATTERN.fullmatch(task_id)
            if match is None:
                raise ValueError(f"unexpected 10k task id: {task_id}")
            if task_id in seen_tasks:
                raise ValueError(f"duplicate 10k task id: {task_id}")
            seen_tasks.add(task_id)
            groups = match.groupdict()
            model = MODEL_NAMES[groups["model"]]
            head = HEAD_NAMES[groups["head"]]
            lane = groups["lane"]
            seed = int(groups["seed"])
            if (
                str(manifest_row["model"]) != model
                or str(manifest_row["head"]) != head
                or str(manifest_row["lane"]) != lane
                or int(manifest_row["seed"]) != seed
            ):
                raise ValueError(f"manifest identity mismatch for {task_id}")
            run_dir = _resolve(root, manifest_row["run_dir"]).resolve()
            required = {
                name: run_dir / name
                for name in (
                    "run.log",
                    "run.meta.json",
                    "config.resolved.yaml",
                    "results.json",
                    "metrics.json",
                    "history.csv",
                    "model.pt",
                    "n40_profile.csv",
                    "n40_profile.json",
                )
            }
            missing = [name for name, path in required.items() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"{task_id} missing artifacts: {missing}")
            result = _read_json(required["results.json"])
            metrics = _read_json(required["metrics.json"])
            meta = _read_json(required["run.meta.json"])
            profile = _read_json(required["n40_profile.json"])
            if str(result.get("status")) not in EXPECTED_STATUSES:
                raise ValueError(f"{task_id} has inadmissible status {result.get('status')!r}")
            if bool(result.get("test_executed")):
                raise ValueError(f"{task_id} read test before head selection")
            if str(result.get("model")) != model or int(result.get("seed", -1)) != seed:
                raise ValueError(f"result identity mismatch for {task_id}")
            if profile.get("status") != "profile_collected" or int(profile.get("sample_count", 0)) < 1:
                raise ValueError(f"{task_id} is missing a valid N40 GPU profile")
            if len(profile.get("gpu_uuids") or []) != 1:
                raise ValueError(f"{task_id} did not use exactly one profiled GPU")
            declared_checkpoint_hash = str(result.get("checkpoint_sha256") or "")
            if len(declared_checkpoint_hash) != 64:
                raise ValueError(f"{task_id} is missing its checkpoint SHA-256")
            actual_checkpoint_hash = (
                _sha256(required["model.pt"])
                if verify_checkpoint_hashes
                else declared_checkpoint_hash
            )
            if actual_checkpoint_hash != declared_checkpoint_hash:
                raise ValueError(f"checkpoint hash mismatch for {task_id}")
            metric_key = (
                "validation_equal_lane_bands_2_10_mre_pct"
                if lane == "J"
                else "bands_2_10_mre_pct"
            )
            metric = float(metrics[metric_key])
            if not metric >= 0.0:
                raise ValueError(f"non-finite or negative validation metric for {task_id}")
            parameter_count = int(metrics["parameter_count"])
            if not 2_000_000 <= parameter_count <= 8_000_000:
                raise ValueError(f"{task_id} violates the preregistered 2-8M parameter budget")
            rows.append(
                {
                    "task_id": task_id,
                    "wave": f"10K-W{groups['wave']}",
                    "model": model,
                    "head": head,
                    "lane": lane,
                    "seed": seed,
                    "validation_bands_2_10_mre_pct": metric,
                    "parameter_count": parameter_count,
                    "peak_gpu_memory_mb": float(metrics["peak_gpu_memory_mb"]),
                    "train_time_s": float(metrics["train_time_s"]),
                    "stop_epoch": int(metrics["stop_epoch"]),
                    "run_dir": str(run_dir.relative_to(root)),
                    "dataset_sha256": str(meta["dataset_sha256"]) if lane != "J" else "dual_lane",
                    "results_sha256": _sha256(required["results.json"]),
                    "metrics_sha256": _sha256(required["metrics.json"]),
                    "config_sha256": _sha256(required["config.resolved.yaml"]),
                    "checkpoint_sha256": actual_checkpoint_hash,
                    "checkpoint_hash_verified": bool(verify_checkpoint_hashes),
                }
            )
    if len(rows) != 54:
        raise RuntimeError(f"validation matrix requires 54 tasks, got {len(rows)}")
    return sorted(rows, key=lambda row: str(row["task_id"]))


def summarize_validation(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the 18 model/head/lane three-seed summaries."""

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["head"]), str(row["lane"]))
        grouped.setdefault(key, []).append(row)
    if len(grouped) != 18:
        raise RuntimeError(f"expected 18 validation groups, got {len(grouped)}")
    output: list[dict[str, Any]] = []
    for (model, head, lane), members in sorted(grouped.items()):
        members = sorted(members, key=lambda row: int(row["seed"]))
        seeds = sorted(int(row["seed"]) for row in members)
        if seeds != [0, 1, 2]:
            raise RuntimeError(f"{model}/{head}/{lane} seeds are {seeds}, expected [0, 1, 2]")
        values = [float(row["validation_bands_2_10_mre_pct"]) for row in members]
        output.append(
            {
                "model": model,
                "head": head,
                "lane": lane,
                "seed_0": values[0],
                "seed_1": values[1],
                "seed_2": values[2],
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values),
                "worst_seed": max(values),
                "parameter_count": int(members[0]["parameter_count"]),
                "peak_gpu_memory_mb_max": max(float(row["peak_gpu_memory_mb"]) for row in members),
                "train_time_s_sum": sum(float(row["train_time_s"]) for row in members),
            }
        )
    return output


def freeze_head_selection(
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the preregistered validation-only output-head rule."""

    by_key = {
        (str(row["model"]), str(row["lane"]), str(row["head"])): row
        for row in summaries
    }
    selection_keys = [
        (model, lane)
        for model in (MODEL_NAMES[key] for key in ("MM", "RM", "CX", "TF"))
        for lane in ("B", "G")
    ] + [(MODEL_NAMES["DL"], "J")]
    selections: list[dict[str, Any]] = []
    for model, lane in selection_keys:
        dense = by_key[(model, lane, "dense")]
        query = by_key[(model, lane, "k_conditioned")]
        dense_mean = float(dense["mean"])
        query_mean = float(query["mean"])
        relative_query_improvement = (dense_mean - query_mean) / dense_mean
        if relative_query_improvement >= 0.01:
            selected = "k_conditioned"
            reason = "query_validation_mean_relative_improvement_at_least_1pct"
        elif abs(relative_query_improvement) <= 0.01:
            raise RuntimeError(
                f"{model}/{lane} requires the preregistered 32/128 transfer tie-break; "
                "head selection is not yet frozen"
            )
        else:
            selected = "dense"
            reason = "query_not_better_by_1pct_and_no_tie_break_is_applicable"
        selected_runs = [
            {
                "seed": int(row["seed"]),
                "run_dir": str(row["run_dir"]),
                "checkpoint_sha256": str(row["checkpoint_sha256"]),
                "results_sha256": str(row["results_sha256"]),
            }
            for row in rows
            if str(row["model"]) == model
            and str(row["lane"]) == lane
            and str(row["head"]) == selected
        ]
        selections.append(
            {
                "model": model,
                "lane": lane,
                "selected_head": selected,
                "dense_validation_mean": dense_mean,
                "query_validation_mean": query_mean,
                "query_relative_improvement": relative_query_improvement,
                "selection_reason": reason,
                "transfer_tie_break_used": False,
                "selected_runs": sorted(selected_runs, key=lambda item: int(item["seed"])),
            }
        )
    return {
        "schema_version": "phc2d.full_cell_upgrade_head_selection.v1",
        "status": "head_selection_frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_split": "validation",
        "test_read_during_selection": False,
        "rule": {
            "query_relative_improvement_min": 0.01,
            "tie_band_absolute_relative": 0.01,
            "tie_break": "32_128_zero_shot_only_when_within_tie_band",
            "default": "dense",
        },
        "validation_task_count": len(rows),
        "selections": selections,
    }


def run_selection(
    campaign_dirs: Sequence[str | Path],
    *,
    output_dir: str | Path,
    repository_root: str | Path = ".",
    verify_checkpoint_hashes: bool = True,
) -> Path:
    """Audit the matrix and write the immutable head-selection evidence."""

    root = Path(repository_root).resolve()
    out = _resolve(root, output_dir)
    out.mkdir(parents=True, exist_ok=False)
    rows = collect_validation_rows(
        campaign_dirs,
        repository_root=root,
        verify_checkpoint_hashes=verify_checkpoint_hashes,
    )
    summaries = summarize_validation(rows)
    lock = freeze_head_selection(rows, summaries)
    _write_csv(out / "validation_runs.csv", rows)
    _write_csv(out / "validation_summary.csv", summaries)
    _write_csv(out / "history.csv", summaries)
    (out / "head_selection.lock.json").write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    audit = {
        "status": "pass",
        "expected_task_count": 54,
        "actual_task_count": len(rows),
        "expected_summary_count": 18,
        "actual_summary_count": len(summaries),
        "checkpoint_hashes_verified": bool(verify_checkpoint_hashes),
        "test_read_count": 0,
    }
    (out / "integrity_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "validation_task_count": len(rows),
                "validation_summary_count": len(summaries),
                "train_time_s_sum": sum(float(row["train_time_s"]) for row in rows),
                "peak_gpu_memory_mb_max": max(float(row["peak_gpu_memory_mb"]) for row in rows),
                "checkpoint_hashes_verified": bool(verify_checkpoint_hashes),
                "test_read_count": 0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "config.resolved.yaml").write_text(
        json.dumps(
            {
                "campaign_dirs": [str(value) for value in campaign_dirs],
                "selection_split": "validation",
                "evaluation_split": None,
                "checkpoint_hash_verification": bool(verify_checkpoint_hashes),
                "head_rule": lock["rule"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "results.json").write_text(
        json.dumps(
            {
                "status": "10k_validation_aggregation_and_head_selection_success",
                "test_executed": False,
                "next_gate": "frozen_head_10k_held_out_test",
                "head_selection_lock_sha256": _sha256(out / "head_selection.lock.json"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "run.meta.json").write_text(
        json.dumps(
            {
                "script": "python -m phc2d.forward.full_cell_upgrade_selection",
                "campaign_dirs": [str(value) for value in campaign_dirs],
                "selection_split": "validation",
                "evaluation_split": None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "run.log").write_text(
        "10k validation aggregation passed; head selection frozen without reading test.\n",
        encoding="utf-8",
    )
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--skip-checkpoint-hash-verification", action="store_true")
    args = parser.parse_args(argv)
    print(
        run_selection(
            args.campaign_dir,
            output_dir=args.output_dir,
            repository_root=args.repository_root,
            verify_checkpoint_hashes=not args.skip_checkpoint_hash_verification,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "collect_validation_rows",
    "freeze_head_selection",
    "run_selection",
    "summarize_validation",
]
