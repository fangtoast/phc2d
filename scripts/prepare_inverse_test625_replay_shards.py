"""Prepare real-MPB replay shard configs for the V2 inverse 625-test run.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml


EXPERIMENT_ID = "inverse_mpb_replay_mixture_v2_10000_test625"
DEFAULT_DATASET = Path("data/query_dataset_v2_pilot_10000_mpb.npz")
DEFAULT_BASE_CONFIG = Path("configs/research/inverse_mpb_replay_mixture_v2_10000_40.yaml")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return payload


def write_yaml(path: Path, payload: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=False)


MappingLike = dict[str, Any]


def load_idx_test(dataset_path: Path) -> list[int]:
    with np.load(dataset_path) as data:
        if "idx_test" not in data:
            raise ValueError(f"Dataset is missing idx_test: {dataset_path}")
        return [int(item) for item in np.asarray(data["idx_test"]).reshape(-1)]


def load_candidate_target_indices(candidates_path: Path) -> list[int]:
    payload = read_json(candidates_path)
    targets = payload.get("targets") if isinstance(payload, dict) else None
    if not isinstance(targets, list) or not targets:
        raise ValueError(f"candidates.json must contain a non-empty targets list: {candidates_path}")
    indices: list[int] = []
    missing: list[int] = []
    for pos, record in enumerate(targets):
        if not isinstance(record, dict) or "target_index" not in record:
            missing.append(pos)
            continue
        indices.append(int(record["target_index"]))
    if missing:
        raise ValueError(f"Missing target_index in candidates targets at positions: {missing[:20]}")
    unique = sorted(set(indices))
    if len(unique) != len(indices):
        duplicates = sorted(index for index in unique if indices.count(index) > 1)
        raise ValueError(f"candidates.json contains duplicate target_index values: {duplicates[:20]}")
    return unique


def require_exact_test_split(candidate_indices: Sequence[int], idx_test: Sequence[int]) -> None:
    candidate_list = [int(item) for item in candidate_indices]
    test_list = [int(item) for item in idx_test]
    if candidate_list == test_list:
        return
    missing = sorted(set(test_list) - set(candidate_list))
    extra = sorted(set(candidate_list) - set(test_list))
    raise ValueError(
        "Candidate target indices do not exactly equal dataset idx_test. "
        f"candidate_count={len(candidate_list)} idx_test_count={len(test_list)} "
        f"missing={missing[:20]} extra={extra[:20]}"
    )


def chunked(values: Sequence[int], size: int) -> list[list[int]]:
    if size <= 0:
        raise ValueError("shard size must be positive")
    return [list(values[start : start + size]) for start in range(0, len(values), size)]


def default_output_root(root: Path, timestamp: str) -> Path:
    date = timestamp[:8]
    return root / "runs" / date / f"{EXPERIMENT_ID}_plan_{timestamp}"


def build_shard_config(
    base_cfg: MappingLike,
    *,
    shard_id: int,
    shard_indices: Sequence[int],
    candidates_path: Path,
    ranked_candidates_path: Path | None,
    target_indices_path: Path,
    resolved_indices_path: Path,
    run_dir: Path,
    root: Path,
) -> MappingLike:
    cfg = deepcopy(base_cfg)
    cfg.setdefault("experiment", {})
    cfg["experiment"]["experiment_id"] = f"{EXPERIMENT_ID}_shard{shard_id:02d}"
    cfg["experiment"]["family"] = "research"
    cfg["experiment"]["stage"] = "inverse_candidate_real_mpb_replay_test625_shard"

    cfg.setdefault("dataset", {})
    cfg["dataset"]["target_split"] = "test"

    cfg.setdefault("replay", {})
    replay = cfg["replay"]
    replay.update(
        {
            "enabled": True,
            "backend": "real_mpb",
            "require_real_mpb": True,
            "environment": "WSL Ubuntu + conda meep_env",
            "source_experiment_id": "inverse_mixture_generator_v2_10000_test625_sample",
            "candidates_path": rel(candidates_path, root),
            "selection_unit": "target_index",
            "within_target_record_policy": "best_surrogate_mre_bands_2_10",
            "replay_top_m": 5,
            "n_targets": int(len(shard_indices)),
            "target_indices_path": rel(target_indices_path, root),
            "write_target_indices_path": rel(resolved_indices_path, root),
            "selection_seed": None,
            "selection_strategy": f"idx_test_exact_shard_{shard_id:02d}",
            "primary_band_range": [2, 10],
            "preflight_override": False,
            "diagnostic_only": False,
        }
    )
    if ranked_candidates_path is not None and ranked_candidates_path.exists():
        replay["ranked_candidates_path"] = rel(ranked_candidates_path, root)
    else:
        replay.pop("ranked_candidates_path", None)

    cfg.setdefault("output", {})
    cfg["output"]["runs_dir"] = "runs"
    cfg["output"]["run_dir"] = rel(run_dir, root)
    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path, help="625-target inverse sample candidates.json.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, type=Path, help="Dataset containing idx_test.")
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG, type=Path, help="40-target formal replay config to adapt.")
    parser.add_argument("--output-root", type=Path, help="Plan output root. Defaults to runs/{date}/..._plan_{timestamp}.")
    parser.add_argument("--shard-size", default=50, type=int, help="Targets per shard.")
    parser.add_argument("--timestamp", help="Override timestamp in YYYYMMDDHHMMSS format.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d%H%M%S")
    candidates_path = (root / args.candidates).resolve() if not args.candidates.is_absolute() else args.candidates.resolve()
    dataset_path = (root / args.dataset).resolve() if not args.dataset.is_absolute() else args.dataset.resolve()
    base_config_path = (root / args.base_config).resolve() if not args.base_config.is_absolute() else args.base_config.resolve()
    output_root = (
        (root / args.output_root).resolve() if args.output_root and not args.output_root.is_absolute() else args.output_root
    )
    output_root = output_root.resolve() if output_root is not None else default_output_root(root, timestamp)

    candidate_indices = load_candidate_target_indices(candidates_path)
    idx_test = load_idx_test(dataset_path)
    require_exact_test_split(candidate_indices, idx_test)

    base_cfg = read_yaml(base_config_path)
    shards = chunked(candidate_indices, args.shard_size)
    target_index_dir = output_root / "target_indices"
    resolved_index_dir = output_root / "target_indices_resolved"
    config_dir = output_root / "configs"
    run_date = timestamp[:8]

    ranked_candidates_path = candidates_path.with_name("ranked_candidates.csv")
    manifest_shards: list[dict[str, Any]] = []
    commands: list[str] = []

    for shard_id, shard_indices in enumerate(shards):
        shard_name = f"shard_{shard_id:02d}"
        target_indices_path = target_index_dir / f"{shard_name}_target_indices.json"
        resolved_indices_path = resolved_index_dir / f"{shard_name}_resolved_target_indices.json"
        config_path = config_dir / f"{shard_name}.yaml"
        run_dir = root / "runs" / run_date / f"{EXPERIMENT_ID}_{shard_name}_{timestamp}"

        write_json(
            target_indices_path,
            {
                "version": 1,
                "experiment_id": EXPERIMENT_ID,
                "shard_id": shard_id,
                "shard_name": shard_name,
                "selection_unit": "target_index",
                "target_count": len(shard_indices),
                "target_indices": list(shard_indices),
                "source_candidates": rel(candidates_path, root),
                "dataset_path": rel(dataset_path, root),
            },
        )
        shard_cfg = build_shard_config(
            base_cfg,
            shard_id=shard_id,
            shard_indices=shard_indices,
            candidates_path=candidates_path,
            ranked_candidates_path=ranked_candidates_path,
            target_indices_path=target_indices_path,
            resolved_indices_path=resolved_indices_path,
            run_dir=run_dir,
            root=root,
        )
        write_yaml(config_path, shard_cfg)
        command = f"PYTHONPATH=src python -m phc2d.cli replay mpb --config {rel(config_path, root)}"
        commands.append(command)
        manifest_shards.append(
            {
                "shard_id": shard_id,
                "shard_name": shard_name,
                "target_count": len(shard_indices),
                "target_index_first": int(shard_indices[0]),
                "target_index_last": int(shard_indices[-1]),
                "target_indices_path": rel(target_indices_path, root),
                "resolved_target_indices_path": rel(resolved_indices_path, root),
                "config_path": rel(config_path, root),
                "run_dir": rel(run_dir, root),
                "command": command,
            }
        )

    write_json(
        output_root / "shard_manifest.json",
        {
            "version": 1,
            "experiment_id": EXPERIMENT_ID,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "planned",
            "validation_boundary": "formal_inverse_candidate_mpb_replay",
            "replay_backend": "real_mpb",
            "require_real_mpb": True,
            "dataset_path": rel(dataset_path, root),
            "candidates_path": rel(candidates_path, root),
            "base_config_path": rel(base_config_path, root),
            "target_split": "test",
            "target_count": len(candidate_indices),
            "target_index_first": int(candidate_indices[0]),
            "target_index_last": int(candidate_indices[-1]),
            "shard_size": int(args.shard_size),
            "shard_count": len(shards),
            "replay_top_m": 5,
            "primary_band_range": [2, 10],
            "shards": manifest_shards,
        },
    )
    (output_root / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")

    print(f"Wrote shard plan: {rel(output_root, root)}")
    print(f"Validated exact idx_test target_count={len(candidate_indices)} shard_count={len(shards)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
