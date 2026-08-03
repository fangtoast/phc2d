"""Merge independent V2 physical-k query dataset shards.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from phc2d.data.forward_dataset import ForwardDatasetBundle, parse_metadata_jsonl, save_forward_dataset
from phc2d.data.k_path import hash_k_path_points, hash_k_path_spec, save_k_path_spec
from scripts.run_query_dataset_v2_pilot_generation import (
    ROOT,
    _build_k_path_spec,
    _display_path,
    _effective_split_counts,
    _file_hash,
    _git_commit,
    _jsonable,
    _metadata_paths,
    _resolve_split_counts,
    _sha256_text,
    _split_indices,
    _split_labels,
    _write_json,
    _write_jsonl,
    _write_yaml,
    load_query_pilot_config,
)


def _read_jsonl_scalar(raw: Any) -> list[dict[str, Any]]:
    return parse_metadata_jsonl(raw)


def merge_shards(
    config_path: str | Path,
    *,
    shards: list[str | Path],
    out: str | Path | None = None,
    run_dir: str | Path | None = None,
    allow_partial: bool = False,
) -> Path:
    """Merge shard NPZ files into the standard forward dataset schema."""

    if not shards:
        raise ValueError("At least one shard path is required")
    started = time.perf_counter()
    config_path = Path(config_path)
    cfg = load_query_pilot_config(config_path)
    dataset_cfg = dict(cfg.get("dataset") or {})
    sample_count = int(dataset_cfg["sample_count"])
    output_path = Path(out or str(dataset_cfg["output_path"]))
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    run_root = Path(run_dir) if run_dir is not None else output_path.parent / "metadata" / f"{output_path.stem}_merge_run"
    if not run_root.is_absolute():
        run_root = ROOT / run_root
    run_root.mkdir(parents=True, exist_ok=True)

    resolved_yaml = yaml.safe_dump(_jsonable(cfg), sort_keys=False, allow_unicode=True)
    config_hash = _sha256_text(resolved_yaml)
    pieces: list[dict[str, Any]] = []
    for shard in shards:
        shard_path = Path(shard)
        if not shard_path.is_absolute():
            shard_path = ROOT / shard_path
        with np.load(shard_path, allow_pickle=False) as data:
            indices = np.asarray(data["sample_index"], dtype=np.int64)
            metadata = _read_jsonl_scalar(data["metadata_jsonl"]) if "metadata_jsonl" in data else []
            pieces.append(
                {
                    "path": shard_path,
                    "indices": indices,
                    "rho": np.asarray(data["rho"], dtype=np.float32),
                    "Wh": np.asarray(data["Wh"], dtype=np.float32),
                    "metadata": metadata,
                }
            )

    rows: list[tuple[int, np.ndarray, np.ndarray, dict[str, Any], Path]] = []
    seen: set[int] = set()
    for piece in pieces:
        indices = piece["indices"]
        rho = piece["rho"]
        wh = piece["Wh"]
        metadata = piece["metadata"]
        if len(indices) != int(rho.shape[0]) or len(indices) != int(wh.shape[0]):
            raise ValueError(f"Shard length mismatch: {piece['path']}")
        for offset, sample_index in enumerate(indices.tolist()):
            idx = int(sample_index)
            if idx in seen:
                raise ValueError(f"Duplicate sample_index {idx} in {piece['path']}")
            seen.add(idx)
            meta = dict(metadata[offset]) if offset < len(metadata) else {"sample_index": idx}
            rows.append((idx, rho[offset], wh[offset], meta, piece["path"]))

    rows.sort(key=lambda item: item[0])
    expected = list(range(sample_count))
    actual = [idx for idx, *_ in rows]
    if actual != expected and not allow_partial:
        missing = sorted(set(expected) - set(actual))[:10]
        raise ValueError(f"Shards do not cover the configured dataset; first missing sample indices: {missing}")

    rho_array = np.stack([row[1] for row in rows], axis=0).astype(np.float32)
    wh_array = np.stack([row[2] for row in rows], axis=0).astype(np.float32)
    metadata_rows = [row[3] for row in rows]
    if actual == expected:
        labels = _split_labels(_effective_split_counts(_resolve_split_counts(dataset_cfg), sample_count))
        idx_train, idx_val, idx_test = _split_indices(labels)
    else:
        labels = [str(row[3].get("split", "")) for row in rows]
        idx_train, idx_val, idx_test = _split_indices(labels)

    bundle = ForwardDatasetBundle(
        rho=rho_array,
        Wh=wh_array,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        config_yaml=resolved_yaml,
        metadata=metadata_rows,
    )
    save_forward_dataset(output_path, bundle)
    k_path_spec = _build_k_path_spec(
        cfg,
        config_path=config_path,
        config_hash=config_hash,
        run_id=run_root.name,
        status="shards_merged",
        git_commit=_git_commit(),
    )
    sidecar_path, manifest_path = _metadata_paths(output_path)
    save_k_path_spec(k_path_spec, sidecar_path)
    manifest = {
        "dataset_path": _display_path(output_path),
        "dataset_id": output_path.stem,
        "dataset_file_hash": _file_hash(output_path),
        "Wh_shape": list(wh_array.shape),
        "rho_shape": list(rho_array.shape),
        "split": {
            "train": int(len(idx_train)),
            "val": int(len(idx_val)),
            "test": int(len(idx_test)),
        },
        "k_path_sidecar": sidecar_path.name,
        "k_path_hash": hash_k_path_spec(k_path_spec),
        "k_points_hash": hash_k_path_points(k_path_spec.k_points),
        "config_hash": config_hash,
        "sample_id_policy": dataset_cfg.get("sample_id_policy"),
        "sample_metadata": _display_path(run_root / "sample_metadata.jsonl"),
        "provenance_status": "shards_merged",
        "physical_verification_status": "generated_not_replayed",
        "formal_full_2a_allowed": False,
        "association_note": "Merged from independent Query Dataset V2 MPB shards; replay validation is still separate.",
    }
    _write_yaml(manifest_path, manifest)
    _write_jsonl(run_root / "sample_metadata.jsonl", metadata_rows)
    _write_json(
        run_root / "metrics.json",
        {
            "physical_validation_decision": "shards_merged",
            "real_mpb_generation_completed": True,
            "wall_time_total_s": float(time.perf_counter() - started),
            "sample_count": int(rho_array.shape[0]),
            "configured_sample_count": sample_count,
            "shard_count": int(len(shards)),
            "allow_partial": bool(allow_partial),
        },
    )
    _write_json(
        run_root / "results.json",
        {
            "action": "query_dataset_v2_shard_merge",
            "status": "shards_merged",
            "dataset_written": True,
            "dataset_path": _display_path(output_path),
            "dataset_manifest": _display_path(manifest_path),
            "k_path_sidecar": _display_path(sidecar_path),
            "sample_count": int(rho_array.shape[0]),
        },
    )
    return output_path


def _discover_shards(shard_dir: Path | None, pattern: str, explicit: list[Path]) -> list[Path]:
    shards = list(explicit)
    if shard_dir is not None:
        shards.extend(sorted(shard_dir.glob(pattern)))
    return shards


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Merge independent Query Dataset V2 MPB shards.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--shard-dir", type=Path, default=None)
    parser.add_argument("--pattern", default="shard_*.npz")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("shards", type=Path, nargs="*")
    args = parser.parse_args(argv)
    shards = _discover_shards(args.shard_dir, args.pattern, list(args.shards))
    return merge_shards(
        args.config,
        shards=shards,
        out=args.out,
        run_dir=args.run_dir,
        allow_partial=args.allow_partial,
    )


if __name__ == "__main__":
    main()
