"""Audit and backfill candidate physical k-path metadata for Full-2A.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from phc2d.data import (
    KPathMetadata,
    build_square_gxm_reduced_spec,
    hash_k_path_spec,
    save_k_path_spec,
    validate_k_path_spec,
)


ROOT = Path(__file__).resolve().parents[1]
DATE = "20260618"
SOURCE_COMMIT = "6c1b95e029441d1d8f47265f594cd87077e3663d"
SOURCE_RUN = "runs/20260610/dataset_gen_paper_like_50k_202606102000"
SOURCE_PATHS = {
    "run_meta": f"{SOURCE_RUN}/run.meta.json",
    "run_log": f"{SOURCE_RUN}/run.log",
    "k_path_code": "src/pod_deeponet/data_gen/k_path.py",
    "mpb_solver_code": "src/pod_deeponet/data_gen/mpb_solver.py",
    "generator_code": "src/pod_deeponet/data_gen/generate_dataset.py",
}


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M")


def _run_dir(experiment_id: str, stamp: str) -> Path:
    out = ROOT / "runs" / DATE / f"{experiment_id}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    return out


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_show(path: str) -> str:
    return subprocess.check_output(["git", "show", f"{SOURCE_COMMIT}:{path}"], cwd=ROOT, text=True, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: Any) -> None:
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _dataset_shape(path: Path) -> tuple[int, int, int]:
    with np.load(path, allow_pickle=False) as data:
        shape = tuple(int(v) for v in data["Wh"].shape)
    if len(shape) != 3:
        raise ValueError(f"Unexpected Wh shape for {path}: {shape}")
    return shape


def _candidate_record(spec_hash: str, spec_path: Path) -> dict[str, Any]:
    return {
        "source_type": "git_history_plus_generation_run",
        "source_path": SOURCE_PATHS,
        "git_commit": SOURCE_COMMIT,
        "config_run_id": SOURCE_RUN,
        "dataset_association_evidence": [
            "run.meta.json records data/paper_like_50k_mpb.npz generation with backend=mpb, target_total=50000, structure_generator=paper_like_compact",
            "NPZ embedded config_yaml matches backend=mpb, mpb.resolution=64, fem.n_bands=10, k_path.n_total=31, output_path=../../data/paper_like_50k_mpb.npz",
            "historical generate_dataset.py passes cfg['k_path']['n_total'] into solve_bands_for_rho_backend",
            "historical mpb_solver.py converts high_symmetry_k_points Γ-X-M-Γ to MPB reduced coordinates via a/(2*pi)",
        ],
        "node_labels": ["Gamma", "X", "M", "Gamma_end"],
        "node_coordinates": {
            "Gamma": [0.0, 0.0],
            "X": [0.5, 0.0],
            "M": [0.5, 0.5],
            "Gamma_end": [0.0, 0.0],
        },
        "coordinate_basis": "MPB reduced reciprocal coordinates for square lattice",
        "segment_order": ["Gamma-X", "X-M", "M-Gamma_end"],
        "points_per_segment": [11, 11, 11],
        "endpoint_inclusion_policy": "X and M shared without duplicate rows; Gamma appears at both start and closure endpoint",
        "total_point_count": 31,
        "repeated_nodes": {"Gamma": [0, 30]},
        "spec_path": str(spec_path.relative_to(ROOT)),
        "spec_hash": spec_hash,
        "confidence_level": "high_candidate_unverified",
        "acceptance_rejection_reason": "Accepted as candidate sidecar source, but not accepted as physically_verified until 5-sample WSL MPB replay matches stored Wh within justified tolerance.",
    }


def main() -> None:
    started = time.perf_counter()
    stamp = _now_stamp()
    run_dir = _run_dir("k_path_provenance_audit", stamp)
    validation_dir = _run_dir("k_path_physical_validation", stamp)
    metadata_dir = ROOT / "data" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    run_meta_path = ROOT / SOURCE_PATHS["run_meta"]
    paper50 = ROOT / "data" / "paper_like_50k_mpb.npz"
    wh_shape = _dataset_shape(paper50)
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    config_text = ""
    with np.load(paper50, allow_pickle=False) as data:
        raw = data["config_yaml"]
        config_text = str(raw.item() if hasattr(raw, "item") else raw)

    source_config_hash = json.dumps(yaml.safe_load(config_text) or {}, sort_keys=True, ensure_ascii=False).encode("utf-8")
    import hashlib

    metadata = KPathMetadata(
        source_type="git_history_plus_generation_run",
        source_path=f"{SOURCE_PATHS['run_meta']} + git:{SOURCE_PATHS['k_path_code']} + git:{SOURCE_PATHS['mpb_solver_code']}",
        source_commit=SOURCE_COMMIT,
        source_run_id=SOURCE_RUN,
        source_config_hash=hashlib.sha256(source_config_hash).hexdigest(),
        verification_status="candidate_unverified",
        verification_run_id=str(validation_dir.relative_to(ROOT)),
        notes=(
            "Do not treat this sidecar as physically_verified until WSL MPB replay aligns with stored Wh.",
            "The historical config path in run.meta.json was local to the legacy/reproduction working directory and is not present as a committed file at the source commit.",
        ),
    )
    spec = build_square_gxm_reduced_spec(n_total=wh_shape[1], lattice_constant=1.0, metadata=metadata)
    problems = validate_k_path_spec(spec, expected_nk=wh_shape[1])
    if problems:
        raise RuntimeError("; ".join(problems))
    spec_path = metadata_dir / "paper_like_square_lattice_gxm_kpath_v1.json"
    save_k_path_spec(spec, spec_path)
    spec_hash = hash_k_path_spec(spec)

    manifests: list[dict[str, Any]] = []
    for dataset_name in ("paper_like_50k_mpb.npz", "paper_like_64_mpb.npz", "tiny.npz"):
        dataset_path = ROOT / "data" / dataset_name
        if not dataset_path.exists():
            continue
        shape = _dataset_shape(dataset_path)
        manifest = {
            "dataset_path": f"data/{dataset_name}",
            "dataset_id": Path(dataset_name).stem,
            "dataset_file_hash": _sha256_file(dataset_path),
            "Wh_shape": list(shape),
            "k_path_sidecar": spec_path.name,
            "k_path_hash": spec_hash,
            "k_points_hash": spec.k_points_hash,
            "config_hash": metadata.source_config_hash if dataset_name == "paper_like_50k_mpb.npz" else "",
            "sample_id_policy": "derived_from_index_when_metadata_jsonl_absent",
            "split_manifest": "embedded idx_train/idx_val/idx_test arrays",
            "provenance_status": "candidate_unverified",
            "physical_verification_status": "blocked_pending_5_sample_wsl_mpb_replay",
            "association_note": (
                "paper_like_50k is directly associated through the generation run; "
                "paper_like_64 and tiny share shape/config family only and remain candidate references."
            ),
        }
        manifest_path = metadata_dir / f"{Path(dataset_name).stem}.manifest.yaml"
        _write_yaml(manifest_path, manifest)
        manifests.append({"path": str(manifest_path.relative_to(ROOT)), **manifest})

    candidates = [_candidate_record(spec_hash, spec_path)]
    decision = {
        "status": "candidate_found_but_formal_gate_blocked",
        "authoritative_source_found": True,
        "source_linked_to_current_forward_dataset": True,
        "versioned_k_path_spec_generated": True,
        "k_points_count_matches_Wh_Nk": True,
        "coordinate_system_documented": True,
        "node_segment_metadata_available": True,
        "duplicate_coordinate_policy_documented": True,
        "physical_validation_completed": False,
        "verification_status": "candidate_unverified",
        "formal_full_2a_allowed": False,
        "blocker": "5-sample WSL MPB physical validation has not been completed in this stage.",
        "next_recommended_task": "Run deterministic 5-sample WSL MPB physical validation against stored Wh before physical-k GPU preflight or Formal Full-2A.",
    }
    _write_json(run_dir / "candidates.json", candidates)
    _write_json(run_dir / "provenance_decision.json", decision)
    _write_yaml(run_dir / "config.resolved.yaml", {"source_commit": SOURCE_COMMIT, "datasets": manifests, "k_path_sidecar": str(spec_path.relative_to(ROOT))})
    _write_json(
        run_dir / "run.meta.json",
        {
            "script": "scripts/run_k_path_provenance_audit.py",
            "run_dir": str(run_dir.relative_to(ROOT)),
            "utc": datetime.now(UTC).isoformat(),
            "wall_time_total_s": time.perf_counter() - started,
        },
    )
    log_lines.extend(
        [
            "K-path provenance audit completed.",
            f"source_commit={SOURCE_COMMIT}",
            f"k_path_sidecar={spec_path.relative_to(ROOT)}",
            "decision=candidate_found_but_formal_gate_blocked",
        ]
    )
    (run_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    sample_rows = [
        {"split": "train", "sample_id": int(v)} for v in np.load(paper50, allow_pickle=False)["idx_train"][:2]
    ]
    with np.load(paper50, allow_pickle=False) as data:
        sample_rows.append({"split": "val", "sample_id": int(data["idx_val"][0])})
        sample_rows.extend({"split": "test", "sample_id": int(v)} for v in data["idx_test"][:2])
    with (validation_dir / "replay_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "sample_id"])
        writer.writeheader()
        writer.writerows(sample_rows)
    _write_json(
        validation_dir / "comparison.json",
        {
            "status": "blocked",
            "reason": "MPB replay not executed in this audit slice; source candidate must be validated before formal query benchmark.",
            "k_path_id": spec.k_path_id,
            "k_path_hash": spec_hash,
            "sample_plan": sample_rows,
        },
    )
    for name in ("per_sample_metrics.csv", "per_k_metrics.csv", "failure_cases.csv", "history.csv"):
        (validation_dir / name).write_text("status,reason\nblocked,pending_5_sample_wsl_mpb_replay\n", encoding="utf-8")
    _write_json(validation_dir / "metrics.json", {"status": "blocked", "mpb_replay_completed": False})
    _write_json(validation_dir / "results.json", {"status": "blocked", "formal_validation": False})
    _write_json(validation_dir / "run.meta.json", {"script": "scripts/run_k_path_provenance_audit.py", "run_dir": str(validation_dir.relative_to(ROOT))})
    _write_yaml(validation_dir / "config.resolved.yaml", {"k_path_sidecar": str(spec_path.relative_to(ROOT)), "sample_plan": sample_rows})
    (validation_dir / "run.log").write_text("Physical validation blocked: pending WSL MPB replay.\n", encoding="utf-8")

    print(run_dir)


if __name__ == "__main__":
    main()
