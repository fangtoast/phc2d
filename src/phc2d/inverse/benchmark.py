"""Single-seed fair inverse benchmark orchestration with real-MPB replay.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from phc2d.data import load_forward_dataset, recover_k_path
from phc2d.inverse.config import GenerativeInverseConfig, load_inverse_run_config
from phc2d.runs import ensure_figures_dir, make_run_dir, setup_logging
from phc2d.solvers.mpb import MpbBandSolverConfig, solve_bands_for_rho_mpb
from phc2d.solvers.mpb_parallel import (
    MPBParallelConfig,
    MPBWorkerJob,
    hash_array,
    run_mpb_worker_jobs,
)

SolverFn = Callable[..., tuple[np.ndarray, dict[str, Any]]]
METHODS = ("retrieval", "direct_inverse", "single_mode_generator", "mixture_generator")
TRAINABLE_METHODS = METHODS[1:]
EVIDENCE_STATUS = "single_seed_formal_replay"
BRIDGE_EVIDENCE_STATUS = "single_seed_adjustment_pilot"
BRIDGE_TEST625_EVIDENCE_STATUS = "single_seed_adjustment_test625_replay"
SURROGATE_SCREEN_EVIDENCE_STATUS = "config_ready_surrogate_screen"
SURROGATE_OBJECTIVE_EVIDENCE_STATUS = "config_ready_surrogate_objective"
POSTREPAIR_EVIDENCE_STATUS = "single_seed_20_target_pilot"
ALLOWED_EVIDENCE_STATUSES = (
    EVIDENCE_STATUS,
    BRIDGE_EVIDENCE_STATUS,
    BRIDGE_TEST625_EVIDENCE_STATUS,
)
SURROGATE_ABLATION_EVIDENCE_STATUSES = (
    SURROGATE_SCREEN_EVIDENCE_STATUS,
    SURROGATE_OBJECTIVE_EVIDENCE_STATUS,
)
SURROGATE_ABLATION_MODELS = (
    "mlp",
    "res_mlp_wide",
    "lifted_cnn",
    "fno2d_band_v2",
    "mlp_query",
    "deep_onet_query_v2",
    "fno2d_query_v2",
)


def _mapping(value: Any, *, name: str = "value") -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _evidence_status(cfg: Mapping[str, Any]) -> str:
    return str(_protocol(cfg).get("evidence_status", EVIDENCE_STATUS))


def _is_objective_bridge(cfg: Mapping[str, Any]) -> bool:
    training = _mapping(cfg.get("training"), name="training")
    return str(training.get("mode", "reconstruction")).strip().lower() == "objective_bridge"


def _is_objective_bridge_test625(cfg: Mapping[str, Any]) -> bool:
    evaluation = _mapping(cfg.get("evaluation"), name="evaluation")
    return _is_objective_bridge(cfg) and str(evaluation.get("scope", "")).strip().lower() == (
        "full_test625_extension"
    )


def _is_surrogate_ablation(cfg: Mapping[str, Any]) -> bool:
    matrix = _mapping(cfg.get("surrogate_matrix"), name="surrogate_matrix")
    return bool(matrix.get("variants") or matrix.get("source_config"))


def _is_postrepair_validation(cfg: Mapping[str, Any]) -> bool:
    """Return whether *cfg* selects the frozen post-repair validation lane."""

    protocol = _protocol(cfg)
    return str(protocol.get("lane", "")).strip().lower() == "postrepair_validation"


def _postrepair_variants(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate and return the seven post-repair inverse-recipe entries."""

    matrix = _mapping(cfg.get("surrogate_matrix"), name="surrogate_matrix")
    raw = matrix.get("variants")
    if not isinstance(raw, list) or not raw:
        raise ValueError("post-repair surrogate_matrix.variants must be a non-empty list")
    variants = [
        _mapping(item, name=f"surrogate_matrix.variants[{index}]")
        for index, item in enumerate(raw)
    ]
    ids = tuple(str(item.get("id", "")).strip() for item in variants)
    if ids != SURROGATE_ABLATION_MODELS:
        raise ValueError(
            "post-repair variants must list the seven approved inverse recipes in order: "
            f"{list(SURROGATE_ABLATION_MODELS)!r}"
        )
    if str(matrix.get("reference_id", "")).strip() != "fno2d_band_v2":
        raise ValueError("post-repair reference_id must remain fno2d_band_v2")
    expected_epochs = tuple(
        int(item)
        for item in _mapping(cfg.get("checkpoint_selection"), name="checkpoint_selection").get(
            "epochs", ()
        )
    )
    for variant in variants:
        variant_id = str(variant["id"])
        primary = _mapping(variant.get("primary_ranker"), name=f"{variant_id}.primary_ranker")
        if str(primary.get("model_name", "")).strip() != variant_id:
            raise ValueError(f"post-repair primary ranker identity mismatch for {variant_id!r}")
        expected_kind = "query" if variant_id.endswith("_query") or "query_v2" in variant_id else "forward"
        if str(primary.get("kind", "")).strip().lower() != expected_kind:
            raise ValueError(f"post-repair primary ranker kind mismatch for {variant_id!r}")
        checkpoints = variant.get("generator_checkpoints")
        if not isinstance(checkpoints, list) or tuple(int(item.get("epoch", -1)) for item in checkpoints) != expected_epochs:
            raise ValueError(
                f"post-repair generator checkpoints for {variant_id!r} must match epochs "
                f"{list(expected_epochs)!r}"
            )
    return variants


def _checkpoint_denylist(cfg: Mapping[str, Any]) -> tuple[set[str], tuple[str, ...]]:
    denylist = _mapping(
        cfg.get("withdrawn_checkpoint_denylist"),
        name="withdrawn_checkpoint_denylist",
    )
    hashes = {str(item).strip().lower() for item in denylist.get("sha256", ())}
    prefixes = tuple(
        str(Path(str(item))).replace("\\", "/").rstrip("/").lower()
        for item in denylist.get("path_prefixes", ())
    )
    return hashes, prefixes


def _require_checkpoint_allowed(
    cfg: Mapping[str, Any],
    path: str | Path,
    expected_hash: str,
    *,
    label: str,
) -> str:
    """Verify checkpoint lineage and reject withdrawn paths or hashes."""

    actual_hash = _require_hash(path, expected_hash, label=label)
    denied_hashes, denied_prefixes = _checkpoint_denylist(cfg)
    normalized_path = str(Path(path)).replace("\\", "/").lower()
    if actual_hash in denied_hashes:
        raise ValueError(f"{label} is withdrawn and denied by SHA-256: {actual_hash}")
    if any(normalized_path == prefix or normalized_path.startswith(f"{prefix}/") for prefix in denied_prefixes):
        raise ValueError(f"{label} is withdrawn and denied by path: {path}")
    return actual_hash


def _surrogate_variants(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    matrix = _mapping(cfg.get("surrogate_matrix"), name="surrogate_matrix")
    raw_variants = matrix.get("variants")
    if not raw_variants and matrix.get("source_config"):
        source_path = Path(str(matrix["source_config"]))
        expected_hash = str(matrix.get("source_config_sha256", "")).strip()
        if expected_hash:
            _require_hash(source_path, expected_hash, label="surrogate matrix source config")
        source_cfg = _read_yaml(source_path)
        source_matrix = _mapping(
            source_cfg.get("surrogate_matrix"),
            name=f"{source_path}:surrogate_matrix",
        )
        raw_variants = source_matrix.get("variants")
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError("surrogate_matrix.variants must be a non-empty list")
    variants = [
        _mapping(item, name=f"surrogate_matrix.variants[{index}]")
        for index, item in enumerate(raw_variants)
    ]
    ids = [str(item.get("id", "")).strip() for item in variants]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("surrogate_matrix variant ids must be non-empty and unique")
    model_names = [str(item.get("model_name", "")).strip() for item in variants]
    if tuple(model_names) != SURROGATE_ABLATION_MODELS:
        raise ValueError(
            "surrogate_matrix must list the seven frozen A2 variants in the approved order: "
            f"{list(SURROGATE_ABLATION_MODELS)!r}"
        )
    reference_id = str(matrix.get("reference_id", "")).strip()
    if reference_id != "fno2d_band_v2" or reference_id not in ids:
        raise ValueError("surrogate_matrix.reference_id must be fno2d_band_v2")
    for variant in variants:
        variant_id = str(variant["id"])
        kind = str(variant.get("kind", "")).strip().lower()
        expected_kind = (
            "query"
            if variant_id in {"mlp_query", "deep_onet_query_v2", "fno2d_query_v2"}
            else "forward"
        )
        if kind != expected_kind:
            raise ValueError(
                f"surrogate {variant_id!r} must use kind={expected_kind!r}, got {kind!r}"
            )
        if int(variant.get("seed", -1)) != 0:
            raise ValueError(f"surrogate {variant_id!r} must use validation-selected seed 0")
        if not str(variant.get("input_repr", "")).strip():
            raise ValueError(f"surrogate {variant_id!r} requires input_repr")
        if kind == "query" and str(variant.get("protocol", "")).strip() != "unseen_structure_all_k":
            raise ValueError(
                f"query surrogate {variant_id!r} must use protocol=unseen_structure_all_k"
            )
    return variants


def _variant_surrogate_config(variant: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("kind", "model_name", "input_repr", "protocol", "seed")
    result = {
        key: variant[key]
        for key in keys
        if key in variant and variant[key] is not None
    }
    result["checkpoint_path"] = str(variant["checkpoint_path"])
    return result


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _selected_training_run(cfg: Mapping[str, Any], *, run: Path) -> Path:
    evaluation = _mapping(cfg.get("evaluation"), name="evaluation")
    selected = str(evaluation.get("selected_training_run", "")).strip()
    return Path(selected) if selected else run


def _read_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return _mapping(payload, name=str(path))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(dict(row)))


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(path: str | Path, expected: str, *, label: str) -> str:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"{label} not found: {target}")
    actual = _sha256_file(target)
    if not expected or actual != str(expected).lower():
        raise ValueError(f"{label} SHA-256 mismatch: expected={expected!r}, actual={actual}")
    return actual


def _protocol(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(cfg.get("protocol"), name="protocol")


def _dataset_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(cfg.get("dataset"), name="dataset")


def _output_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(cfg.get("output"), name="output")


def _target_indices(cfg: Mapping[str, Any]) -> list[int]:
    manifest_path = Path(str(_dataset_cfg(cfg)["target_manifest_path"]))
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    indices = payload.get("target_indices")
    if isinstance(indices, list):
        return [int(item) for item in indices]
    index_range = payload.get("target_index_range")
    if isinstance(index_range, Mapping):
        start = int(index_range["start"])
        stop_exclusive = int(index_range["stop_exclusive"])
        step = int(index_range.get("step", 1))
        if step <= 0 or stop_exclusive <= start:
            raise ValueError("target manifest target_index_range must be increasing")
        return list(range(start, stop_exclusive, step))
    raise ValueError("target manifest must contain target_indices or target_index_range")


def _solver_config(cfg: Mapping[str, Any]) -> MpbBandSolverConfig:
    raw = _mapping(cfg.get("solver"), name="solver")
    return MpbBandSolverConfig(
        num_bands=int(raw["num_bands"]),
        resolution=int(raw["resolution"]),
        polarization=str(raw["polarization"]),
        epsilon_air=float(raw["epsilon_air"]),
        epsilon_material=float(raw["epsilon_material"]),
        lattice_constant=float(raw["lattice_constant"]),
        eigensolver_tolerance=float(raw["eigensolver_tolerance"]),
        grid_shape=tuple(int(item) for item in raw["grid_shape"]),  # type: ignore[arg-type]
        band_guard_count=int(raw.get("band_guard_count", 0)),
        band_order_policy=str(raw.get("band_order_policy", "frequency_ascending")),
    )


def _solver_payload(value: MpbBandSolverConfig) -> dict[str, Any]:
    payload = {key: item for key, item in asdict(value).items() if item is not None}
    if payload.get("grid_shape") is not None:
        payload["grid_shape"] = list(payload["grid_shape"])
    if payload.get("field_band_indices_one_based") is not None:
        payload["field_band_indices_one_based"] = list(payload["field_band_indices_one_based"])
    return payload


def _resolve_run_dir(cfg: Mapping[str, Any], run_dir: str | Path | None) -> Path:
    if run_dir is not None:
        target = Path(run_dir)
        target.mkdir(parents=True, exist_ok=True)
        return target
    output = _output_cfg(cfg)
    explicit = output.get("run_dir")
    if explicit:
        target = Path(str(explicit))
        target.mkdir(parents=True, exist_ok=True)
        return target
    return make_run_dir(
        str(_protocol(cfg).get("id", "a2_inverse_single_seed_v1")),
        runs_dir=Path(str(output.get("runs_dir", "runs"))),
    )


def _write_standard_scaffold(run_dir: Path, cfg: Mapping[str, Any], *, stage: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    ensure_figures_dir(run_dir)
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(_jsonable(cfg), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    if not (run_dir / "history.csv").exists():
        _write_csv(run_dir / "history.csv", [{"stage": stage, "status": "started"}])


def run_a2_surrogate_preflight(
    config_path: str | Path,
    *,
    run_dir: str | Path | None = None,
) -> Path:
    """Validate the seven-surrogate ablation lineage without training or replay."""

    cfg = _read_yaml(config_path)
    protocol = _protocol(cfg)
    evidence_status = _evidence_status(cfg)
    if evidence_status not in SURROGATE_ABLATION_EVIDENCE_STATUSES:
        raise ValueError(
            "surrogate ablation protocol.evidence_status must be one of "
            f"{list(SURROGATE_ABLATION_EVIDENCE_STATUSES)!r}"
        )
    if str(protocol.get("selection_split", "")).strip().lower() not in {"val", "validation"}:
        raise ValueError("surrogate ablation selection_split must be val")
    if str(protocol.get("final_evaluation_split", "")).strip().lower() != "test":
        raise ValueError("surrogate ablation final_evaluation_split must be test")
    if not bool(protocol.get("forbid_test_driven_selection", False)):
        raise ValueError("surrogate ablation must set forbid_test_driven_selection=true")

    run = _resolve_run_dir(cfg, run_dir)
    _write_standard_scaffold(run, cfg, stage="surrogate_preflight")
    logger = setup_logging(
        log_file=run / "run.log",
        run_meta={
            "script": "phc2d inverse benchmark",
            "stage": "surrogate_preflight",
            "pid": os.getpid(),
        },
    )

    dataset_cfg = _dataset_cfg(cfg)
    dataset_path = Path(str(dataset_cfg["path"]))
    dataset_hash = _require_hash(
        dataset_path,
        str(dataset_cfg["expected_sha256"]),
        label="dataset",
    )
    bundle = load_forward_dataset(dataset_path)
    actual_splits = {
        "train": int(len(bundle.idx_train)),
        "val": int(len(bundle.idx_val)),
        "test": int(len(bundle.idx_test)),
    }
    expected_splits = {
        key: int(value)
        for key, value in _mapping(
            dataset_cfg["expected_split_sizes"],
            name="dataset.expected_split_sizes",
        ).items()
    }
    if actual_splits != expected_splits:
        raise ValueError(f"dataset split mismatch: expected={expected_splits}, actual={actual_splits}")

    target_manifest_path = Path(str(dataset_cfg["target_manifest_path"]))
    target_manifest_hash = _require_hash(
        target_manifest_path,
        str(dataset_cfg["target_manifest_sha256"]),
        label="validation target manifest",
    )
    target_payload = json.loads(target_manifest_path.read_text(encoding="utf-8"))
    target_indices = _target_indices(cfg)
    expected_target_count = int(dataset_cfg["expected_target_count"])
    if len(target_indices) != expected_target_count or len(set(target_indices)) != len(target_indices):
        raise ValueError("validation target manifest count/uniqueness check failed")
    validation_set = {int(item) for item in np.asarray(bundle.idx_val, dtype=np.int64)}
    if any(int(item) not in validation_set for item in target_indices):
        raise ValueError("surrogate selection manifest contains an index outside the validation split")
    if str(target_payload.get("dataset_split", "")).strip().lower() not in {"val", "validation"}:
        raise ValueError("surrogate selection target manifest must declare dataset_split=val")
    if str(target_payload.get("dataset_sha256", "")).lower() != dataset_hash:
        raise ValueError("validation target manifest dataset_sha256 mismatch")

    pilot_cfg = _mapping(cfg.get("pilot"), name="pilot")
    pilot_positions = [int(item) for item in pilot_cfg.get("target_positions", ())]
    if len(pilot_positions) != 20 or len(set(pilot_positions)) != len(pilot_positions):
        raise ValueError("pilot.target_positions must contain 20 unique positions")
    if any(item < 0 or item >= len(target_indices) for item in pilot_positions):
        raise ValueError("pilot.target_positions contains an out-of-range position")

    k_path = recover_k_path(bundle)
    expected_k_hash = str(dataset_cfg["k_path_sha256"]).lower()
    if not k_path.is_physical or str(k_path.k_path_hash).lower() != expected_k_hash:
        raise ValueError(
            f"physical k-path mismatch: expected={expected_k_hash}, actual={k_path.k_path_hash}"
        )
    if not k_path.has_segment_metadata:
        raise ValueError("surrogate ablation requires segment-aware physical k-path metadata")
    if not k_path.repeated_endpoint_indices:
        raise ValueError("surrogate ablation requires the preserved repeated endpoint Gamma")
    if int(k_path.repeated_endpoint_indices[-1]) != int(bundle.nk) - 1:
        raise ValueError("the repeated endpoint Gamma must remain the final physical k query")
    if not np.allclose(k_path.coordinates[0], k_path.coordinates[-1]):
        raise ValueError("the first and final physical k coordinates must preserve Gamma repetition")

    variants = _surrogate_variants(cfg)
    variant_lineage: list[dict[str, Any]] = []
    for variant in variants:
        variant_id = str(variant["id"])
        checkpoint = Path(str(variant["checkpoint_path"]))
        checkpoint_hash = _require_hash(
            checkpoint,
            str(variant["sha256"]),
            label=f"{variant_id} checkpoint",
        )
        if str(variant["kind"]) == "query":
            query_k_hash = str(variant.get("k_path_sha256", "")).lower()
            if query_k_hash != expected_k_hash:
                raise ValueError(
                    f"query surrogate {variant_id!r} k-path hash mismatch: "
                    f"expected={expected_k_hash}, actual={query_k_hash}"
                )
        variant_lineage.append(
            {
                "id": variant_id,
                "kind": str(variant["kind"]),
                "model_name": str(variant["model_name"]),
                "input_repr": str(variant["input_repr"]),
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "seed": int(variant["seed"]),
                "protocol": variant.get("protocol"),
                "k_path_sha256": variant.get("k_path_sha256"),
                "role_group": str(variant.get("role_group", "")),
            }
        )

    candidate_generation = _mapping(
        cfg.get("candidate_generation"),
        name="candidate_generation",
    )
    generator = _mapping(
        candidate_generation.get("generator_checkpoint"),
        name="candidate_generation.generator_checkpoint",
    )
    generator_path = Path(str(generator["path"]))
    generator_hash = _require_hash(
        generator_path,
        str(generator["sha256"]),
        label="reconstruction-trained mixture generator checkpoint",
    )
    if int(generator.get("seed", -1)) != 0:
        raise ValueError("surrogate screen generator checkpoint must use seed 0")
    if int(candidate_generation.get("n_candidates", 0)) != int(protocol.get("requested_k", 0)):
        raise ValueError("candidate_generation.n_candidates must equal protocol.requested_k")
    if str(candidate_generation.get("source_training_mode", "")).strip() != "reconstruction":
        raise ValueError(
            "candidate generation must use a reconstruction-trained checkpoint, not an objective-tuned one"
        )

    boundaries = _mapping(cfg.get("boundaries"), name="boundaries")
    for key in ("pino_status", "field_inverse_status", "physics_aware_rerank_status"):
        if str(boundaries.get(key, "")).strip().lower() != "blocked":
            raise ValueError(f"boundaries.{key} must remain blocked")
    if bool(boundaries.get("b4_checkpoint_allowed", False)):
        raise ValueError("B4 field/PINO checkpoints are not allowed in the A2 surrogate ablation")

    solver = _solver_config(cfg)
    solver_hash = _sha256_json(_solver_payload(solver))
    expected_solver_hash = str(_mapping(cfg.get("solver"), name="solver")["protocol_sha256"]).lower()
    if solver_hash != expected_solver_hash:
        raise ValueError(f"solver protocol hash mismatch: expected={expected_solver_hash}, actual={solver_hash}")

    surrogate_manifest = {
        "version": 1,
        "status": "frozen",
        "reference_id": "fno2d_band_v2",
        "variant_count": len(variant_lineage),
        "variants": variant_lineage,
        "field_models_excluded": ["unet_field_v2", "fno_field_v2", "adapted_pino"],
        "field_inverse_status": "blocked_not_in_a2_surrogate_matrix",
    }
    lineage = {
        "protocol_config_path": str(config_path),
        "protocol_config_sha256": _sha256_file(config_path),
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "split_sizes": actual_splits,
        "target_manifest_path": str(target_manifest_path),
        "target_manifest_sha256": target_manifest_hash,
        "target_indices_sha256": _sha256_json(target_indices),
        "target_count": len(target_indices),
        "pilot_target_positions": pilot_positions,
        "generator_checkpoint_path": str(generator_path),
        "generator_checkpoint_sha256": generator_hash,
        "surrogate_manifest_sha256": _sha256_json(surrogate_manifest),
        "k_path_id": k_path.k_path_id,
        "k_path_sha256": k_path.k_path_hash,
        "k_path_source": k_path.source,
        "k_query_mode": "physical_coordinates_only_no_index_fallback",
        "k_point_count": int(bundle.nk),
        "repeated_endpoint_indices": [int(item) for item in k_path.repeated_endpoint_indices],
        "solver_protocol_sha256": solver_hash,
        "code_revision": _git_revision(),
        "evidence_status": evidence_status,
    }
    shutil.copyfile(target_manifest_path, run / "target_manifest_copy.json")
    _atomic_json(run / "surrogate_manifest.json", surrogate_manifest)
    _atomic_json(run / "lineage.json", lineage)
    _atomic_json(
        run / "preflight_decision.json",
        {
            "status": "pass",
            "next_status": "frozen_candidate_sampling",
            "evidence_status": evidence_status,
            "training_authorized": evidence_status == SURROGATE_OBJECTIVE_EVIDENCE_STATUS,
            "sampling_authorized": True,
            "surrogate_screen_authorized": True,
            "real_mpb_pilot_authorized_after_screen": True,
            "formal_replay_authorized": False,
            "test_split_access_authorized": False,
            "cross_seed_claims_authorized": False,
            "pino_or_physics_rerank_authorized": False,
        },
    )
    _atomic_json(
        run / "metrics.json",
        {
            "preflight_pass": True,
            "target_count": len(target_indices),
            "pilot_target_count": len(pilot_positions),
            "surrogate_variant_count": len(variant_lineage),
        },
    )
    _atomic_json(
        run / "results.json",
        {
            "status": "preflight_passed",
            "evidence_status": evidence_status,
            "validation_boundary": "config_and_lineage_only_no_training_no_mpb",
        },
    )
    _atomic_json(
        run / "run.meta.json",
        {
            "script": "phc2d inverse benchmark",
            "stage": "surrogate_preflight",
            "status": "completed",
            "pid": os.getpid(),
            "end_timestamp": datetime.now().astimezone().isoformat(),
        },
    )
    logger.info("A2 surrogate-ablation preflight passed: %s", run)
    return run


def run_a2_postrepair_preflight(
    config_path: str | Path,
    *,
    run_dir: str | Path | None = None,
) -> Path:
    """Validate frozen post-repair checkpoints, rankers, targets, and physical k lineage."""

    cfg = _read_yaml(config_path)
    if not _is_postrepair_validation(cfg):
        raise ValueError("post-repair preflight requires protocol.lane=postrepair_validation")
    protocol = _protocol(cfg)
    dataset_cfg = _dataset_cfg(cfg)
    run = _resolve_run_dir(cfg, run_dir)
    _write_standard_scaffold(run, cfg, stage="postrepair_preflight")

    dataset_path = Path(str(dataset_cfg["path"]))
    dataset_hash = _require_hash(
        dataset_path,
        str(dataset_cfg["expected_sha256"]),
        label="post-repair dataset",
    )
    bundle = load_forward_dataset(dataset_path)
    expected_splits = {
        key: int(value)
        for key, value in _mapping(
            dataset_cfg["expected_split_sizes"],
            name="dataset.expected_split_sizes",
        ).items()
    }
    actual_splits = {
        "train": int(len(bundle.idx_train)),
        "val": int(len(bundle.idx_val)),
        "test": int(len(bundle.idx_test)),
    }
    if actual_splits != expected_splits:
        raise ValueError(f"dataset split mismatch: expected={expected_splits}, actual={actual_splits}")

    target_manifest_path = Path(str(dataset_cfg["target_manifest_path"]))
    target_manifest_hash = _require_hash(
        target_manifest_path,
        str(dataset_cfg["target_manifest_sha256"]),
        label="post-repair validation target manifest",
    )
    target_payload = json.loads(target_manifest_path.read_text(encoding="utf-8"))
    target_indices = _target_indices(cfg)
    if len(target_indices) != int(dataset_cfg["expected_target_count"]):
        raise ValueError("post-repair target manifest count mismatch")
    if len(set(target_indices)) != len(target_indices):
        raise ValueError("post-repair target manifest contains duplicate target IDs")
    val_set = {int(item) for item in np.asarray(bundle.idx_val, dtype=np.int64)}
    if any(int(item) not in val_set for item in target_indices):
        raise ValueError("post-repair target manifest accesses a non-validation target")
    if str(target_payload.get("dataset_split", "")).strip().lower() not in {"val", "validation"}:
        raise ValueError("post-repair target manifest must declare dataset_split=val")
    if str(target_payload.get("dataset_sha256", "")).lower() != dataset_hash:
        raise ValueError("post-repair target manifest dataset SHA-256 mismatch")

    k_path = recover_k_path(bundle)
    expected_k_hash = str(dataset_cfg["k_path_sha256"]).lower()
    if not k_path.is_physical or str(k_path.k_path_hash).lower() != expected_k_hash:
        raise ValueError(
            f"physical k-path mismatch: expected={expected_k_hash}, actual={k_path.k_path_hash}"
        )
    if not k_path.has_segment_metadata:
        raise ValueError("post-repair query rankers require segment-aware physical k metadata")
    if tuple(k_path.repeated_endpoint_indices) != (int(bundle.nk) - 1,):
        raise ValueError("post-repair k path must preserve exactly the final repeated Gamma endpoint")
    if not np.allclose(k_path.coordinates[0], k_path.coordinates[-1]):
        raise ValueError("post-repair k path must preserve the first/final Gamma coordinates")
    if bool(dataset_cfg.get("allow_index_query_fallback", True)):
        raise ValueError("post-repair validation must explicitly prohibit index-query fallback")

    common = _mapping(cfg.get("common_diagnostic_ranker"), name="common_diagnostic_ranker")
    common_hash = _require_checkpoint_allowed(
        cfg,
        str(common["checkpoint_path"]),
        str(common["sha256"]),
        label="common FNO diagnostic ranker",
    )
    variants_manifest: list[dict[str, Any]] = []
    for variant in _postrepair_variants(cfg):
        variant_id = str(variant["id"])
        training_path = Path(str(variant["training_config_path"]))
        training_hash = _require_hash(
            training_path,
            str(variant["training_config_sha256"]),
            label=f"{variant_id} training config",
        )
        primary = _mapping(variant["primary_ranker"], name=f"{variant_id}.primary_ranker")
        primary_hash = _require_checkpoint_allowed(
            cfg,
            str(primary["checkpoint_path"]),
            str(primary["sha256"]),
            label=f"{variant_id} primary ranker",
        )
        if str(primary.get("kind")) == "query" and str(primary.get("k_path_sha256", "")).lower() != expected_k_hash:
            raise ValueError(f"query primary ranker {variant_id!r} k-path hash mismatch")
        checkpoint_rows: list[dict[str, Any]] = []
        for checkpoint in variant["generator_checkpoints"]:
            entry = _mapping(checkpoint, name=f"{variant_id}.generator_checkpoint")
            checkpoint_hash = _require_checkpoint_allowed(
                cfg,
                str(entry["path"]),
                str(entry["sha256"]),
                label=f"{variant_id} epoch-{int(entry['epoch'])} generator checkpoint",
            )
            checkpoint_rows.append(
                {
                    "epoch": int(entry["epoch"]),
                    "path": str(entry["path"]),
                    "sha256": checkpoint_hash,
                }
            )
        variants_manifest.append(
            {
                "id": variant_id,
                "role_group": str(variant["role_group"]),
                "source_lane": str(variant["source_lane"]),
                "training_config_path": str(training_path),
                "training_config_sha256": training_hash,
                "primary_ranker": {
                    **primary,
                    "sha256": primary_hash,
                },
                "generator_checkpoints": checkpoint_rows,
            }
        )

    manifest = {
        "version": 2,
        "status": "frozen",
        "protocol_id": str(protocol["id"]),
        "comparison_interpretation": "seven_inverse_recipes_not_pure_surrogate_architecture_ablation",
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "target_manifest_path": str(target_manifest_path),
        "target_manifest_sha256": target_manifest_hash,
        "target_indices_sha256": _sha256_json(target_indices),
        "target_count": len(target_indices),
        "latent_seed": int(_mapping(cfg["candidate_protocol"], name="candidate_protocol")["latent_seed"]),
        "k_query_mode": "physical_coordinates_only_no_index_fallback",
        "k_path_sha256": expected_k_hash,
        "repeated_gamma_endpoint_indices": [int(item) for item in k_path.repeated_endpoint_indices],
        "common_diagnostic_ranker": {**common, "sha256": common_hash},
        "variants": variants_manifest,
        "withdrawn_checkpoint_denylist": cfg["withdrawn_checkpoint_denylist"],
        "test100_accessed": False,
        "test625_accessed": False,
        "field_pino_accessed": False,
    }
    shutil.copyfile(target_manifest_path, run / "target_manifest_copy.json")
    _atomic_json(run / "postrepair_surrogate_manifest.json", manifest)
    decision = {
        "status": "pass",
        "next_status": "validation_checkpoint_selection",
        "selection_authorized": True,
        "real_mpb_smoke_authorized": False,
        "pilot_replay_authorized": False,
        "formal_or_multiseed_authorized": False,
        "evidence_status": "config_and_lineage_ready_no_candidate_sampling_no_mpb",
    }
    _atomic_json(run / "preflight_decision.json", decision)
    _atomic_json(run / "results.json", decision)
    _atomic_json(
        run / "metrics.json",
        {
            "status": "ok",
            "variant_count": len(variants_manifest),
            "generator_checkpoint_count": sum(
                len(item["generator_checkpoints"]) for item in variants_manifest
            ),
            "target_count": len(target_indices),
            "physical_k_path_verified": True,
            "withdrawn_checkpoint_rejected_by_default": True,
        },
    )
    return run


def run_a2_preflight(config_path: str | Path, *, run_dir: str | Path | None = None) -> Path:
    """Validate all frozen A2 inputs and write a reject-by-default lineage decision."""

    cfg = _read_yaml(config_path)
    if _is_postrepair_validation(cfg):
        return run_a2_postrepair_preflight(config_path, run_dir=run_dir)
    if _is_surrogate_ablation(cfg):
        return run_a2_surrogate_preflight(config_path, run_dir=run_dir)
    protocol = _protocol(cfg)
    evidence_status = _evidence_status(cfg)
    if evidence_status not in ALLOWED_EVIDENCE_STATUSES:
        raise ValueError(f"protocol.evidence_status must be one of {list(ALLOWED_EVIDENCE_STATUSES)!r}")
    if [int(item) for item in protocol.get("seeds", [])] != [0]:
        raise ValueError("A2 single-seed protocol requires exactly seeds: [0]")
    if tuple(protocol.get("methods", ())) != METHODS:
        raise ValueError(f"A2 methods must be exactly {list(METHODS)}")

    run = _resolve_run_dir(cfg, run_dir)
    _write_standard_scaffold(run, cfg, stage="preflight")
    logger = setup_logging(
        log_file=run / "run.log",
        run_meta={"script": "phc2d inverse benchmark", "stage": "preflight", "pid": os.getpid()},
    )
    dataset_cfg = _dataset_cfg(cfg)
    dataset_path = Path(str(dataset_cfg["path"]))
    dataset_hash = _require_hash(dataset_path, str(dataset_cfg["expected_sha256"]), label="dataset")
    bundle = load_forward_dataset(dataset_path)
    actual_splits = {
        "train": int(len(bundle.idx_train)),
        "val": int(len(bundle.idx_val)),
        "test": int(len(bundle.idx_test)),
    }
    expected_splits = {key: int(value) for key, value in _mapping(dataset_cfg["expected_split_sizes"]).items()}
    if actual_splits != expected_splits:
        raise ValueError(f"dataset split mismatch: expected={expected_splits}, actual={actual_splits}")

    target_manifest_path = Path(str(dataset_cfg["target_manifest_path"]))
    target_manifest_hash = _require_hash(
        target_manifest_path,
        str(dataset_cfg["target_manifest_sha256"]),
        label="target manifest",
    )
    target_payload = json.loads(target_manifest_path.read_text(encoding="utf-8"))
    target_indices = _target_indices(cfg)
    if len(target_indices) != int(dataset_cfg["expected_target_count"]) or len(set(target_indices)) != len(target_indices):
        raise ValueError("target manifest count/uniqueness check failed")
    test_set = {int(item) for item in np.asarray(bundle.idx_test, dtype=np.int64)}
    if any(index not in test_set for index in target_indices):
        raise ValueError("target manifest contains an index outside the test split")
    if str(target_payload.get("dataset_sha256", "")).lower() != dataset_hash:
        raise ValueError("target manifest dataset_sha256 does not match the frozen dataset")

    checkpoints = _mapping(cfg.get("checkpoints"), name="checkpoints")
    checkpoint_lineage: dict[str, Any] = {}
    for key in ("ranking_forward", "diagnostic_query"):
        entry = _mapping(checkpoints.get(key), name=f"checkpoints.{key}")
        path = Path(str(entry["path"]))
        checkpoint_lineage[key] = {
            "path": str(path),
            "sha256": _require_hash(path, str(entry["sha256"]), label=f"{key} checkpoint"),
            "model_name": str(entry["model_name"]),
            "seed": int(entry["seed"]),
            "role": str(entry["role"]),
        }
        if int(entry["seed"]) != 0:
            raise ValueError(f"{key} must use seed 0")
    training_cfg = _mapping(cfg.get("training"), name="training")
    warm_start_lineage: dict[str, Any] = {}
    if _is_objective_bridge(cfg):
        methods_to_train = tuple(str(item) for item in training_cfg.get("methods_to_train", ()))
        if methods_to_train != ("single_mode_generator", "mixture_generator"):
            raise ValueError(
                "objective bridge requires methods_to_train: [single_mode_generator, mixture_generator]"
            )
        warm_starts = _mapping(training_cfg.get("warm_start"), name="training.warm_start")
        for method in (*methods_to_train, "direct_inverse"):
            entry = _mapping(warm_starts.get(method), name=f"training.warm_start.{method}")
            path = Path(str(entry["path"]))
            warm_start_lineage[method] = {
                "path": str(path),
                "sha256": _require_hash(path, str(entry["sha256"]), label=f"{method} warm-start checkpoint"),
            }

    baseline = _mapping(cfg.get("baseline_protocol"), name="baseline_protocol")
    baseline_hash = _require_hash(
        Path(str(baseline["path"])),
        str(baseline["sha256"]),
        label="baseline protocol",
    )
    parent_protocol_lineage: dict[str, Any] = {}
    parent_path_value = protocol.get("parent_protocol")
    parent_hash_value = protocol.get("parent_protocol_sha256")
    if parent_path_value and parent_hash_value:
        parent_path = Path(str(parent_path_value))
        parent_protocol_lineage = {
            "path": str(parent_path),
            "sha256": _require_hash(
                parent_path,
                str(parent_hash_value),
                label="parent A2 protocol",
            ),
        }
    if _is_objective_bridge(cfg) and not _is_objective_bridge_test625(cfg):
        pilot_cfg = _mapping(cfg.get("pilot"), name="pilot")
        for key in ("baseline_per_target_metrics_path", "baseline_main_table_path"):
            baseline_artifact = Path(str(pilot_cfg[key]))
            if not baseline_artifact.is_file():
                raise FileNotFoundError(f"missing frozen A2 baseline artifact: {baseline_artifact}")
    evaluation_lineage: dict[str, Any] = {}
    if _is_objective_bridge_test625(cfg):
        if int(dataset_cfg["expected_target_count"]) != int(actual_splits["test"]):
            raise ValueError("test625 extension must cover the complete frozen test split")
        evaluation_cfg = _mapping(cfg.get("evaluation"), name="evaluation")
        selected_run = _selected_training_run(cfg, run=run)
        if not selected_run.is_dir():
            raise FileNotFoundError(f"selected bridge training run not found: {selected_run}")
        parent_results = Path(str(evaluation_cfg["parent_aggregate_results_path"]))
        parent_results_hash = _require_hash(
            parent_results,
            str(evaluation_cfg["parent_aggregate_results_sha256"]),
            label="parent bridge aggregate results",
        )
        parent_payload = json.loads(parent_results.read_text(encoding="utf-8"))
        if str(parent_payload.get("status")) != "adjustment_pilot_full_100_completed":
            raise ValueError("test625 extension requires a completed 100-target bridge parent")
        parent_replay_metrics = Path(str(evaluation_cfg["parent_replay_metrics_path"]))
        parent_replay_metrics_hash = _require_hash(
            parent_replay_metrics,
            str(evaluation_cfg["parent_replay_metrics_sha256"]),
            label="parent bridge replay metrics",
        )
        parent_replay_payload = json.loads(parent_replay_metrics.read_text(encoding="utf-8"))
        if int(parent_replay_payload.get("failed_count", -1)) != 0:
            raise ValueError("test625 cache seed requires a zero-failure parent replay")
        historical_path = Path(str(evaluation_cfg["historical_baseline_per_target_path"]))
        historical_hash = _require_hash(
            historical_path,
            str(evaluation_cfg["historical_baseline_per_target_sha256"]),
            label="historical test625 baseline",
        )
        selected_checkpoints = _mapping(
            evaluation_cfg.get("selected_checkpoints"),
            name="evaluation.selected_checkpoints",
        )
        selected_lineage: dict[str, Any] = {}
        for method in [str(item) for item in training_cfg["methods_to_train"]]:
            expected = _mapping(
                selected_checkpoints.get(method),
                name=f"evaluation.selected_checkpoints.{method}",
            )
            selected_manifest = selected_run / "train" / method / "selected_checkpoint.json"
            if not selected_manifest.is_file():
                raise FileNotFoundError(
                    f"selected bridge checkpoint manifest not found for {method}: {selected_manifest}"
                )
            selected_payload = json.loads(selected_manifest.read_text(encoding="utf-8"))
            checkpoint = Path(str(selected_payload["checkpoint_path"]))
            checkpoint_hash = _require_hash(
                checkpoint,
                str(expected["sha256"]),
                label=f"{method} selected bridge checkpoint",
            )
            if checkpoint != Path(str(expected["path"])):
                raise ValueError(
                    f"{method} selected checkpoint path mismatch: "
                    f"manifest={checkpoint}, config={expected['path']}"
                )
            selected_lineage[method] = {
                "path": str(checkpoint),
                "sha256": checkpoint_hash,
                "selected_manifest_path": str(selected_manifest),
                "selected_manifest_sha256": _sha256_file(selected_manifest),
            }
        evaluation_lineage = {
            "scope": "full_test625_extension",
            "selected_training_run": str(selected_run),
            "selected_checkpoints": selected_lineage,
            "parent_aggregate_results_path": str(parent_results),
            "parent_aggregate_results_sha256": parent_results_hash,
            "parent_replay_metrics_path": str(parent_replay_metrics),
            "parent_replay_metrics_sha256": parent_replay_metrics_hash,
            "historical_baseline_per_target_path": str(historical_path),
            "historical_baseline_per_target_sha256": historical_hash,
        }
    k_path = recover_k_path(bundle)
    if not k_path.is_physical or not k_path.k_path_hash:
        raise ValueError("A2 requires a physically verified k-path with a stable hash")
    if str(dataset_cfg["k_path_sha256"]).lower() != str(k_path.k_path_hash).lower():
        raise ValueError(
            f"k-path hash mismatch: expected={dataset_cfg['k_path_sha256']}, actual={k_path.k_path_hash}"
        )

    solver = _solver_config(cfg)
    solver_hash = _sha256_json(_solver_payload(solver))
    expected_solver_hash = str(_mapping(cfg.get("solver"), name="solver")["protocol_sha256"]).lower()
    if solver_hash != expected_solver_hash:
        raise ValueError(f"solver protocol hash mismatch: expected={expected_solver_hash}, actual={solver_hash}")

    lineage = {
        "protocol_config_path": str(config_path),
        "protocol_config_sha256": _sha256_file(config_path),
        "baseline_protocol_path": str(baseline["path"]),
        "baseline_protocol_sha256": baseline_hash,
        "parent_protocol": parent_protocol_lineage,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "split_sizes": actual_splits,
        "target_manifest_path": str(target_manifest_path),
        "target_manifest_sha256": target_manifest_hash,
        "target_indices_sha256": _sha256_json(target_indices),
        "target_count": len(target_indices),
        "checkpoints": checkpoint_lineage,
        "warm_start_checkpoints": warm_start_lineage,
        "evaluation": evaluation_lineage,
        "k_path_id": k_path.k_path_id,
        "k_path_sha256": k_path.k_path_hash,
        "k_path_source": k_path.source,
        "solver_protocol_sha256": solver_hash,
        "solver": _solver_payload(solver),
        "code_revision": _git_revision(),
        "seed": 0,
        "evidence_status": evidence_status,
        "pino_inverse_status": "blocked_not_in_a2",
    }
    shutil.copyfile(target_manifest_path, run / "target_manifest_copy.json")
    _atomic_json(run / "lineage.json", lineage)
    decision = {
        "status": "pass",
        "next_status": "candidate_sampling"
        if _is_objective_bridge_test625(cfg)
        else "a2_0_smoke_required",
        "evidence_status": evidence_status,
        "training_authorized": not _is_objective_bridge_test625(cfg),
        "sampling_authorized": True,
        "real_mpb_smoke_authorized": True,
        "formal_replay_authorized_after_smoke": True,
        "cross_seed_claims_authorized": False,
        "pino_or_physics_rerank_authorized": False,
    }
    _atomic_json(run / "preflight_decision.json", decision)
    _atomic_json(run / "metrics.json", {"preflight_pass": True, "target_count": len(target_indices)})
    _atomic_json(run / "results.json", {"status": "preflight_passed", "evidence_status": evidence_status})
    _atomic_json(
        run / "run.meta.json",
        {
            "script": "phc2d inverse benchmark",
            "stage": "preflight",
            "status": "completed",
            "pid": os.getpid(),
            "end_timestamp": datetime.now().astimezone().isoformat(),
        },
    )
    logger.info("A2 preflight passed: %s", run)
    return run


def _git_revision() -> str:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _method_train_config(
    cfg: Mapping[str, Any],
    method: str,
    *,
    run_dir: Path,
) -> dict[str, Any]:
    training = _mapping(cfg.get("training"), name="training")
    method_cfg = _mapping(_mapping(cfg.get("methods"), name="methods")[method], name=f"methods.{method}")
    checkpoint = _mapping(cfg.get("checkpoints"), name="checkpoints")["ranking_forward"]
    return {
        "experiment": {
            "experiment_id": f"{_protocol(cfg)['id']}_{method}_train",
            "family": "research",
            "stage": "a2_single_seed_validation_selection",
        },
        "dataset": {
            "path": str(_dataset_cfg(cfg)["path"]),
            "condition_target": "Wh",
            "output_repr": "rho_wedge",
            "target_split": "val",
        },
        "model": {
            "name": method_cfg["model_name"],
            "seed": 0,
            "seeds": [0],
            "mode_count": int(method_cfg.get("mode_count", 1)),
            "latent_dim": int(method_cfg.get("latent_dim", 32)),
            "hidden_dim": int(method_cfg.get("hidden_dim", 256)),
            "depth": int(method_cfg.get("depth", 4)),
        },
        "surrogate": {
            "kind": "forward",
            "checkpoint_path": str(checkpoint["path"]),
            "model_name": str(checkpoint["model_name"]),
            "input_repr": str(checkpoint.get("input_repr", "epsilon_grid")),
        },
        "runtime": dict(_mapping(cfg.get("runtime"), name="runtime")),
        "inverse": {
            "train_epochs": int(training["epochs"]),
            "checkpoint_epochs": [int(item) for item in training["checkpoint_epochs"]],
            "train_batch_size": int(training["batch_size"]),
            "learning_rates": [float(item) for item in training["learning_rates"]],
            "learning_rate": float(training["learning_rates"][0]),
            "optimizer": str(training.get("optimizer", "adamw")),
            "weight_decay": float(training.get("weight_decay", 1.0e-5)),
            "n_candidates": int(_protocol(cfg)["requested_k"]),
            "surrogate_top_m": int(_protocol(cfg)["replay_top_m"]),
            "threshold": float(_protocol(cfg)["threshold"]),
            "candidate_separation": {
                "enabled": True,
                "deduplicate_binary": True,
                "min_hamming": str(_protocol(cfg)["minimum_normalized_hamming"]),
            },
            "allow_mock_generator": False,
            "allow_mock_surrogate": False,
        },
        "replay": {"enabled": False},
        "output": {"runs_dir": "runs", "run_dir": str(run_dir)},
    }


def _method_objective_bridge_config(
    cfg: Mapping[str, Any],
    method: str,
    *,
    run_dir: Path,
) -> dict[str, Any]:
    training = _mapping(cfg.get("training"), name="training")
    objective = dict(_mapping(training.get("objective"), name="training.objective"))
    diagnostic = _mapping(_mapping(cfg.get("checkpoints"), name="checkpoints")["diagnostic_query"])
    objective["secondary_surrogate"] = {
        "kind": "query",
        "checkpoint_path": str(diagnostic["path"]),
        "model_name": str(diagnostic["model_name"]),
        "input_repr": str(diagnostic.get("input_repr", "epsilon_grid")),
        "protocol": str(diagnostic.get("protocol", "unseen_structure_all_k")),
    }
    stage_cfg = _method_train_config(cfg, method, run_dir=run_dir)
    inverse = _mapping(stage_cfg["inverse"], name="inverse")
    warm_start = _mapping(
        _mapping(training.get("warm_start"), name="training.warm_start")[method],
        name=f"training.warm_start.{method}",
    )
    learning_rate = float(training["learning_rate"])
    inverse.update(
        {
            "train_epochs": int(training["epochs"]),
            "checkpoint_epochs": [int(item) for item in training["checkpoint_epochs"]],
            "learning_rate": learning_rate,
            "learning_rates": [learning_rate],
            "generator_checkpoint_path": str(warm_start["path"]),
            "ranking": {
                "error_metric": "bands_2_10_mre",
                "objective": objective,
            },
        }
    )
    stage_cfg["experiment"]["stage"] = "a2_single_seed_objective_bridge"
    stage_cfg["inverse"] = inverse
    return stage_cfg


def _lr_label(value: float) -> str:
    return f"lr{float(value):.0e}".replace("+", "").replace("-", "m")


def _binary_unique_separated(
    candidates: np.ndarray,
    *,
    threshold: float,
    minimum_hamming: float,
) -> tuple[np.ndarray, dict[str, float]]:
    raw = np.asarray(candidates, dtype=np.float32).reshape(int(candidates.shape[0]), -1)
    valid_mask = np.isfinite(raw).all(axis=1) & (raw >= 0.0).all(axis=1) & (raw <= 1.0).all(axis=1)
    valid = raw[valid_mask]
    binary = (valid >= float(threshold)).astype(np.float32)
    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    for row in binary:
        key = np.ascontiguousarray(row, dtype=np.uint8).tobytes()
        if key in seen:
            continue
        seen.add(key)
        if unique and min(float(np.mean(row != old)) for old in unique) < float(minimum_hamming):
            continue
        unique.append(row)
    selected = np.stack(unique, axis=0) if unique else np.empty((0, raw.shape[1]), dtype=np.float32)
    pairwise: list[float] = []
    for left in range(len(unique)):
        for right in range(left + 1, len(unique)):
            pairwise.append(float(np.mean(unique[left] != unique[right])))
    return selected, {
        "raw_k": int(raw.shape[0]),
        "valid_k": int(valid.shape[0]),
        "effective_unique_k": int(selected.shape[0]),
        "validity_rate": float(valid.shape[0] / max(raw.shape[0], 1)),
        "unique_rate": float(selected.shape[0] / max(valid.shape[0], 1)),
        "hamming_mean": float(np.mean(pairwise)) if pairwise else 0.0,
        "hamming_min": float(np.min(pairwise)) if pairwise else 0.0,
    }


def _continuous_candidate_diagnostics(
    candidates: np.ndarray,
    *,
    threshold: float,
    mode_count: int,
) -> dict[str, float]:
    flat = np.asarray(candidates, dtype=np.float32).reshape(int(candidates.shape[0]), -1)
    resolved_modes = max(1, int(mode_count))
    mode_separation = 0.0
    if resolved_modes > 1:
        centers = [
            np.mean(flat[np.arange(len(flat)) % resolved_modes == mode], axis=0)
            for mode in range(resolved_modes)
        ]
        pairwise = [
            float(np.mean(np.abs(centers[left] - centers[right])))
            for left in range(resolved_modes)
            for right in range(left + 1, resolved_modes)
        ]
        mode_separation = float(np.mean(pairwise)) if pairwise else 0.0
    return {
        "continuous_candidate_std": float(np.std(flat, axis=0).mean()),
        "threshold_margin_fraction": float(np.mean(np.abs(flat - float(threshold)) < 0.05)),
        "mode_center_separation": mode_separation,
    }


def _bands_mre(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = np.asarray(prediction, dtype=np.float64).reshape(int(prediction.shape[0]), *target.shape)
    tgt = np.asarray(target, dtype=np.float64)
    band_start = 1 if int(tgt.shape[-1]) >= 2 else 0
    delta = np.abs(pred[:, :, band_start:] - tgt[None, :, band_start:])
    denom = np.maximum(np.abs(tgt[None, :, band_start:]), 1.0e-8)
    return np.mean(delta / denom, axis=(1, 2))


def _evaluate_checkpoint(
    config_path: Path,
    checkpoint_path: Path,
    *,
    target_limit: int | None = None,
    warm_start_path: Path | None = None,
    diagnostic_surrogate_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from phc2d.inverse.runner import (
        _TrainedConditionalGenerator,
        _build_surrogate,
        _load_targets,
        _require_torch,
    )
    from phc2d.training import resolve_device, synchronize_device

    inverse_cfg = load_inverse_run_config(config_path)
    torch = _require_torch()
    device_info = resolve_device(inverse_cfg.runtime, torch_module=torch)
    device = device_info.torch_device
    bundle = load_forward_dataset(inverse_cfg.dataset_path)
    indices, all_targets, candidate_dim, _ = _load_targets(inverse_cfg)
    if target_limit is not None:
        indices = indices[: int(target_limit)]
    generator = _TrainedConditionalGenerator(inverse_cfg, checkpoint_path, torch=torch, device=device)
    parameter_count = int(sum(int(parameter.numel()) for parameter in generator.net.parameters()))
    target_shape = tuple(int(item) for item in np.asarray(all_targets).shape[1:])
    surrogate = _build_surrogate(
        inverse_cfg,
        input_dim=int(candidate_dim),
        target_shape=target_shape,
        torch=torch,
        device=device,
    )
    objective = _mapping(inverse_cfg.ranking.get("objective"), name="inverse.ranking.objective")
    secondary_entry = _mapping(objective.get("secondary_surrogate"), name="secondary_surrogate")
    diagnostic_surrogate = None
    resolved_diagnostic = dict(diagnostic_surrogate_cfg or secondary_entry)
    if resolved_diagnostic:
        diagnostic_cfg = replace(inverse_cfg, surrogate=resolved_diagnostic)
        diagnostic_surrogate = _build_surrogate(
            diagnostic_cfg,
            input_dim=int(candidate_dim),
            target_shape=target_shape,
            torch=torch,
            device=device,
        )
    minimum_hamming = 1.0 / float(np.asarray(bundle.rho).reshape(len(bundle.rho), -1).shape[1])
    errors: list[float] = []
    diagnostic_errors: list[float] = []
    cross_gaps: list[float] = []
    validity: list[float] = []
    unique: list[float] = []
    hamming: list[float] = []
    continuous_std: list[float] = []
    threshold_margin_fraction: list[float] = []
    mode_separation: list[float] = []
    sampling_times: list[float] = []
    primary_ranking_times: list[float] = []
    diagnostic_ranking_times: list[float] = []
    started = time.perf_counter()
    for index in indices:
        target = np.asarray(bundle.Wh[int(index)], dtype=np.float32)
        condition = torch.as_tensor(target.reshape(-1), dtype=torch.float32, device=device)
        sample_started = time.perf_counter()
        raw = generator.sample_tensor(condition, target_id=f"val_{int(index)}", target_index=int(index))
        synchronize_device(device, torch_module=torch)
        sampling_times.append(float(time.perf_counter() - sample_started))
        raw_array = raw.detach().cpu().numpy()
        candidates, metrics = _binary_unique_separated(
            raw_array,
            threshold=float(inverse_cfg.threshold),
            minimum_hamming=minimum_hamming,
        )
        mode_count = max(1, int(getattr(generator.net, "mode_count", 1)))
        continuous = _continuous_candidate_diagnostics(
            raw_array,
            threshold=float(inverse_cfg.threshold),
            mode_count=mode_count,
        )
        continuous_std.append(float(continuous["continuous_candidate_std"]))
        threshold_margin_fraction.append(float(continuous["threshold_margin_fraction"]))
        mode_separation.append(float(continuous["mode_center_separation"]))
        if candidates.size == 0:
            errors.append(float("inf"))
            diagnostic_errors.append(float("inf"))
            cross_gaps.append(float("inf"))
        else:
            candidate_tensor = torch.as_tensor(candidates, dtype=torch.float32, device=device)
            primary_started = time.perf_counter()
            prediction = surrogate.predict_tensor(candidate_tensor)
            synchronize_device(device, torch_module=torch)
            primary_ranking_times.append(float(time.perf_counter() - primary_started))
            primary_errors = _bands_mre(prediction.detach().cpu().numpy(), target)
            best_index = int(np.argmin(primary_errors))
            synchronize_device(device, torch_module=torch)
            errors.append(float(primary_errors[best_index]))
            if diagnostic_surrogate is None:
                diagnostic_errors.append(float(primary_errors[best_index]))
                cross_gaps.append(0.0)
            else:
                diagnostic_started = time.perf_counter()
                diagnostic_prediction = diagnostic_surrogate.predict_tensor(candidate_tensor)
                synchronize_device(device, torch_module=torch)
                diagnostic_ranking_times.append(float(time.perf_counter() - diagnostic_started))
                common_errors = _bands_mre(
                    diagnostic_prediction.detach().cpu().numpy(),
                    target,
                )
                diagnostic_errors.append(float(common_errors[best_index]))
                cross_gaps.append(
                    float(abs(float(common_errors[best_index]) - float(primary_errors[best_index])))
                )
        validity.append(float(metrics["validity_rate"]))
        unique.append(float(metrics["effective_unique_k"]))
        hamming.append(float(metrics["hamming_mean"]))
    latent_weight_change = 0.0
    if warm_start_path is not None:
        warm_payload = torch.load(warm_start_path, map_location="cpu", weights_only=False)
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        warm_state = warm_payload["state_dict"]
        checkpoint_state = checkpoint_payload["state_dict"]
        latent_dim = int(checkpoint_payload.get("latent_dim", 0))
        if latent_dim > 0 and "decoder" in checkpoint_state:
            warm_weight = warm_state["decoder"]["0.weight"][:, -latent_dim:]
            checkpoint_weight = checkpoint_state["decoder"]["0.weight"][:, -latent_dim:]
            latent_weight_change = float((checkpoint_weight - warm_weight).abs().max().item())
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "parameter_count": parameter_count,
        "validation_target_count": int(len(indices)),
        "validation_best_of_k_surrogate_mre": float(np.mean(errors)),
        "validation_primary_selected_common_fno_mre": float(np.mean(diagnostic_errors)),
        "cross_surrogate_gap_mean": float(np.mean(cross_gaps)),
        "validity_rate": float(np.mean(validity)),
        "effective_unique_k": float(np.mean(unique)),
        "effective_unique_k_p10": float(np.percentile(unique, 10.0)),
        "hamming_mean": float(np.mean(hamming)),
        "continuous_candidate_std": float(np.mean(continuous_std)),
        "threshold_margin_fraction": float(np.mean(threshold_margin_fraction)),
        "mode_center_separation": float(np.mean(mode_separation)),
        "latent_weight_change_max": latent_weight_change,
        "candidate_sampling_time_s": float(np.sum(sampling_times)),
        "primary_surrogate_ranking_time_s": float(np.sum(primary_ranking_times)),
        "common_fno_diagnostic_ranking_time_s": float(np.sum(diagnostic_ranking_times)),
        "eval_time_s": float(time.perf_counter() - started),
    }


def _bridge_gate_failures(
    row: Mapping[str, Any],
    *,
    method: str,
    training: Mapping[str, Any],
) -> list[str]:
    gates = _mapping(training.get("candidate_health_gates"), name="training.candidate_health_gates")
    baselines = _mapping(training.get("baseline_validation_mre"), name="training.baseline_validation_mre")
    failures: list[str] = []
    checks = (
        ("validity_rate", float(row["validity_rate"]), float(gates["validity_rate_min"]), "min"),
        (
            "effective_unique_k",
            float(row["effective_unique_k"]),
            float(gates["effective_k_mean_min"]),
            "min",
        ),
        (
            "effective_unique_k_p10",
            float(row["effective_unique_k_p10"]),
            float(gates["effective_k_p10_min"]),
            "min",
        ),
        (
            "continuous_candidate_std",
            float(row["continuous_candidate_std"]),
            float(gates["continuous_candidate_std_min"]),
            "min",
        ),
        (
            "threshold_margin_fraction",
            float(row["threshold_margin_fraction"]),
            float(gates["threshold_margin_fraction_max"]),
            "max",
        ),
        (
            "latent_weight_change_max",
            float(row["latent_weight_change_max"]),
            float(gates["latent_weight_change_min"]),
            "min",
        ),
        (
            "latent_input_gradient_norm",
            float(row.get("latent_input_gradient_norm", 0.0)),
            float(gates["latent_gradient_norm_min"]),
            "min",
        ),
        (
            "mode_coverage_fraction",
            float(row.get("mode_coverage_fraction", 0.0)),
            float(gates["mode_coverage_fraction_min"]),
            "min",
        ),
        (
            "cross_surrogate_gap_mean",
            float(row["cross_surrogate_gap_mean"]),
            float(gates["cross_surrogate_gap_max"]),
            "max",
        ),
    )
    for name, value, threshold, direction in checks:
        if not math.isfinite(value):
            failures.append(f"{name}: non_finite")
        elif direction == "min" and value < threshold:
            failures.append(f"{name}: {value} < {threshold}")
        elif direction == "max" and value > threshold:
            failures.append(f"{name}: {value} > {threshold}")
    relative_improvement = float(gates["validation_mre_relative_improvement_min"])
    maximum_mre = float(baselines[method]) * (1.0 - relative_improvement)
    actual_mre = float(row["validation_best_of_k_surrogate_mre"])
    if not math.isfinite(actual_mre) or actual_mre > maximum_mre:
        failures.append(
            f"validation_best_of_k_surrogate_mre: {actual_mre} > {maximum_mre}"
        )
    return failures


def _history_row_for_epoch(path: Path, *, epoch: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if int(row["epoch"]) == int(epoch)]
    if len(matches) != 1:
        raise ValueError(f"expected one objective history row for epoch {epoch}, got {len(matches)}")
    return matches[0]


def _run_a2_objective_bridge_train(
    cfg: Mapping[str, Any],
    *,
    run: Path,
    config_path: str | Path,
) -> Path:
    from phc2d.inverse.runner import run_inverse_objective_train

    training = _mapping(cfg.get("training"), name="training")
    training_root = run / "train"
    training_root.mkdir(parents=True, exist_ok=True)
    methods_to_train = [str(item) for item in training["methods_to_train"]]
    checkpoint_epochs = [int(item) for item in training["checkpoint_epochs"]]
    learning_rate = float(training["learning_rate"])
    validation_limit = training.get("validation_target_limit")
    selection_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    blocked_methods: list[str] = []

    direct_dir = training_root / "direct_inverse"
    direct_dir.mkdir(parents=True, exist_ok=True)
    direct_cfg = _method_train_config(cfg, "direct_inverse", run_dir=direct_dir)
    direct_cfg_path = training_root / "direct_inverse.yaml"
    direct_cfg_path.write_text(yaml.safe_dump(direct_cfg, sort_keys=False), encoding="utf-8")
    direct_warm = _mapping(
        _mapping(training["warm_start"], name="training.warm_start")["direct_inverse"],
        name="training.warm_start.direct_inverse",
    )
    direct_selection = {
        "method": "direct_inverse",
        "checkpoint_path": str(direct_warm["path"]),
        "checkpoint_sha256": str(direct_warm["sha256"]),
        "frozen_baseline": True,
        "selection_split": "historical_a2_validation",
    }
    _atomic_json(direct_dir / "selected_checkpoint.json", direct_selection)
    selections.append(direct_selection)

    for method in methods_to_train:
        method_dir = training_root / method
        stage_cfg = _method_objective_bridge_config(cfg, method, run_dir=method_dir)
        method_cfg_path = training_root / f"{method}.yaml"
        method_cfg_path.write_text(yaml.safe_dump(stage_cfg, sort_keys=False), encoding="utf-8")
        if not (method_dir / "results.json").is_file():
            run_inverse_objective_train(method_cfg_path)
        warm_start = Path(
            str(
                _mapping(
                    _mapping(training["warm_start"], name="training.warm_start")[method],
                    name=f"training.warm_start.{method}",
                )["path"]
            )
        )
        method_rows: list[dict[str, Any]] = []
        for epoch in checkpoint_epochs:
            checkpoint = method_dir / f"objective_checkpoint_epoch_{epoch}_seed0_{_lr_label(learning_rate)}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"missing objective validation checkpoint: {checkpoint}")
            cache_path = method_dir / "validation" / f"{checkpoint.stem}.json"
            if cache_path.is_file():
                row = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                row = _evaluate_checkpoint(
                    method_cfg_path,
                    checkpoint,
                    target_limit=int(validation_limit) if validation_limit is not None else None,
                    warm_start_path=warm_start,
                )
                history_row = _history_row_for_epoch(method_dir / "history.csv", epoch=epoch)
                row.update(
                    {
                        "method": method,
                        "learning_rate": learning_rate,
                        "epoch": epoch,
                        "latent_input_gradient_norm": float(
                            history_row.get("latent_input_gradient_norm", 0.0)
                        ),
                        "mode_coverage_fraction": float(
                            history_row.get("mode_coverage_fraction", 0.0)
                        ),
                    }
                )
                failures = _bridge_gate_failures(row, method=method, training=training)
                row["candidate_health_pass"] = not failures
                row["candidate_health_failures"] = failures
                _atomic_json(cache_path, row)
            method_rows.append(row)
            selection_rows.append(row)
        eligible = [row for row in method_rows if bool(row.get("candidate_health_pass", False))]
        if not eligible:
            blocked_methods.append(method)
            continue
        best = min(eligible, key=lambda row: float(row["validation_best_of_k_surrogate_mre"]))
        _atomic_json(method_dir / "selected_checkpoint.json", best)
        selections.append(best)

    _write_csv(training_root / "validation_history.csv", selection_rows)
    gate_decision = {
        "protocol": str(_protocol(cfg)["id"]),
        "config_path": str(config_path),
        "status": "pass" if not blocked_methods else "blocked_candidate_health",
        "blocked_methods": blocked_methods,
        "methods_to_train": methods_to_train,
        "seed": 0,
        "evidence_status": _evidence_status(cfg),
        "selection_split": "val",
        "test_accessed_for_selection": False,
    }
    _atomic_json(training_root / "candidate_health_decision.json", gate_decision)
    if blocked_methods:
        _atomic_json(
            training_root / "results.json",
            {"status": "blocked_candidate_health", "blocked_methods": blocked_methods},
        )
        _atomic_json(
            training_root / "metrics.json",
            {
                "candidate_health_pass": False,
                "blocked_method_count": len(blocked_methods),
                "checkpoint_evaluation_count": len(selection_rows),
            },
        )
        return training_root

    manifest = {
        "protocol": "a2_single_seed_objective_bridge_validation_v1",
        "selection_split": "val",
        "final_evaluation_split": "test",
        "test_accessed_for_selection": False,
        "seed": 0,
        "evidence_status": _evidence_status(cfg),
        "selections": selections,
    }
    _atomic_json(training_root / "validation_selection_manifest.json", manifest)
    _atomic_json(
        training_root / "metrics.json",
        {
            "candidate_health_pass": True,
            "checkpoint_evaluation_count": len(selection_rows),
            "selected_method_count": len(selections),
        },
    )
    _atomic_json(training_root / "results.json", {"status": "validation_selection_completed"})
    return training_root


def run_a2_train(config_path: str | Path, *, run_dir: str | Path) -> Path:
    """Train 3 LR x seed 0 for each trainable method and select on validation only."""

    from phc2d.inverse.runner import run_inverse_train

    cfg = _read_yaml(config_path)
    run = Path(run_dir)
    if not (run / "preflight_decision.json").is_file():
        raise FileNotFoundError("preflight_decision.json is required before A2 training")
    if _is_objective_bridge_test625(cfg):
        raise RuntimeError("test625 objective-bridge extension is evaluation-only; training is prohibited")
    if _is_objective_bridge(cfg):
        return _run_a2_objective_bridge_train(cfg, run=run, config_path=config_path)
    training_root = run / "train"
    training_root.mkdir(parents=True, exist_ok=True)
    selection_rows: list[dict[str, Any]] = []
    validation_limit = _mapping(cfg.get("training"), name="training").get("validation_target_limit")
    for method in TRAINABLE_METHODS:
        method_dir = training_root / method
        stage_cfg = _method_train_config(cfg, method, run_dir=method_dir)
        config_out = training_root / f"{method}.yaml"
        config_out.write_text(yaml.safe_dump(stage_cfg, sort_keys=False), encoding="utf-8")
        if not (method_dir / "results.json").is_file():
            run_inverse_train(config_out)
        checkpoint_epochs = [int(item) for item in _mapping(cfg.get("training"))["checkpoint_epochs"]]
        learning_rates = [float(item) for item in _mapping(cfg.get("training"))["learning_rates"]]
        method_rows: list[dict[str, Any]] = []
        for learning_rate in learning_rates:
            for epoch in checkpoint_epochs:
                checkpoint = method_dir / f"checkpoint_epoch_{epoch}_seed0_{_lr_label(learning_rate)}.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"missing validation checkpoint: {checkpoint}")
                cache_path = method_dir / "validation" / f"{checkpoint.stem}.json"
                if cache_path.is_file():
                    row = json.loads(cache_path.read_text(encoding="utf-8"))
                else:
                    row = _evaluate_checkpoint(
                        config_out,
                        checkpoint,
                        target_limit=int(validation_limit) if validation_limit is not None else None,
                    )
                    row.update({"method": method, "learning_rate": learning_rate, "epoch": epoch})
                    _atomic_json(cache_path, row)
                method_rows.append(row)
                selection_rows.append(row)
        best = min(method_rows, key=lambda row: float(row["validation_best_of_k_surrogate_mre"]))
        _atomic_json(method_dir / "selected_checkpoint.json", best)

    selected = []
    for method in TRAINABLE_METHODS:
        selected.append(json.loads((training_root / method / "selected_checkpoint.json").read_text(encoding="utf-8")))
    manifest = {
        "protocol": "a2_single_seed_validation_selected_checkpoint_v1",
        "selection_split": "val",
        "final_evaluation_split": "test",
        "test_accessed_for_selection": False,
        "seed": 0,
        "evidence_status": _evidence_status(cfg),
        "selections": selected,
    }
    _write_csv(training_root / "validation_history.csv", selection_rows)
    _atomic_json(training_root / "validation_selection_manifest.json", manifest)
    _atomic_json(
        training_root / "metrics.json",
        {
            "validation_case_count": len(TRAINABLE_METHODS)
            * len(_mapping(cfg.get("training"))["learning_rates"]),
            "expected_validation_case_count": 9,
            "checkpoint_evaluation_count": len(selection_rows),
            "selected_method_count": len(selected),
        },
    )
    _atomic_json(training_root / "results.json", {"status": "validation_selection_completed"})
    return training_root


def _load_selected_generators(
    cfg: Mapping[str, Any],
    run: Path,
    *,
    methods: Sequence[str] = TRAINABLE_METHODS,
) -> tuple[Any, dict[str, Any], dict[str, GenerativeInverseConfig]]:
    from phc2d.inverse.runner import _TrainedConditionalGenerator, _require_torch
    from phc2d.training import resolve_device

    torch = _require_torch()
    configs: dict[str, GenerativeInverseConfig] = {}
    generators: dict[str, Any] = {}
    selected_run = _selected_training_run(cfg, run=run)
    for method in methods:
        method_cfg_path = selected_run / "train" / f"{method}.yaml"
        selected_path = selected_run / "train" / method / "selected_checkpoint.json"
        if not selected_path.is_file():
            raise FileNotFoundError(f"selection manifest missing for {method}: {selected_path}")
        inverse_cfg = load_inverse_run_config(method_cfg_path)
        device = resolve_device(inverse_cfg.runtime, torch_module=torch).torch_device
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        checkpoint = Path(str(selected["checkpoint_path"]))
        _require_hash(checkpoint, str(selected["checkpoint_sha256"]), label=f"{method} selected checkpoint")
        configs[method] = inverse_cfg
        generators[method] = _TrainedConditionalGenerator(
            inverse_cfg,
            checkpoint,
            torch=torch,
            device=device,
        )
    return torch, generators, configs


def _fake_candidates(method: str, *, candidate_dim: int, requested_k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if method in {"retrieval", "direct_inverse"}:
        row = rng.integers(0, 2, size=(1, candidate_dim), dtype=np.int8).astype(np.float32)
        return np.repeat(row, requested_k if method == "direct_inverse" else 1, axis=0)
    base = rng.random((requested_k, candidate_dim), dtype=np.float32)
    if method == "single_mode_generator":
        base = 0.7 * base + 0.3 * base[:1]
    return base


def _candidate_manifest_arrays(
    *,
    candidates: np.ndarray,
    predictions: np.ndarray,
    diagnostic_predictions: np.ndarray,
    errors: np.ndarray,
    diagnostic_errors: np.ndarray,
    cross_surrogate_gap: np.ndarray,
    ranking: np.ndarray,
    top: np.ndarray,
    storage: str,
) -> dict[str, np.ndarray]:
    if storage == "top_m_only":
        return {
            "candidates": candidates[top],
            "surrogate_predictions": predictions[top],
            "diagnostic_query_predictions": diagnostic_predictions[top],
            "surrogate_errors_bands_2_10": errors[top],
            "diagnostic_query_errors_bands_2_10": diagnostic_errors[top],
            "cross_surrogate_gap_bands_2_10": cross_surrogate_gap[top],
            "ranked_indices": np.arange(len(top), dtype=np.int64),
            "selected_indices": np.arange(len(top), dtype=np.int64),
            "selected_source_indices": top,
        }
    return {
        "candidates": candidates,
        "surrogate_predictions": predictions,
        "diagnostic_query_predictions": diagnostic_predictions,
        "surrogate_errors_bands_2_10": errors,
        "diagnostic_query_errors_bands_2_10": diagnostic_errors,
        "cross_surrogate_gap_bands_2_10": cross_surrogate_gap,
        "ranked_indices": ranking,
        "selected_indices": top,
        "selected_source_indices": top,
    }


def run_a2_sample(
    config_path: str | Path,
    *,
    run_dir: str | Path,
    fake_models: bool = False,
    target_limit: int | None = None,
) -> Path:
    """Create and atomically freeze 4-method candidate records for paired targets."""

    from phc2d.inverse.runner import _ForwardFrozenSurrogate, _QueryFrozenSurrogate, _require_torch
    from phc2d.training import resolve_device, synchronize_device

    cfg = _read_yaml(config_path)
    run = Path(run_dir)
    candidate_dir = run / ("smoke_fake" if fake_models else "candidates")
    candidate_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = candidate_dir / "candidate_manifest.frozen.json"
    if frozen_path.is_file():
        return candidate_dir
    dataset = load_forward_dataset(_dataset_cfg(cfg)["path"])
    indices = _target_indices(cfg)
    if target_limit is not None:
        indices = indices[: int(target_limit)]
    protocol = _protocol(cfg)
    sampling_cfg = _mapping(cfg.get("sampling"), name="sampling")
    sample_methods = tuple(str(item) for item in sampling_cfg.get("methods", METHODS))
    if not sample_methods or len(set(sample_methods)) != len(sample_methods):
        raise ValueError("sampling.methods must be a non-empty unique method list")
    if any(method not in METHODS for method in sample_methods):
        raise ValueError(f"sampling.methods must be a subset of {list(METHODS)!r}")
    manifest_candidate_storage = str(
        sampling_cfg.get("manifest_candidate_storage", "all_unique")
    ).strip().lower()
    if manifest_candidate_storage not in {"all_unique", "top_m_only"}:
        raise ValueError("sampling.manifest_candidate_storage must be all_unique or top_m_only")
    requested_k = int(protocol["requested_k"])
    replay_top_m = int(protocol["replay_top_m"])
    threshold = float(protocol["threshold"])
    candidate_dim = int(np.asarray(dataset.rho).reshape(len(dataset.rho), -1).shape[1])
    min_hamming = 1.0 / float(candidate_dim)
    torch = _require_torch()
    ranking_entry = _mapping(_mapping(cfg.get("checkpoints"))["ranking_forward"])
    selected_run = _selected_training_run(cfg, run=run)
    runtime_method = next(
        (method for method in sample_methods if method in TRAINABLE_METHODS),
        "direct_inverse",
    )
    base_inverse_cfg = GenerativeInverseConfig(
        dataset_path=Path(str(_dataset_cfg(cfg)["path"])),
        output_repr="rho_wedge",
        surrogate={
            "kind": "forward",
            "checkpoint_path": str(ranking_entry["path"]),
            "model_name": str(ranking_entry["model_name"]),
            "input_repr": str(ranking_entry.get("input_repr", "epsilon_grid")),
        },
        runtime=load_inverse_run_config(selected_run / "train" / f"{runtime_method}.yaml").runtime
        if not fake_models
        else GenerativeInverseConfig().runtime,
        allow_mock_surrogate=False,
    )
    device_info = resolve_device(base_inverse_cfg.runtime, torch_module=torch)
    surrogate = None if fake_models else _ForwardFrozenSurrogate(
        base_inverse_cfg,
        torch=torch,
        device=device_info.torch_device,
    )
    diagnostic_surrogate = None
    diagnostic_entry = _mapping(_mapping(cfg.get("checkpoints"))["diagnostic_query"])
    if not fake_models:
        diagnostic_cfg = replace(
            base_inverse_cfg,
            surrogate={
                "kind": "query",
                "checkpoint_path": str(diagnostic_entry["path"]),
                "model_name": str(diagnostic_entry["model_name"]),
                "input_repr": str(diagnostic_entry.get("input_repr", "epsilon_grid")),
                "protocol": str(diagnostic_entry.get("protocol", "unseen_structure_all_k")),
            },
        )
        diagnostic_surrogate = _QueryFrozenSurrogate(
            diagnostic_cfg,
            torch=torch,
            device=device_info.torch_device,
        )
    generators: dict[str, Any] = {}
    method_configs: dict[str, GenerativeInverseConfig] = {}
    if not fake_models:
        generator_methods = [method for method in sample_methods if method in TRAINABLE_METHODS]
        torch, generators, method_configs = _load_selected_generators(
            cfg,
            run,
            methods=generator_methods,
        )

    train_indices = np.asarray(dataset.idx_train, dtype=np.int64)
    train_targets = np.asarray(dataset.Wh[train_indices], dtype=np.float32)
    train_rho = np.asarray(dataset.rho[train_indices], dtype=np.float32)
    test_set = {int(item) for item in np.asarray(dataset.idx_test, dtype=np.int64)}
    if any(int(item) not in test_set for item in indices):
        raise ValueError("candidate targets must remain inside the test split")

    records: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for target_position, target_index in enumerate(indices):
        target = np.asarray(dataset.Wh[int(target_index)], dtype=np.float32)
        for method_position, method in enumerate(sample_methods):
            started = time.perf_counter()
            if fake_models:
                raw = _fake_candidates(
                    method,
                    candidate_dim=candidate_dim,
                    requested_k=requested_k,
                    seed=target_position * 100 + method_position,
                )
                sampling_time = time.perf_counter() - started
            elif method == "retrieval":
                errors = _bands_mre(train_targets, target)
                nearest_train_local = np.argsort(errors, kind="stable")[:requested_k]
                raw = train_rho[np.asarray(nearest_train_local, dtype=np.int64)].reshape(
                    len(nearest_train_local),
                    -1,
                )
                sampling_time = time.perf_counter() - started
            else:
                method_cfg = method_configs[method]
                condition = torch.as_tensor(
                    target.reshape(-1),
                    dtype=torch.float32,
                    device=generators[method].device,
                )
                raw_tensor = generators[method].sample_tensor(
                    condition,
                    target_id=f"test_{int(target_index)}",
                    target_index=int(target_index),
                )
                synchronize_device(generators[method].device, torch_module=torch)
                raw = raw_tensor.detach().cpu().numpy()
                sampling_time = time.perf_counter() - started
            mode_count = (
                max(1, int(getattr(generators[method].net, "mode_count", 1)))
                if method in generators
                else 1
            )
            continuous_diagnostics = _continuous_candidate_diagnostics(
                raw,
                threshold=threshold,
                mode_count=mode_count,
            )
            candidates, uniqueness = _binary_unique_separated(
                raw,
                threshold=threshold,
                minimum_hamming=min_hamming,
            )
            if candidates.size == 0:
                raise ValueError(f"{method} target {target_index} has no valid unique candidates")
            ranking_started = time.perf_counter()
            if fake_models:
                rng = np.random.default_rng(target_position * 1000 + method_position)
                predictions = target[None, :, :] * (
                    1.0 + rng.normal(0.0, 0.02, size=(len(candidates), *target.shape))
                )
            else:
                tensor = torch.as_tensor(candidates, dtype=torch.float32, device=device_info.torch_device)
                predictions = surrogate.predict_tensor(tensor).detach().cpu().numpy().reshape(
                    len(candidates), *target.shape
                )
                diagnostic_predictions = diagnostic_surrogate.predict_tensor(tensor).detach().cpu().numpy().reshape(
                    len(candidates), *target.shape
                )
                synchronize_device(device_info.torch_device, torch_module=torch)
            if fake_models:
                diagnostic_predictions = predictions.copy()
            errors = _bands_mre(predictions, target)
            diagnostic_errors = _bands_mre(diagnostic_predictions, target)
            cross_surrogate_gap = np.mean(
                np.abs(
                    np.asarray(predictions, dtype=np.float64)[:, :, 1:10]
                    - np.asarray(diagnostic_predictions, dtype=np.float64)[:, :, 1:10]
                )
                / np.maximum(np.abs(target[None, :, 1:10]), 1.0e-8),
                axis=(1, 2),
            )
            ranking = np.argsort(errors, kind="stable")
            top = ranking[: min(replay_top_m, len(ranking))]
            ranking_time = time.perf_counter() - ranking_started
            manifest_arrays = _candidate_manifest_arrays(
                candidates=candidates,
                predictions=predictions,
                diagnostic_predictions=diagnostic_predictions,
                errors=errors,
                diagnostic_errors=diagnostic_errors,
                cross_surrogate_gap=cross_surrogate_gap,
                ranking=ranking,
                top=top,
                storage=manifest_candidate_storage,
            )
            record = {
                "record_id": f"{method}_target_{int(target_index)}",
                "method": method,
                "target_id": f"test_{int(target_index)}",
                "target_index": int(target_index),
                "target_position": int(target_position),
                "split": "test",
                "condition": {"Wh": target},
                "requested_k": requested_k,
                "raw_k": int(uniqueness["raw_k"]),
                "effective_unique_k": int(uniqueness["effective_unique_k"]),
                "manifest_candidate_storage": manifest_candidate_storage,
                **manifest_arrays,
                "best_of_k_surrogate_mre": float(np.min(errors)),
                "best_of_k_diagnostic_query_mre": float(np.min(diagnostic_errors)),
                "best_of_k_cross_surrogate_gap": float(np.min(cross_surrogate_gap)),
                **continuous_diagnostics,
                "sampling_time_s": float(sampling_time),
                "ranking_time_s": float(ranking_time),
                "lineage": {
                    "dataset_sha256": str(_dataset_cfg(cfg)["expected_sha256"]),
                    "target_manifest_sha256": str(_dataset_cfg(cfg)["target_manifest_sha256"]),
                    "ranking_checkpoint_path": str(ranking_entry["path"]),
                    "ranking_checkpoint_sha256": str(ranking_entry["sha256"]),
                    "diagnostic_query_checkpoint_path": str(diagnostic_entry["path"]),
                    "diagnostic_query_checkpoint_sha256": str(diagnostic_entry["sha256"]),
                    "generator_checkpoint": None
                    if method == "retrieval" or fake_models
                    else json.loads(
                        (selected_run / "train" / method / "selected_checkpoint.json").read_text(
                            encoding="utf-8"
                        )
                    ),
                    "retrieval_library_split": "train" if method == "retrieval" else None,
                },
                **uniqueness,
            }
            records.append(record)
            runtime_rows.append(
                {
                    "method": method,
                    "target_index": int(target_index),
                    "candidate_sampling_time_s": sampling_time,
                    "surrogate_ranking_time_s": ranking_time,
                    "effective_unique_k": int(uniqueness["effective_unique_k"]),
                }
            )

    manifest = {
        "version": 1,
        "status": "frozen",
        "frozen_at": datetime.now().astimezone().isoformat(),
        "evidence_status": "fake_mpb_smoke" if fake_models else _evidence_status(cfg),
        "seed": 0,
        "methods": list(sample_methods),
        "target_count": len(indices),
        "record_count": len(records),
        "requested_k": requested_k,
        "replay_top_m": replay_top_m,
        "threshold": threshold,
        "minimum_normalized_hamming": min_hamming,
        "manifest_candidate_storage": manifest_candidate_storage,
        "ranking_checkpoint_path": str(ranking_entry["path"]),
        "ranking_checkpoint_sha256": str(ranking_entry["sha256"]),
        "records": records,
    }
    candidate_health: dict[str, Any] = {}
    candidate_blocked_methods: list[str] = []
    if _is_objective_bridge(cfg) and not fake_models:
        training_cfg = _mapping(cfg.get("training"), name="training")
        gates = _mapping(
            training_cfg.get("candidate_health_gates"),
            name="training.candidate_health_gates",
        )
        for method in [str(item) for item in training_cfg["methods_to_train"]]:
            method_records = [record for record in records if record["method"] == method]
            effective = np.asarray(
                [float(record["effective_unique_k"]) for record in method_records],
                dtype=np.float64,
            )
            health_row = {
                "method": method,
                "target_count": len(method_records),
                "validity_rate": float(
                    np.mean(
                        [
                            float(record["valid_k"]) / max(float(record["raw_k"]), 1.0)
                            for record in method_records
                        ]
                    )
                ),
                "effective_k_mean": float(np.mean(effective)),
                "effective_k_p10": float(np.percentile(effective, 10.0)),
                "continuous_candidate_std": float(
                    np.mean([float(record["continuous_candidate_std"]) for record in method_records])
                ),
                "threshold_margin_fraction": float(
                    np.mean([float(record["threshold_margin_fraction"]) for record in method_records])
                ),
            }
            failures = []
            for name, value, threshold_value, direction in (
                (
                    "validity_rate",
                    health_row["validity_rate"],
                    float(gates["validity_rate_min"]),
                    "min",
                ),
                (
                    "effective_k_mean",
                    health_row["effective_k_mean"],
                    float(gates["effective_k_mean_min"]),
                    "min",
                ),
                (
                    "effective_k_p10",
                    health_row["effective_k_p10"],
                    float(gates["effective_k_p10_min"]),
                    "min",
                ),
                (
                    "continuous_candidate_std",
                    health_row["continuous_candidate_std"],
                    float(gates["continuous_candidate_std_min"]),
                    "min",
                ),
                (
                    "threshold_margin_fraction",
                    health_row["threshold_margin_fraction"],
                    float(gates["threshold_margin_fraction_max"]),
                    "max",
                ),
            ):
                if direction == "min" and value < threshold_value:
                    failures.append(f"{name}: {value} < {threshold_value}")
                elif direction == "max" and value > threshold_value:
                    failures.append(f"{name}: {value} > {threshold_value}")
            health_row["pass"] = not failures
            health_row["failures"] = failures
            candidate_health[method] = health_row
            if failures:
                candidate_blocked_methods.append(method)
    _atomic_json(frozen_path, manifest)
    _atomic_json(candidate_dir / "candidate_manifest.sha256.json", {"sha256": _sha256_file(frozen_path)})
    _write_csv(candidate_dir / "runtime_ledger.csv", runtime_rows)
    _atomic_json(
        candidate_dir / "metrics.json",
        {
            "target_count": len(indices),
            "record_count": len(records),
            "expected_record_count": len(indices) * len(sample_methods),
            "candidate_sampling_time_s": float(sum(row["candidate_sampling_time_s"] for row in runtime_rows)),
            "surrogate_ranking_time_s": float(sum(row["surrogate_ranking_time_s"] for row in runtime_rows)),
        },
    )
    if _is_objective_bridge(cfg) and not fake_models:
        _atomic_json(
            candidate_dir / "candidate_health_decision.json",
            {
                "status": "pass" if not candidate_blocked_methods else "blocked_candidate_health",
                "blocked_methods": candidate_blocked_methods,
                "methods": candidate_health,
                "test_mpb_accessed": False,
            },
        )
    _atomic_json(
        candidate_dir / "results.json",
        {
            "status": "candidate_manifest_frozen"
            if not candidate_blocked_methods
            else "blocked_candidate_health",
            "blocked_methods": candidate_blocked_methods,
        },
    )
    return candidate_dir


def _fake_solver(
    rho: np.ndarray,
    *,
    k_points: Sequence[Sequence[float]],
    config: MpbBandSolverConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    seed = int(hash_array(np.asarray(rho))[:8], 16)
    rng = np.random.default_rng(seed)
    bands = np.sort(
        rng.uniform(0.15, 0.85, size=(len(k_points), int(config.num_bands))),
        axis=1,
    ).astype(np.float32)
    return bands, {"backend": "fake_mpb", "seed": seed}


def run_a2_replay(
    config_path: str | Path,
    *,
    run_dir: str | Path,
    fake_mpb: bool = False,
    smoke: bool = False,
    pilot: bool = False,
    solver_fn: SolverFn | None = None,
) -> Path:
    """Replay frozen top-M candidates with structure-hash caching and shard resume."""

    cfg = _read_yaml(config_path)
    run = Path(run_dir)
    formal_candidate_dir = run / "candidates"
    candidate_dir = (
        run / "smoke_fake"
        if fake_mpb or (smoke and not (formal_candidate_dir / "candidate_manifest.frozen.json").is_file())
        else formal_candidate_dir
    )
    manifest_path = candidate_dir / "candidate_manifest.frozen.json"
    hash_path = candidate_dir / "candidate_manifest.sha256.json"
    expected_manifest_hash = json.loads(hash_path.read_text(encoding="utf-8"))["sha256"]
    _require_hash(manifest_path, expected_manifest_hash, label="frozen candidate manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _is_objective_bridge(cfg) and not fake_mpb and not smoke:
        health_path = candidate_dir / "candidate_health_decision.json"
        health = json.loads(health_path.read_text(encoding="utf-8"))
        if str(health.get("status")) != "pass":
            raise RuntimeError("candidate-health gate did not pass; MPB replay is prohibited")
        if not pilot:
            if _is_objective_bridge_test625(cfg):
                evaluation_cfg = _mapping(cfg.get("evaluation"), name="evaluation")
                parent_path = Path(str(evaluation_cfg["parent_aggregate_results_path"]))
                _require_hash(
                    parent_path,
                    str(evaluation_cfg["parent_aggregate_results_sha256"]),
                    label="parent bridge aggregate results",
                )
                parent = json.loads(parent_path.read_text(encoding="utf-8"))
                if str(parent.get("status")) != "adjustment_pilot_full_100_completed":
                    raise RuntimeError(
                        "test625 replay requires a completed 100-target objective-bridge parent"
                    )
            else:
                promotion_path = run / "pilot" / "promotion_decision.json"
                if not promotion_path.is_file():
                    raise FileNotFoundError(
                        "objective-bridge full replay requires pilot/promotion_decision.json"
                    )
                promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
                if str(promotion.get("status")) != "promote_full_100":
                    raise RuntimeError(
                        "mixture pilot did not pass; objective-bridge full replay is prohibited"
                    )
    replay_dir = run / (
        "replay_fake"
        if fake_mpb
        else ("replay_smoke" if smoke else ("replay_pilot" if pilot else "replay"))
    )
    replay_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (run / "replay" / "structure_cache") if pilot else replay_dir / "structure_cache"
    references: list[dict[str, Any]] = []
    unique: dict[str, np.ndarray] = {}
    references_by_method: dict[str, int] = {method: 0 for method in METHODS}
    max_targets = 2 if smoke else None
    max_candidates = 2 if smoke else int(_protocol(cfg)["replay_top_m"])
    replay_cfg = _mapping(cfg.get("replay"), name="replay")
    seeded_cache_pair_count = 0
    if not fake_mpb and not smoke and not pilot:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for source_value in replay_cfg.get("seed_structure_cache_dirs", ()):
            source_dir = Path(str(source_value))
            if not source_dir.is_dir():
                raise FileNotFoundError(f"seed structure cache not found: {source_dir}")
            for source_json in source_dir.glob("*.json"):
                source_npy = source_json.with_suffix(".npy")
                if not source_npy.is_file():
                    raise FileNotFoundError(
                        f"seed structure cache is missing bands array: {source_npy}"
                    )
                payload = json.loads(source_json.read_text(encoding="utf-8"))
                if str(payload.get("status")) != "computed" or str(payload.get("error", "")):
                    raise ValueError(f"seed structure cache contains a failed result: {source_json}")
                destination_json = cache_dir / source_json.name
                destination_npy = cache_dir / source_npy.name
                if not destination_json.exists():
                    shutil.copy2(source_json, destination_json)
                if not destination_npy.exists():
                    shutil.copy2(source_npy, destination_npy)
                seeded_cache_pair_count += 1
    replay_methods = (
        set(METHODS)
        if fake_mpb
        else {str(item) for item in replay_cfg.get("methods", METHODS)}
    )
    pilot_positions = {
        int(item)
        for item in _mapping(cfg.get("pilot"), name="pilot").get("target_positions", ())
    }
    for record in manifest["records"]:
        if str(record["method"]) not in replay_methods:
            continue
        if pilot and int(record["target_position"]) not in pilot_positions:
            continue
        if max_targets is not None and int(record["target_position"]) >= max_targets:
            continue
        candidates = np.asarray(record["candidates"], dtype=np.float32)
        for rank_position, candidate_index in enumerate(record["selected_indices"][:max_candidates]):
            rho = np.asarray(candidates[int(candidate_index)], dtype=np.float32).reshape(-1)
            structure_hash = hash_array(rho)
            unique.setdefault(structure_hash, rho)
            references.append(
                {
                    "record_id": record["record_id"],
                    "method": record["method"],
                    "target_index": int(record["target_index"]),
                    "rank_position": int(rank_position),
                    "candidate_index": int(candidate_index),
                    "structure_sha256": structure_hash,
                }
            )
            references_by_method[str(record["method"])] += 1
    per_method_cap = int(_mapping(cfg.get("replay"), name="replay").get("per_method_solve_cap", 500))
    for method, count in references_by_method.items():
        if count > per_method_cap:
            raise ValueError(f"{method} replay references exceed configured cap {per_method_cap}: {count}")
    bundle = load_forward_dataset(_dataset_cfg(cfg)["path"])
    k_path = recover_k_path(bundle)
    solver_config = _solver_config(cfg)
    jobs = [
        MPBWorkerJob(
            sample_id=structure_hash,
            sample_index=position,
            split="candidate",
            rho=rho,
            k_points=np.asarray(k_path.coordinates, dtype=np.float64).tolist(),
            solver_config=solver_config,
            rho_hash=structure_hash,
            geometry_hash=structure_hash,
        )
        for position, (structure_hash, rho) in enumerate(sorted(unique.items()))
    ]
    parallel = MPBParallelConfig(
        workers=int(replay_cfg.get("workers_requested", 8)),
        min_workers=1,
        max_workers=int(replay_cfg.get("workers_requested", 8)),
        worker_backend=str(replay_cfg.get("worker_backend", "isolated_subprocess")),
        fail_fast=bool(replay_cfg.get("fail_fast", True)),
        suppress_solver_output=bool(replay_cfg.get("suppress_solver_output", True)),
    )
    chosen_solver = solver_fn or (_fake_solver if fake_mpb else solve_bands_for_rho_mpb)
    results, summary = run_mpb_worker_jobs(
        jobs,
        config=parallel,
        shard_dir=cache_dir,
        solver_fn=chosen_solver,
    )
    by_hash = {result.rho_hash: result for result in results}
    replay_rows: list[dict[str, Any]] = []
    replay_band_rows: list[dict[str, Any]] = []
    records_by_id = {str(record["record_id"]): record for record in manifest["records"]}
    for reference in references:
        result = by_hash[reference["structure_sha256"]]
        target_record = records_by_id[reference["record_id"]]
        target = np.asarray(target_record["condition"]["Wh"], dtype=np.float32)
        if result.bands is None:
            shard_npy = cache_dir / f"{result.sample_id}.npy"
            if shard_npy.is_file():
                bands = np.asarray(np.load(shard_npy, allow_pickle=False), dtype=np.float32)
            else:
                bands = np.empty((0, 0), dtype=np.float32)
        else:
            bands = np.asarray(result.bands, dtype=np.float32)
        error = float(_bands_mre(bands[None, :, :], target)[0]) if bands.size else None
        replay_rows.append(
            {
                **reference,
                "status": result.status,
                "resumed": bool(result.resumed),
                "mpb_replay_mre": error,
                "mpb_replay_time_s": float(result.solve_time_s),
                "worker_pid": int(result.pid),
                "solver_metadata": result.solve_metadata,
            }
        )
        if bands.size:
            for band_index in range(int(bands.shape[1])):
                band_delta = np.abs(
                    np.asarray(bands[:, band_index], dtype=np.float64)
                    - np.asarray(target[:, band_index], dtype=np.float64)
                )
                band_relative = band_delta / np.maximum(
                    np.abs(np.asarray(target[:, band_index], dtype=np.float64)),
                    1.0e-8,
                )
                replay_band_rows.append(
                    {
                        **reference,
                        "band_one_based": int(band_index + 1),
                        "mpb_replay_mre": float(np.mean(band_relative)),
                        "mpb_replay_mae": float(np.mean(band_delta)),
                    }
                )
    _atomic_json(replay_dir / "replay_references.json", references)
    _write_csv(replay_dir / "replay_candidates.csv", replay_rows)
    _write_csv(replay_dir / "per_candidate_band_metrics.csv", replay_band_rows)
    _atomic_json(replay_dir / "worker_summary.json", summary)
    _atomic_json(
        replay_dir / "metrics.json",
        {
            **summary,
            "reference_count": len(references),
            "unique_structure_count": len(unique),
            "structure_cache_hit_count": len(references) - len(unique),
            "seeded_structure_cache_pair_count": seeded_cache_pair_count,
            "evidence_status": "fake_mpb_smoke"
            if fake_mpb
            else (
                "real_mpb_smoke"
                if smoke
                else ("single_seed_adjustment_pilot_replay" if pilot else _evidence_status(cfg))
            ),
        },
    )
    _atomic_json(
        replay_dir / "results.json",
        {
            "status": "replay_completed" if summary["failed_count"] == 0 else "replay_failed",
            "fake_mpb": bool(fake_mpb),
            "smoke": bool(smoke),
            "pilot": bool(pilot),
        },
    )
    if smoke and not fake_mpb and summary["failed_count"] == 0:
        total_solver_time = float(sum(float(result.solve_time_s) for result in results))
        resolved_workers = max(int(summary["workers_resolved"]), 1)
        preflight_path = run / "preflight_decision.json"
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        preflight.update(
            {
                "status": "ready_to_execute",
                "next_status": "a2_1_training",
                "real_mpb_smoke_pass": True,
                "real_mpb_smoke_unique_structures": len(unique),
                "real_mpb_smoke_solver_time_s_total": total_solver_time,
                "real_mpb_smoke_solver_time_per_candidate_s": total_solver_time / max(len(results), 1),
                "real_mpb_smoke_parallel_wall_lower_bound_s": total_solver_time / resolved_workers,
                "real_mpb_smoke_last_invocation_wall_time_s": float(summary["worker_wall_time_s"]),
                "real_mpb_smoke_last_invocation_resumed_count": int(summary["resumed_count"]),
            }
        )
        _atomic_json(preflight_path, preflight)
    return replay_dir


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _bootstrap_ci(values: Sequence[float], *, seed: int = 0, iterations: int = 10_000) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(arr), size=(iterations, len(arr)))
    means = np.mean(arr[draws], axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _best_replay_by_method_target(path: Path) -> dict[tuple[str, int], float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        if str(row.get("status")) not in {"computed", "completed"} or not row.get("mpb_replay_mre"):
            continue
        key = (str(row["method"]), int(row["target_index"]))
        grouped.setdefault(key, []).append(float(row["mpb_replay_mre"]))
    return {key: min(values) for key, values in grouped.items()}


def run_a2_pilot_aggregate(config_path: str | Path, *, run_dir: str | Path) -> Path:
    """Evaluate the 20-target objective-bridge replay and freeze its promotion decision."""

    cfg = _read_yaml(config_path)
    if not _is_objective_bridge(cfg):
        raise ValueError("pilot aggregate is available only for objective-bridge protocols")
    run = Path(run_dir)
    pilot_cfg = _mapping(cfg.get("pilot"), name="pilot")
    replay_dir = run / "replay_pilot"
    best = _best_replay_by_method_target(replay_dir / "replay_candidates.csv")
    worker_metrics = json.loads((replay_dir / "metrics.json").read_text(encoding="utf-8"))
    baseline_path = Path(str(pilot_cfg["baseline_per_target_metrics_path"]))
    with baseline_path.open("r", encoding="utf-8", newline="") as handle:
        baseline_rows = list(csv.DictReader(handle))
    baseline = {
        (str(row["method"]), int(row["target_index"])): float(row["best_of_k_replay_mre"])
        for row in baseline_rows
    }
    candidate_manifest = json.loads(
        (run / "candidates" / "candidate_manifest.frozen.json").read_text(encoding="utf-8")
    )
    surrogate = {
        (str(record["method"]), int(record["target_index"])): float(
            record["best_of_k_surrogate_mre"]
        )
        for record in candidate_manifest["records"]
    }
    positions = {int(item) for item in pilot_cfg["target_positions"]}
    target_by_position = {
        int(record["target_position"]): int(record["target_index"])
        for record in candidate_manifest["records"]
    }
    target_indices = [target_by_position[position] for position in sorted(positions)]
    promotion_cfg = _mapping(pilot_cfg.get("promotion"), name="pilot.promotion")
    method_rows: list[dict[str, Any]] = []
    for method in [str(item) for item in _mapping(cfg.get("replay"), name="replay")["methods"]]:
        new_values = np.asarray([best[(method, index)] for index in target_indices], dtype=np.float64)
        old_values = np.asarray([baseline[(method, index)] for index in target_indices], dtype=np.float64)
        gap_values = np.asarray(
            [
                abs(best[(method, index)] - surrogate[(method, index)])
                for index in target_indices
            ],
            dtype=np.float64,
        )
        method_gate = _mapping(promotion_cfg[method], name=f"pilot.promotion.{method}")
        relative_improvement = float(1.0 - float(np.mean(new_values)) / float(np.mean(old_values)))
        wins = int(np.sum(new_values < old_values))
        mean_gap = float(np.mean(gap_values))
        failures = []
        if relative_improvement < float(method_gate["relative_mean_improvement_min"]):
            failures.append("relative_mean_improvement_below_gate")
        if wins < int(method_gate["wins_min"]):
            failures.append("paired_win_count_below_gate")
        if mean_gap > float(method_gate["surrogate_replay_gap_max"]):
            failures.append("surrogate_replay_gap_above_gate")
        method_rows.append(
            {
                "method": method,
                "target_count": len(target_indices),
                "new_replay_mean": float(np.mean(new_values)),
                "baseline_replay_mean": float(np.mean(old_values)),
                "relative_mean_improvement": relative_improvement,
                "paired_wins": wins,
                "surrogate_replay_gap_mean": mean_gap,
                "pass": not failures,
                "failures": failures,
            }
        )
    mixture_pass = next(
        bool(row["pass"]) for row in method_rows if row["method"] == "mixture_generator"
    )
    solver_failure_count = int(worker_metrics.get("failed_count", 0))
    decision = {
        "status": "promote_full_100"
        if mixture_pass and solver_failure_count == 0
        else "pilot_completed_no_promotion",
        "mixture_promotion_pass": mixture_pass,
        "solver_failure_count": solver_failure_count,
        "target_indices": target_indices,
        "method_rows": method_rows,
        "evidence_status": "single_seed_adjustment_pilot_replay",
    }
    out = run / "pilot"
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "promotion_decision.json", decision)
    _write_csv(out / "method_metrics.csv", method_rows)
    _atomic_json(out / "metrics.json", decision)
    _atomic_json(out / "results.json", {"status": decision["status"]})
    report = [
        "# A2 objective bridge 20-target pilot",
        "",
        f"- status: `{decision['status']}`",
        f"- solver failures: `{decision['solver_failure_count']}`",
        "",
        "| method | new mean | baseline mean | relative improvement | wins | replay gap | pass |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in method_rows:
        report.append(
            f"| {row['method']} | {row['new_replay_mean'] * 100:.3f}% | "
            f"{row['baseline_replay_mean'] * 100:.3f}% | "
            f"{row['relative_mean_improvement'] * 100:.1f}% | {row['paired_wins']}/"
            f"{row['target_count']} | {row['surrogate_replay_gap_mean'] * 100:.3f} pp | "
            f"{row['pass']} |"
        )
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return out


def _sign_test_paired(left: Sequence[float], right: Sequence[float]) -> float:
    delta = np.asarray(left) - np.asarray(right)
    positive = int(np.sum(delta > 0))
    negative = int(np.sum(delta < 0))
    n = positive + negative
    if n == 0:
        return 1.0
    tail = min(positive, negative)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / float(2**n)
    return min(1.0, 2.0 * probability)


def _holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for position, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - position) * float(value)))
        adjusted[name] = running
    return adjusted


def _run_a2_objective_bridge_test625_aggregate(
    cfg: Mapping[str, Any],
    *,
    run: Path,
) -> Path:
    """Aggregate the evaluation-only bridge replay against the historical test625 mixture."""

    aggregate_dir = run / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    replay_path = run / "replay" / "replay_candidates.csv"
    with replay_path.open("r", encoding="utf-8", newline="") as handle:
        replay_rows = list(csv.DictReader(handle))
    if any(str(row.get("status")) not in {"computed", "completed"} for row in replay_rows):
        raise ValueError("test625 bridge aggregation requires every replay candidate to be computed")
    best = _best_replay_by_method_target(replay_path)
    manifest = json.loads(
        (run / "candidates" / "candidate_manifest.frozen.json").read_text(encoding="utf-8")
    )
    records = {
        (str(record["method"]), int(record["target_index"])): record
        for record in manifest["records"]
    }
    replay_methods = [str(item) for item in _mapping(cfg.get("replay"), name="replay")["methods"]]
    expected_targets = int(_dataset_cfg(cfg)["expected_target_count"])
    target_indices = sorted(
        {
            int(record["target_index"])
            for record in manifest["records"]
            if str(record["method"]) == replay_methods[0]
        }
    )
    if len(target_indices) != expected_targets:
        raise ValueError(
            f"objective-bridge test625 aggregate has {len(target_indices)} paired targets; "
            f"expected {expected_targets}"
        )

    evaluation_cfg = _mapping(cfg.get("evaluation"), name="evaluation")
    historical_path = Path(str(evaluation_cfg["historical_baseline_per_target_path"]))
    _require_hash(
        historical_path,
        str(evaluation_cfg["historical_baseline_per_target_sha256"]),
        label="historical test625 baseline",
    )
    with historical_path.open("r", encoding="utf-8", newline="") as handle:
        historical_rows = list(csv.DictReader(handle))
    historical = {
        int(row["target_index"]): float(row["best_of_k_replay_mre_bands_2_10_pct"]) / 100.0
        for row in historical_rows
    }
    historical_surrogate = {
        int(row["target_index"]): float(row["best_of_k_surrogate_mre_bands_2_10_pct"]) / 100.0
        for row in historical_rows
    }
    if set(target_indices) != set(historical):
        raise ValueError("historical test625 baseline target set does not match current manifest")

    per_target: list[dict[str, Any]] = []
    per_method: dict[str, dict[int, float]] = {}
    method_rows: list[dict[str, Any]] = []
    historical_values = [historical[index] for index in target_indices]
    historical_ci_low, historical_ci_high = _bootstrap_ci(historical_values, seed=0)
    method_rows.append(
        {
            "method": "historical_objective_mixture",
            "source": "inverse_mixture_generator_v2_10000_test625_seed2",
            "selected_learning_rate": 0.0001,
            "selected_epoch": "",
            "replay_mean": float(np.mean(historical_values)),
            "replay_mean_pct": float(np.mean(historical_values) * 100.0),
            "replay_median": float(np.median(historical_values)),
            "replay_median_pct": float(np.median(historical_values) * 100.0),
            "replay_p10": _percentile(historical_values, 10),
            "replay_p10_pct": _percentile(historical_values, 10) * 100.0,
            "replay_p90": _percentile(historical_values, 90),
            "replay_p90_pct": _percentile(historical_values, 90) * 100.0,
            "paired_bootstrap_95_ci_low": historical_ci_low,
            "paired_bootstrap_95_ci_high": historical_ci_high,
            "paired_bootstrap_95_ci_low_pct": historical_ci_low * 100.0,
            "paired_bootstrap_95_ci_high_pct": historical_ci_high * 100.0,
            "surrogate_mre_mean": float(
                np.mean([historical_surrogate[index] for index in target_indices])
            ),
            "surrogate_mre_mean_pct": float(
                np.mean([historical_surrogate[index] for index in target_indices]) * 100.0
            ),
            "surrogate_replay_gap_mean": float(
                np.mean(
                    [
                        historical[index] - historical_surrogate[index]
                        for index in target_indices
                    ]
                )
            ),
            "surrogate_replay_gap_mean_pct": float(
                np.mean(
                    [
                        historical[index] - historical_surrogate[index]
                        for index in target_indices
                    ]
                )
                * 100.0
            ),
            "effective_k_mean": float(evaluation_cfg.get("historical_effective_k_mean", 0.0)),
            "effective_m_mean": int(_protocol(cfg)["replay_top_m"]),
        }
    )

    selected_run = _selected_training_run(cfg, run=run)
    for method in replay_methods:
        method_target: dict[int, float] = {}
        for target_index in target_indices:
            key = (method, target_index)
            if key not in best:
                raise ValueError(f"missing bridge replay result for {method} target {target_index}")
            record = records[key]
            value = float(best[key])
            method_target[target_index] = value
            per_target.append(
                {
                    "method": method,
                    "target_index": target_index,
                    "best_of_k_replay_mre": value,
                    "best_of_k_surrogate_mre": float(record["best_of_k_surrogate_mre"]),
                    "surrogate_replay_gap": value - float(record["best_of_k_surrogate_mre"]),
                    "best_of_k_diagnostic_query_mre": float(
                        record["best_of_k_diagnostic_query_mre"]
                    ),
                    "best_of_k_cross_surrogate_gap": float(
                        record["best_of_k_cross_surrogate_gap"]
                    ),
                    "effective_k": int(record["effective_unique_k"]),
                    "effective_m": min(
                        int(_protocol(cfg)["replay_top_m"]),
                        int(record["effective_unique_k"]),
                    ),
                    "continuous_candidate_std": float(record["continuous_candidate_std"]),
                    "threshold_margin_fraction": float(record["threshold_margin_fraction"]),
                    "candidate_sampling_time_s": float(record["sampling_time_s"]),
                    "surrogate_ranking_time_s": float(record["ranking_time_s"]),
                }
            )
        per_method[method] = method_target
        rows = [row for row in per_target if row["method"] == method]
        values = [float(row["best_of_k_replay_mre"]) for row in rows]
        ci_low, ci_high = _bootstrap_ci(values, seed=0)
        selection = json.loads(
            (
                selected_run / "train" / method / "selected_checkpoint.json"
            ).read_text(encoding="utf-8")
        )
        method_rows.append(
            {
                "method": method,
                "source": "objective_bridge_seed0_v1_frozen_test625",
                "selected_learning_rate": float(selection["learning_rate"]),
                "selected_epoch": int(selection["epoch"]),
                "replay_mean": float(np.mean(values)),
                "replay_mean_pct": float(np.mean(values) * 100.0),
                "replay_median": float(np.median(values)),
                "replay_median_pct": float(np.median(values) * 100.0),
                "replay_p10": _percentile(values, 10),
                "replay_p10_pct": _percentile(values, 10) * 100.0,
                "replay_p90": _percentile(values, 90),
                "replay_p90_pct": _percentile(values, 90) * 100.0,
                "paired_bootstrap_95_ci_low": ci_low,
                "paired_bootstrap_95_ci_high": ci_high,
                "paired_bootstrap_95_ci_low_pct": ci_low * 100.0,
                "paired_bootstrap_95_ci_high_pct": ci_high * 100.0,
                "surrogate_mre_mean": float(
                    np.mean([float(row["best_of_k_surrogate_mre"]) for row in rows])
                ),
                "surrogate_mre_mean_pct": float(
                    np.mean([float(row["best_of_k_surrogate_mre"]) for row in rows]) * 100.0
                ),
                "surrogate_replay_gap_mean": float(
                    np.mean([float(row["surrogate_replay_gap"]) for row in rows])
                ),
                "surrogate_replay_gap_mean_pct": float(
                    np.mean([float(row["surrogate_replay_gap"]) for row in rows]) * 100.0
                ),
                "diagnostic_query_mre_mean_pct": float(
                    np.mean([float(row["best_of_k_diagnostic_query_mre"]) for row in rows])
                    * 100.0
                ),
                "cross_surrogate_gap_mean_pct": float(
                    np.mean([float(row["best_of_k_cross_surrogate_gap"]) for row in rows])
                    * 100.0
                ),
                "effective_k_mean": float(
                    np.mean([float(row["effective_k"]) for row in rows])
                ),
                "effective_m_mean": float(
                    np.mean([float(row["effective_m"]) for row in rows])
                ),
                "continuous_candidate_std_mean": float(
                    np.mean([float(row["continuous_candidate_std"]) for row in rows])
                ),
                "threshold_margin_fraction_mean": float(
                    np.mean([float(row["threshold_margin_fraction"]) for row in rows])
                ),
                "parameter_count": int(selection["parameter_count"]),
                "candidate_sampling_time_s": float(
                    sum(float(row["candidate_sampling_time_s"]) for row in rows)
                ),
                "surrogate_ranking_time_s": float(
                    sum(float(row["surrogate_ranking_time_s"]) for row in rows)
                ),
            }
        )

    comparisons: list[dict[str, Any]] = []
    historical_array = np.asarray(historical_values, dtype=np.float64)
    for method in replay_methods:
        current = np.asarray(
            [per_method[method][index] for index in target_indices],
            dtype=np.float64,
        )
        delta = current - historical_array
        ci_low, ci_high = _bootstrap_ci(delta, seed=0)
        comparisons.append(
            {
                "comparison": f"{method}_bridge_seed0_vs_historical_mixture_seed2",
                "target_count": expected_targets,
                "bridge_mean": float(np.mean(current)),
                "historical_mean": float(np.mean(historical_array)),
                "relative_mean_improvement": float(
                    1.0 - float(np.mean(current)) / float(np.mean(historical_array))
                ),
                "paired_wins": int(np.sum(current < historical_array)),
                "paired_mean_delta": float(np.mean(delta)),
                "paired_bootstrap_95_ci_low": ci_low,
                "paired_bootstrap_95_ci_high": ci_high,
                "sign_test_p": _sign_test_paired(current, historical_array),
            }
        )
    single = np.asarray(
        [per_method["single_mode_generator"][index] for index in target_indices],
        dtype=np.float64,
    )
    mixture = np.asarray(
        [per_method["mixture_generator"][index] for index in target_indices],
        dtype=np.float64,
    )
    delta = mixture - single
    ci_low, ci_high = _bootstrap_ci(delta, seed=0)
    comparisons.append(
        {
            "comparison": "bridge_mixture_vs_single_mode_generator",
            "target_count": expected_targets,
            "paired_mean_delta": float(np.mean(delta)),
            "paired_bootstrap_95_ci_low": ci_low,
            "paired_bootstrap_95_ci_high": ci_high,
            "sign_test_p": _sign_test_paired(mixture, single),
        }
    )

    _write_csv(aggregate_dir / "main_table.csv", method_rows)
    _write_csv(aggregate_dir / "per_target_metrics.csv", per_target)
    _write_csv(aggregate_dir / "paired_comparisons.csv", comparisons)
    metrics = {
        "status": "adjustment_test625_completed",
        "evidence_status": _evidence_status(cfg),
        "seed": 0,
        "target_count": expected_targets,
        "evaluation_scope": "complete_test_split",
        "training_reused_without_update": True,
        "historical_baseline_protocol_matched": False,
        "cross_seed_stability_claim_authorized": False,
        "pino_or_physics_rerank_included": False,
        "method_rows": method_rows,
        "comparisons": comparisons,
    }
    _atomic_json(aggregate_dir / "metrics.json", metrics)
    _atomic_json(aggregate_dir / "results.json", metrics)
    report = [
        "# A2 objective-bridge seed0 test625 replay",
        "",
        "- status: `adjustment_test625_completed`",
        f"- evidence_status: `{_evidence_status(cfg)}`",
        "- 当前 bridge checkpoint、A1 ranker 与候选规则均冻结，未重新训练。",
        "- 历史 mixture 使用 seed2 与旧 ranker；相同 625 targets 支持性能配对比较，"
        "但不能把差异归因于单一因素。",
        "",
        "| method | source | replay mean (%) | median (%) | p10/p90 (%) | effective K/M |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        report.append(
            f"| {row['method']} | {row.get('source', '-')} | "
            f"{float(row['replay_mean_pct']):.3f} | "
            f"{float(row['replay_median_pct']):.3f} | "
            f"{float(row['replay_p10_pct']):.3f}/{float(row['replay_p90_pct']):.3f} | "
            f"{float(row.get('effective_k_mean', 0.0)):.2f}/"
            f"{float(row.get('effective_m_mean', 0.0)):.2f} |"
        )
    report.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "| comparison | mean delta | bootstrap 95% CI | sign p |",
            "|---|---:|---|---:|",
        ]
    )
    for row in comparisons:
        report.append(
            f"| {row['comparison']} | {float(row['paired_mean_delta']):.6f} | "
            f"[{float(row['paired_bootstrap_95_ci_low']):.6f}, "
            f"{float(row['paired_bootstrap_95_ci_high']):.6f}] | "
            f"{float(row['sign_test_p']):.6g} |"
        )
    (aggregate_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return aggregate_dir


def _run_a2_objective_bridge_aggregate(
    cfg: Mapping[str, Any],
    *,
    run: Path,
) -> Path:
    """Aggregate the promoted 100-target bridge replay without rewriting frozen baselines."""

    aggregate_dir = run / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    replay_path = run / "replay" / "replay_candidates.csv"
    with replay_path.open("r", encoding="utf-8", newline="") as handle:
        replay_rows = list(csv.DictReader(handle))
    if any(str(row.get("status")) not in {"computed", "completed"} for row in replay_rows):
        raise ValueError("objective-bridge aggregation requires every replay candidate to be computed")
    best = _best_replay_by_method_target(replay_path)
    manifest = json.loads(
        (run / "candidates" / "candidate_manifest.frozen.json").read_text(encoding="utf-8")
    )
    records = {
        (str(record["method"]), int(record["target_index"])): record
        for record in manifest["records"]
    }
    replay_methods = [str(item) for item in _mapping(cfg.get("replay"), name="replay")["methods"]]
    expected_targets = int(_dataset_cfg(cfg)["expected_target_count"])
    target_indices = sorted(
        {
            int(record["target_index"])
            for record in manifest["records"]
            if str(record["method"]) == replay_methods[0]
        }
    )
    if len(target_indices) != expected_targets:
        raise ValueError(
            f"objective-bridge aggregate has {len(target_indices)} paired targets; expected {expected_targets}"
        )

    pilot_cfg = _mapping(cfg.get("pilot"), name="pilot")
    baseline_per_target_path = Path(str(pilot_cfg["baseline_per_target_metrics_path"]))
    baseline_main_table_path = Path(str(pilot_cfg["baseline_main_table_path"]))
    with baseline_per_target_path.open("r", encoding="utf-8", newline="") as handle:
        baseline_target_rows = list(csv.DictReader(handle))
    with baseline_main_table_path.open("r", encoding="utf-8", newline="") as handle:
        baseline_main_rows = list(csv.DictReader(handle))
    baseline_target = {
        (str(row["method"]), int(row["target_index"])): float(row["best_of_k_replay_mre"])
        for row in baseline_target_rows
    }
    baseline_main = {str(row["method"]): dict(row) for row in baseline_main_rows}

    per_target: list[dict[str, Any]] = []
    per_method: dict[str, dict[int, float]] = {}
    method_rows: list[dict[str, Any]] = []
    for frozen_method in ("retrieval", "direct_inverse"):
        row = dict(baseline_main[frozen_method])
        row["source"] = "frozen_a2_single_seed_v1"
        method_rows.append(row)
        per_method[frozen_method] = {
            int(target_index): float(value)
            for (method, target_index), value in baseline_target.items()
            if method == frozen_method
        }

    for method in replay_methods:
        method_target: dict[int, float] = {}
        for target_index in target_indices:
            key = (method, target_index)
            if key not in best:
                raise ValueError(f"missing bridge replay result for {method} target {target_index}")
            record = records[key]
            value = float(best[key])
            method_target[target_index] = value
            per_target.append(
                {
                    "method": method,
                    "target_index": target_index,
                    "best_of_k_replay_mre": value,
                    "best_of_k_surrogate_mre": float(record["best_of_k_surrogate_mre"]),
                    "surrogate_replay_gap": value - float(record["best_of_k_surrogate_mre"]),
                    "best_of_k_diagnostic_query_mre": float(
                        record["best_of_k_diagnostic_query_mre"]
                    ),
                    "best_of_k_cross_surrogate_gap": float(
                        record["best_of_k_cross_surrogate_gap"]
                    ),
                    "effective_k": int(record["effective_unique_k"]),
                    "effective_m": min(
                        int(_protocol(cfg)["replay_top_m"]),
                        int(record["effective_unique_k"]),
                    ),
                    "continuous_candidate_std": float(record["continuous_candidate_std"]),
                    "threshold_margin_fraction": float(record["threshold_margin_fraction"]),
                    "candidate_sampling_time_s": float(record["sampling_time_s"]),
                    "surrogate_ranking_time_s": float(record["ranking_time_s"]),
                }
            )
        per_method[method] = method_target
        rows = [row for row in per_target if row["method"] == method]
        values = [float(row["best_of_k_replay_mre"]) for row in rows]
        ci_low, ci_high = _bootstrap_ci(values, seed=0)
        selection = json.loads(
            (run / "train" / method / "selected_checkpoint.json").read_text(encoding="utf-8")
        )
        method_rows.append(
            {
                "method": method,
                "source": "objective_bridge_seed0_v1",
                "selected_learning_rate": float(selection["learning_rate"]),
                "selected_epoch": int(selection["epoch"]),
                "replay_mean": float(np.mean(values)),
                "replay_mean_pct": float(np.mean(values) * 100.0),
                "replay_median": float(np.median(values)),
                "replay_median_pct": float(np.median(values) * 100.0),
                "replay_p10": _percentile(values, 10),
                "replay_p10_pct": _percentile(values, 10) * 100.0,
                "replay_p90": _percentile(values, 90),
                "replay_p90_pct": _percentile(values, 90) * 100.0,
                "paired_bootstrap_95_ci_low": ci_low,
                "paired_bootstrap_95_ci_high": ci_high,
                "paired_bootstrap_95_ci_low_pct": ci_low * 100.0,
                "paired_bootstrap_95_ci_high_pct": ci_high * 100.0,
                "surrogate_mre_mean": float(
                    np.mean([float(row["best_of_k_surrogate_mre"]) for row in rows])
                ),
                "surrogate_mre_mean_pct": float(
                    np.mean([float(row["best_of_k_surrogate_mre"]) for row in rows]) * 100.0
                ),
                "surrogate_replay_gap_mean": float(
                    np.mean([float(row["surrogate_replay_gap"]) for row in rows])
                ),
                "surrogate_replay_gap_mean_pct": float(
                    np.mean([float(row["surrogate_replay_gap"]) for row in rows]) * 100.0
                ),
                "diagnostic_query_mre_mean_pct": float(
                    np.mean([float(row["best_of_k_diagnostic_query_mre"]) for row in rows])
                    * 100.0
                ),
                "cross_surrogate_gap_mean_pct": float(
                    np.mean([float(row["best_of_k_cross_surrogate_gap"]) for row in rows])
                    * 100.0
                ),
                "effective_k_mean": float(
                    np.mean([float(row["effective_k"]) for row in rows])
                ),
                "effective_m_mean": float(
                    np.mean([float(row["effective_m"]) for row in rows])
                ),
                "continuous_candidate_std_mean": float(
                    np.mean([float(row["continuous_candidate_std"]) for row in rows])
                ),
                "threshold_margin_fraction_mean": float(
                    np.mean([float(row["threshold_margin_fraction"]) for row in rows])
                ),
                "parameter_count": int(selection["parameter_count"]),
                "candidate_sampling_time_s": float(
                    sum(float(row["candidate_sampling_time_s"]) for row in rows)
                ),
                "surrogate_ranking_time_s": float(
                    sum(float(row["surrogate_ranking_time_s"]) for row in rows)
                ),
            }
        )

    comparisons: list[dict[str, Any]] = []
    for method in replay_methods:
        new_values = np.asarray(
            [per_method[method][index] for index in target_indices],
            dtype=np.float64,
        )
        old_values = np.asarray(
            [baseline_target[(method, index)] for index in target_indices],
            dtype=np.float64,
        )
        delta = new_values - old_values
        ci_low, ci_high = _bootstrap_ci(delta, seed=0)
        comparisons.append(
            {
                "comparison": f"{method}_bridge_vs_original_a2",
                "target_count": expected_targets,
                "bridge_mean": float(np.mean(new_values)),
                "original_a2_mean": float(np.mean(old_values)),
                "relative_mean_improvement": float(
                    1.0 - float(np.mean(new_values)) / float(np.mean(old_values))
                ),
                "paired_wins": int(np.sum(new_values < old_values)),
                "paired_mean_delta": float(np.mean(delta)),
                "paired_bootstrap_95_ci_low": ci_low,
                "paired_bootstrap_95_ci_high": ci_high,
                "sign_test_p": _sign_test_paired(new_values, old_values),
            }
        )
    mixture = per_method["mixture_generator"]
    for baseline_method in ("retrieval", "direct_inverse", "single_mode_generator"):
        left = np.asarray([mixture[index] for index in target_indices], dtype=np.float64)
        right = np.asarray(
            [
                (
                    per_method["single_mode_generator"][index]
                    if baseline_method == "single_mode_generator"
                    else per_method[baseline_method][index]
                )
                for index in target_indices
            ],
            dtype=np.float64,
        )
        delta = left - right
        ci_low, ci_high = _bootstrap_ci(delta, seed=0)
        comparisons.append(
            {
                "comparison": f"bridge_mixture_vs_{baseline_method}",
                "target_count": expected_targets,
                "paired_mean_delta": float(np.mean(delta)),
                "paired_bootstrap_95_ci_low": ci_low,
                "paired_bootstrap_95_ci_high": ci_high,
                "sign_test_p": _sign_test_paired(left, right),
            }
        )

    _write_csv(aggregate_dir / "main_table.csv", method_rows)
    _write_csv(aggregate_dir / "per_target_metrics.csv", per_target)
    _write_csv(aggregate_dir / "paired_comparisons.csv", comparisons)
    metrics = {
        "status": "adjustment_pilot_full_100_completed",
        "evidence_status": _evidence_status(cfg),
        "seed": 0,
        "target_count": expected_targets,
        "original_a1_a2_history_preserved": True,
        "cross_seed_stability_claim_authorized": False,
        "pino_or_physics_rerank_included": False,
        "method_rows": method_rows,
        "comparisons": comparisons,
    }
    _atomic_json(aggregate_dir / "metrics.json", metrics)
    _atomic_json(aggregate_dir / "results.json", metrics)
    report = [
        "# A2 objective-bridge seed0 full replay",
        "",
        "- status: `adjustment_pilot_full_100_completed`",
        f"- evidence_status: `{_evidence_status(cfg)}`",
        "- A1 checkpoints、retrieval 和 direct 均冻结；这里只替换 single/mixture。",
        "- 这是单 seed 调整证据，不提供跨 seed 稳定性结论。",
        "",
        "| method | source | replay mean (%) | median (%) | p10/p90 (%) | effective K/M |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        report.append(
            f"| {row['method']} | {row.get('source', '-')} | "
            f"{float(row['replay_mean_pct']):.3f} | "
            f"{float(row['replay_median_pct']):.3f} | "
            f"{float(row['replay_p10_pct']):.3f}/{float(row['replay_p90_pct']):.3f} | "
            f"{float(row['effective_k_mean']):.2f}/{float(row['effective_m_mean']):.2f} |"
        )
    report.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "| comparison | mean delta | bootstrap 95% CI | sign p |",
            "|---|---:|---|---:|",
        ]
    )
    for row in comparisons:
        report.append(
            f"| {row['comparison']} | {float(row['paired_mean_delta']):.6f} | "
            f"[{float(row['paired_bootstrap_95_ci_low']):.6f}, "
            f"{float(row['paired_bootstrap_95_ci_high']):.6f}] | "
            f"{float(row['sign_test_p']):.6g} |"
        )
    (aggregate_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return aggregate_dir


def run_a2_aggregate(config_path: str | Path, *, run_dir: str | Path) -> Path:
    """Aggregate paired 100-target replay metrics and write the formal A2 report."""

    cfg = _read_yaml(config_path)
    run = Path(run_dir)
    if _is_objective_bridge_test625(cfg):
        return _run_a2_objective_bridge_test625_aggregate(cfg, run=run)
    if _is_objective_bridge(cfg):
        return _run_a2_objective_bridge_aggregate(cfg, run=run)
    manifest = json.loads((run / "candidates" / "candidate_manifest.frozen.json").read_text(encoding="utf-8"))
    replay_rows = list(csv.DictReader((run / "replay" / "replay_candidates.csv").open(encoding="utf-8")))
    replay_band_rows = list(
        csv.DictReader((run / "replay" / "per_candidate_band_metrics.csv").open(encoding="utf-8"))
    )
    if any(row["status"] != "computed" for row in replay_rows):
        raise ValueError("formal aggregation requires every replay candidate to be computed")
    by_record: dict[str, list[dict[str, str]]] = {}
    for row in replay_rows:
        by_record.setdefault(row["record_id"], []).append(row)
    candidate_records = {record["record_id"]: record for record in manifest["records"]}
    per_target: list[dict[str, Any]] = []
    for record_id, rows in by_record.items():
        record = candidate_records[record_id]
        errors = [float(row["mpb_replay_mre"]) for row in rows]
        best_pos = int(np.argmin(errors))
        per_target.append(
            {
                "method": record["method"],
                "target_index": int(record["target_index"]),
                "best_of_k_replay_mre": float(errors[best_pos]),
                "best_structure_sha256": str(rows[best_pos]["structure_sha256"]),
                "best_of_k_surrogate_mre": float(record["best_of_k_surrogate_mre"]),
                "surrogate_replay_gap": float(errors[best_pos] - record["best_of_k_surrogate_mre"]),
                "best_of_k_diagnostic_query_mre": float(record["best_of_k_diagnostic_query_mre"]),
                "best_of_k_cross_surrogate_gap": float(record["best_of_k_cross_surrogate_gap"]),
                "effective_k": int(record["effective_unique_k"]),
                "effective_m": len(rows),
                "candidate_sampling_time_s": float(record["sampling_time_s"]),
                "surrogate_ranking_time_s": float(record["ranking_time_s"]),
                "mpb_replay_time_s": float(sum(float(row["mpb_replay_time_s"]) for row in rows)),
            }
        )
    expected_targets = int(_dataset_cfg(cfg)["expected_target_count"])
    for method in METHODS:
        count = sum(row["method"] == method for row in per_target)
        if count != expected_targets:
            raise ValueError(f"{method} has {count} paired targets; expected {expected_targets}")

    method_rows: list[dict[str, Any]] = []
    per_method: dict[str, dict[int, float]] = {}
    for method in METHODS:
        rows = sorted((row for row in per_target if row["method"] == method), key=lambda row: row["target_index"])
        values = [float(row["best_of_k_replay_mre"]) for row in rows]
        ci_low, ci_high = _bootstrap_ci(values, seed=0)
        per_method[method] = {int(row["target_index"]): float(row["best_of_k_replay_mre"]) for row in rows}
        selection = None
        if method in TRAINABLE_METHODS:
            selection = json.loads((run / "train" / method / "selected_checkpoint.json").read_text(encoding="utf-8"))
        method_cfg = _mapping(_mapping(cfg.get("methods"))[method])
        method_rows.append(
            {
                "method": method,
                "selected_learning_rate": None if selection is None else float(selection["learning_rate"]),
                "selected_epoch": None if selection is None else int(selection["epoch"]),
                "replay_mean": float(np.mean(values)),
                "replay_mean_pct": float(np.mean(values) * 100.0),
                "replay_median": float(np.median(values)),
                "replay_median_pct": float(np.median(values) * 100.0),
                "replay_p10": _percentile(values, 10),
                "replay_p10_pct": _percentile(values, 10) * 100.0,
                "replay_p90": _percentile(values, 90),
                "replay_p90_pct": _percentile(values, 90) * 100.0,
                "paired_bootstrap_95_ci_low": ci_low,
                "paired_bootstrap_95_ci_high": ci_high,
                "paired_bootstrap_95_ci_low_pct": ci_low * 100.0,
                "paired_bootstrap_95_ci_high_pct": ci_high * 100.0,
                "surrogate_mre_mean": float(np.mean([row["best_of_k_surrogate_mre"] for row in rows])),
                "surrogate_mre_mean_pct": float(
                    np.mean([row["best_of_k_surrogate_mre"] for row in rows]) * 100.0
                ),
                "surrogate_replay_gap_mean": float(np.mean([row["surrogate_replay_gap"] for row in rows])),
                "surrogate_replay_gap_mean_pct": float(
                    np.mean([row["surrogate_replay_gap"] for row in rows]) * 100.0
                ),
                "diagnostic_query_mre_mean_pct": float(
                    np.mean([row["best_of_k_diagnostic_query_mre"] for row in rows]) * 100.0
                ),
                "cross_surrogate_gap_mean_pct": float(
                    np.mean([row["best_of_k_cross_surrogate_gap"] for row in rows]) * 100.0
                ),
                "effective_k_mean": float(np.mean([row["effective_k"] for row in rows])),
                "effective_m_mean": float(np.mean([row["effective_m"] for row in rows])),
                "parameter_count": method_cfg.get("parameter_count")
                if selection is None
                else int(selection["parameter_count"]),
                "candidate_sampling_time_s": float(sum(row["candidate_sampling_time_s"] for row in rows)),
                "surrogate_ranking_time_s": float(sum(row["surrogate_ranking_time_s"] for row in rows)),
                "mpb_replay_time_s": float(sum(row["mpb_replay_time_s"] for row in rows)),
            }
        )

    comparisons: list[dict[str, Any]] = []
    mixture_p: dict[str, float] = {}
    mixture = per_method["mixture_generator"]
    target_order = sorted(mixture)
    for other in METHODS[:-1]:
        left = [mixture[index] for index in target_order]
        right = [per_method[other][index] for index in target_order]
        p = _sign_test_paired(left, right)
        mixture_p[other] = p
        delta = np.asarray(left) - np.asarray(right)
        ci_low, ci_high = _bootstrap_ci(delta, seed=0)
        comparisons.append(
            {
                "comparison": f"mixture_generator_vs_{other}",
                "paired_mean_delta": float(np.mean(delta)),
                "paired_mean_delta_pct": float(np.mean(delta) * 100.0),
                "paired_bootstrap_95_ci_low": ci_low,
                "paired_bootstrap_95_ci_high": ci_high,
                "paired_bootstrap_95_ci_low_pct": ci_low * 100.0,
                "paired_bootstrap_95_ci_high_pct": ci_high * 100.0,
                "sign_test_p": p,
            }
        )
    adjusted = _holm_adjust(mixture_p)
    for row in comparisons:
        other = str(row["comparison"]).replace("mixture_generator_vs_", "")
        row["holm_adjusted_p"] = adjusted[other]

    best_keys = {
        (str(row["method"]), int(row["target_index"]), str(row["best_structure_sha256"]))
        for row in per_target
    }
    selected_band_rows = [
        row
        for row in replay_band_rows
        if (str(row["method"]), int(row["target_index"]), str(row["structure_sha256"])) in best_keys
    ]
    per_band: list[dict[str, Any]] = []
    for method in METHODS:
        for band_one_based in range(1, int(_solver_config(cfg).num_bands) + 1):
            values = [
                float(row["mpb_replay_mre"])
                for row in selected_band_rows
                if row["method"] == method and int(row["band_one_based"]) == band_one_based
            ]
            if values:
                per_band.append(
                    {
                        "method": method,
                        "band_one_based": band_one_based,
                        "target_count": len(values),
                        "mpb_replay_mre_mean": float(np.mean(values)),
                        "mpb_replay_mre_median": float(np.median(values)),
                    }
                )

    _write_csv(run / "aggregate" / "main_table.csv", method_rows)
    _write_csv(run / "aggregate" / "per_target_metrics.csv", per_target)
    _write_csv(run / "aggregate" / "per_band_metrics.csv", per_band)
    _write_csv(run / "aggregate" / "paired_comparisons.csv", comparisons)
    metrics = {
        "status": "formal_completed",
        "evidence_status": _evidence_status(cfg),
        "seed": 0,
        "target_count": expected_targets,
        "cross_seed_stability_claim_authorized": False,
        "pino_or_physics_rerank_included": False,
        "mre_storage_unit": "fraction_with_explicit_pct_companion_fields",
        "method_rows": method_rows,
        "comparisons": comparisons,
    }
    _atomic_json(run / "aggregate" / "metrics.json", metrics)
    _atomic_json(run / "aggregate" / "results.json", metrics)
    report_lines = [
        "<!--",
        "  A2 single-seed fair inverse benchmark report.",
        "  Author: fangtoast",
        "  Email: fangtoast@foxmail.com",
        "  Copyright (c) 2026 fangtoast. All rights reserved.",
        "-->",
        "",
        "# A2 单 Seed 逆向设计公平对照",
        "",
        f"- 状态：`formal_completed`",
        f"- 证据：`{_evidence_status(cfg)}`",
        "- 固定 seed：`0`；固定 paired test targets：`100`。",
        "- 本报告不提供跨 seed 稳定性证据，也不纳入 PINO/field residual/physics-aware rerank。",
        "",
        "## 主表",
        "",
        "| method | selected LR | epoch | replay mean (%) | median | p10 | p90 | bootstrap 95% CI (%) | effective K/M |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in method_rows:
        report_lines.append(
            f"| {row['method']} | {row['selected_learning_rate'] or '-'} | {row['selected_epoch'] or '-'} | "
            f"{row['replay_mean_pct']:.3f} | {row['replay_median_pct']:.3f} | "
            f"{row['replay_p10_pct']:.3f} | {row['replay_p90_pct']:.3f} | "
            f"[{row['paired_bootstrap_95_ci_low_pct']:.3f}, "
            f"{row['paired_bootstrap_95_ci_high_pct']:.3f}] | {row['effective_k_mean']:.2f}/"
            f"{row['effective_m_mean']:.2f} |"
        )
    report_lines.extend(
        [
            "",
            "## Surrogate、规模与运行时间",
            "",
            "| method | ranker MRE (%) | query diagnostic MRE (%) | surrogate–replay gap (pp) | params | sample (s) | rank (s) | solver sum (h) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in method_rows:
        report_lines.append(
            f"| {row['method']} | {row['surrogate_mre_mean_pct']:.3f} | "
            f"{row['diagnostic_query_mre_mean_pct']:.3f} | "
            f"{row['surrogate_replay_gap_mean_pct']:.3f} | {row['parameter_count']} | "
            f"{row['candidate_sampling_time_s']:.3f} | {row['surrogate_ranking_time_s']:.3f} | "
            f"{row['mpb_replay_time_s'] / 3600.0:.3f} |"
        )
    comparison_by_name = {str(row["comparison"]): row for row in comparisons}
    vs_retrieval = comparison_by_name["mixture_generator_vs_retrieval"]
    vs_direct = comparison_by_name["mixture_generator_vs_direct_inverse"]
    vs_single = comparison_by_name["mixture_generator_vs_single_mode_generator"]
    report_lines.extend(
        [
            "",
            "## Paired 方法比较",
            "",
            "| comparison (mixture − baseline) | mean delta (pp) | bootstrap 95% CI (pp) | sign-test p | Holm p |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for row in comparisons:
        report_lines.append(
            f"| {row['comparison']} | {row['paired_mean_delta_pct']:.3f} | "
            f"[{row['paired_bootstrap_95_ci_low_pct']:.3f}, "
            f"{row['paired_bootstrap_95_ci_high_pct']:.3f}] | "
            f"{row['sign_test_p']:.6g} | {row['holm_adjusted_p']:.6g} |"
        )
    report_lines.extend(
        [
            "",
            "## 结论边界",
            "",
            f"- retrieval 的 replay mean 最低；mixture 相对 retrieval 更差 "
            f"`{vs_retrieval['paired_mean_delta_pct']:.3f}` pp，不能声称生成模型总体优于检索基线。",
            f"- mixture 优于 direct inverse `{abs(vs_direct['paired_mean_delta_pct']):.3f}` pp，"
            f"Holm p=`{vs_direct['holm_adjusted_p']:.6g}`。",
            f"- mixture 相对 single-mode 的均值改善为 "
            f"`{abs(vs_single['paired_mean_delta_pct']):.3f}` pp，但 Holm p="
            f"`{vs_single['holm_adjusted_p']:.4f}`，本协议下不能声称显著架构优势。",
            "- mixture 的 effective-K 略高于 single-mode；候选预算与 mode 架构贡献须留待 A3 消融。",
            "- `formal_completed` 只表示该 single-seed 协议完整执行。",
            "",
        ]
    )
    (run / "aggregate" / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return run / "aggregate"


def _surrogate_ablation_sample_config(
    cfg: Mapping[str, Any],
    *,
    run_dir: Path,
) -> dict[str, Any]:
    protocol = _protocol(cfg)
    dataset = _dataset_cfg(cfg)
    candidate_generation = _mapping(
        cfg.get("candidate_generation"),
        name="candidate_generation",
    )
    generator = _mapping(
        candidate_generation.get("generator_checkpoint"),
        name="candidate_generation.generator_checkpoint",
    )
    generator_model = _mapping(
        candidate_generation.get("model"),
        name="candidate_generation.model",
    )
    variants = {str(item["id"]): item for item in _surrogate_variants(cfg)}
    reference_id = str(_mapping(cfg["surrogate_matrix"], name="surrogate_matrix")["reference_id"])
    reference = variants[reference_id]
    return {
        "experiment": {
            "experiment_id": f"{protocol['id']}_frozen_candidate_pool",
            "family": "research",
            "stage": "a2_surrogate_ablation_frozen_candidate_sampling",
        },
        "dataset": {
            "path": str(dataset["path"]),
            "condition_target": "Wh",
            "output_repr": "rho_wedge",
            "target_split": "val",
            "target_manifest_path": str(dataset["target_manifest_path"]),
        },
        "model": {
            "name": str(generator_model.get("name", "mixture_conditional_generator")),
            "seed": int(generator.get("seed", 0)),
            "seeds": [int(generator.get("seed", 0))],
            "mode_count": int(generator_model.get("mode_count", 4)),
            "latent_dim": int(generator_model.get("latent_dim", 32)),
            "hidden_dim": int(generator_model.get("hidden_dim", 256)),
            "depth": int(generator_model.get("depth", 4)),
        },
        "surrogate": _variant_surrogate_config(reference),
        "runtime": dict(_mapping(cfg.get("runtime"), name="runtime")),
        "inverse": {
            "generator_checkpoint_path": str(generator["path"]),
            "n_candidates": int(candidate_generation["n_candidates"]),
            "surrogate_top_m": int(protocol["replay_top_m"]),
            "threshold": float(protocol["threshold"]),
            "candidate_separation": {
                "enabled": True,
                "deduplicate_binary": True,
                "min_hamming": str(protocol["minimum_normalized_hamming"]),
            },
            "allow_mock_generator": False,
            "allow_mock_surrogate": False,
            "ranking": {"error_metric": "bands_2_10_mre"},
        },
        "replay": {"enabled": False},
        "output": {"runs_dir": "runs", "run_dir": str(run_dir)},
    }


def run_a2_surrogate_sample(config_path: str | Path, *, run_dir: str | Path) -> Path:
    """Freeze one reconstruction-generator candidate pool for all seven rankers."""

    from phc2d.inverse.runner import _candidate_pool_from_record, run_inverse_sample

    cfg = _read_yaml(config_path)
    run = Path(run_dir)
    decision_path = run / "preflight_decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError("preflight_decision.json is required before surrogate sampling")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if str(decision.get("status")) != "pass" or not bool(decision.get("sampling_authorized")):
        raise RuntimeError("surrogate sampling is not authorized by preflight_decision.json")

    candidate_dir = run / "candidates"
    derived_config = _surrogate_ablation_sample_config(cfg, run_dir=candidate_dir)
    config_out = run / "candidate_generation.yaml"
    config_out.write_text(
        yaml.safe_dump(derived_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    if not (candidate_dir / "candidates.json").is_file():
        run_inverse_sample(config_out)

    candidate_path = candidate_dir / "candidates.json"
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("frozen candidate pool contains no targets")
    expected_indices = _target_indices(cfg)
    actual_indices = [int(item["target_index"]) for item in targets]
    if actual_indices != expected_indices:
        raise ValueError("frozen candidate pool target order does not match the validation manifest")
    identity_rows: list[dict[str, Any]] = []
    health_rows: list[dict[str, Any]] = []
    for target in targets:
        pool = _candidate_pool_from_record(target)
        raw_metrics = _mapping(target.get("raw_metrics"), name="target.raw_metrics")
        filtered_metrics = _mapping(
            target.get("filtered_metrics"),
            name="target.filtered_metrics",
        )
        identity_rows.append(
            {
                "target_id": str(target.get("target_id", "")),
                "target_index": int(target["target_index"]),
                "requested_k": int(_protocol(cfg)["requested_k"]),
                "raw_k": int(raw_metrics.get("n_candidates", len(target.get("raw_candidates", ())))),
                "candidate_count": int(pool.shape[0]),
                "candidate_sha256": hash_array(np.asarray(pool, dtype=np.float32)),
            }
        )
        health_rows.append(
            {
                "target_id": str(target.get("target_id", "")),
                "target_index": int(target["target_index"]),
                "raw_k": int(raw_metrics.get("n_candidates", len(target.get("raw_candidates", ())))),
                "effective_k": int(pool.shape[0]),
                "validity_rate": float(filtered_metrics.get("validity_rate", 0.0)),
                "unique_rate": float(filtered_metrics.get("unique_rate", 0.0)),
                "hamming_diversity": float(filtered_metrics.get("diversity_hamming", 0.0)),
                "nearest_neighbor_hamming_p10": float(
                    filtered_metrics.get("nearest_neighbor_hamming_p10", 0.0)
                ),
                "fill_fraction": float(filtered_metrics.get("fill_fraction", 0.0)),
            }
        )
    candidate_generation = _mapping(
        cfg.get("candidate_generation"),
        name="candidate_generation",
    )
    generator = _mapping(
        candidate_generation.get("generator_checkpoint"),
        name="candidate_generation.generator_checkpoint",
    )
    manifest = {
        "version": 1,
        "status": "frozen",
        "validation_boundary": "validation_candidate_pool_no_mpb",
        "candidates_path": str(candidate_path),
        "candidates_sha256": _sha256_file(candidate_path),
        "generator_checkpoint_path": str(generator["path"]),
        "generator_checkpoint_sha256": str(generator["sha256"]),
        "generator_training_mode": str(candidate_generation["source_training_mode"]),
        "dataset_sha256": str(_dataset_cfg(cfg)["expected_sha256"]),
        "target_manifest_path": str(_dataset_cfg(cfg)["target_manifest_path"]),
        "target_manifest_sha256": str(_dataset_cfg(cfg)["target_manifest_sha256"]),
        "target_count": len(targets),
        "target_indices_sha256": _sha256_json(actual_indices),
        "candidate_identity_sha256": _sha256_json(identity_rows),
        "candidate_health": {
            "validity_rate_mean": float(
                np.mean([row["validity_rate"] for row in health_rows])
            ),
            "unique_rate_mean": float(np.mean([row["unique_rate"] for row in health_rows])),
            "effective_k_mean": float(np.mean([row["effective_k"] for row in health_rows])),
            "effective_k_p10": float(
                np.percentile([row["effective_k"] for row in health_rows], 10.0)
            ),
            "effective_k_min": int(min(row["effective_k"] for row in health_rows)),
            "hamming_diversity_mean": float(
                np.mean([row["hamming_diversity"] for row in health_rows])
            ),
            "nearest_neighbor_hamming_p10_mean": float(
                np.mean([row["nearest_neighbor_hamming_p10"] for row in health_rows])
            ),
        },
        "targets": identity_rows,
    }
    _write_csv(run / "candidate_health.csv", health_rows)
    _atomic_json(run / "candidate_pool_manifest.json", manifest)
    return candidate_dir


def _surrogate_ablation_rerank_config(
    cfg: Mapping[str, Any],
    *,
    candidates_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    protocol = _protocol(cfg)
    variants = _surrogate_variants(cfg)
    reference_id = str(_mapping(cfg["surrogate_matrix"], name="surrogate_matrix")["reference_id"])
    strategies = [
        {"id": str(variant["id"]), **_variant_surrogate_config(variant)}
        for variant in variants
    ]
    return {
        "experiment": {
            "experiment_id": f"{protocol['id']}_seven_surrogate_rerank",
            "family": "research",
            "stage": "inverse_surrogate_rerank_no_mpb_meep",
        },
        "dataset": {
            "path": str(_dataset_cfg(cfg)["path"]),
            "condition_target": "Wh",
            "output_repr": "rho_wedge",
            "target_split": "val",
            "target_manifest_path": str(_dataset_cfg(cfg)["target_manifest_path"]),
        },
        "model": {"name": "mixture_conditional_generator", "seed": 0, "seeds": [0]},
        "surrogate": _variant_surrogate_config(
            next(item for item in variants if str(item["id"]) == reference_id)
        ),
        "runtime": dict(_mapping(cfg.get("runtime"), name="runtime")),
        "inverse": {
            "n_candidates": int(protocol["requested_k"]),
            "surrogate_top_m": int(protocol["replay_top_m"]),
            "threshold": float(protocol["threshold"]),
            "allow_mock_generator": False,
            "allow_mock_surrogate": False,
            "ranking": {"error_metric": "bands_2_10_mre"},
        },
        "rerank": {
            "candidates_path": str(candidates_path),
            "top_m": int(protocol["replay_top_m"]),
            "primary_strategy": reference_id,
            "strategies": strategies,
        },
        "replay": {"enabled": False},
        "metrics": {
            "include": [
                "best_of_k_surrogate_mre_bands_2_10_pct",
                "selection_overlap_with_primary",
                "surrogate_ranking_time_s",
                "inference_time_per_sample_ms",
                "wall_time_total_s",
                "peak_gpu_memory_mb",
            ]
        },
        "output": {"runs_dir": "runs", "run_dir": str(run_dir)},
    }


def _spearman_permutation(left: Sequence[int], right: Sequence[int]) -> float:
    if len(left) != len(right) or set(left) != set(right):
        raise ValueError("rank permutations must contain the same candidate indices")
    n = len(left)
    if n < 2:
        return 1.0
    left_rank = {int(value): index for index, value in enumerate(left)}
    right_rank = {int(value): index for index, value in enumerate(right)}
    sum_squared = sum((left_rank[value] - right_rank[value]) ** 2 for value in left_rank)
    return float(1.0 - 6.0 * sum_squared / (n * (n * n - 1)))


def _summarize_surrogate_screen(screen_dir: Path, *, reference_id: str) -> None:
    per_target = _read_csv(screen_dir / "per_target_strategy_metrics.csv")
    primary_by_target: dict[int, list[int]] = {}
    for row in per_target:
        if str(row["strategy_id"]) == reference_id:
            primary_by_target[int(row["target_index"])] = [
                int(item) for item in json.loads(str(row["ranked_indices"]))
            ]
    correlation_by_strategy: dict[str, list[float]] = {}
    overlap_by_strategy: dict[str, list[float]] = {}
    for row in per_target:
        strategy_id = str(row["strategy_id"])
        target_index = int(row["target_index"])
        primary = primary_by_target.get(target_index)
        if primary is None:
            raise ValueError(f"missing primary ranking for target {target_index}")
        current = [int(item) for item in json.loads(str(row["ranked_indices"]))]
        correlation_by_strategy.setdefault(strategy_id, []).append(
            _spearman_permutation(primary, current)
        )
        overlap_by_strategy.setdefault(strategy_id, []).append(
            float(row["selection_overlap_with_primary"])
        )
    leaderboard = _read_csv(screen_dir / "per_strategy_metrics.csv")
    candidate_manifest_path = screen_dir.parent / "candidate_pool_manifest.json"
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    candidate_health = _mapping(
        candidate_manifest.get("candidate_health"),
        name="candidate_pool_manifest.candidate_health",
    )
    enriched: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for row in leaderboard:
        strategy_id = str(row["strategy_id"])
        correlations = correlation_by_strategy.get(strategy_id, [])
        overlaps = overlap_by_strategy.get(strategy_id, [])
        mean_correlation = float(np.mean(correlations)) if correlations else float("nan")
        mean_overlap = float(np.mean(overlaps)) if overlaps else float("nan")
        enriched.append(
            {
                **row,
                "rank_spearman_with_primary": mean_correlation,
                "selection_overlap_with_primary": mean_overlap,
                **{
                    f"candidate_{key}": value
                    for key, value in candidate_health.items()
                },
            }
        )
        comparison_rows.append(
            {
                "strategy_id": strategy_id,
                "primary_strategy_id": reference_id,
                "target_count": len(correlations),
                "rank_spearman_with_primary": mean_correlation,
                "selection_overlap_with_primary": mean_overlap,
            }
        )
    _write_csv(screen_dir / "strategy_metrics.csv", enriched)
    _write_csv(screen_dir / "selection_overlap.csv", comparison_rows)


def run_a2_surrogate_screen(config_path: str | Path, *, run_dir: str | Path) -> Path:
    """Rerank one frozen validation candidate pool with all seven surrogates."""

    from phc2d.inverse.runner import run_inverse_rerank

    cfg = _read_yaml(config_path)
    run = Path(run_dir)
    candidate_manifest_path = run / "candidate_pool_manifest.json"
    if not candidate_manifest_path.is_file():
        raise FileNotFoundError("candidate_pool_manifest.json is required before surrogate screening")
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    candidates_path = Path(str(candidate_manifest["candidates_path"]))
    _require_hash(
        candidates_path,
        str(candidate_manifest["candidates_sha256"]),
        label="frozen candidate pool",
    )
    screen_dir = run / "screen"
    derived_config = _surrogate_ablation_rerank_config(
        cfg,
        candidates_path=candidates_path,
        run_dir=screen_dir,
    )
    config_out = run / "surrogate_screen.yaml"
    config_out.write_text(
        yaml.safe_dump(derived_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    if not (screen_dir / "strategy_candidates.json").is_file():
        run_inverse_rerank(config_out)
    reference_id = str(_mapping(cfg["surrogate_matrix"], name="surrogate_matrix")["reference_id"])
    _summarize_surrogate_screen(screen_dir, reference_id=reference_id)
    return screen_dir


def _is_wsl_meep_env() -> bool:
    return bool(os.environ.get("WSL_DISTRO_NAME")) and os.environ.get("CONDA_DEFAULT_ENV") == "meep_env"


def _surrogate_pilot_replay_config(
    cfg: Mapping[str, Any],
    *,
    strategy_candidates_path: Path,
    target_indices_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    protocol = _protocol(cfg)
    pilot = _mapping(cfg.get("pilot"), name="pilot")
    bundle = load_forward_dataset(_dataset_cfg(cfg)["path"])
    first_primary_band = 2 if int(bundle.nb) >= 2 else 1
    return {
        "experiment": {
            "experiment_id": f"{protocol['id']}_pilot_replay",
            "family": "research",
            "stage": "inverse_strategy_candidate_mpb_pilot",
        },
        "dataset": {
            "path": str(_dataset_cfg(cfg)["path"]),
            "condition_target": "Wh",
            "output_repr": "rho_wedge",
            "target_split": "val",
        },
        "model": {"name": "mixture_conditional_generator", "seed": 0, "seeds": [0]},
        "inverse": {
            "n_candidates": int(protocol["requested_k"]),
            "surrogate_top_m": int(protocol["replay_top_m"]),
            "threshold": float(protocol["threshold"]),
        },
        "replay": {
            "enabled": True,
            "backend": str(pilot.get("backend", "real_mpb")),
            "require_real_mpb": True,
            "environment": "WSL Ubuntu + conda meep_env",
            "candidates_path": str(strategy_candidates_path),
            "target_indices_path": str(target_indices_path),
            "n_targets": len(pilot["target_positions"]),
            "selection_unit": "target_index",
            "replay_top_m": int(protocol["replay_top_m"]),
            "primary_band_range": [first_primary_band, int(bundle.nb)],
            "timeout_seconds": float(pilot.get("timeout_seconds", 900.0)),
        },
        "solver": dict(_mapping(cfg.get("solver"), name="solver")),
        "metrics": {
            "band_ranges": {
                "bands_1_10": [1, int(bundle.nb)],
                "bands_2_10": [first_primary_band, int(bundle.nb)],
            }
        },
        "output": {"runs_dir": "runs", "run_dir": str(run_dir)},
    }


def run_a2_surrogate_pilot_replay(
    config_path: str | Path,
    *,
    run_dir: str | Path,
    solver_fn: SolverFn | None = None,
) -> Path:
    """Replay the deduplicated union of seven Top-M selections on validation targets."""

    from phc2d.replay import run_replay_mpb

    cfg = _read_yaml(config_path)
    run = Path(run_dir)
    strategy_candidates_path = run / "screen" / "strategy_candidates.json"
    if not strategy_candidates_path.is_file():
        raise FileNotFoundError("screen/strategy_candidates.json is required before pilot replay")
    pilot = _mapping(cfg.get("pilot"), name="pilot")
    all_indices = _target_indices(cfg)
    candidate_manifest_path = run / "candidate_pool_manifest.json"
    if not candidate_manifest_path.is_file():
        raise FileNotFoundError(
            "candidate_pool_manifest.json is required before pilot replay"
        )
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    candidate_targets = list(candidate_manifest.get("targets", ()))
    replay_top_m = int(_protocol(cfg)["replay_top_m"])
    coverage_shortfalls = [
        {
            "target_position": int(position),
            "target_index": int(candidate_targets[int(position)]["target_index"]),
            "effective_k": int(candidate_targets[int(position)]["candidate_count"]),
            "required_top_m": replay_top_m,
        }
        for position in pilot["target_positions"]
        if int(candidate_targets[int(position)]["candidate_count"]) < replay_top_m
    ]
    if coverage_shortfalls:
        _atomic_json(
            run / "pilot_replay_blocked.json",
            {
                "status": "blocked_candidate_coverage",
                "reason": "one or more fixed pilot targets cannot supply Top-M candidates",
                "shortfalls": coverage_shortfalls,
                "real_mpb_started": False,
            },
        )
        raise RuntimeError(
            "20-target pilot replay is blocked because the frozen candidate pool "
            "does not provide Top-M candidates for every fixed pilot target"
        )
    pilot_indices = [all_indices[int(position)] for position in pilot["target_positions"]]
    pilot_manifest_path = run / "pilot_target_manifest.json"
    _atomic_json(
        pilot_manifest_path,
        {
            "version": 1,
            "selection_unit": "target_index",
            "selection_policy": "fixed_four_validation_strata_positions",
            "dataset_sha256": str(_dataset_cfg(cfg)["expected_sha256"]),
            "dataset_split": "val",
            "target_indices": pilot_indices,
        },
    )
    replay_dir = run / "pilot_replay"
    replay_config = _surrogate_pilot_replay_config(
        cfg,
        strategy_candidates_path=strategy_candidates_path,
        target_indices_path=pilot_manifest_path,
        run_dir=replay_dir,
    )
    if solver_fn is None and not _is_wsl_meep_env():
        raise RuntimeError(
            "real-MPB surrogate pilot replay must run inside WSL Ubuntu "
            "with CONDA_DEFAULT_ENV=meep_env"
        )
    config_out = run / "pilot_replay.yaml"
    config_out.write_text(
        yaml.safe_dump(replay_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return run_replay_mpb(config_out, solver_fn=solver_fn)


def run_a2_surrogate_smoke_replay(
    config_path: str | Path,
    *,
    run_dir: str | Path,
    real_mpb: bool,
    solver_fn: SolverFn | None = None,
) -> Path:
    """Run the fixed two-target, two-candidate replay gate for seven strategies."""

    from phc2d.replay import run_replay_mpb

    cfg = _read_yaml(config_path)
    run = Path(run_dir)
    strategy_candidates_path = run / "screen" / "strategy_candidates.json"
    if not strategy_candidates_path.is_file():
        raise FileNotFoundError("screen/strategy_candidates.json is required before replay smoke")
    pilot = _mapping(cfg.get("pilot"), name="pilot")
    all_indices = _target_indices(cfg)
    smoke_positions = [int(item) for item in pilot["target_positions"][:2]]
    smoke_indices = [all_indices[position] for position in smoke_positions]
    smoke_name = "real_mpb_smoke" if real_mpb else "fake_mpb_smoke"
    smoke_manifest_path = run / f"{smoke_name}_target_manifest.json"
    _atomic_json(
        smoke_manifest_path,
        {
            "version": 1,
            "selection_unit": "target_index",
            "selection_policy": "first_two_fixed_pilot_positions",
            "dataset_sha256": str(_dataset_cfg(cfg)["expected_sha256"]),
            "dataset_split": "val",
            "target_indices": smoke_indices,
        },
    )
    smoke_dir = run / smoke_name
    replay_config = _surrogate_pilot_replay_config(
        cfg,
        strategy_candidates_path=strategy_candidates_path,
        target_indices_path=smoke_manifest_path,
        run_dir=smoke_dir,
    )
    replay_config["experiment"]["experiment_id"] = f"{_protocol(cfg)['id']}_{smoke_name}"
    replay_config["experiment"]["stage"] = (
        "inverse_strategy_candidate_real_mpb_smoke"
        if real_mpb
        else "inverse_strategy_candidate_fake_mpb_smoke"
    )
    replay_config["replay"].update(
        {
            "backend": "real_mpb" if real_mpb else "fake_mpb",
            "require_real_mpb": bool(real_mpb),
            "n_targets": 2,
            "replay_top_m": 2,
            "environment": (
                "WSL Ubuntu + conda meep_env"
                if real_mpb
                else "deterministic fake-MPB test boundary"
            ),
        }
    )
    if real_mpb and solver_fn is None and not _is_wsl_meep_env():
        raise RuntimeError(
            "two-target real-MPB smoke must run inside WSL Ubuntu "
            "with CONDA_DEFAULT_ENV=meep_env"
        )
    config_out = run / f"{smoke_name}.yaml"
    config_out.write_text(
        yaml.safe_dump(replay_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return run_replay_mpb(config_out, solver_fn=solver_fn)


def _mean_ci(values: Sequence[float], *, seed: int, iterations: int) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    array = np.asarray(values, dtype=np.float64)
    if array.size == 1:
        value = float(array[0])
        return value, value
    rng = np.random.default_rng(int(seed))
    samples = rng.choice(array, size=(int(iterations), int(array.size)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _promotion_choice(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("promotion group has no candidates")
    ordered = sorted(
        (dict(item) for item in candidates),
        key=lambda item: float(item["best_of_k_replay_mre_bands_2_10_pct"]),
    )
    best = ordered[0]
    overlap_group = [
        item
        for item in ordered
        if float(item["ci_low_pct"]) <= float(best["ci_high_pct"])
        and float(item["ci_high_pct"]) >= float(best["ci_low_pct"])
    ]
    return min(
        overlap_group,
        key=lambda item: (
            abs(float(item["surrogate_replay_gap_bands_2_10_pct"])),
            float(item["inference_time_per_sample_ms"]),
            str(item["strategy_id"]),
        ),
    )


def run_a2_surrogate_promote(config_path: str | Path, *, run_dir: str | Path) -> Path:
    """Promote the FNO reference plus one forward and one query challenger."""

    cfg = _read_yaml(config_path)
    run = Path(run_dir)
    replay_dir = run / "pilot_replay"
    replay_metrics_path = replay_dir / "metrics.json"
    if not replay_metrics_path.is_file():
        raise FileNotFoundError("pilot_replay/metrics.json is required before promotion")
    replay_metrics = json.loads(replay_metrics_path.read_text(encoding="utf-8"))
    if str(replay_metrics.get("replay_backend")) != "real_mpb":
        raise RuntimeError("surrogate promotion requires replay_backend=real_mpb")
    if str(replay_metrics.get("validation_boundary")) != "formal_inverse_candidate_mpb_replay":
        raise RuntimeError("surrogate promotion requires a formal real-MPB validation boundary")
    if int(replay_metrics.get("targets", 0)) != 20:
        raise RuntimeError("surrogate promotion requires complete coverage of 20 pilot targets")
    if int(replay_metrics.get("strategy_count", 0)) != len(SURROGATE_ABLATION_MODELS):
        raise RuntimeError("surrogate promotion requires all seven ranking strategies")
    if float(replay_metrics.get("mpb_replay_success_rate", 0.0)) != 1.0:
        raise RuntimeError("surrogate promotion requires mpb_replay_success_rate=1.0")

    target_rows = _read_csv(replay_dir / "per_target_replay_metrics.csv")
    replay_rows = _read_csv(replay_dir / "per_strategy_replay_metrics.csv")
    if len(target_rows) != 20 * len(SURROGATE_ABLATION_MODELS):
        raise RuntimeError("surrogate promotion requires 20 target rows for each strategy")
    if any(int(row["target_count"]) != 20 for row in replay_rows):
        raise RuntimeError("surrogate promotion requires target_count=20 for every strategy")
    screen_rows = {
        str(row["strategy_id"]): row
        for row in _read_csv(run / "screen" / "strategy_metrics.csv")
    }
    statistics = _mapping(cfg.get("statistics"), name="statistics")
    iterations = int(statistics.get("bootstrap_iterations", 10_000))
    seed = int(statistics.get("bootstrap_seed", 0))
    enriched: list[dict[str, Any]] = []
    for row in replay_rows:
        strategy_id = str(row["strategy_id"])
        values = [
            float(item["best_of_k_replay_mre_bands_2_10_pct"])
            for item in target_rows
            if str(item["strategy_id"]) == strategy_id
        ]
        ci_low, ci_high = _mean_ci(values, seed=seed, iterations=iterations)
        screen = screen_rows[strategy_id]
        enriched.append(
            {
                **row,
                "ci_low_pct": ci_low,
                "ci_high_pct": ci_high,
                "inference_time_per_sample_ms": float(screen["inference_time_per_sample_ms"]),
                "surrogate_ranking_time_s": float(screen["surrogate_ranking_time_s"]),
            }
        )

    variants = _surrogate_variants(cfg)
    roles = {str(item["id"]): str(item.get("role_group", "")) for item in variants}
    reference_id = str(_mapping(cfg["surrogate_matrix"], name="surrogate_matrix")["reference_id"])
    reference = next(item for item in enriched if str(item["strategy_id"]) == reference_id)
    forward = _promotion_choice(
        [
            item
            for item in enriched
            if roles.get(str(item["strategy_id"])) == "forward_challenger"
        ]
    )
    query = _promotion_choice(
        [
            item
            for item in enriched
            if roles.get(str(item["strategy_id"])) == "query_challenger"
        ]
    )
    promoted_ids = [reference_id, str(forward["strategy_id"]), str(query["strategy_id"])]
    decision = {
        "version": 1,
        "status": "pass",
        "evidence_status": "validation_real_mpb_pilot",
        "selection_split": "val",
        "test_accessed_for_selection": False,
        "reference_id": reference_id,
        "promoted_surrogate_ids": promoted_ids,
        "reference": reference,
        "best_forward_challenger": forward,
        "best_query_challenger": query,
        "tie_break_policy": [
            "overlapping_bootstrap_ci",
            "minimum_absolute_surrogate_replay_gap",
            "minimum_inference_time",
            "lexical_strategy_id",
        ],
        "formal_claims_authorized": False,
        "cross_seed_claims_authorized": False,
    }
    _write_csv(run / "pilot_replay_metrics.csv", enriched)
    _atomic_json(run / "promotion_decision.json", decision)
    _atomic_json(
        run / "objective_arm_manifest.json",
        {
            "version": 1,
            "status": "selected_from_validation_pilot",
            "surrogate_ids": promoted_ids,
            "promotion_decision_sha256": _sha256_json(decision),
        },
    )
    return run


def _postrepair_checkpoint_eval_config(
    cfg: Mapping[str, Any],
    *,
    variant: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    """Build one private validation config without changing the public CLI."""

    training_path = Path(str(variant["training_config_path"]))
    resolved = _read_yaml(training_path)
    protocol = _protocol(cfg)
    dataset = dict(_mapping(resolved.get("dataset"), name="training.dataset"))
    dataset.update(
        {
            "path": str(_dataset_cfg(cfg)["path"]),
            "target_split": "val",
            "target_limit": None,
            "target_selection_seed": None,
            "target_manifest_path": str(_dataset_cfg(cfg)["target_manifest_path"]),
        }
    )
    inverse = dict(_mapping(resolved.get("inverse"), name="training.inverse"))
    inverse.update(
        {
            "generator_checkpoint_path": str(checkpoint["path"]),
            "n_candidates": int(protocol["requested_k"]),
            "surrogate_top_m": int(protocol["replay_top_m"]),
            "seed": int(protocol["seed"]),
            "seeds": [int(protocol["seed"])],
            "candidate_seed_offset": 0,
            "surrogate_seed_offset": 0,
            "replay_seed_offset": 0,
            "allow_mock_generator": False,
            "allow_mock_surrogate": False,
        }
    )
    primary = {
        key: value
        for key, value in _mapping(
            variant["primary_ranker"],
            name=f"{variant['id']}.primary_ranker",
        ).items()
        if key not in {"sha256", "k_path_sha256", "read_only", "id"}
    }
    return {
        **resolved,
        "experiment": {
            "experiment_id": f"{protocol['id']}_{variant['id']}_epoch{int(checkpoint['epoch'])}_validation",
            "family": "research",
            "stage": "postrepair_validation_checkpoint_selection",
        },
        "dataset": dataset,
        "inverse": inverse,
        "surrogate": primary,
        "runtime": dict(_mapping(cfg.get("runtime"), name="runtime")),
        "replay": {"enabled": False},
        "output": {"runs_dir": "runs", "run_dir": str(run_dir)},
    }


def _postrepair_gate_failures(
    row: Mapping[str, Any],
    *,
    gates: Mapping[str, Any],
) -> list[str]:
    checks = (
        ("validity_rate", float(row["validity_rate"]), float(gates["validity_rate_min"]), "min"),
        ("effective_unique_k", float(row["effective_unique_k"]), float(gates["effective_k_mean_min"]), "min"),
        ("effective_unique_k_p10", float(row["effective_unique_k_p10"]), float(gates["effective_k_p10_min"]), "min"),
        ("continuous_candidate_std", float(row["continuous_candidate_std"]), float(gates["continuous_candidate_std_min"]), "min"),
        ("threshold_margin_fraction", float(row["threshold_margin_fraction"]), float(gates["threshold_margin_fraction_max"]), "max"),
        ("latent_input_gradient_norm", float(row["latent_input_gradient_norm"]), float(gates["latent_gradient_norm_min"]), "min"),
        ("mode_coverage_fraction", float(row["mode_coverage_fraction"]), float(gates["mode_coverage_fraction_min"]), "min"),
    )
    failures: list[str] = []
    for name, actual, threshold, direction in checks:
        if not math.isfinite(actual):
            failures.append(f"{name}:non_finite")
        elif direction == "min" and actual < threshold:
            failures.append(f"{name}:{actual}<{threshold}")
        elif direction == "max" and actual > threshold:
            failures.append(f"{name}:{actual}>{threshold}")
    if bool(gates.get("require_finite_training_metrics", True)) and not bool(row["training_metrics_finite"]):
        failures.append("training_metrics:non_finite")
    if bool(gates.get("require_cuda_no_fallback", True)):
        if str(row["resolved_device"]).strip().lower().startswith("cuda") is False:
            failures.append(f"resolved_device:{row['resolved_device']}")
        if str(row.get("device_fallback_reason", "")).strip() not in {"", "None", "null"}:
            failures.append(f"device_fallback_reason:{row['device_fallback_reason']}")
    for name in (
        "validation_best_of_k_surrogate_mre",
        "validation_primary_selected_common_fno_mre",
        "candidate_sampling_time_s",
        "primary_surrogate_ranking_time_s",
        "common_fno_diagnostic_ranking_time_s",
    ):
        if not math.isfinite(float(row[name])):
            failures.append(f"{name}:non_finite")
    return failures


def _select_postrepair_checkpoint(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    """Apply the frozen hierarchical checkpoint rule without accessing test data."""

    if not rows:
        raise ValueError("post-repair checkpoint selection requires at least one row")
    eligible = [dict(row) for row in rows if str(row["gate_status"]) == "pass"]
    if not eligible:
        return min((dict(row) for row in rows), key=lambda row: int(row["epoch"])), (
            "blocked_candidate_health"
        )
    return min(
        eligible,
        key=lambda row: (
            float(row["selection_score_max_primary_common_mre"]),
            float(row["threshold_margin_fraction"]),
            float(row["sampling_plus_ranking_time_s"]),
            int(row["epoch"]),
        ),
    ), "selected"


def run_a2_postrepair_select(config_path: str | Path, *, run_dir: str | Path) -> Path:
    """Evaluate epochs 10/30/50 and select one healthy checkpoint per inverse recipe."""

    cfg = _read_yaml(config_path)
    run = Path(run_dir)
    preflight_path = run / "preflight_decision.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("preflight_decision.json is required before checkpoint selection")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if str(preflight.get("status")) != "pass" or not bool(preflight.get("selection_authorized")):
        raise RuntimeError("post-repair checkpoint selection is not authorized by preflight")

    selection_cfg = _mapping(cfg.get("checkpoint_selection"), name="checkpoint_selection")
    gates = _mapping(selection_cfg.get("hard_gates"), name="checkpoint_selection.hard_gates")
    common = {
        key: value
        for key, value in _mapping(
            cfg.get("common_diagnostic_ranker"),
            name="common_diagnostic_ranker",
        ).items()
        if key not in {"sha256", "read_only", "id", "k_path_sha256"}
    }
    selection_dir = run / "checkpoint_selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for variant in _postrepair_variants(cfg):
        variant_id = str(variant["id"])
        training_path = Path(str(variant["training_config_path"]))
        history_path = training_path.parent / "history.csv"
        metrics_path = training_path.parent / "metrics.json"
        training_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        numeric_training_values = [
            float(value)
            for value in training_metrics.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        training_metrics_finite = bool(numeric_training_values) and all(
            math.isfinite(value) for value in numeric_training_values
        )
        variant_rows: list[dict[str, Any]] = []
        for checkpoint in variant["generator_checkpoints"]:
            entry = _mapping(checkpoint, name=f"{variant_id}.generator_checkpoint")
            epoch = int(entry["epoch"])
            eval_config = _postrepair_checkpoint_eval_config(
                cfg,
                variant=variant,
                checkpoint=entry,
                run_dir=selection_dir / variant_id / f"epoch_{epoch}",
            )
            eval_config_path = selection_dir / f"{variant_id}_epoch_{epoch}.yaml"
            eval_config_path.write_text(
                yaml.safe_dump(eval_config, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            evaluated = _evaluate_checkpoint(
                eval_config_path,
                Path(str(entry["path"])),
                diagnostic_surrogate_cfg=common,
            )
            history = _history_row_for_epoch(history_path, epoch=epoch)
            row: dict[str, Any] = {
                "variant_id": variant_id,
                "role_group": str(variant["role_group"]),
                "source_lane": str(variant["source_lane"]),
                "epoch": epoch,
                **evaluated,
                "latent_input_gradient_norm": float(history.get("latent_input_gradient_norm", 0.0)),
                "mode_coverage_fraction": float(history.get("mode_coverage_fraction", 0.0)),
                "training_objective_total_loss": float(history.get("objective_total_loss", float("nan"))),
                "training_metrics_finite": training_metrics_finite,
                "resolved_device": str(training_metrics.get("resolved_device", "")),
                "device_fallback_reason": training_metrics.get("device_fallback_reason"),
            }
            failures = _postrepair_gate_failures(row, gates=gates)
            row["selection_score_max_primary_common_mre"] = max(
                float(row["validation_best_of_k_surrogate_mre"]),
                float(row["validation_primary_selected_common_fno_mre"]),
            )
            row["sampling_plus_ranking_time_s"] = sum(
                float(row[name])
                for name in (
                    "candidate_sampling_time_s",
                    "primary_surrogate_ranking_time_s",
                    "common_fno_diagnostic_ranking_time_s",
                )
            )
            row["gate_status"] = "pass" if not failures else "blocked_candidate_health"
            row["gate_failures"] = json.dumps(failures, ensure_ascii=False)
            variant_rows.append(row)
            all_rows.append(row)
        chosen, selection_status = _select_postrepair_checkpoint(variant_rows)
        eligible = selection_status == "selected"
        selected = {
            **chosen,
            "selection_status": selection_status,
            "selected_checkpoint_path": str(chosen["checkpoint_path"]) if eligible else "",
            "selected_checkpoint_sha256": str(chosen["checkpoint_sha256"]) if eligible else "",
        }
        selected_rows.append(selected)
        arm_dir = selection_dir / variant_id
        arm_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(arm_dir / "selected_checkpoint.json", selected)

    _write_csv(run / "candidate_health.csv", all_rows)
    _write_csv(run / "primary_common_diagnostic_metrics.csv", all_rows)
    _write_csv(run / "checkpoint_selection.csv", selected_rows)
    selected_manifest = {
        "version": 2,
        "status": "selection_complete",
        "evidence_status": "validation_checkpoint_selection_no_mpb",
        "selected_variant_count": sum(
            str(row["selection_status"]) == "selected" for row in selected_rows
        ),
        "blocked_variant_count": sum(
            str(row["selection_status"]) != "selected" for row in selected_rows
        ),
        "target_indices_sha256": _sha256_json(_target_indices(cfg)),
        "latent_seed": int(_protocol(cfg)["seed"]),
        "variants": selected_rows,
        "real_mpb_smoke_authorized": any(
            str(row["selection_status"]) == "selected" for row in selected_rows
        ),
        "pilot_replay_authorized": False,
    }
    _atomic_json(run / "validation_selected_checkpoints.json", selected_manifest)
    return run


def _postrepair_selected_rows(run: Path) -> list[dict[str, Any]]:
    path = run / "validation_selected_checkpoints.json"
    if not path.is_file():
        raise FileNotFoundError("validation_selected_checkpoints.json is required")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        dict(item)
        for item in payload.get("variants", ())
        if str(item.get("selection_status")) == "selected"
    ]


def run_a2_postrepair_screen(config_path: str | Path, *, run_dir: str | Path) -> Path:
    """Generate one selected candidate pool per healthy recipe and merge replay lineage."""

    from phc2d.inverse.runner import run_inverse_sample

    cfg = _read_yaml(config_path)
    run = Path(run_dir)
    selected_rows = _postrepair_selected_rows(run)
    if not selected_rows:
        _atomic_json(
            run / "candidate_pool_manifest.json",
            {
                "version": 2,
                "status": "blocked_candidate_health",
                "selected_variant_count": 0,
                "real_mpb_smoke_authorized": False,
            },
        )
        return run
    variants = {str(item["id"]): item for item in _postrepair_variants(cfg)}
    target_indices = _target_indices(cfg)
    arm_payloads: dict[str, dict[str, Any]] = {}
    arm_lineage: list[dict[str, Any]] = []
    for selected in selected_rows:
        variant_id = str(selected["variant_id"])
        variant = variants[variant_id]
        checkpoint = {
            "epoch": int(selected["epoch"]),
            "path": str(selected["selected_checkpoint_path"]),
            "sha256": str(selected["selected_checkpoint_sha256"]),
        }
        arm_dir = run / "candidate_pools" / variant_id
        sample_cfg = _postrepair_checkpoint_eval_config(
            cfg,
            variant=variant,
            checkpoint=checkpoint,
            run_dir=arm_dir,
        )
        sample_cfg["experiment"]["stage"] = "postrepair_selected_candidate_sampling"
        sample_cfg["inverse"]["surrogate_top_m"] = int(_protocol(cfg)["replay_top_m"])
        sample_path = run / "candidate_pools" / f"{variant_id}.yaml"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(
            yaml.safe_dump(sample_cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        if not (arm_dir / "candidates.json").is_file():
            run_inverse_sample(sample_path)
        candidate_path = arm_dir / "candidates.json"
        payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        actual_indices = [int(item["target_index"]) for item in payload.get("targets", ())]
        if actual_indices != target_indices:
            raise ValueError(f"{variant_id} candidate target IDs do not match the frozen manifest")
        arm_payloads[variant_id] = payload
        arm_lineage.append(
            {
                "variant_id": variant_id,
                "selected_epoch": int(selected["epoch"]),
                "generator_checkpoint_path": str(checkpoint["path"]),
                "generator_checkpoint_sha256": str(checkpoint["sha256"]),
                "candidates_path": str(candidate_path),
                "candidates_sha256": _sha256_file(candidate_path),
            }
        )

    merged_targets: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    reference_id = str(_mapping(cfg["surrogate_matrix"], name="surrogate_matrix")["reference_id"])
    for position, target_index in enumerate(target_indices):
        union: list[list[float]] = []
        union_hash_to_index: dict[str, int] = {}
        rankings: dict[str, Any] = {}
        selected_hashes: dict[str, set[str]] = {}
        target_template: dict[str, Any] | None = None
        for selected in selected_rows:
            variant_id = str(selected["variant_id"])
            target = dict(arm_payloads[variant_id]["targets"][position])
            if int(target["target_index"]) != int(target_index):
                raise ValueError("candidate target order drifted across inverse recipes")
            target_template = target_template or target
            candidates = np.asarray(target.get("candidates", ()), dtype=np.float32)
            metadata = list(target.get("metadata", ()))
            if candidates.ndim != 2 or int(candidates.shape[0]) == 0:
                raise ValueError(f"{variant_id} target {target_index} has no valid candidates")
            if metadata and [int(item.get("candidate_index", -1)) for item in metadata] != sorted(
                int(item.get("candidate_index", -1)) for item in metadata
            ):
                raise ValueError(f"{variant_id} latent candidate IDs are not deterministic")
            local_to_union: dict[int, int] = {}
            for local_index, candidate in enumerate(candidates):
                structure_hash = hash_array(np.asarray(candidate, dtype=np.float32))
                union_index = union_hash_to_index.get(structure_hash)
                if union_index is None:
                    union_index = len(union)
                    union_hash_to_index[structure_hash] = union_index
                    union.append(np.asarray(candidate, dtype=np.float32).tolist())
                local_to_union[local_index] = union_index
            selected_local = [int(item) for item in target.get("selected_indices", ())]
            ranked_local = [int(item) for item in target.get("ranked_indices", ())]
            selected_union = [local_to_union[item] for item in selected_local]
            ranked_union = [local_to_union[item] for item in ranked_local]
            selected_hashes[variant_id] = {
                hash_array(np.asarray(candidates[item], dtype=np.float32)) for item in selected_local
            }
            rankings[variant_id] = {
                "strategy_id": variant_id,
                "strategy_kind": str(variants[variant_id]["primary_ranker"]["kind"]),
                "selected_indices": selected_union,
                "ranked_indices": ranked_union,
                "selected_surrogate_predictions": target.get("selected_surrogate_predictions", []),
                "best_surrogate_prediction": target.get("best_surrogate_prediction", []),
                "surrogate_metrics": target.get("surrogate_metrics", {}),
                "generator_checkpoint_sha256": str(selected["selected_checkpoint_sha256"]),
                "selected_epoch": int(selected["epoch"]),
            }
            for local_index, union_index in local_to_union.items():
                lineage_rows.append(
                    {
                        "target_index": int(target_index),
                        "variant_id": variant_id,
                        "candidate_index": int(local_index),
                        "latent_id": f"seed{int(_protocol(cfg)['seed'])}_candidate{int(local_index):03d}",
                        "union_candidate_index": int(union_index),
                        "structure_sha256": hash_array(
                            np.asarray(candidates[local_index], dtype=np.float32)
                        ),
                    }
                )
        assert target_template is not None
        reference_hashes = selected_hashes.get(reference_id, set())
        for variant_id, hashes in selected_hashes.items():
            overlap_rows.append(
                {
                    "target_index": int(target_index),
                    "variant_id": variant_id,
                    "reference_id": reference_id,
                    "top_m_overlap_count": len(reference_hashes & hashes),
                    "top_m_overlap_fraction": float(
                        len(reference_hashes & hashes) / max(len(reference_hashes | hashes), 1)
                    ),
                }
            )
        merged_targets.append(
            {
                "target_id": f"postrepair_val_{int(target_index):05d}",
                "target_index": int(target_index),
                "local_pos": int(position),
                "condition": target_template.get("condition", {}),
                "sample_id": str(target_template.get("sample_id", f"target_{int(target_index)}")),
                "split": "val",
                "dataset_id": Path(str(_dataset_cfg(cfg)["path"])).stem,
                "candidates": union,
                "candidate_union_sha256": _sha256_json(union),
                "strategy_rankings": rankings,
            }
        )

    strategy_payload = {
        "version": 2,
        "kind": "postrepair_inverse_recipe_candidate_union",
        "dataset_path": str(_dataset_cfg(cfg)["path"]),
        "validation_boundary": "postrepair_validation_candidate_pools_no_mpb_meep",
        "top_m": int(_protocol(cfg)["replay_top_m"]),
        "primary_strategy": reference_id,
        "strategies": [
            {
                "strategy_id": str(item["variant_id"]),
                "generator_checkpoint_sha256": str(item["selected_checkpoint_sha256"]),
            }
            for item in selected_rows
        ],
        "targets": merged_targets,
    }
    screen_dir = run / "screen"
    screen_dir.mkdir(parents=True, exist_ok=True)
    strategy_path = screen_dir / "strategy_candidates.json"
    _atomic_json(strategy_path, strategy_payload)
    _write_csv(run / "candidate_lineage.csv", lineage_rows)
    _write_csv(run / "selection_overlap.csv", overlap_rows)
    target_id_hash = _sha256_json([int(item["target_index"]) for item in merged_targets])
    shared_latent_ids = [
        f"target{int(target_index):05d}_seed{int(_protocol(cfg)['seed'])}_candidate{candidate_index:03d}"
        for target_index in target_indices
        for candidate_index in range(int(_protocol(cfg)["requested_k"]))
    ]
    manifest = {
        "version": 2,
        "status": "candidate_pools_ready",
        "validation_boundary": "validation_selected_candidate_pools_no_mpb",
        "target_count": len(merged_targets),
        "selected_variant_count": len(selected_rows),
        "target_indices_sha256": target_id_hash,
        "latent_ids_sha256": _sha256_json(shared_latent_ids),
        "strategy_candidates_path": str(strategy_path),
        "strategy_candidates_sha256": _sha256_file(strategy_path),
        "arms": arm_lineage,
        "joint_structure_deduplication": True,
        "real_mpb_smoke_authorized": True,
        "pilot_replay_authorized": False,
    }
    _atomic_json(run / "candidate_pool_manifest.json", manifest)
    return screen_dir


def _postrepair_replay_config(
    cfg: Mapping[str, Any],
    *,
    candidates_path: Path,
    target_manifest_path: Path,
    run_dir: Path,
    top_m: int,
    n_targets: int,
    stage: str,
) -> dict[str, Any]:
    bundle = load_forward_dataset(str(_dataset_cfg(cfg)["path"]))
    first_primary_band = 2 if int(bundle.nb) >= 2 else 1
    return {
        "experiment": {
            "experiment_id": f"{_protocol(cfg)['id']}_{stage}",
            "family": "research",
            "stage": stage,
        },
        "dataset": {
            "path": str(_dataset_cfg(cfg)["path"]),
            "condition_target": "Wh",
            "output_repr": "rho_wedge",
            "target_split": "val",
        },
        "model": {"name": "mixture_conditional_generator", "seed": 0, "seeds": [0]},
        "inverse": {
            "n_candidates": int(_protocol(cfg)["requested_k"]),
            "surrogate_top_m": int(top_m),
            "threshold": float(_protocol(cfg)["threshold"]),
        },
        "replay": {
            "enabled": True,
            "backend": "real_mpb",
            "require_real_mpb": True,
            "environment": "WSL Ubuntu + conda meep_env",
            "strategy_candidates_path": str(candidates_path),
            "target_indices_path": str(target_manifest_path),
            "n_targets": int(n_targets),
            "selection_unit": "target_index",
            "replay_top_m": int(top_m),
            "primary_band_range": [first_primary_band, int(bundle.nb)],
            "timeout_seconds": float(
                _mapping(cfg.get("pilot"), name="pilot").get("timeout_seconds", 900.0)
            ),
        },
        "solver": dict(_mapping(cfg.get("solver"), name="solver")),
        "metrics": {
            "band_ranges": {
                "bands_1_10": [1, int(bundle.nb)],
                "bands_2_10": [first_primary_band, int(bundle.nb)],
            }
        },
        "output": {"runs_dir": "runs", "run_dir": str(run_dir)},
    }


def _postrepair_fixed_manifest(
    cfg: Mapping[str, Any],
    *,
    positions: Sequence[int],
    path: Path,
    policy: str,
) -> list[int]:
    all_indices = _target_indices(cfg)
    indices = [all_indices[int(position)] for position in positions]
    _atomic_json(
        path,
        {
            "version": 2,
            "selection_unit": "target_index",
            "selection_policy": policy,
            "dataset_sha256": str(_dataset_cfg(cfg)["expected_sha256"]),
            "dataset_split": "val",
            "target_indices": indices,
        },
    )
    return indices


def run_a2_postrepair_smoke_replay(
    config_path: str | Path,
    *,
    run_dir: str | Path,
    solver_fn: SolverFn | None = None,
) -> Path:
    """Run and decide the fixed two-target real-MPB gate for healthy recipes only."""

    from phc2d.replay import run_replay_mpb

    cfg = _read_yaml(config_path)
    run = Path(run_dir)
    manifest_path = run / "candidate_pool_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not bool(manifest.get("real_mpb_smoke_authorized")):
        raise RuntimeError("candidate-health selection did not authorize real-MPB smoke")
    candidates_path = Path(str(manifest["strategy_candidates_path"]))
    _require_hash(
        candidates_path,
        str(manifest["strategy_candidates_sha256"]),
        label="post-repair strategy candidates",
    )
    smoke = _mapping(cfg.get("smoke"), name="smoke")
    positions = [int(item) for item in smoke["target_positions"]]
    target_manifest = run / "real_mpb_smoke_target_manifest.json"
    _postrepair_fixed_manifest(
        cfg,
        positions=positions,
        path=target_manifest,
        policy="fixed_two_target_postrepair_smoke",
    )
    replay_dir = run / "real_mpb_smoke"
    replay_cfg = _postrepair_replay_config(
        cfg,
        candidates_path=candidates_path,
        target_manifest_path=target_manifest,
        run_dir=replay_dir,
        top_m=int(smoke["top_m"]),
        n_targets=len(positions),
        stage="postrepair_real_mpb_smoke",
    )
    replay_path = run / "real_mpb_smoke.yaml"
    replay_path.write_text(
        yaml.safe_dump(replay_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    if solver_fn is None and not _is_wsl_meep_env():
        raise RuntimeError(
            "post-repair real-MPB smoke must run inside WSL Ubuntu with CONDA_DEFAULT_ENV=meep_env"
        )
    run_replay_mpb(replay_path, solver_fn=solver_fn)
    metrics = json.loads((replay_dir / "metrics.json").read_text(encoding="utf-8"))
    strategy_rows = _read_csv(replay_dir / "per_strategy_replay_metrics.csv")
    finite = all(
        math.isfinite(float(value))
        for row in strategy_rows
        for key, value in row.items()
        if key != "strategy_id" and str(value).strip() != ""
    )
    gaps = [
        abs(float(row["surrogate_replay_gap_bands_2_10_pct"])) for row in strategy_rows
    ]
    failures: list[str] = []
    if float(metrics.get("mpb_replay_success_rate", 0.0)) != 1.0:
        failures.append("solver_failure_nonzero")
    if not finite:
        failures.append("non_finite_replay_metric")
    if float(np.mean(gaps)) > float(smoke["mean_surrogate_replay_gap_max_pct"]):
        failures.append(
            f"mean_absolute_surrogate_replay_gap_pct:{float(np.mean(gaps))}>"
            f"{float(smoke['mean_surrogate_replay_gap_max_pct'])}"
        )
    decision = {
        "status": "pass" if not failures else "blocked",
        "evidence_status": "two_target_real_mpb_pipeline_smoke",
        "conclusion_scope": str(smoke["conclusion_scope"]),
        "selected_variant_count": len(strategy_rows),
        "target_count": len(positions),
        "solver_failure_count": 0
        if float(metrics.get("mpb_replay_success_rate", 0.0)) == 1.0
        else 1,
        "metrics_finite": finite,
        "mean_absolute_surrogate_replay_gap_pct": float(np.mean(gaps)),
        "failures": failures,
        "pilot_replay_authorized": not failures,
        "model_superiority_claim_authorized": False,
    }
    _atomic_json(run / "real_mpb_smoke_decision.json", decision)
    return replay_dir


def run_a2_postrepair_pilot_replay(
    config_path: str | Path,
    *,
    run_dir: str | Path,
    solver_fn: SolverFn | None = None,
) -> Path:
    """Run the fixed four-stratum 20-target Top-5 pilot after a passing smoke."""

    from phc2d.replay import run_replay_mpb

    cfg = _read_yaml(config_path)
    run = Path(run_dir)
    smoke_path = run / "real_mpb_smoke_decision.json"
    if not smoke_path.is_file():
        raise FileNotFoundError("real_mpb_smoke_decision.json is required before the pilot")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if str(smoke.get("status")) != "pass" or not bool(smoke.get("pilot_replay_authorized")):
        raise RuntimeError("20-target pilot is blocked by the real-MPB smoke decision")
    manifest = json.loads((run / "candidate_pool_manifest.json").read_text(encoding="utf-8"))
    candidates_path = Path(str(manifest["strategy_candidates_path"]))
    pilot = _mapping(cfg.get("pilot"), name="pilot")
    positions = [int(item) for item in pilot["target_positions"]]
    target_manifest = run / "pilot_target_manifest.json"
    _postrepair_fixed_manifest(
        cfg,
        positions=positions,
        path=target_manifest,
        policy="fixed_four_strata_20_target_postrepair_pilot",
    )
    replay_dir = run / "pilot_replay"
    replay_cfg = _postrepair_replay_config(
        cfg,
        candidates_path=candidates_path,
        target_manifest_path=target_manifest,
        run_dir=replay_dir,
        top_m=int(pilot["top_m"]),
        n_targets=len(positions),
        stage="postrepair_single_seed_20_target_pilot",
    )
    replay_path = run / "pilot_replay.yaml"
    replay_path.write_text(
        yaml.safe_dump(replay_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    if solver_fn is None and not _is_wsl_meep_env():
        raise RuntimeError(
            "post-repair 20-target pilot must run inside WSL Ubuntu with CONDA_DEFAULT_ENV=meep_env"
        )
    return run_replay_mpb(replay_path, solver_fn=solver_fn)


def run_a2_postrepair_promote(config_path: str | Path, *, run_dir: str | Path) -> Path:
    """Aggregate the pilot, add common-FNO diagnostics, and retain the FNO reference."""

    from phc2d.inverse.runner import _build_surrogate, _require_torch
    from phc2d.training import resolve_device

    cfg = _read_yaml(config_path)
    run = Path(run_dir)
    replay_dir = run / "pilot_replay"
    metrics_path = replay_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError("pilot_replay/metrics.json is required before promotion")
    replay_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if str(replay_metrics.get("replay_backend")) != "real_mpb":
        raise RuntimeError("post-repair promotion requires replay_backend=real_mpb")
    if int(replay_metrics.get("targets", 0)) != 20:
        raise RuntimeError("post-repair promotion requires complete 20-target coverage")
    if float(replay_metrics.get("mpb_replay_success_rate", 0.0)) != 1.0:
        raise RuntimeError("post-repair promotion requires solver success rate 1.0")

    selected_rows = _postrepair_selected_rows(run)
    selected_ids = [str(item["variant_id"]) for item in selected_rows]
    target_rows = _read_csv(replay_dir / "per_target_replay_metrics.csv")
    if len(target_rows) != 20 * len(selected_ids):
        raise RuntimeError("post-repair pilot target/strategy coverage is incomplete")
    replay_by_key = {
        (str(row["strategy_id"]), int(row["target_index"])): row for row in target_rows
    }
    candidates = json.loads(
        (run / "screen" / "strategy_candidates.json").read_text(encoding="utf-8")
    )
    pilot_indices = set(
        json.loads((run / "pilot_target_manifest.json").read_text(encoding="utf-8"))[
            "target_indices"
        ]
    )

    variants = {str(item["id"]): item for item in _postrepair_variants(cfg)}
    first_variant = variants[selected_ids[0]]
    first_checkpoint = {
        "epoch": int(selected_rows[0]["epoch"]),
        "path": str(selected_rows[0]["selected_checkpoint_path"]),
    }
    common_config_payload = _postrepair_checkpoint_eval_config(
        cfg,
        variant=first_variant,
        checkpoint=first_checkpoint,
        run_dir=run / "common_fno_diagnostic",
    )
    common_entry = {
        key: value
        for key, value in _mapping(
            cfg.get("common_diagnostic_ranker"),
            name="common_diagnostic_ranker",
        ).items()
        if key not in {"sha256", "read_only", "id", "k_path_sha256"}
    }
    common_config_payload["surrogate"] = common_entry
    common_config_path = run / "common_fno_diagnostic.yaml"
    common_config_path.write_text(
        yaml.safe_dump(common_config_payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    inverse_cfg = load_inverse_run_config(common_config_path)
    torch = _require_torch()
    device_info = resolve_device(inverse_cfg.runtime, torch_module=torch)
    bundle = load_forward_dataset(inverse_cfg.dataset_path)
    candidate_dim = int(np.asarray(bundle.rho).reshape(len(bundle.rho), -1).shape[1])
    common_surrogate = _build_surrogate(
        inverse_cfg,
        input_dim=candidate_dim,
        target_shape=tuple(int(item) for item in np.asarray(bundle.Wh).shape[1:]),
        torch=torch,
        device=device_info.torch_device,
    )
    lineage = _read_csv(run / "candidate_lineage.csv")
    membership: dict[tuple[int, str], set[int]] = {}
    for row in lineage:
        membership.setdefault((int(row["target_index"]), str(row["variant_id"])), set()).add(
            int(row["union_candidate_index"])
        )

    detailed: list[dict[str, Any]] = []
    common_overlap_rows: list[dict[str, Any]] = []
    for target in candidates["targets"]:
        target_index = int(target["target_index"])
        if target_index not in pilot_indices:
            continue
        pool = np.asarray(target["candidates"], dtype=np.float32)
        pool_tensor = torch.as_tensor(pool, dtype=torch.float32, device=device_info.torch_device)
        common_prediction = common_surrogate.predict_tensor(pool_tensor).detach().cpu().numpy()
        target_matrix = np.asarray(bundle.Wh[target_index], dtype=np.float32)
        common_errors = _bands_mre(common_prediction, target_matrix)
        for variant_id in selected_ids:
            ranking = _mapping(
                target["strategy_rankings"][variant_id],
                name=f"strategy_rankings.{variant_id}",
            )
            selected_indices = [int(item) for item in ranking["selected_indices"]]
            primary_predictions = np.asarray(
                ranking["selected_surrogate_predictions"],
                dtype=np.float32,
            )
            primary_errors = _bands_mre(primary_predictions, target_matrix)
            common_selected_errors = common_errors[np.asarray(selected_indices, dtype=np.int64)]
            replay = replay_by_key[(variant_id, target_index)]
            replay_mre = float(replay["best_of_k_replay_mre_bands_2_10_pct"])
            primary_mre = float(np.min(primary_errors) * 100.0)
            common_mre = float(np.min(common_selected_errors) * 100.0)
            detailed.append(
                {
                    "variant_id": variant_id,
                    "target_index": target_index,
                    "replay_mre_pct": replay_mre,
                    "primary_surrogate_mre_pct": primary_mre,
                    "common_fno_mre_pct": common_mre,
                    "primary_replay_gap_pct": replay_mre - primary_mre,
                    "common_fno_replay_gap_pct": replay_mre - common_mre,
                }
            )
            arm_indices = membership[(target_index, variant_id)]
            primary_ranked = [
                int(item) for item in ranking["ranked_indices"] if int(item) in arm_indices
            ]
            common_ranked = sorted(arm_indices, key=lambda item: float(common_errors[item]))
            common_top = common_ranked[: int(_protocol(cfg)["replay_top_m"])]
            primary_top = primary_ranked[: int(_protocol(cfg)["replay_top_m"])]
            common_overlap_rows.append(
                {
                    "comparison": "primary_vs_common_fno",
                    "target_index": target_index,
                    "variant_id": variant_id,
                    "reference_id": "common_fno_diagnostic",
                    "top_m_overlap_count": len(set(primary_top) & set(common_top)),
                    "top_m_overlap_fraction": float(
                        len(set(primary_top) & set(common_top))
                        / max(len(set(primary_top) | set(common_top)), 1)
                    ),
                    "paired_rank_spearman": _spearman_permutation(
                        primary_ranked,
                        common_ranked,
                    ),
                }
            )
    _write_csv(replay_dir / "primary_common_per_target_metrics.csv", detailed)
    existing_overlap = _read_csv(run / "selection_overlap.csv")
    _write_csv(run / "selection_overlap.csv", [*existing_overlap, *common_overlap_rows])

    statistics = _mapping(cfg.get("statistics"), name="statistics")
    seed = int(statistics.get("bootstrap_seed", 0))
    iterations = int(statistics.get("bootstrap_iterations", 10_000))
    selection_by_id = {str(row["variant_id"]): row for row in selected_rows}
    replay_strategy_by_id = {
        str(row["strategy_id"]): row
        for row in _read_csv(replay_dir / "per_strategy_replay_metrics.csv")
    }
    reference_id = str(_mapping(cfg["promotion"], name="promotion")["reference_id"])
    reference_by_target = {
        int(row["target_index"]): float(row["replay_mre_pct"])
        for row in detailed
        if str(row["variant_id"]) == reference_id
    }
    aggregate: list[dict[str, Any]] = []
    for variant_id in selected_ids:
        rows = [row for row in detailed if str(row["variant_id"]) == variant_id]
        replay_values = [float(row["replay_mre_pct"]) for row in rows]
        ci_low, ci_high = _mean_ci(replay_values, seed=seed, iterations=iterations)
        paired_deltas = [
            float(row["replay_mre_pct"]) - reference_by_target[int(row["target_index"])]
            for row in rows
        ]
        paired_ci_low, paired_ci_high = _mean_ci(
            paired_deltas,
            seed=seed,
            iterations=iterations,
        )
        selected = selection_by_id[variant_id]
        sample_metrics = json.loads(
            (run / "candidate_pools" / variant_id / "metrics.json").read_text(
                encoding="utf-8"
            )
        )
        allocated_replay_time = float(
            replay_strategy_by_id[variant_id]["mpb_replay_time_s"]
        )
        total_runtime = (
            float(sample_metrics["wall_time_total_s"])
            + float(selected["common_fno_diagnostic_ranking_time_s"])
            + allocated_replay_time
        )
        aggregate.append(
            {
                "variant_id": variant_id,
                "role_group": str(selected["role_group"]),
                "target_count": len(rows),
                "replay_mre_mean_pct": float(np.mean(replay_values)),
                "replay_mre_ci_low_pct": ci_low,
                "replay_mre_ci_high_pct": ci_high,
                "paired_replay_delta_vs_fno_mean_pct": float(np.mean(paired_deltas)),
                "paired_replay_delta_vs_fno_ci_low_pct": paired_ci_low,
                "paired_replay_delta_vs_fno_ci_high_pct": paired_ci_high,
                "primary_surrogate_mre_mean_pct": float(
                    np.mean([float(row["primary_surrogate_mre_pct"]) for row in rows])
                ),
                "common_fno_mre_mean_pct": float(
                    np.mean([float(row["common_fno_mre_pct"]) for row in rows])
                ),
                "primary_replay_gap_mean_pct": float(
                    np.mean([float(row["primary_replay_gap_pct"]) for row in rows])
                ),
                "common_fno_replay_gap_mean_pct": float(
                    np.mean([float(row["common_fno_replay_gap_pct"]) for row in rows])
                ),
                "solver_success_rate": 1.0,
                "candidate_sampling_time_s": float(sample_metrics["candidate_sampling_time_s"]),
                "primary_surrogate_ranking_time_s": float(
                    sample_metrics["surrogate_ranking_time_s"]
                ),
                "common_fno_diagnostic_ranking_time_s": float(
                    selected["common_fno_diagnostic_ranking_time_s"]
                ),
                "allocated_mpb_replay_time_s": allocated_replay_time,
                "shared_mpb_wall_time_s": float(replay_metrics["mpb_replay_time_s"]),
                "total_runtime_s": total_runtime,
                "peak_gpu_memory_mb": float(sample_metrics["peak_gpu_memory_mb"]),
            }
        )
    _write_csv(run / "pilot_replay_metrics.csv", aggregate)

    reference = next(row for row in aggregate if str(row["variant_id"]) == reference_id)
    challengers = [row for row in aggregate if str(row["variant_id"]) != reference_id]
    if challengers:
        replay_best = min(challengers, key=lambda row: float(row["replay_mre_mean_pct"]))
        overlap = [
            row
            for row in challengers
            if float(row["replay_mre_ci_low_pct"]) <= float(replay_best["replay_mre_ci_high_pct"])
            and float(row["replay_mre_ci_high_pct"]) >= float(replay_best["replay_mre_ci_low_pct"])
        ]
        best_challenger = min(
            overlap,
            key=lambda row: (
                max(
                    abs(float(row["primary_replay_gap_mean_pct"])),
                    abs(float(row["common_fno_replay_gap_mean_pct"])),
                ),
                -float(row["solver_success_rate"]),
                float(row["total_runtime_s"]),
            ),
        )
    else:
        best_challenger = None
    supports_replacement = bool(
        best_challenger is not None
        and float(best_challenger["replay_mre_mean_pct"])
        < float(reference["replay_mre_mean_pct"])
    )
    blocked_ids = [
        str(item["variant_id"])
        for item in json.loads(
            (run / "validation_selected_checkpoints.json").read_text(encoding="utf-8")
        )["variants"]
        if str(item["selection_status"]) != "selected"
    ]
    promotion_cfg = _mapping(cfg.get("promotion"), name="promotion")
    decision = {
        "version": 2,
        "status": "pass",
        "evidence_status": POSTREPAIR_EVIDENCE_STATUS,
        "reference_id": reference_id,
        "reference_retained": True,
        "reference": reference,
        "best_challenger": best_challenger,
        "blocked_candidate_health_ids": blocked_ids,
        "supports_replacing_fno": supports_replacement,
        "conclusion": (
            f"pilot evidence supports challenger {best_challenger['variant_id']}"
            if supports_replacement and best_challenger is not None
            else str(promotion_cfg["no_replacement_message"])
        ),
        "formal_or_multiseed_authorized": False,
        "automatic_100_target_expansion_authorized": False,
    }
    _atomic_json(run / "promotion_decision.json", decision)
    return run


def _surrogate_objective_arm_config(
    cfg: Mapping[str, Any],
    *,
    variant: Mapping[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    training = _mapping(cfg.get("training"), name="training")
    objective = dict(_mapping(training.get("objective"), name="training.objective"))
    objective["cross_surrogate_weight"] = 0.0
    objective.pop("secondary_surrogate", None)
    warm_start = _mapping(training.get("warm_start"), name="training.warm_start")
    model = _mapping(training.get("model"), name="training.model")
    dataset = _dataset_cfg(cfg)
    return {
        "experiment": {
            "experiment_id": f"{_protocol(cfg)['id']}_{variant['id']}_seed0",
            "family": "research",
            "stage": "a2_surrogate_primary_only_objective_bridge",
        },
        "dataset": {
            "path": str(dataset["path"]),
            "condition_target": "Wh",
            "output_repr": "rho_wedge",
            "target_split": "val",
            "target_manifest_path": str(dataset["target_manifest_path"]),
        },
        "model": {
            "name": str(model.get("name", "mixture_conditional_generator")),
            "seed": 0,
            "seeds": [0],
            "mode_count": int(model.get("mode_count", 4)),
            "latent_dim": int(model.get("latent_dim", 32)),
            "hidden_dim": int(model.get("hidden_dim", 256)),
            "depth": int(model.get("depth", 4)),
        },
        "surrogate": _variant_surrogate_config(variant),
        "runtime": dict(_mapping(cfg.get("runtime"), name="runtime")),
        "inverse": {
            "train_epochs": int(training["epochs"]),
            "checkpoint_epochs": [int(item) for item in training["checkpoint_epochs"]],
            "train_batch_size": int(training["batch_size"]),
            "learning_rate": float(training["learning_rate"]),
            "learning_rates": [float(training["learning_rate"])],
            "optimizer": str(training.get("optimizer", "adamw")),
            "weight_decay": float(training.get("weight_decay", 1.0e-5)),
            "generator_checkpoint_path": str(warm_start["path"]),
            "n_candidates": int(_protocol(cfg)["requested_k"]),
            "surrogate_top_m": int(_protocol(cfg)["replay_top_m"]),
            "threshold": float(_protocol(cfg)["threshold"]),
            "candidate_separation": {
                "enabled": True,
                "deduplicate_binary": True,
                "min_hamming": str(_protocol(cfg)["minimum_normalized_hamming"]),
            },
            "allow_mock_generator": False,
            "allow_mock_surrogate": False,
            "ranking": {
                "error_metric": "bands_2_10_mre",
                "objective": objective,
            },
        },
        "evaluation": {
            "selection_split": "val",
            "target_manifest_path": str(dataset["target_manifest_path"]),
            "other_surrogates_role": "read_only_diagnostic_no_backpropagation",
        },
        "replay": {"enabled": False},
        "output": {"runs_dir": "runs", "run_dir": str(run_dir)},
    }


def run_a2_surrogate_derive_objective(
    config_path: str | Path,
    *,
    run_dir: str | Path,
) -> Path:
    """Write primary-only objective configs for selected or all frequency surrogates."""

    cfg = _read_yaml(config_path)
    run = Path(run_dir)
    selection = _mapping(cfg.get("selection"), name="selection")
    variants = {str(item["id"]): item for item in _surrogate_variants(cfg)}
    selection_mode = str(selection.get("mode", "promoted_three"))
    decision_path: Path | None = None
    if selection_mode == "all_frequency_surrogates":
        promoted_ids = list(variants)
        required_count = int(selection.get("required_promoted_arm_count", 7))
        if len(promoted_ids) != required_count:
            raise ValueError(
                f"all-frequency objective derivation expected {required_count} variants, "
                f"found {len(promoted_ids)}"
            )
    else:
        explicit = str(selection.get("promotion_decision_path", "")).strip()
        decision_path = Path(explicit) if explicit else run / "promotion_decision.json"
        if not decision_path.is_file():
            raise FileNotFoundError(f"promotion decision not found: {decision_path}")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if str(decision.get("status")) != "pass":
            raise RuntimeError("objective config derivation requires a passing promotion decision")
        promoted_ids = [str(item) for item in decision["promoted_surrogate_ids"]]
        if len(promoted_ids) != 3 or promoted_ids[0] != "fno2d_band_v2":
            raise ValueError("promotion decision must contain FNO reference plus two challengers")
    objective_root = run / "objective"
    config_root = objective_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    arm_rows: list[dict[str, Any]] = []
    for variant_id in promoted_ids:
        variant = variants[variant_id]
        arm_dir = objective_root / variant_id
        arm_cfg = _surrogate_objective_arm_config(cfg, variant=variant, run_dir=arm_dir)
        arm_config_path = config_root / f"{variant_id}.yaml"
        arm_config_path.write_text(
            yaml.safe_dump(arm_cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        arm_rows.append(
            {
                "surrogate_id": variant_id,
                "kind": str(variant["kind"]),
                "model_name": str(variant["model_name"]),
                "config_path": str(arm_config_path),
                "run_dir": str(arm_dir),
                "seed": 0,
                "cross_surrogate_weight": 0.0,
                "status": "config_ready_not_trained",
            }
        )
    manifest = {
        "version": 1,
        "status": "config_ready_not_trained",
        "evidence_status": SURROGATE_OBJECTIVE_EVIDENCE_STATUS,
        "selection_mode": selection_mode,
        "promotion_decision_path": str(decision_path) if decision_path else None,
        "promotion_decision_sha256": _sha256_file(decision_path) if decision_path else None,
        "arms": arm_rows,
        "historical_dual_surrogate_control_role": "read_only_separate_control",
        "test_accessed_for_selection": False,
    }
    _atomic_json(objective_root / "objective_arm_manifest.json", manifest)
    return objective_root


def run_a2_surrogate_derive_formal(
    config_path: str | Path,
    *,
    run_dir: str | Path,
) -> Path:
    """Write a gated two-model, three-seed formal execution manifest."""

    cfg = _read_yaml(config_path)
    run = Path(run_dir)
    formal = _mapping(cfg.get("formal"), name="formal")
    decision_value = str(formal.get("objective_promotion_decision_path", "")).strip()
    decision_path = Path(decision_value) if decision_value else run / "objective_promotion_decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError(
            "formal derivation requires objective_promotion_decision.json from the seed-0 "
            f"validation MPB pilot: {decision_path}"
        )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    challenger_id = str(decision.get("best_challenger_id", "")).strip()
    if not challenger_id or challenger_id == "fno2d_band_v2":
        raise ValueError("objective promotion decision must identify one non-FNO challenger")
    variants = {str(item["id"]): item for item in _surrogate_variants(cfg)}
    if challenger_id not in variants:
        raise ValueError(f"unknown formal challenger id: {challenger_id}")
    seeds = [int(item) for item in formal.get("seeds", [0, 1, 2])]
    if seeds != [0, 1, 2]:
        raise ValueError("formal surrogate comparison requires seeds [0, 1, 2]")
    test_manifest = Path(str(formal["test_target_manifest_path"]))
    test_manifest_hash = _require_hash(
        test_manifest,
        str(formal["test_target_manifest_sha256"]),
        label="formal test100 target manifest",
    )
    steps: list[dict[str, Any]] = []
    for surrogate_id in ("fno2d_band_v2", challenger_id):
        for seed in seeds:
            steps.append(
                {
                    "surrogate_id": surrogate_id,
                    "seed": seed,
                    "reconstruction_pretrain_required": seed != 0,
                    "objective_finetune_required": True,
                    "validation_checkpoint_selection_required": True,
                    "test_target_manifest_path": str(test_manifest),
                    "requested_k": int(_protocol(cfg)["requested_k"]),
                    "replay_top_m": int(_protocol(cfg)["replay_top_m"]),
                    "mpb_solve_cap": int(formal.get("mpb_solve_cap_per_arm_seed", 500)),
                    "status": "blocked_pending_seed_specific_checkpoint",
                }
            )
    formal_root = run / "formal"
    _atomic_json(
        formal_root / "formal_execution_manifest.json",
        {
            "version": 1,
            "status": "config_ready_execution_blocked",
            "reference_id": "fno2d_band_v2",
            "challenger_id": challenger_id,
            "seeds": seeds,
            "test_target_manifest_path": str(test_manifest),
            "test_target_manifest_sha256": test_manifest_hash,
            "selection_split": "val",
            "final_evaluation_split": "test",
            "test_accessed_for_selection": False,
            "formal_completed": False,
            "steps": steps,
        },
    )
    return formal_root


def run_a2_benchmark(
    config_path: str | Path,
    *,
    stage: str,
    run_dir: str | Path | None = None,
    solver_fn: SolverFn | None = None,
) -> Path:
    """Run one resumable A2 stage: preflight/train/sample/replay/aggregate/smoke."""

    normalized = str(stage).strip().lower()
    cfg = _read_yaml(config_path)
    surrogate_ablation = _is_surrogate_ablation(cfg)
    postrepair_validation = _is_postrepair_validation(cfg)
    if normalized == "preflight":
        return run_a2_preflight(config_path, run_dir=run_dir)
    if run_dir is None:
        raise ValueError(f"--run-dir is required for A2 stage {normalized!r}")
    if postrepair_validation:
        if normalized == "sample":
            return run_a2_postrepair_select(config_path, run_dir=run_dir)
        if normalized == "screen":
            return run_a2_postrepair_screen(config_path, run_dir=run_dir)
        if normalized == "real-smoke":
            return run_a2_postrepair_smoke_replay(
                config_path,
                run_dir=run_dir,
                solver_fn=solver_fn,
            )
        if normalized == "pilot-replay":
            return run_a2_postrepair_pilot_replay(
                config_path,
                run_dir=run_dir,
                solver_fn=solver_fn,
            )
        if normalized == "promote":
            return run_a2_postrepair_promote(config_path, run_dir=run_dir)
        raise ValueError(f"unknown A2 post-repair validation stage: {stage!r}")
    if surrogate_ablation:
        if normalized == "fake-smoke":
            return run_a2_surrogate_smoke_replay(
                config_path,
                run_dir=run_dir,
                real_mpb=False,
                solver_fn=solver_fn,
            )
        if normalized == "real-smoke":
            return run_a2_surrogate_smoke_replay(
                config_path,
                run_dir=run_dir,
                real_mpb=True,
                solver_fn=solver_fn,
            )
        if normalized == "sample":
            return run_a2_surrogate_sample(config_path, run_dir=run_dir)
        if normalized == "screen":
            return run_a2_surrogate_screen(config_path, run_dir=run_dir)
        if normalized == "pilot-replay":
            return run_a2_surrogate_pilot_replay(
                config_path,
                run_dir=run_dir,
                solver_fn=solver_fn,
            )
        if normalized == "promote":
            return run_a2_surrogate_promote(config_path, run_dir=run_dir)
        if normalized == "derive-objective":
            return run_a2_surrogate_derive_objective(config_path, run_dir=run_dir)
        if normalized == "derive-formal":
            return run_a2_surrogate_derive_formal(config_path, run_dir=run_dir)
        raise ValueError(f"unknown A2 surrogate-ablation stage: {stage!r}")
    if normalized == "train":
        return run_a2_train(config_path, run_dir=run_dir)
    if normalized == "sample":
        return run_a2_sample(config_path, run_dir=run_dir)
    if normalized == "pilot-replay":
        return run_a2_replay(config_path, run_dir=run_dir, pilot=True, solver_fn=solver_fn)
    if normalized == "pilot-aggregate":
        return run_a2_pilot_aggregate(config_path, run_dir=run_dir)
    if normalized == "replay":
        return run_a2_replay(config_path, run_dir=run_dir, solver_fn=solver_fn)
    if normalized == "aggregate":
        return run_a2_aggregate(config_path, run_dir=run_dir)
    if normalized == "fake-smoke":
        run_a2_sample(config_path, run_dir=run_dir, fake_models=True, target_limit=2)
        run_a2_replay(config_path, run_dir=run_dir, fake_mpb=True, solver_fn=solver_fn)
        return Path(run_dir) / "replay_fake"
    if normalized == "real-smoke":
        return run_a2_replay(config_path, run_dir=run_dir, smoke=True, solver_fn=solver_fn)
    raise ValueError(f"unknown A2 benchmark stage: {stage!r}")


__all__ = [
    "EVIDENCE_STATUS",
    "METHODS",
    "run_a2_aggregate",
    "run_a2_benchmark",
    "run_a2_preflight",
    "run_a2_pilot_aggregate",
    "run_a2_postrepair_pilot_replay",
    "run_a2_postrepair_preflight",
    "run_a2_postrepair_promote",
    "run_a2_postrepair_screen",
    "run_a2_postrepair_select",
    "run_a2_postrepair_smoke_replay",
    "run_a2_replay",
    "run_a2_sample",
    "run_a2_surrogate_derive_formal",
    "run_a2_surrogate_derive_objective",
    "run_a2_surrogate_pilot_replay",
    "run_a2_surrogate_promote",
    "run_a2_surrogate_sample",
    "run_a2_surrogate_screen",
    "run_a2_surrogate_smoke_replay",
    "run_a2_train",
]
