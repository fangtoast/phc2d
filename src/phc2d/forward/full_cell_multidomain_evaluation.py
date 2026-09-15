"""Frozen-cache six-domain evaluation for EXP-FWD-021.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from phc2d.forward.full_cell_multidomain_contracts import (
    TRAINING_SUCCESS_STATUSES,
    audit_training_runs,
    read_csv,
    sha256_array,
    sha256_file,
    write_csv,
)
from phc2d.forward.full_cell_multidomain_training import forward_full_wh
from phc2d.forward.global_local_bridge_evaluation import cell_metrics, paired_bootstrap
from phc2d.models import (
    build_model,
    describe_model_identity,
    get_model_factory_identity,
    get_model_spec,
)
from phc2d.runs import (
    ensure_figures_dir,
    setup_logging,
    write_metrics,
    write_resolved_config,
    write_results,
    write_run_meta,
)
from phc2d.training.device import DeviceConfig, resolve_device


def evaluation_plan_rows(training_rows: Sequence[Mapping[str, Any]], targets: Sequence[str]) -> list[dict[str, Any]]:
    """Expand 63 checkpoints into a collision-free 378-cell evaluation plan."""

    rows: list[dict[str, Any]] = []
    for training in training_rows:
        for target in targets:
            index = len(rows)
            rows.append(
                {
                    "cell_index": index,
                    "cell_id": f"{training['task_id']}__to__{target}",
                    "task_id": training["task_id"],
                    "regime": training["regime"],
                    "model": training["model"],
                    "seed": training["seed"],
                    "target_domain": target,
                    "checkpoint_path": training["checkpoint_path"],
                    "checkpoint_sha256": training["checkpoint_sha256"],
                    "cell_dir": f"evaluation/cells/{training['regime']}/{training['model']}_s0/{target}",
                }
            )
    if len(rows) != 378 or len({row["cell_id"] for row in rows}) != 378:
        raise RuntimeError("EXP-FWD-021 must contain exactly 378 unique evaluation cells")
    return rows


def _load_campaign(campaign_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    campaign = Path(campaign_dir).resolve()
    payload = yaml.safe_load((campaign / "config.resolved.yaml").read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("resolved campaign config must be a mapping")
    return campaign, dict(payload)


def _atomic_npz(path: Path, values: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, path)


def _torch_load(torch: Any, path: Path, *, map_location: Any = "cpu") -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # pragma: no cover - older torch compatibility
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint payload is not a mapping: {path}")
    return dict(payload)


def freeze_evaluation(campaign_dir: str | Path) -> Path:
    """Freeze 63 accepted checkpoints and read each held-out target once into cache."""

    campaign, profile = _load_campaign(campaign_dir)
    audit_training_runs(campaign / "task_manifest.csv")
    evaluation = campaign / "evaluation"
    if evaluation.exists() and any(evaluation.iterdir()):
        raise FileExistsError(f"evaluation directory is not empty: {evaluation}")
    evaluation.mkdir(parents=True, exist_ok=True)
    ensure_figures_dir(evaluation)
    import torch

    normalizers = {
        row["regime"]: row for row in read_csv(campaign / "normalizer_manifest.csv")
    }
    subsets_by_regime: dict[str, dict[str, str]] = defaultdict(dict)
    for subset in read_csv(campaign / "subset_manifest.csv"):
        subsets_by_regime[subset["regime"]][subset["domain"]] = subset["indices_sha256"]
    training_rows: list[dict[str, Any]] = []
    for row in read_csv(campaign / "task_manifest.csv"):
        run = campaign / row["run_dir"]
        result = json.loads((run / "results.json").read_text(encoding="utf-8"))
        checkpoint = run / "model.pt"
        if result.get("status") not in TRAINING_SUCCESS_STATUSES:
            raise RuntimeError(f"unaccepted training status for {row['task_id']}")
        if result.get("checkpoint_sha256") != sha256_file(checkpoint):
            raise RuntimeError(f"checkpoint result hash mismatch for {row['task_id']}")
        payload = _torch_load(torch, checkpoint)
        provenance = dict(payload.get("multidomain_provenance") or {})
        identity = dict(payload.get("model_identity") or {})
        if payload.get("model_name") != row["model"] or provenance.get("regime") != row["regime"]:
            raise RuntimeError(f"checkpoint task identity mismatch for {row['task_id']}")
        if identity.get("factory_function_identity") != get_model_factory_identity(row["model"]):
            raise RuntimeError(f"checkpoint factory identity mismatch for {row['task_id']}")
        if provenance.get("subset_hashes") != subsets_by_regime[row["regime"]]:
            raise RuntimeError(f"checkpoint subset provenance mismatch for {row['task_id']}")
        normalizer = normalizers[row["regime"]]
        for key in ("normalizer_file_sha256", "target_mean_sha256", "target_std_sha256"):
            if provenance.get(key) != normalizer[key]:
                raise RuntimeError(f"checkpoint {key} mismatch for {row['task_id']}")
        training_rows.append(
            {
                **row,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "training_status": result["status"],
            }
        )
    if len({row["checkpoint_sha256"] for row in training_rows}) != 63:
        raise RuntimeError("checkpoint hash collision detected")
    write_csv(evaluation / "checkpoint_manifest.csv", training_rows)

    import h5py

    dataset_rows = {row["domain"]: row for row in read_csv(campaign / "dataset_manifest.csv")}
    cache_rows: list[dict[str, Any]] = []
    targets = list(profile["evaluation"]["targets"])
    for domain in targets:
        dataset = dataset_rows[domain]
        source = Path(dataset["dataset_path"])
        with h5py.File(source, "r") as handle:
            indices = np.asarray(handle["idx_test"], dtype=np.int64)
            inputs = np.asarray(handle[dataset["grid_field"]][indices], dtype=np.float32)
            targets_array = np.asarray(handle["Wh"][indices], dtype=np.float32)
            sample_ids = np.asarray(handle["sample_id"][indices]).astype("U")
            k_points = np.asarray(handle["k_points"], dtype=np.float32)
        if not np.isfinite(inputs).all() or not np.isfinite(targets_array).all():
            raise RuntimeError(f"non-finite held-out cache values for {domain}")
        relative = Path("cache") / f"{domain}.npz"
        _atomic_npz(
            evaluation / relative,
            {
                "inputs": inputs,
                "targets": targets_array,
                "sample_ids": sample_ids,
                "indices": indices,
                "k_points": k_points,
            },
        )
        cache_rows.append(
            {
                "target_domain": domain,
                "sample_count": len(indices),
                "cache_path": str(relative).replace("\\", "/"),
                "cache_sha256": sha256_file(evaluation / relative),
                "indices_sha256": sha256_array(indices, dtype=np.int64),
                "sample_ids_sha256": sha256_array(sample_ids.astype("S")),
                "k_points_sha256": sha256_array(k_points, dtype=np.float32),
                "source_dataset_sha256": dataset["dataset_sha256"],
                "original_test_target_read_count": 1,
            }
        )
    write_csv(evaluation / "cache_manifest.csv", cache_rows)
    plan = evaluation_plan_rows(training_rows, targets)
    write_csv(evaluation / "evaluation_manifest.csv", plan)
    write_resolved_config(
        evaluation, yaml.safe_dump(profile, sort_keys=False, allow_unicode=True)
    )
    write_run_meta(
        evaluation,
        {
            "script": "python -m phc2d.forward.full_cell_multidomain_campaign freeze-evaluation",
            "experiment_id": "EXP-FWD-021",
            "checkpoint_count": 63,
            "evaluation_cell_count": 378,
        },
    )
    (evaluation / "run.log").write_text(
        "Frozen 63 checkpoints and six read-once held-out test caches for 378 cells.\n",
        encoding="utf-8",
    )
    write_csv(evaluation / "history.csv", [{"phase": "freeze_evaluation", "status": "complete"}])
    write_metrics(
        evaluation,
        {
            "checkpoint_count": 63,
            "test_cache_count": 6,
            "evaluation_cell_count": 378,
            "original_test_target_reads_per_domain": 1,
        },
    )
    write_results(
        evaluation,
        {
            "status": "evaluation_frozen_pending_cells",
            "scientific_boundary": "seed0_exploratory_surrogate_only",
            "model_promotion": False,
            "physical_claim": False,
        },
    )
    return evaluation


def _predict_batches(
    model: Any,
    inputs: np.ndarray,
    *,
    layout: str,
    batch_size: int,
    device: Any,
    k_points: Any,
    mean: Any,
    std: Any,
    model_spec: Any,
    torch: Any,
) -> tuple[np.ndarray, float]:
    predictions: list[np.ndarray] = []
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size]
            if layout == "flat":
                batch = batch.reshape(len(batch), -1)
            tensor = torch.as_tensor(batch, dtype=torch.float32, device=device)
            normalized = forward_full_wh(
                model,
                tensor,
                model_spec=model_spec,
                k_points=k_points,
                lane_index=None,
                torch=torch,
            )
            if not bool(torch.isfinite(normalized).all().item()):
                raise RuntimeError("non-finite evaluation prediction")
            decoded = normalized * std + mean
            predictions.append(decoded.detach().cpu().numpy())
    return np.concatenate(predictions, axis=0), time.perf_counter() - started


def run_evaluation_task(
    campaign_dir: str | Path,
    cell_index: int,
    *,
    device_override: str | None = None,
) -> Path:
    """Evaluate one manifest-addressed checkpoint-to-target cell from frozen cache."""

    campaign, profile = _load_campaign(campaign_dir)
    evaluation = campaign / "evaluation"
    rows = [
        row
        for row in read_csv(evaluation / "evaluation_manifest.csv")
        if int(row["cell_index"]) == int(cell_index)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"cell_index {cell_index} must resolve exactly once")
    row = rows[0]
    output = campaign / row["cell_dir"]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"evaluation cell directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    ensure_figures_dir(output)
    setup_logging(
        log_file=output / "run.log",
        run_meta={
            "script": "python -m phc2d.forward.full_cell_multidomain_campaign run-evaluation-task",
            "experiment_id": "EXP-FWD-021",
            "cell_id": row["cell_id"],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_node": os.environ.get("SLURM_JOB_NODELIST"),
        },
    )
    checkpoint_path = Path(row["checkpoint_path"])
    if sha256_file(checkpoint_path) != row["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint hash mismatch for {row['cell_id']}")
    cache_rows = [
        item
        for item in read_csv(evaluation / "cache_manifest.csv")
        if item["target_domain"] == row["target_domain"]
    ]
    if len(cache_rows) != 1:
        raise RuntimeError("target cache must resolve exactly once")
    cache_row = cache_rows[0]
    cache_path = evaluation / cache_row["cache_path"]
    if sha256_file(cache_path) != cache_row["cache_sha256"]:
        raise RuntimeError(f"cache hash mismatch for {row['target_domain']}")

    import torch

    runtime = dict(profile["runtime"])
    if device_override is not None:
        runtime["device"] = device_override
        runtime["require_accelerator"] = str(device_override).startswith("cuda")
    device_info = resolve_device(DeviceConfig.from_mapping(runtime), torch_module=torch)
    device = device_info.torch_device
    checkpoint = _torch_load(torch, checkpoint_path, map_location="cpu")
    if checkpoint.get("model_name") != row["model"]:
        raise RuntimeError("checkpoint model name mismatch")
    provenance = dict(checkpoint.get("multidomain_provenance") or {})
    if provenance.get("regime") != row["regime"]:
        raise RuntimeError("checkpoint regime mismatch")
    model_spec = get_model_spec(row["model"])
    model_params = dict(checkpoint["model_params"])
    model_kwargs = {
        **model_params,
        "input_shape": tuple(checkpoint["input_shape"]),
        "output_shape": tuple(checkpoint["output_shape"]),
    }
    k_points_np = np.asarray(checkpoint["k_points"], dtype=np.float32)
    if sha256_array(k_points_np, dtype=np.float32) != checkpoint["k_points_sha256"]:
        raise RuntimeError("checkpoint k-point hash mismatch")
    if model_spec.accepts_k_points:
        model_kwargs["k_points"] = k_points_np
    model = build_model(row["model"], **model_kwargs)
    identity = describe_model_identity(
        model, row["model"], model_spec, model_kwargs=model_params
    )
    expected_identity = dict(checkpoint["model_identity"])
    for key in ("factory_function_identity", "state_dict_schema_hash", "architecture_fingerprint"):
        if identity[key] != expected_identity[key]:
            raise RuntimeError(f"checkpoint model identity mismatch: {key}")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    mean = torch.as_tensor(checkpoint["target_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(checkpoint["target_std"], dtype=torch.float32, device=device)
    if sha256_array(mean.detach().cpu().numpy(), dtype=np.float32) != provenance["target_mean_sha256"]:
        raise RuntimeError("checkpoint normalizer mean hash mismatch")
    if sha256_array(std.detach().cpu().numpy(), dtype=np.float32) != provenance["target_std_sha256"]:
        raise RuntimeError("checkpoint normalizer std hash mismatch")
    with np.load(cache_path) as cache:
        inputs = np.asarray(cache["inputs"], dtype=np.float32)
        targets = np.asarray(cache["targets"], dtype=np.float32)
        sample_ids = np.asarray(cache["sample_ids"]).astype("U")
        cache_k_points = np.asarray(cache["k_points"], dtype=np.float32)
    if not np.array_equal(cache_k_points, k_points_np):
        raise RuntimeError("cache and checkpoint k-point grids differ")
    predictions, elapsed = _predict_batches(
        model,
        inputs,
        layout=model_spec.input_layout,
        batch_size=int(profile["training"]["micro_batch_size"]),
        device=device,
        k_points=torch.as_tensor(k_points_np, dtype=torch.float32, device=device),
        mean=mean,
        std=std,
        model_spec=model_spec,
        torch=torch,
    )
    metrics, per_band, per_k, arrays = cell_metrics(predictions, targets)
    for item in per_band:
        item.update({"cell_id": row["cell_id"], "target_domain": row["target_domain"]})
    for item in per_k:
        item.update({"cell_id": row["cell_id"], "target_domain": row["target_domain"]})
    write_csv(output / "per_band_metrics.csv", per_band)
    write_csv(output / "per_k_metrics.csv", per_k)
    _atomic_npz(
        output / "sample_metrics.npz",
        {
            **arrays,
            "sample_ids": sample_ids,
        },
    )
    write_resolved_config(
        output,
        yaml.safe_dump(
            {**profile, "active_evaluation_cell": row},
            sort_keys=False,
            allow_unicode=True,
        ),
    )
    throughput = len(inputs) / max(elapsed, 1.0e-12)
    write_csv(
        output / "history.csv",
        [{"phase": "evaluation", "sample_count": len(inputs), "elapsed_s": elapsed}],
    )
    write_metrics(
        output,
        {
            **metrics,
            "sample_count": len(inputs),
            "eval_time_s": elapsed,
            "throughput_samples_per_s": throughput,
            "inference_time_per_sample_ms": 1000.0 / max(throughput, 1.0e-12),
            "checkpoint_sha256": row["checkpoint_sha256"],
            "cache_sha256": cache_row["cache_sha256"],
            "model_identity": identity,
            "normalizer_file_sha256": provenance["normalizer_file_sha256"],
        },
    )
    write_results(
        output,
        {
            "status": "evaluation_cell_success",
            "experiment_id": "EXP-FWD-021",
            "cell_id": row["cell_id"],
            "regime": row["regime"],
            "model": row["model"],
            "target_domain": row["target_domain"],
            "seed": int(row["seed"]),
            "checkpoint_sha256": row["checkpoint_sha256"],
            "cache_sha256": cache_row["cache_sha256"],
            "per_band_record_count": len(per_band),
            "scientific_boundary": "seed0_exploratory_surrogate_only",
            "model_promotion": False,
            "physical_claim": False,
        },
    )
    return output


def evaluation_status(campaign_dir: str | Path) -> dict[str, Any]:
    """Return a read-only completeness snapshot and fail on corrupt completed cells."""

    campaign, _ = _load_campaign(campaign_dir)
    manifest = read_csv(campaign / "evaluation" / "evaluation_manifest.csv")
    completed = 0
    missing: list[str] = []
    for row in manifest:
        cell = campaign / row["cell_dir"]
        result_path = cell / "results.json"
        if not result_path.is_file():
            missing.append(row["cell_id"])
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "evaluation_cell_success":
            raise RuntimeError(f"failed evaluation cell: {row['cell_id']}")
        per_band = read_csv(cell / "per_band_metrics.csv")
        if len(per_band) != 10 or {int(item["band_one_based"]) for item in per_band} != set(range(1, 11)):
            raise RuntimeError(f"invalid per-band records: {row['cell_id']}")
        metrics = json.loads((cell / "metrics.json").read_text(encoding="utf-8"))
        numeric = [metrics[key] for key in ("band1_mre_pct", "bands_2_10_mre_pct", "bands_1_10_mre_pct")]
        if not np.isfinite(np.asarray(numeric, dtype=np.float64)).all():
            raise RuntimeError(f"non-finite evaluation metrics: {row['cell_id']}")
        completed += 1
    return {
        "status": "evaluation_complete" if completed == 378 else "evaluation_incomplete",
        "expected_cells": 378,
        "completed_cells": completed,
        "missing_cells": missing,
    }


def _transfer_group(profile: Mapping[str, Any], regime: str, target: str) -> str:
    sources = set(profile["regimes"][regime]["domains"])
    if target in sources:
        return "training_source_domain"
    family, lane = target.rsplit("_", 1)
    source_families = {item.rsplit("_", 1)[0] for item in sources}
    source_lanes = {item.rsplit("_", 1)[1] for item in sources}
    if family in source_families and lane not in source_lanes:
        return "cross_lane"
    if family not in source_families and lane not in source_lanes:
        return "double_shift"
    return "unseen_domain"


def finalize_evaluation(campaign_dir: str | Path) -> Path:
    """Finalize only a complete finite 63-by-6 matrix with paired evidence."""

    campaign, profile = _load_campaign(campaign_dir)
    status = evaluation_status(campaign)
    if status["status"] != "evaluation_complete":
        raise RuntimeError(f"cannot finalize incomplete evaluation: {status['completed_cells']}/378")
    evaluation = campaign / "evaluation"
    rows: list[dict[str, Any]] = []
    sample_paths: dict[tuple[str, str, str], Path] = {}
    for item in read_csv(evaluation / "evaluation_manifest.csv"):
        cell = campaign / item["cell_dir"]
        metrics = json.loads((cell / "metrics.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "regime": item["regime"],
                "model": item["model"],
                "target_domain": item["target_domain"],
                "transfer_group": _transfer_group(profile, item["regime"], item["target_domain"]),
                "band1_mre_pct": metrics["band1_mre_pct"],
                "bands_2_10_mre_pct": metrics["bands_2_10_mre_pct"],
                "bands_1_10_mre_pct": metrics["bands_1_10_mre_pct"],
                "throughput_samples_per_s": metrics["throughput_samples_per_s"],
            }
        )
        sample_paths[(item["regime"], item["model"], item["target_domain"])] = cell / "sample_metrics.npz"
    write_csv(evaluation / "cell_matrix.csv", rows)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["regime"], row["model"])].append(row)
    macro_rows: list[dict[str, Any]] = []
    for (regime, model), values in grouped.items():
        if len(values) != 6:
            raise RuntimeError(f"macro group is incomplete: {regime}/{model}")
        macro_rows.append(
            {
                "regime": regime,
                "model": model,
                "target_domain_count": 6,
                "macro_weighting": "equal_target_domain",
                "band1_mre_pct": float(np.mean([row["band1_mre_pct"] for row in values])),
                "bands_2_10_mre_pct": float(np.mean([row["bands_2_10_mre_pct"] for row in values])),
                "bands_1_10_mre_pct": float(np.mean([row["bands_1_10_mre_pct"] for row in values])),
            }
        )
    write_csv(evaluation / "six_domain_macro.csv", macro_rows)
    bootstrap_rows: list[dict[str, Any]] = []
    reference = str(profile["evaluation"]["paired_bootstrap_reference_model"])
    replicates = int(profile["evaluation"]["bootstrap_replicates"])
    for regime in profile["regimes"]:
        for target in profile["evaluation"]["targets"]:
            with np.load(sample_paths[(regime, reference, target)]) as payload:
                baseline = np.asarray(payload["sample_mre_bands_2_10_pct"], dtype=np.float64)
                baseline_ids = np.asarray(payload["sample_ids"]).astype("U")
            for model in profile["models"]:
                if model == reference:
                    continue
                with np.load(sample_paths[(regime, model, target)]) as payload:
                    candidate = np.asarray(payload["sample_mre_bands_2_10_pct"], dtype=np.float64)
                    candidate_ids = np.asarray(payload["sample_ids"]).astype("U")
                if not np.array_equal(candidate_ids, baseline_ids):
                    raise RuntimeError(f"paired sample identity mismatch: {regime}/{model}/{target}")
                bootstrap_rows.append(
                    {
                        "regime": regime,
                        "target_domain": target,
                        "candidate_model": model,
                        "reference_model": reference,
                        **paired_bootstrap(
                            candidate,
                            baseline,
                            key=f"EXP-FWD-021:{regime}:{model}:{target}",
                            replicates=replicates,
                        ),
                    }
                )
    write_csv(evaluation / "paired_bootstrap.csv", bootstrap_rows)
    ranks_210 = sorted(macro_rows, key=lambda row: float(row["bands_2_10_mre_pct"]))
    ranks_110 = sorted(macro_rows, key=lambda row: float(row["bands_1_10_mre_pct"]))
    conflict = (
        (ranks_210[0]["regime"], ranks_210[0]["model"])
        != (ranks_110[0]["regime"], ranks_110[0]["model"])
    )
    training_metrics = []
    for task in read_csv(campaign / "task_manifest.csv"):
        training_metrics.append(
            json.loads((campaign / task["run_dir"] / "metrics.json").read_text(encoding="utf-8"))
        )
    summary = {
        "status": "finalized_seed0_exploratory_surrogate_evidence",
        "training_task_count": 63,
        "evaluation_cell_count": 378,
        "per_band_record_count": 3780,
        "six_domain_macro_weighting": "equal_target_domain",
        "bands_2_10_best": {"regime": ranks_210[0]["regime"], "model": ranks_210[0]["model"]},
        "bands_1_10_best": {"regime": ranks_110[0]["regime"], "model": ranks_110[0]["model"]},
        "ranking_conflict": conflict,
        "seed_count": 1,
        "sample_bootstrap_is_not_cross_seed_stability": True,
        "scientific_boundary": "seed0_exploratory_surrogate_only",
        "model_promotion": False,
        "physical_claim": False,
    }
    write_metrics(
        evaluation,
        {
            **summary,
            "training_time_s_total": float(sum(float(item["train_time_s"]) for item in training_metrics)),
            "peak_gpu_memory_mb_max": float(max(float(item["peak_gpu_memory_mb"]) for item in training_metrics)),
        },
    )
    write_results(evaluation, summary)
    write_csv(
        evaluation / "history.csv",
        [
            {"phase": "freeze_evaluation", "status": "complete"},
            {"phase": "finalize", "status": "complete", "evaluation_cells": 378},
        ],
    )
    with (evaluation / "run.log").open("a", encoding="utf-8") as handle:
        handle.write("Finalized complete 63-by-6 seed0 exploratory surrogate matrix.\n")
    return evaluation


__all__ = [
    "evaluation_plan_rows",
    "evaluation_status",
    "finalize_evaluation",
    "freeze_evaluation",
    "run_evaluation_task",
]
