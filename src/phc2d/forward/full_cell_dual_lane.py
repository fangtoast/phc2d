"""Balanced binary/graded Full-Wh training for the dual-lane operator.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from phc2d.forward.full_cell import (
    _FullCellH5Dataset,
    _evaluate,
    _epoch_100_continue_decision,
    _git_commit,
    _plot_history,
    _read_contract,
    _sha256_checkpoint,
    _write_history,
)
from phc2d.models import build_model, describe_model_identity, get_model_spec
from phc2d.runs import (
    RUN_LOG_NAME,
    ensure_figures_dir,
    make_run_dir,
    setup_logging,
    validate_run_artifacts,
    write_metrics,
    write_resolved_config,
    write_results,
    write_run_meta,
)
from phc2d.training import DeviceConfig, resolve_device


LANES = ("binary", "graded")


def _validate_and_resolve(
    config: Mapping[str, Any],
    *,
    seed: int,
    admission_probe: bool,
    head_kind: str,
) -> dict[str, Any]:
    experiment = dict(config.get("experiment") or {})
    datasets = dict(config.get("datasets") or {})
    representation = dict(config.get("representation") or {})
    model_cfg = dict(config.get("model") or {})
    training = dict(config.get("training") or {})
    runtime = dict(config.get("runtime") or {})
    output = dict(config.get("output") or {})
    if set(datasets) != set(LANES):
        raise ValueError("datasets must contain exactly binary and graded")
    if int(seed) not in {0, 1, 2}:
        raise ValueError("dual-lane seeds are fixed to 0, 1, 2")
    if head_kind not in {"dense", "k_conditioned"}:
        raise ValueError("head_kind must be dense or k_conditioned")
    model_name = str(model_cfg.get("name", "dual_lane_band_operator_v1"))
    spec = get_model_spec(model_name)
    if not spec.accepts_lane_id or spec.input_layout != "image" or spec.output_target != "Wh":
        raise ValueError(f"model {model_name!r} is not a dual-lane Full-Wh image model")
    micro_batch = int(training.get("micro_batch_size", 64))
    if micro_batch <= 0 or micro_batch % 2:
        raise ValueError("dual-lane micro_batch_size must be a positive even number")
    if int(training.get("effective_batch_size", micro_batch)) != micro_batch:
        raise ValueError("dual-lane runner currently requires effective_batch_size=micro_batch_size")
    if str(training.get("optimizer", "")).lower() != "adamw":
        raise ValueError("dual-lane training requires AdamW")
    if str(runtime.get("precision", "")).lower() != "fp32":
        raise ValueError("dual-lane training requires FP32")
    if bool(runtime.get("amp_enabled")) or bool(runtime.get("tf32_enabled")):
        raise ValueError("dual-lane training forbids AMP and TF32")
    max_epochs = int(training.get("max_epochs", 100))
    evidence_status = str(experiment.get("evidence_status", ""))
    short_screen = evidence_status == "single_seed_short_budget_screen"
    if short_screen:
        if admission_probe:
            raise ValueError("short-budget screening and admission_probe are mutually exclusive")
        if int(seed) != 0 or max_epochs != 20:
            raise ValueError("dual-lane short-budget screening requires seed=0 and max_epochs=20")
        if bool(training.get("epoch_100_continue_gate", False)):
            raise ValueError("dual-lane short-budget screening forbids the epoch-100 continuation gate")
        if int(training.get("early_stopping_patience", 0)) != 0:
            raise ValueError("dual-lane short-budget screening requires early_stopping_patience=0")
    elif not admission_probe and max_epochs not in {100, 200}:
        raise ValueError("formal dual-lane max_epochs must be 100 or 200")
    gate_enabled = bool(training.get("epoch_100_continue_gate", False))
    if not admission_probe and gate_enabled:
        if max_epochs != 200:
            raise ValueError("dual-lane epoch-100 continuation gate requires max_epochs=200")
        if float(training.get("continue_gate_min_relative_improvement", 0.005)) != 0.005:
            raise ValueError("preregistered continuation threshold must equal 0.005")
        if int(training.get("early_stopping_patience", 0)) != 15:
            raise ValueError("dual-lane epoch-100 continuation requires patience 15")
    params = dict(model_cfg.get("params") or {})
    params["head_kind"] = head_kind
    experiment_id = str(experiment.get("experiment_id", "full_cell_forward_upgrade_dual_lane_10k_v1"))
    return {
        "experiment": {
            **experiment,
            "experiment_id": experiment_id,
            "lane": "joint",
            "seed": int(seed),
            "admission_probe": bool(admission_probe),
            "formal_training_started": not admission_probe,
            "mpb_meep_used": False,
        },
        "datasets": {lane: dict(datasets[lane]) for lane in LANES},
        "representation": {**representation, "layout": "image"},
        "model": {"name": model_name, "params": params},
        "training": {
            **training,
            "seed": int(seed),
            "max_epochs": 1 if admission_probe else max_epochs,
            "max_train_batches": 1 if admission_probe else None,
        },
        "runtime": runtime,
        "output": output,
    }


def _lane_contracts(resolved: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for lane in LANES:
        lane_resolved = {
            "dataset": resolved["datasets"][lane],
            "representation": resolved["representation"],
            "training": resolved["training"],
        }
        contracts[lane] = _read_contract(lane_resolved)
    reference = contracts["binary"]
    graded = contracts["graded"]
    for key in ("grid_shape", "target_shape", "k_points_sha256"):
        if reference[key] != graded[key]:
            raise ValueError(f"dual-lane {key} mismatch: {reference[key]!r} != {graded[key]!r}")
    if reference["k_points"] is None:
        raise ValueError("dual-lane datasets must contain k_points")
    return contracts


def _dataset(contract: Mapping[str, Any], resolved: Mapping[str, Any], indices: Any) -> Any:
    return _FullCellH5Dataset(
        path=contract["path"],
        indices=indices,
        grid_field=contract["grid_field"],
        layout="image",
        sampling=str(resolved["representation"].get("sampling", "continuous")),
        threshold=resolved["representation"].get("threshold"),
        target_mean=contract["target_mean"],
        target_std=contract["target_std"],
    )


def run_full_cell_dual_lane(
    config_path: Path,
    *,
    seed: int,
    head_kind: str,
    run_dir: Path,
    admission_probe: bool = False,
) -> Path:
    """Train or probe one balanced dual-lane model checkpoint."""

    config_file = Path(config_path)
    payload = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("dual-lane config root must be a mapping")
    resolved = _validate_and_resolve(
        payload,
        seed=seed,
        admission_probe=admission_probe,
        head_kind=head_kind,
    )
    out = make_run_dir(
        str(resolved["experiment"]["experiment_id"]),
        runs_dir=resolved["output"].get("runs_dir", "runs"),
        run_dir=run_dir,
    )
    logger = setup_logging(
        log_file=out / RUN_LOG_NAME,
        run_meta={
            "script": "python -m phc2d.forward.full_cell_dual_lane",
            "config_path": str(config_file),
            "model": resolved["model"]["name"],
            "head_kind": head_kind,
            "seed": int(seed),
            "validation_boundary": "dual_lane_admission_probe" if admission_probe else "dual_lane_training",
        },
    )
    write_resolved_config(out, yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))
    ensure_figures_dir(out)

    import torch
    from torch.utils.data import DataLoader

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device_info = resolve_device(DeviceConfig.from_mapping(resolved["runtime"]), torch_module=torch)
    device = device_info.torch_device
    contracts = _lane_contracts(resolved)
    spec = get_model_spec(str(resolved["model"]["name"]))
    reference = contracts["binary"]
    model = build_model(
        spec.name,
        input_shape=reference["grid_shape"],
        output_shape=reference["target_shape"],
        k_points=reference["k_points"],
        **dict(resolved["model"]["params"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(resolved["training"]["learning_rate"]),
        weight_decay=float(resolved["training"].get("weight_decay", 1.0e-5)),
    )
    loss_fn = torch.nn.MSELoss()
    half_batch = int(resolved["training"]["micro_batch_size"]) // 2
    loader_kwargs = {
        "batch_size": half_batch,
        "num_workers": int(device_info.num_workers),
        "pin_memory": bool(device_info.pin_memory),
    }
    train_loaders = {
        lane: DataLoader(
            _dataset(contracts[lane], resolved, contracts[lane]["selected_train"]),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + lane_index),
            **loader_kwargs,
        )
        for lane_index, lane in enumerate(LANES)
    }
    val_datasets = {
        lane: _dataset(contracts[lane], resolved, contracts[lane]["idx_val"])
        for lane in LANES
    }
    val_loaders = {
        lane: DataLoader(val_datasets[lane], shuffle=False, **loader_kwargs)
        for lane in LANES
    }
    means = {
        lane: torch.as_tensor(contracts[lane]["target_mean"], dtype=torch.float32, device=device)
        for lane in LANES
    }
    stds = {
        lane: torch.as_tensor(contracts[lane]["target_std"], dtype=torch.float32, device=device)
        for lane in LANES
    }
    k_points = torch.as_tensor(reference["k_points"], dtype=torch.float32, device=device)

    history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_epoch = 0
    stale_epochs = 0
    optimizer_steps = 0
    samples_seen = 0
    wall_start = time.perf_counter()
    train_start = time.perf_counter()
    gate_enabled = bool(resolved["training"].get("epoch_100_continue_gate", False)) and not admission_probe
    gate_record: dict[str, Any] | None = None
    epoch100_snapshot: Path | None = None
    for epoch in range(1, int(resolved["training"]["max_epochs"]) + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_sum = 0.0
        train_count = 0
        for batch_index, ((xb, yb), (xg, yg)) in enumerate(
            zip(train_loaders["binary"], train_loaders["graded"], strict=True),
            start=1,
        ):
            inputs = torch.cat((xb, xg), dim=0).to(device, non_blocking=True)
            targets = torch.cat((yb, yg), dim=0).to(device, non_blocking=True)
            lane_id = torch.cat(
                (
                    torch.zeros(xb.shape[0], dtype=torch.long),
                    torch.ones(xg.shape[0], dtype=torch.long),
                )
            ).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs, k_points=k_points, lane_id=lane_id)
            loss = loss_fn(prediction, targets)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"non-finite dual-lane loss at epoch {epoch}, batch {batch_index}")
            loss.backward()
            if any(
                parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
                for parameter in model.parameters()
            ):
                raise RuntimeError(f"non-finite dual-lane gradient at epoch {epoch}, batch {batch_index}")
            optimizer.step()
            optimizer_steps += 1
            batch_count = int(inputs.shape[0])
            train_sum += float(loss.detach().cpu()) * batch_count
            train_count += batch_count
            samples_seen += batch_count
            if resolved["training"].get("max_train_batches") is not None:
                break
        evaluations = {
            lane: _evaluate(
                model,
                val_loaders[lane],
                device=device,
                mean=means[lane],
                std=stds[lane],
                model_spec=spec,
                k_points=k_points,
                lane_index=lane_index,
                torch=torch,
            )
            for lane_index, lane in enumerate(LANES)
        }
        score = float(np.mean([evaluations[lane]["bands_2_10_mre_pct"] for lane in LANES]))
        row = {
            "epoch": epoch,
            "train_loss": train_sum / max(train_count, 1),
            "val_loss": float(np.mean([evaluations[lane]["normalized_mse"] for lane in LANES])),
            "val_bands_2_10_mre_pct": score,
            "binary_val_bands_2_10_mre_pct": evaluations["binary"]["bands_2_10_mre_pct"],
            "graded_val_bands_2_10_mre_pct": evaluations["graded"]["bands_2_10_mre_pct"],
            "learning_rate": float(resolved["training"]["learning_rate"]),
            "epoch_time_s": time.perf_counter() - epoch_start,
            "wall_time_s": time.perf_counter() - wall_start,
            "optimizer_steps": optimizer_steps,
            "samples_seen": samples_seen,
        }
        history.append(row)
        improved = score < best_score
        if improved:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            checkpoint_tmp = out / "model.pt.tmp"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_name": spec.name,
                    "model_params": dict(resolved["model"]["params"]),
                    "model_identity": describe_model_identity(
                        model, spec.name, spec, model_kwargs=resolved["model"]["params"]
                    ),
                    "input_shape": reference["grid_shape"],
                    "output_shape": reference["target_shape"],
                    "k_points": torch.as_tensor(reference["k_points"], dtype=torch.float32),
                    "k_points_sha256": reference["k_points_sha256"],
                    "lane_mapping": {"binary": 0, "graded": 1},
                    "target_normalizers": {
                        lane: {
                            "mean": torch.as_tensor(contracts[lane]["target_mean"]),
                            "std": torch.as_tensor(contracts[lane]["target_std"]),
                        }
                        for lane in LANES
                    },
                    "epoch": epoch,
                    "validation_equal_lane_mre_pct": best_score,
                    "resolved_config": resolved,
                },
                checkpoint_tmp,
            )
            os.replace(checkpoint_tmp, out / "model.pt")
        else:
            stale_epochs += 1
        _write_history(out / "history.csv", history)
        logger.info(
            "epoch=%d train_loss=%.6g binary_mre=%.6g graded_mre=%.6g equal_lane_mre=%.6g",
            epoch,
            row["train_loss"],
            row["binary_val_bands_2_10_mre_pct"],
            row["graded_val_bands_2_10_mre_pct"],
            score,
        )
        if gate_enabled and epoch == 100:
            gate_record = _epoch_100_continue_decision(
                history,
                threshold=float(resolved["training"].get("continue_gate_min_relative_improvement", 0.005)),
            )
            epoch100_snapshot = out / "epoch100.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": 100,
                    "checkpoint_role": "epoch_100_gate_snapshot",
                    "gate_record": gate_record,
                },
                epoch100_snapshot,
            )
            logger.info(
                "epoch-100 gate relative_improvement=%.8f threshold=%.8f decision=%s",
                gate_record["relative_improvement"],
                gate_record["threshold"],
                gate_record["decision"],
            )
            if gate_record["decision"] == "stop_at_100":
                break
            stale_epochs = 0
        patience = int(resolved["training"].get("early_stopping_patience", 15))
        patience_active = not gate_enabled or (epoch > 100 and gate_record is not None)
        if not admission_probe and patience_active and patience > 0 and stale_epochs >= patience:
            break

    train_time = time.perf_counter() - train_start
    stop_epoch = int(history[-1]["epoch"])
    stop_snapshot = epoch100_snapshot
    if gate_enabled and stop_epoch > 100:
        stop_snapshot = out / "continuation_stop.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "epoch": stop_epoch,
                "checkpoint_role": "post_gate_actual_stop_snapshot",
                "gate_record": gate_record,
            },
            stop_snapshot,
        )
    checkpoint = out / "model.pt"
    if not checkpoint.is_file():
        raise RuntimeError("best-validation dual-lane checkpoint was not written")
    try:
        checkpoint_payload = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        checkpoint_payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_payload["state_dict"])
    final_evaluations = {
        lane: _evaluate(
            model,
            val_loaders[lane],
            device=device,
            mean=means[lane],
            std=stds[lane],
            model_spec=spec,
            k_points=k_points,
            lane_index=lane_index,
            torch=torch,
        )
        for lane_index, lane in enumerate(LANES)
    }
    final_score = float(
        np.mean([final_evaluations[lane]["bands_2_10_mre_pct"] for lane in LANES])
    )
    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device_info.is_cuda else 0.0
    metrics = {
        "validation_equal_lane_bands_2_10_mre_pct": final_score,
        "binary_validation_bands_2_10_mre_pct": final_evaluations["binary"]["bands_2_10_mre_pct"],
        "graded_validation_bands_2_10_mre_pct": final_evaluations["graded"]["bands_2_10_mre_pct"],
        "best_epoch": int(checkpoint_payload["epoch"]),
        "stop_epoch": stop_epoch,
        "epoch_100_continue_gate": gate_record,
        "train_time_s": train_time,
        "wall_time_total_s": time.perf_counter() - wall_start,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "peak_gpu_memory_mb": peak_mb,
        "optimizer_steps": optimizer_steps,
        "samples_seen": samples_seen,
        "binary_n_train": len(train_loaders["binary"].dataset),
        "graded_n_train": len(train_loaders["graded"].dataset),
        "binary_n_val": len(val_datasets["binary"]),
        "graded_n_val": len(val_datasets["graded"]),
        "model_identity": describe_model_identity(
            model, spec.name, spec, model_kwargs=resolved["model"]["params"]
        ),
    }
    _plot_history(out, history, monitor_label="equal-lane validation")
    write_metrics(out, metrics)
    (out / "model_summary.json").write_text(
        json.dumps(
            {
                "model": spec.name,
                "parameter_count": metrics["parameter_count"],
                "input_shape": list(reference["grid_shape"]),
                "output_shape": list(reference["target_shape"]),
                "best_validation_checkpoint": str(checkpoint),
                "best_validation_checkpoint_sha256": _sha256_checkpoint(checkpoint),
                "model_identity": metrics["model_identity"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    status = "dual_lane_admission_probe_pass" if admission_probe else "dual_lane_training_success"
    write_results(
        out,
        {
            "status": status,
            "formal_training_started": not admission_probe,
            "mpb_meep_used": False,
            "test_executed": False,
            "model": spec.name,
            "model_identity": metrics["model_identity"],
            "head_kind": head_kind,
            "seed": int(seed),
            "lane": "joint",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256_checkpoint(checkpoint),
            "best_epoch": int(checkpoint_payload["epoch"]),
            "stop_epoch": stop_epoch,
            "epoch_100_continue_gate": gate_record,
            "epoch_100_checkpoint_path": str(epoch100_snapshot) if epoch100_snapshot else None,
            "epoch_100_checkpoint_sha256": _sha256_checkpoint(epoch100_snapshot) if epoch100_snapshot else None,
            "actual_stop_checkpoint_path": str(stop_snapshot) if stop_snapshot else None,
            "actual_stop_checkpoint_sha256": _sha256_checkpoint(stop_snapshot) if stop_snapshot else None,
            "selection_metric": "equal_lane_validation_bands_2_10_mre_pct",
            "next_gate": "dual_lane_admission_decision" if admission_probe else "10k_head_selection",
        },
    )
    write_run_meta(
        out,
        {
            "script": "python -m phc2d.forward.full_cell_dual_lane",
            "config_path": str(config_file),
            "deployed_git_commit": _git_commit(),
            "dataset_paths": {lane: str(contracts[lane]["path"]) for lane in LANES},
            "dataset_sha256": {lane: contracts[lane]["dataset_sha256"] for lane in LANES},
            "idx_train_hash": {lane: contracts[lane]["idx_train_hash"] for lane in LANES},
            "idx_val_hash": {lane: contracts[lane]["idx_val_hash"] for lane in LANES},
            "idx_test_hash": {lane: contracts[lane]["idx_test_hash"] for lane in LANES},
            "k_points_sha256": reference["k_points_sha256"],
            "model": spec.name,
            "head_kind": head_kind,
            "seed": int(seed),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_node": os.environ.get("SLURM_JOB_NODELIST"),
            **device_info.metadata(peak_gpu_memory_mb=peak_mb),
        },
    )
    missing = validate_run_artifacts(out)
    if missing:
        raise RuntimeError(f"incomplete dual-lane run artifact contract: {missing}")
    logger.info("terminal_status=%s run_dir=%s", status, out)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--head-kind", choices=("dense", "k_conditioned"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--admission-probe", action="store_true")
    args = parser.parse_args(argv)
    print(
        run_full_cell_dual_lane(
            args.config,
            seed=args.seed,
            head_kind=args.head_kind,
            run_dir=args.run_dir,
            admission_probe=args.admission_probe,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["run_full_cell_dual_lane"]
