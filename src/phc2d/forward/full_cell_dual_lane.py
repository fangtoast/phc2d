"""Balanced binary/graded Full-Wh training for the dual-lane operator.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from phc2d.forward.full_cell import (
    _evaluate,
    _epoch_100_continue_decision,
    _git_commit,
    _plot_history,
    _sha256_checkpoint,
    _write_history,
)
from phc2d.forward.dual_lane_contracts import (
    LANES,
    dual_lane_convergence_record as _dual_lane_convergence_record,
    make_dual_lane_dataset as _dataset,
    read_dual_lane_contracts as _lane_contracts,
    validate_and_resolve_dual_lane as _validate_and_resolve,
)
from phc2d.forward.dual_lane_recovery import recover_fixed_200_finalizer as _recover_fixed_200_finalizer
from phc2d.forward.gap_aware_loss import (
    GapAwareLossConfig,
    gap_aware_spectrum_loss,
    resolve_gap_aware_loss_config,
)
from phc2d.forward.frequency_weighted_loss import (
    FrequencyWeightedLossConfig,
    bounded_inverse_frequency_loss,
    precompute_frequency_centers,
    resolve_frequency_weighted_loss_config,
)
from phc2d.forward.frequency_weighted_metrics import evaluate_frequency_weighted_predictions
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


@dataclass
class _DualLaneTrainingContext:
    """Objects shared by the epoch engine and artifact finalizer."""

    resolved: Mapping[str, Any]
    out: Path
    logger: Any
    model: Any
    spec: Any
    contracts: Mapping[str, Mapping[str, Any]]
    reference: Mapping[str, Any]
    optimizer: Any
    train_loaders: Mapping[str, Any]
    val_datasets: Mapping[str, Any]
    val_loaders: Mapping[str, Any]
    means: Mapping[str, Any]
    stds: Mapping[str, Any]
    k_points: Any
    device: Any
    device_info: Any
    torch: Any
    admission_probe: bool
    warm_start: Mapping[str, Any] | None
    wall_start: float
    config_file: Path
    head_kind: str
    seed: int
    loss_config: GapAwareLossConfig | FrequencyWeightedLossConfig | None
    frequency_config: FrequencyWeightedLossConfig | None
    train_frequency_centers: Mapping[str, Any] | None
    val_frequency_centers: Mapping[str, Any] | None


class _FrequencyCenterDataset:
    """Attach start-of-run precomputed centers without changing HDF5 adapters."""

    def __init__(self, base: Any, centers: Any) -> None:
        self.base = base
        self.centers = centers
        if len(base) != len(centers):
            raise ValueError("frequency-center count does not match dataset length")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        inputs, target = self.base[index]
        return inputs, target, self.centers[index]


def recover_fixed_200_finalizer(
    run_dir: str | Path,
    *,
    source_slurm_job_id: str | None = None,
) -> Path:
    """Compatibility facade for artifact-only fixed-200 recovery."""

    return _recover_fixed_200_finalizer(
        run_dir,
        source_slurm_job_id=source_slurm_job_id,
        contract_reader=_lane_contracts,
    )


def _dual_lane_training_loss(
    context: _DualLaneTrainingContext,
    prediction: Any,
    target: Any,
    *,
    binary_batch_size: int,
    frequency_centers: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Compute the legacy base loss or equal-lane gap-aware components."""

    if context.loss_config is None:
        base = context.torch.nn.functional.mse_loss(prediction, target)
        return base, {"full_spectrum": base}
    if binary_batch_size <= 0 or binary_batch_size >= int(target.shape[0]):
        raise ValueError("dual-lane gap-aware loss requires non-empty binary and graded batches")
    if isinstance(context.loss_config, FrequencyWeightedLossConfig):
        if frequency_centers is None:
            raise ValueError("frequency-weighted loss requires one center per training sample")
        lane_terms = {}
        start = 0
        for lane, count in (("binary", binary_batch_size), ("graded", target.shape[0] - binary_batch_size)):
            stop = start + int(count)
            lane_terms[lane] = bounded_inverse_frequency_loss(
                prediction[start:stop], target[start:stop], target_mean=context.means[lane],
                target_std=context.stds[lane], frequency_center=frequency_centers[start:stop],
                config=context.loss_config, torch_module=context.torch,
            )
            start = stop
        names = tuple(lane_terms["binary"].as_dict())
        averaged = {
            name: 0.5 * (lane_terms["binary"].as_dict()[name] + lane_terms["graded"].as_dict()[name])
            for name in names
        }
        return averaged["total"], {name: value for name, value in averaged.items() if name != "total"}
    lane_terms = {}
    start = 0
    for lane, count in (("binary", binary_batch_size), ("graded", target.shape[0] - binary_batch_size)):
        stop = start + int(count)
        lane_terms[lane] = gap_aware_spectrum_loss(
            prediction[start:stop], target[start:stop], target_mean=context.means[lane],
            target_std=context.stds[lane], config=context.loss_config, torch_module=context.torch,
        )
        start = stop
    names = tuple(lane_terms["binary"].as_dict())
    averaged = {
        name: 0.5 * (
            lane_terms["binary"].as_dict()[name] + lane_terms["graded"].as_dict()[name]
        )
        for name in names
    }
    return averaged["total"], {name: value for name, value in averaged.items() if name != "total"}


def _gap_evaluation_kwargs(context: _DualLaneTrainingContext) -> dict[str, int | None]:
    """Expose an explicit evaluation range without coupling metrics to the loss."""

    evaluation = dict(context.resolved.get("evaluation") or {})
    gap_decision = dict(evaluation.get("gap_decision") or {})
    if gap_decision:
        return {
            "gap_lower_band_min_one_based": int(
                gap_decision["lower_band_min_one_based"]
            ),
            "gap_lower_band_max_one_based": int(
                gap_decision["lower_band_max_one_based"]
            ),
        }
    if context.loss_config is None or isinstance(context.loss_config, FrequencyWeightedLossConfig):
        return {}
    return {
        "gap_lower_band_min_one_based": context.loss_config.lower_band_min_one_based,
        "gap_lower_band_max_one_based": context.loss_config.lower_band_max_one_based,
    }



def _run_dual_lane_epochs(context: _DualLaneTrainingContext) -> dict[str, Any]:
    """Execute epochs and write only training-state checkpoints/history."""

    resolved = context.resolved
    warm_start = context.warm_start
    history: list[dict[str, Any]] = [] if warm_start is None else list(warm_start["source_history"])
    best_score = float("inf") if warm_start is None else min(float(row["val_bands_2_10_mre_pct"]) for row in history)
    stale_epochs = 0
    optimizer_steps = 0 if warm_start is None else int(history[-1]["optimizer_steps"])
    samples_seen = 0 if warm_start is None else int(history[-1]["samples_seen"])
    wall_time_offset = 0.0 if warm_start is None else float(history[-1]["wall_time_s"])
    continuation_convergence: dict[str, Any] | None = None
    if warm_start is not None:
        from phc2d.forward.full_cell_warm_start import seed_derivative_best_checkpoint

        seed_derivative_best_checkpoint(warm_start, context.out)
    train_start = time.perf_counter()
    gate_enabled = bool(resolved["training"].get("epoch_100_continue_gate", False)) and not context.admission_probe
    gate_record: dict[str, Any] | None = None
    epoch100_snapshot: Path | None = None
    start_epoch = 1 if warm_start is None else int(warm_start["source_epoch"]) + 1
    for epoch in range(start_epoch, int(resolved["training"]["max_epochs"]) + 1):
        epoch_start = time.perf_counter()
        context.model.train()
        train_sum = 0.0
        train_count = 0
        component_sums: dict[str, float] = {}
        for batch_index, (binary_batch, graded_batch) in enumerate(
            zip(context.train_loaders["binary"], context.train_loaders["graded"], strict=True),
            start=1,
        ):
            if isinstance(context.loss_config, FrequencyWeightedLossConfig):
                xb, yb, cb = binary_batch
                xg, yg, cg = graded_batch
                frequency_centers = context.torch.cat((cb, cg), dim=0).to(
                    context.device, non_blocking=True
                )
            else:
                xb, yb = binary_batch
                xg, yg = graded_batch
                frequency_centers = None
            inputs = context.torch.cat((xb, xg), dim=0).to(context.device, non_blocking=True)
            targets = context.torch.cat((yb, yg), dim=0).to(context.device, non_blocking=True)
            lane_id = context.torch.cat(
                (
                    context.torch.zeros(xb.shape[0], dtype=context.torch.long),
                    context.torch.ones(xg.shape[0], dtype=context.torch.long),
                )
            ).to(context.device, non_blocking=True)
            context.optimizer.zero_grad(set_to_none=True)
            prediction = context.model(inputs, k_points=context.k_points, lane_id=lane_id)
            loss, loss_components = _dual_lane_training_loss(
                context,
                prediction,
                targets,
                binary_batch_size=int(xb.shape[0]),
                frequency_centers=frequency_centers,
            )
            if not bool(context.torch.isfinite(loss).item()):
                raise RuntimeError(f"non-finite dual-lane loss at epoch {epoch}, batch {batch_index}")
            loss.backward()
            if any(
                parameter.grad is not None and not bool(context.torch.isfinite(parameter.grad).all().item())
                for parameter in context.model.parameters()
            ):
                raise RuntimeError(f"non-finite dual-lane gradient at epoch {epoch}, batch {batch_index}")
            context.optimizer.step()
            optimizer_steps += 1
            batch_count = int(inputs.shape[0])
            train_sum += float(loss.detach().cpu()) * batch_count
            for name, value in loss_components.items():
                component_sums[name] = component_sums.get(name, 0.0) + float(
                    value.detach().cpu()
                ) * batch_count
            train_count += batch_count
            samples_seen += batch_count
            if resolved["training"].get("max_train_batches") is not None:
                break
        evaluations = {
            lane: _evaluate(
                context.model,
                context.val_loaders[lane],
                device=context.device,
                mean=context.means[lane],
                std=context.stds[lane],
                model_spec=context.spec,
                k_points=context.k_points,
                lane_index=lane_index,
                torch=context.torch,
                **_gap_evaluation_kwargs(context),
            )
            for lane_index, lane in enumerate(LANES)
        }
        score = float(np.mean([evaluations[lane]["bands_2_10_mre_pct"] for lane in LANES]))
        row = {
            "epoch": epoch,
            "train_loss": train_sum / max(train_count, 1),
            "val_loss": float(np.mean([evaluations[lane]["normalized_mse"] for lane in LANES])),
            "val_band1_mre_pct": float(np.mean([evaluations[lane]["band1_mre_pct"] for lane in LANES])),
            "val_bands_2_10_mre_pct": score,
            "val_bands_1_10_mre_pct": float(np.mean([evaluations[lane]["bands_1_10_mre_pct"] for lane in LANES])),
            "binary_val_bands_2_10_mre_pct": evaluations["binary"]["bands_2_10_mre_pct"],
            "graded_val_bands_2_10_mre_pct": evaluations["graded"]["bands_2_10_mre_pct"],
            "learning_rate": float(resolved["training"]["learning_rate"]),
            "epoch_time_s": time.perf_counter() - epoch_start,
            "wall_time_s": wall_time_offset + time.perf_counter() - context.wall_start,
            "optimizer_steps": optimizer_steps,
            "samples_seen": samples_seen,
        }
        if _gap_evaluation_kwargs(context):
            row.update(
                {
                    f"train_{name}_loss": total / max(train_count, 1)
                    for name, total in component_sums.items()
                }
            )
            for metric in (
                "band_accuracy",
                "upper_edge_k_accuracy",
                "joint_band_k_accuracy",
                "selected_frequency_mre_pct",
                "gap_width_mae",
                "gap_center_mae",
            ):
                row[f"val_gap_{metric}"] = float(
                    np.mean([evaluations[lane][metric] for lane in LANES])
                )
                for lane in LANES:
                    row[f"{lane}_val_gap_{metric}"] = evaluations[lane][metric]
        history.append(row)
        if score < best_score:
            best_score = score
            stale_epochs = 0
            checkpoint_tmp = context.out / "model.pt.tmp"
            context.torch.save(
                {
                    "state_dict": context.model.state_dict(),
                    "model_name": context.spec.name,
                    "model_params": dict(resolved["model"]["params"]),
                    "model_identity": describe_model_identity(context.model, context.spec.name, context.spec, model_kwargs=resolved["model"]["params"]),
                    "input_shape": context.reference["grid_shape"],
                    "output_shape": context.reference["target_shape"],
                    "k_points": context.torch.as_tensor(context.reference["k_points"], dtype=context.torch.float32),
                    "k_points_sha256": context.reference["k_points_sha256"],
                    "lane_mapping": {"binary": 0, "graded": 1},
                    "target_normalizers": {lane: {"mean": context.torch.as_tensor(context.contracts[lane]["target_mean"]), "std": context.torch.as_tensor(context.contracts[lane]["target_std"])} for lane in LANES},
                    "epoch": epoch,
                    "validation_equal_lane_mre_pct": best_score,
                    "resolved_config": resolved,
                    "warm_start": None if warm_start is None else {key: value for key, value in warm_start.items() if key not in {"source_history", "source_result", "terminal_payload"}},
                },
                checkpoint_tmp,
            )
            os.replace(checkpoint_tmp, context.out / "model.pt")
        else:
            stale_epochs += 1
        _write_history(context.out / "history.csv", history)
        context.logger.info(
            "epoch=%d train_loss=%.6g binary_mre=%.6g graded_mre=%.6g equal_lane_mre=%.6g",
            epoch,
            row["train_loss"],
            row["binary_val_bands_2_10_mre_pct"],
            row["graded_val_bands_2_10_mre_pct"],
            score,
        )
        if warm_start is not None and epoch % int(resolved["continuation"]["check_interval"]) == 0:
            from phc2d.forward.full_cell_warm_start import rolling_convergence_record

            curves = {
                "binary": rolling_convergence_record(history, end_epoch=epoch, metric="binary_val_bands_2_10_mre_pct", threshold=float(resolved["continuation"]["relative_threshold"])),
                "graded": rolling_convergence_record(history, end_epoch=epoch, metric="graded_val_bands_2_10_mre_pct", threshold=float(resolved["continuation"]["relative_threshold"])),
                "equal_lane": rolling_convergence_record(history, end_epoch=epoch, metric="val_bands_2_10_mre_pct", threshold=float(resolved["continuation"]["relative_threshold"])),
            }
            continuation_convergence = {**curves["equal_lane"], "contract_metric": "validation_bands_2_10_mre", "curves": curves}
            context.logger.info(
                "dual-lane continuation convergence epoch=%d binary=%s graded=%s equal_lane=%s",
                epoch,
                curves["binary"]["passed"],
                curves["graded"]["passed"],
                curves["equal_lane"]["passed"],
            )
            if bool(resolved["continuation"].get("stop_when_converged", True)) and all(bool(item["passed"]) for item in curves.values()):
                break
        if gate_enabled and epoch == 100:
            gate_record = _epoch_100_continue_decision(
                history,
                threshold=float(resolved["training"].get("continue_gate_min_relative_improvement", 0.005)),
            )
            epoch100_snapshot = context.out / "epoch100.pt"
            context.torch.save({"state_dict": context.model.state_dict(), "epoch": 100, "checkpoint_role": "epoch_100_gate_snapshot", "gate_record": gate_record}, epoch100_snapshot)
            context.logger.info(
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
        if not context.admission_probe and patience_active and patience > 0 and stale_epochs >= patience:
            break
    stop_epoch = int(history[-1]["epoch"])
    stop_snapshot = epoch100_snapshot
    if gate_enabled and stop_epoch > 100:
        stop_snapshot = context.out / "continuation_stop.pt"
        context.torch.save({"state_dict": context.model.state_dict(), "epoch": stop_epoch, "checkpoint_role": "post_gate_actual_stop_snapshot", "gate_record": gate_record}, stop_snapshot)
    return {
        "history": history,
        "optimizer_steps": optimizer_steps,
        "samples_seen": samples_seen,
        "train_time": time.perf_counter() - train_start,
        "stop_epoch": stop_epoch,
        "gate_record": gate_record,
        "epoch100_snapshot": epoch100_snapshot,
        "stop_snapshot": stop_snapshot,
        "continuation_convergence": continuation_convergence,
    }


def _write_terminal_checkpoint(
    context: _DualLaneTrainingContext,
    outcome: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, int | None, Path | None]:
    """Write the fixed-stop or continuation terminal checkpoint."""

    resolved = context.resolved
    stop_epoch = int(outcome["stop_epoch"])
    required_stop_epoch = resolved["training"].get("required_stop_epoch")
    convergence: dict[str, Any] | None = None
    terminal_checkpoint: Path | None = None
    if required_stop_epoch is not None and not context.admission_probe:
        required_stop_epoch = int(required_stop_epoch)
        if stop_epoch != required_stop_epoch:
            raise RuntimeError(f"required terminal epoch {required_stop_epoch}, observed {stop_epoch}")
        convergence_window = tuple(int(value) for value in resolved["training"].get("convergence_window", [181, 200]))
        if convergence_window != (181, 200):
            raise ValueError("EXP-FWD-008 convergence_window must be [181, 200]")
        convergence = _dual_lane_convergence_record(
            outcome["history"],
            threshold=float(resolved["training"].get("convergence_relative_threshold", 0.005)),
            contract_metric=str(resolved["training"].get("convergence_metric", "validation_bands_2_10_mre")),
        )
        checkpoint_role = "terminal_epoch"
    elif context.warm_start is not None and not context.admission_probe:
        convergence = outcome["continuation_convergence"]
        if convergence is None:
            raise RuntimeError("dual-lane continuation ended without a convergence checkpoint")
        required_stop_epoch = None
        checkpoint_role = "continuation_terminal_epoch"
    else:
        return convergence, required_stop_epoch, terminal_checkpoint
    terminal_checkpoint = context.out / f"epoch{stop_epoch}.pt"
    terminal_tmp = terminal_checkpoint.with_suffix(terminal_checkpoint.suffix + ".tmp")
    warm_start_public = None
    if context.warm_start is not None:
        from phc2d.forward.full_cell_warm_start import public_warm_start_metadata

        warm_start_public = public_warm_start_metadata(context.warm_start)
    context.torch.save(
        {
            "state_dict": context.model.state_dict(),
            "model_name": context.spec.name,
            "model_params": dict(resolved["model"]["params"]),
            "model_identity": describe_model_identity(context.model, context.spec.name, context.spec, model_kwargs=resolved["model"]["params"]),
            "input_shape": context.reference["grid_shape"],
            "output_shape": context.reference["target_shape"],
            "k_points": context.torch.as_tensor(context.reference["k_points"], dtype=context.torch.float32),
            "k_points_sha256": context.reference["k_points_sha256"],
            "lane_mapping": {"binary": 0, "graded": 1},
            "target_normalizers": {lane: {"mean": context.torch.as_tensor(context.contracts[lane]["target_mean"]), "std": context.torch.as_tensor(context.contracts[lane]["target_std"])} for lane in LANES},
            "epoch": stop_epoch,
            "checkpoint_role": checkpoint_role,
            "selection_split": "terminal_training_state",
            "convergence": convergence,
            "warm_start": warm_start_public,
            "resolved_config": resolved,
        },
        terminal_tmp,
    )
    os.replace(terminal_tmp, terminal_checkpoint)
    return convergence, required_stop_epoch, terminal_checkpoint


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write stable CSV evidence without introducing a report-only dependency."""

    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(str(field))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_frequency_validation_artifacts(context: _DualLaneTrainingContext) -> dict[str, Any]:
    """Evaluate validation only and emit the EXP-FWD-018 diagnostic artifact set."""

    if context.frequency_config is None:
        return {}
    if context.val_frequency_centers is None:
        raise RuntimeError("frequency-weighted validation centers are unavailable")
    all_band: list[dict[str, Any]] = []
    all_k: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []
    lane_metrics: dict[str, Any] = {}
    context.model.eval()
    for lane_index, lane in enumerate(LANES):
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        with context.torch.no_grad():
            for inputs, normalized_target in context.val_loaders[lane]:
                inputs = inputs.to(context.device, non_blocking=True)
                normalized_target = normalized_target.to(context.device, non_blocking=True)
                lane_id = context.torch.full((inputs.shape[0],), lane_index, dtype=context.torch.long, device=context.device)
                prediction = context.model(inputs, k_points=context.k_points, lane_id=lane_id)
                predictions.append((prediction * context.stds[lane] + context.means[lane]).detach().cpu().numpy())
                targets.append((normalized_target * context.stds[lane] + context.means[lane]).detach().cpu().numpy())
        prediction_array = np.concatenate(predictions)
        target_array = np.concatenate(targets)
        center_array = context.val_frequency_centers[lane].detach().cpu().numpy()
        metrics, per_band, per_k, pair_rows = evaluate_frequency_weighted_predictions(
            prediction_array, target_array,
            frequency_center=center_array,
            target_std=context.stds[lane].detach().cpu().numpy(), config=context.frequency_config,
        )
        np.savez_compressed(
            context.out / f"validation_predictions_{lane}.npz",
            prediction_Wh=prediction_array, target_Wh=target_array, frequency_center=center_array,
            target_std=context.stds[lane].detach().cpu().numpy(),
        )
        lane_metrics[lane] = metrics
        all_band.extend({"lane": lane, **row} for row in per_band)
        all_k.extend({"lane": lane, **row} for row in per_k)
        all_pairs.extend({"lane": lane, **row} for row in pair_rows)
    _write_csv(context.out / "per_band_metrics.csv", all_band)
    _write_csv(context.out / "per_k_metrics.csv", all_k)
    _write_csv(context.out / "frequency_pair_diagnostics.csv", all_pairs)
    (context.out / "window_metrics.json").write_text(json.dumps(lane_metrics, indent=2) + "\n", encoding="utf-8")
    (context.out / "weight_statistics.json").write_text(
        json.dumps({lane: metrics["weight_statistics"] for lane, metrics in lane_metrics.items()}, indent=2) + "\n", encoding="utf-8"
    )
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    for lane in LANES:
        axis.plot([row["band_one_based"] for row in all_band if row["lane"] == lane],
                  [row["mre_pct"] for row in all_band if row["lane"] == lane], marker="o", label=lane)
    axis.set_xlabel("band")
    axis.set_ylabel("validation MRE (%)")
    axis.set_title("frequency-weighted validation spectrum")
    axis.legend()
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(context.out / "figures" / f"frequency_window_validation.{suffix}", dpi=180)
    plt.close(figure)
    return lane_metrics


def _finalize_dual_lane_run(
    context: _DualLaneTrainingContext,
    outcome: Mapping[str, Any],
) -> Path:
    """Evaluate the best checkpoint and write the standard run contract."""

    convergence, required_stop_epoch, terminal_checkpoint = _write_terminal_checkpoint(context, outcome)
    checkpoint = context.out / "model.pt"
    if not checkpoint.is_file():
        raise RuntimeError("best-validation dual-lane checkpoint was not written")
    try:
        checkpoint_payload = context.torch.load(checkpoint, map_location=context.device, weights_only=True)
    except TypeError:
        checkpoint_payload = context.torch.load(checkpoint, map_location=context.device)
    context.model.load_state_dict(checkpoint_payload["state_dict"])
    evaluations = {
        lane: _evaluate(
            context.model,
            context.val_loaders[lane],
            device=context.device,
            mean=context.means[lane],
            std=context.stds[lane],
            model_spec=context.spec,
            k_points=context.k_points,
            lane_index=lane_index,
            torch=context.torch,
            **_gap_evaluation_kwargs(context),
        )
        for lane_index, lane in enumerate(LANES)
    }
    final_score = float(np.mean([evaluations[lane]["bands_2_10_mre_pct"] for lane in LANES]))
    frequency_validation = _write_frequency_validation_artifacts(context)
    peak_mb = float(context.torch.cuda.max_memory_allocated(context.device) / (1024**2)) if context.device_info.is_cuda else 0.0
    warm_start_public = None
    if context.warm_start is not None:
        from phc2d.forward.full_cell_warm_start import public_warm_start_metadata

        warm_start_public = public_warm_start_metadata(context.warm_start)
    metrics = {
        "validation_equal_lane_bands_2_10_mre_pct": final_score,
        "binary_validation_bands_2_10_mre_pct": evaluations["binary"]["bands_2_10_mre_pct"],
        "graded_validation_bands_2_10_mre_pct": evaluations["graded"]["bands_2_10_mre_pct"],
        "best_epoch": int(checkpoint_payload["epoch"]),
        "stop_epoch": int(outcome["stop_epoch"]),
        "epoch_100_continue_gate": outcome["gate_record"],
        "train_time_s": outcome["train_time"],
        "wall_time_total_s": time.perf_counter() - context.wall_start,
        "parameter_count": int(sum(parameter.numel() for parameter in context.model.parameters())),
        "peak_gpu_memory_mb": peak_mb,
        "optimizer_steps": outcome["optimizer_steps"],
        "samples_seen": outcome["samples_seen"],
        "binary_n_train": len(context.train_loaders["binary"].dataset),
        "graded_n_train": len(context.train_loaders["graded"].dataset),
        "binary_n_val": len(context.val_datasets["binary"]),
        "graded_n_val": len(context.val_datasets["graded"]),
        "model_identity": describe_model_identity(context.model, context.spec.name, context.spec, model_kwargs=context.resolved["model"]["params"]),
        "convergence": convergence,
        "required_stop_epoch": required_stop_epoch,
        "terminal_checkpoint_path": str(terminal_checkpoint) if terminal_checkpoint else None,
        "terminal_checkpoint_sha256": _sha256_checkpoint(terminal_checkpoint) if terminal_checkpoint else None,
        "history_epoch_count": len(outcome["history"]),
        "warm_start": warm_start_public,
        "loss_contract": (
            context.loss_config.as_dict()
            if context.loss_config is not None
            else {"name": "equal_lane_normalized_full_matrix_mse"}
        ),
        "frequency_window_validation": frequency_validation or None,
    }
    if _gap_evaluation_kwargs(context):
        metrics["gap_validation"] = {
            lane: {
                key: value
                for key, value in evaluations[lane].items()
                if key
                not in {
                    "mse",
                    "normalized_mse",
                    "mae",
                    "rmse",
                    "bands_2_10_mre_pct",
                    "band1_mre_pct",
                    "bands_1_10_mre_pct",
                    "eval_time_s",
                }
            }
            for lane in LANES
        }
    _plot_history(context.out, outcome["history"], monitor_label="equal-lane validation")
    write_metrics(context.out, metrics)
    (context.out / "model_summary.json").write_text(
        json.dumps(
            {
                "model": context.spec.name,
                "parameter_count": metrics["parameter_count"],
                "input_shape": list(context.reference["grid_shape"]),
                "output_shape": list(context.reference["target_shape"]),
                "best_validation_checkpoint": str(checkpoint),
                "best_validation_checkpoint_sha256": _sha256_checkpoint(checkpoint),
                "terminal_checkpoint": str(terminal_checkpoint) if terminal_checkpoint else None,
                "terminal_checkpoint_sha256": _sha256_checkpoint(terminal_checkpoint) if terminal_checkpoint else None,
                "model_identity": metrics["model_identity"],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    status = "dual_lane_admission_probe_pass" if context.admission_probe else "continuation_training_success" if context.warm_start is not None else "dual_lane_training_success"
    if not context.admission_probe and convergence is not None and not all(bool(item["passed"]) for item in convergence["curves"].values()):
        status = (
            "inconclusive_nonconverged"
            if isinstance(context.loss_config, FrequencyWeightedLossConfig)
            else "training_complete_convergence_not_met"
        )
    write_results(
        context.out,
        {
            "status": status,
            "formal_training_started": not context.admission_probe,
            "mpb_meep_used": False,
            "test_executed": False,
            "model": context.spec.name,
            "model_identity": metrics["model_identity"],
            "head_kind": context.head_kind,
            "seed": context.seed,
            "lane": "joint",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256_checkpoint(checkpoint),
            "terminal_checkpoint_path": str(terminal_checkpoint) if terminal_checkpoint else None,
            "terminal_checkpoint_sha256": _sha256_checkpoint(terminal_checkpoint) if terminal_checkpoint else None,
            "required_stop_epoch": required_stop_epoch,
            "convergence": convergence,
            "history_epoch_count": len(outcome["history"]),
            "warm_start": warm_start_public,
            "loss_contract": metrics["loss_contract"],
            "best_epoch": int(checkpoint_payload["epoch"]),
            "stop_epoch": int(outcome["stop_epoch"]),
            "epoch_100_continue_gate": outcome["gate_record"],
            "epoch_100_checkpoint_path": str(outcome["epoch100_snapshot"]) if outcome["epoch100_snapshot"] else None,
            "epoch_100_checkpoint_sha256": _sha256_checkpoint(outcome["epoch100_snapshot"]) if outcome["epoch100_snapshot"] else None,
            "actual_stop_checkpoint_path": str(outcome["stop_snapshot"]) if outcome["stop_snapshot"] else None,
            "actual_stop_checkpoint_sha256": _sha256_checkpoint(outcome["stop_snapshot"]) if outcome["stop_snapshot"] else None,
            "selection_metric": "equal_lane_validation_bands_2_10_mre_pct",
            "next_gate": (
                "gap_aware_loss_formal_experiment_authorization"
                if context.admission_probe and context.loss_config is not None
                else "dual_lane_admission_decision"
                if context.admission_probe
                else "continuation_checkpoint_freeze"
                if context.warm_start is not None
                else "10k_head_selection"
            ),
        },
    )
    write_run_meta(
        context.out,
        {
            "script": "python -m phc2d.forward.full_cell_dual_lane",
            "config_path": str(context.config_file),
            "deployed_git_commit": _git_commit(),
            "dataset_paths": {lane: str(context.contracts[lane]["path"]) for lane in LANES},
            "dataset_sha256": {lane: context.contracts[lane]["dataset_sha256"] for lane in LANES},
            "idx_train_hash": {lane: context.contracts[lane]["idx_train_hash"] for lane in LANES},
            "idx_val_hash": {lane: context.contracts[lane]["idx_val_hash"] for lane in LANES},
            "idx_test_hash": {lane: context.contracts[lane]["idx_test_hash"] for lane in LANES},
            "k_points_sha256": context.reference["k_points_sha256"],
            "model": context.spec.name,
            "head_kind": context.head_kind,
            "seed": context.seed,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_node": os.environ.get("SLURM_JOB_NODELIST"),
            **context.device_info.metadata(peak_gpu_memory_mb=peak_mb),
        },
    )
    missing = validate_run_artifacts(context.out)
    if missing:
        raise RuntimeError(f"incomplete dual-lane run artifact contract: {missing}")
    context.logger.info("terminal_status=%s run_dir=%s", status, context.out)
    return context.out


def run_full_cell_dual_lane(
    config_path: Path,
    *,
    seed: int,
    head_kind: str,
    run_dir: Path,
    admission_probe: bool = False,
    warm_start_run: str | Path | None = None,
) -> Path:
    """Train or probe one balanced dual-lane model checkpoint."""

    config_file = Path(config_path)
    payload = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("dual-lane config root must be a mapping")
    mutable_payload = dict(payload)
    if warm_start_run is not None:
        mutable_payload["continuation"] = {
            **dict(mutable_payload.get("continuation") or {}),
            "source_run": str(Path(warm_start_run).resolve()),
        }
    resolved = _validate_and_resolve(
        mutable_payload,
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
    warm_start: dict[str, Any] | None = None
    if resolved.get("continuation"):
        from phc2d.forward.full_cell_warm_start import load_weight_only_warm_start

        source_run = resolved["continuation"].get("source_run")
        if not source_run:
            raise ValueError("dual-lane continuation source_run is required")
        warm_start = load_weight_only_warm_start(
            source_run,
            expected_model=spec.name,
            expected_seed=seed,
            torch_module=torch,
            expected_source_experiment=resolved["continuation"].get("source_experiment"),
        )
        model.load_state_dict(warm_start["terminal_payload"]["state_dict"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(resolved["training"]["learning_rate"]),
        weight_decay=float(resolved["training"].get("weight_decay", 1.0e-5)),
    )
    half_batch = int(resolved["training"]["micro_batch_size"]) // 2
    loader_kwargs = {
        "batch_size": half_batch,
        "num_workers": int(device_info.num_workers),
        "pin_memory": bool(device_info.pin_memory),
    }
    loss_config = resolve_frequency_weighted_loss_config(resolved["training"].get("loss"))
    if loss_config is None:
        loss_config = resolve_gap_aware_loss_config(resolved["training"].get("loss"))
    frequency_config = (
        loss_config if isinstance(loss_config, FrequencyWeightedLossConfig)
        else resolve_frequency_weighted_loss_config(
            dict(resolved.get("evaluation") or {}).get("frequency_weighted_diagnostics")
        )
    )
    train_frequency_centers: dict[str, Any] | None = None
    val_frequency_centers: dict[str, Any] | None = None
    if frequency_config is not None:
        train_frequency_centers = {}
        val_frequency_centers = {}
        manifest_rows: list[dict[str, Any]] = []
        for lane in LANES:
            train_centers, train_ids = precompute_frequency_centers(
                contracts[lane]["path"], grid_field=contracts[lane]["grid_field"],
                indices=contracts[lane]["selected_train"], config=frequency_config,
            )
            val_centers, val_ids = precompute_frequency_centers(
                contracts[lane]["path"], grid_field=contracts[lane]["grid_field"],
                indices=contracts[lane]["idx_val"], config=frequency_config,
            )
            train_frequency_centers[lane] = torch.as_tensor(train_centers, dtype=torch.float32)
            val_frequency_centers[lane] = torch.as_tensor(val_centers, dtype=torch.float32)
            manifest_rows.extend(
                {"lane": lane, "split": split, "dataset_index": int(index), "sample_id": str(sample_id),
                 "frequency_center": float(center), "center_kind": frequency_config.center_kind}
                for split, indices, sample_ids, centers in (
                    ("train", contracts[lane]["selected_train"], train_ids, train_centers),
                    ("validation", contracts[lane]["idx_val"], val_ids, val_centers),
                )
                for index, sample_id, center in zip(indices, sample_ids, centers, strict=True)
            )
        with (out / "frequency_center_manifest.jsonl").open("w", encoding="utf-8") as handle:
            for row in manifest_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    train_loaders = {
        lane: DataLoader(
            _FrequencyCenterDataset(_dataset(contracts[lane], resolved, contracts[lane]["selected_train"]), train_frequency_centers[lane])
            if train_frequency_centers is not None
            else _dataset(contracts[lane], resolved, contracts[lane]["selected_train"]),
            shuffle=True, generator=torch.Generator().manual_seed(seed + lane_index), **loader_kwargs,
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
    context = _DualLaneTrainingContext(
        resolved=resolved,
        out=out,
        logger=logger,
        model=model,
        spec=spec,
        contracts=contracts,
        reference=reference,
        optimizer=optimizer,
        train_loaders=train_loaders,
        val_datasets=val_datasets,
        val_loaders=val_loaders,
        means=means,
        stds=stds,
        k_points=k_points,
        device=device,
        device_info=device_info,
        torch=torch,
        admission_probe=admission_probe,
        warm_start=warm_start,
        wall_start=time.perf_counter(),
        config_file=config_file,
        head_kind=head_kind,
        seed=int(seed),
        loss_config=loss_config,
        frequency_config=frequency_config,
        train_frequency_centers=train_frequency_centers,
        val_frequency_centers=val_frequency_centers,
    )
    outcome = _run_dual_lane_epochs(context)
    return _finalize_dual_lane_run(context, outcome)

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--head-kind", choices=("dense", "k_conditioned"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--admission-probe", action="store_true")
    parser.add_argument("--warm-start-run", type=Path)
    args = parser.parse_args(argv)
    print(
        run_full_cell_dual_lane(
            args.config,
            seed=args.seed,
            head_kind=args.head_kind,
            run_dir=args.run_dir,
            admission_probe=args.admission_probe,
            warm_start_run=args.warm_start_run,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["recover_fixed_200_finalizer", "run_full_cell_dual_lane"]
