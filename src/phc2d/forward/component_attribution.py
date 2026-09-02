"""Validation-only frozen diagnostics for Full-Wh component attribution.

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
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from phc2d.forward.full_cell import _forward_model, _lane_index, _read_contract
from phc2d.models import build_model, describe_model_identity, get_model_spec
from phc2d.runs import ensure_figures_dir, make_run_dir, setup_logging, write_metrics, write_results, write_run_meta
from phc2d.training import DeviceConfig, resolve_device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected YAML mapping: {path}")
    return dict(value)


def _load_checkpoint(path: Path, torch: Any) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping) or "state_dict" not in value:
        raise ValueError("checkpoint must contain state_dict")
    return dict(value)


def _mask_names(model_name: str) -> tuple[tuple[str, ...], ...]:
    if model_name == "fno2d_band_v2":
        return ((), ("primary_spectral",), ("band_spectral",), ("band_gate",), ("low_rank",))
    if model_name in {"tfno2d_band_v3", "dual_lane_band_operator_v1"}:
        return ((), ("spectral",), ("local",), ("channel",))
    return ((),)


def _fourier_filter(x: Any, *, keep_low: bool, cutoff: float, torch: Any) -> Any:
    height, width = int(x.shape[-2]), int(x.shape[-1])
    fy = torch.fft.fftfreq(height, device=x.device)[:, None]
    fx = torch.fft.rfftfreq(width, device=x.device)[None, :]
    mask = torch.sqrt(fy * fy + fx * fx) <= float(cutoff)
    if not keep_low:
        mask = ~mask
    spectrum = torch.fft.rfft2(x)
    return torch.fft.irfft2(spectrum * mask[None, None, :, :], s=(height, width))


def _sample_features(x: np.ndarray) -> dict[str, np.ndarray]:
    dx = np.abs(x - np.roll(x, 1, axis=-1))
    dy = np.abs(x - np.roll(x, 1, axis=-2))
    spectrum = np.fft.rfft2(x, axes=(-2, -1))
    fy = np.fft.fftfreq(x.shape[-2])[:, None]
    fx = np.fft.rfftfreq(x.shape[-1])[None, :]
    high = np.sqrt(fy * fy + fx * fx) > 0.25
    energy = np.abs(spectrum) ** 2
    return {
        "boundary_density": (dx.mean(axis=(-2, -1)) + dy.mean(axis=(-2, -1))) / 2.0,
        "gray_ratio": ((x > 0.05) & (x < 0.95)).mean(axis=(-2, -1)),
        "high_frequency_ratio": energy[:, high].sum(axis=1) / np.maximum(energy.sum(axis=(-2, -1)), 1.0e-12),
    }


def _quartile_rows(features: Mapping[str, np.ndarray], errors: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, values in features.items():
        edges = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
        for index in range(4):
            selected = (values >= edges[index]) & ((values <= edges[index + 1]) if index == 3 else (values < edges[index + 1]))
            rows.append({
                "feature": name, "quartile": index + 1, "count": int(selected.sum()),
                "lower": float(edges[index]), "upper": float(edges[index + 1]),
                "bands_2_10_mre_pct": float(errors[selected].mean()) if selected.any() else None,
            })
    return rows


def run_component_diagnostic(
    resolved_config: str | Path,
    checkpoint: str | Path,
    *,
    output_dir: str | Path,
    batch_size: int = 64,
) -> Path:
    """Run frozen perturbations on validation only; never reads target test rows."""

    import h5py
    import torch

    config_path = Path(resolved_config)
    checkpoint_path = Path(checkpoint)
    config = _load_yaml(config_path)
    experiment = dict(config["experiment"])
    model_cfg = dict(config["model"])
    model_name = str(model_cfg["name"])
    if model_name not in {"fno2d_band_v2", "tfno2d_band_v3", "convnext_band_v1", "dual_lane_band_operator_v1"}:
        raise ValueError(f"unsupported frozen diagnostic model: {model_name}")
    out = make_run_dir("EXP-FWD-019_C0", run_dir=output_dir)
    ensure_figures_dir(out)
    setup_logging(log_file=out / "run.log")
    started = time.perf_counter()
    runtime = dict(config.get("runtime") or {})
    runtime["require_accelerator"] = True
    device_info = resolve_device(DeviceConfig.from_mapping(runtime), torch_module=torch)
    device = device_info.torch_device
    contract = _read_contract(config)
    payload = _load_checkpoint(checkpoint_path, torch)
    spec = get_model_spec(model_name)
    params = dict(model_cfg.get("params") or {})
    if spec.accepts_k_points:
        params["k_points"] = contract["k_points"]
    model = build_model(model_name, input_shape=contract["grid_shape"], output_shape=contract["target_shape"], **params)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    mean = np.asarray(payload["target_mean"], dtype=np.float32)
    std = np.asarray(payload["target_std"], dtype=np.float32)
    k_points = torch.as_tensor(contract["k_points"], dtype=torch.float32, device=device) if spec.accepts_k_points else None
    lane_index = _lane_index(experiment.get("lane")) if spec.accepts_lane_id else None
    with h5py.File(contract["path"], "r") as handle:
        indices = np.asarray(handle["idx_val"], dtype=np.int64)
        grids = np.asarray(handle[contract["grid_field"]][indices], dtype=np.float32)
        targets = np.asarray(handle["Wh"][indices], dtype=np.float32)
    predictions: dict[str, list[np.ndarray]] = {"baseline": [], "shift_1_0": [], "shift_0_1": [], "shift_4_4": [], "low_fourier": [], "high_fourier": []}
    for disabled in _mask_names(model_name)[1:]:
        predictions[f"mask_{disabled[0]}"] = []
    with torch.no_grad():
        for start in range(0, len(indices), int(batch_size)):
            xb = torch.as_tensor(grids[start:start + int(batch_size)], dtype=torch.float32, device=device).unsqueeze(1)
            variants = {
                "baseline": xb,
                "shift_1_0": torch.roll(xb, shifts=(1, 0), dims=(-2, -1)),
                "shift_0_1": torch.roll(xb, shifts=(0, 1), dims=(-2, -1)),
                "shift_4_4": torch.roll(xb, shifts=(4, 4), dims=(-2, -1)),
                "low_fourier": _fourier_filter(xb, keep_low=True, cutoff=0.25, torch=torch),
                "high_fourier": _fourier_filter(xb, keep_low=False, cutoff=0.25, torch=torch),
            }
            for name, value in variants.items():
                prediction = _forward_model(model, value, model_spec=spec, k_points=k_points, lane_index=lane_index, torch=torch)
                predictions[name].append((prediction.cpu().numpy() * std + mean).astype(np.float32))
            for disabled in _mask_names(model_name)[1:]:
                model.set_component_mask(disabled)
                prediction = _forward_model(model, xb, model_spec=spec, k_points=k_points, lane_index=lane_index, torch=torch)
                predictions[f"mask_{disabled[0]}"].append((prediction.cpu().numpy() * std + mean).astype(np.float32))
                model.set_component_mask(())
    joined = {name: np.concatenate(parts, axis=0) for name, parts in predictions.items()}
    baseline = joined["baseline"]
    denom = np.maximum(np.abs(targets[:, :, 1:]), 1.0e-8)
    sample_error = (np.abs(baseline[:, :, 1:] - targets[:, :, 1:]) / denom).mean(axis=(1, 2)) * 100.0
    perturbations = {}
    for name, values in joined.items():
        if name == "baseline":
            continue
        perturbations[name] = {
            "prediction_delta_mae": float(np.abs(values - baseline).mean()),
            "prediction_delta_relative_pct": float((np.abs(values - baseline) / np.maximum(np.abs(baseline), 1.0e-8)).mean() * 100.0),
        }
    strata = _quartile_rows(_sample_features(grids), sample_error)
    with (out / "stratified_validation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(strata[0]))
        writer.writeheader(); writer.writerows(strata)
    resolved_snapshot = dict(config)
    resolved_snapshot["diagnostic"] = {"experiment_id": "EXP-FWD-019", "split": "validation", "test_targets_loaded": False, "fourier_cutoff": 0.25}
    (out / "config.resolved.yaml").write_text(yaml.safe_dump(resolved_snapshot, sort_keys=False, allow_unicode=True), encoding="utf-8")
    metrics = {"validation_samples": int(len(indices)), "baseline_bands_2_10_mre_pct": float(sample_error.mean()), "perturbations": perturbations}
    results = {"status": "frozen_validation_diagnostic_success", "experiment_id": "EXP-FWD-019", "stage": "C0", "model": model_name, "lane": experiment.get("lane"), "training_started": False, "test_targets_loaded": False, "mpb_meep_used": False, "elapsed_s": time.perf_counter() - started}
    write_metrics(out, metrics); write_results(out, results)
    write_run_meta(out, {"deployed_git_commit": os.environ.get("PHC2D_DEPLOYED_GIT_COMMIT"), "checkpoint": str(checkpoint_path), "checkpoint_sha256": _sha256(checkpoint_path), "config": str(config_path), "model_identity": describe_model_identity(model, model_name, model_kwargs=params), "resolved_device": str(device), "validation_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest()})
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    print(run_component_diagnostic(args.config, args.checkpoint, output_dir=args.output_dir, batch_size=args.batch_size))


if __name__ == "__main__":
    main()


__all__ = ["run_component_diagnostic"]
