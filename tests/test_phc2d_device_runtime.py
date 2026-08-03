"""Tests for GPU-first runtime device policy and smoke metadata.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from phc2d.forward import run_forward_eval, run_forward_train
from phc2d.inverse import run_inverse_sample
from phc2d.training import DeviceConfig, move_batch_to_device, resolve_device


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")


def _torch():
    import torch

    return torch


def _write_tiny_npz(path: Path) -> None:
    rng = np.random.default_rng(321)
    rho = rng.random((8, 10), dtype=np.float32)
    weights = rng.normal(size=(10, 6), scale=0.05).astype(np.float32)
    wh = (rho @ weights).reshape(8, 3, 2).astype(np.float32)
    np.savez(
        path,
        rho=rho,
        Wh=wh,
        idx_train=np.arange(0, 5, dtype=np.int64),
        idx_val=np.arange(5, 6, dtype=np.int64),
        idx_test=np.arange(6, 8, dtype=np.int64),
    )


def _write_forward_config(path: Path, dataset: Path, runs_dir: Path, *, device: str = "cpu", require: bool = False, checkpoint: Path | None = None) -> None:
    checkpoint_line = f"  checkpoint_path: {checkpoint.as_posix()}\n" if checkpoint else ""
    path.write_text(
        "\n".join(
            [
                "experiment:",
                "  experiment_id: unit_device_forward",
                "dataset:",
                f"  path: {dataset.as_posix()}",
                "  input_repr: rho_wedge",
                "  output_target: Wh",
                "model:",
                "  name: res_mlp",
                "  params:",
                "    hidden_dim: 8",
                "    depth: 1",
                checkpoint_line.rstrip(),
                "training:",
                "  epochs: 1",
                "  batch_size: 2",
                "  learning_rate: 0.001",
                "  seed: 5",
                f"  device: {device}",
                "runtime:",
                f"  device: {device}",
                f"  require_accelerator: {str(require).lower()}",
                "  precision: fp32",
                "  amp_enabled: false",
                "  non_blocking_transfer: true",
                "  record_peak_memory: true",
                "evaluation:",
                "  split: test",
                "output:",
                f"  runs_dir: {runs_dir.as_posix()}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_inverse_config(path: Path, dataset: Path, runs_dir: Path, *, device: str = "cpu", require: bool = False) -> None:
    path.write_text(
        "\n".join(
            [
                "experiment:",
                "  experiment_id: unit_device_inverse",
                "  stage: inverse_sample",
                "dataset:",
                f"  path: {dataset.as_posix()}",
                "  condition_target: Wh",
                "  output_repr: rho_wedge",
                "  target_split: test",
                "model:",
                "  name: conditional_vae_rho",
                "  seed: 7",
                "runtime:",
                f"  device: {device}",
                f"  require_accelerator: {str(require).lower()}",
                "  precision: fp32",
                "  amp_enabled: false",
                "inverse:",
                "  n_candidates: 4",
                "  surrogate_top_m: 2",
                "  candidate_value_range: [0.0, 1.0]",
                "  threshold: 0.5",
                "  morphology_filters:",
                "    min_fill_fraction: 0.05",
                "    max_fill_fraction: 0.95",
                "  ranking:",
                "    morphology_weight: 0.0",
                "    diversity_weight: 0.1",
                "replay:",
                "  enabled: false",
                "output:",
                f"  runs_dir: {runs_dir.as_posix()}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_device_auto_selects_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _torch()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"Unit CUDA {index}")

    info = resolve_device(DeviceConfig(device="auto", require_accelerator=True), torch_module=torch)

    assert info.resolved_device == "cuda:0"
    assert info.device_name == "Unit CUDA 0"
    assert info.accelerator_required is True


def test_device_auto_falls_back_to_cpu_when_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _torch()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    info = resolve_device(DeviceConfig(device="auto", require_accelerator=False), torch_module=torch)
    assert info.resolved_device == "cpu"
    assert info.fallback_reason == "cuda_unavailable_auto_cpu_allowed"


def test_device_auto_fails_when_accelerator_required(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _torch()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="require_accelerator=true"):
        resolve_device(DeviceConfig(device="auto", require_accelerator=True), torch_module=torch)


def test_move_batch_to_device_keeps_nested_tensors_on_one_device() -> None:
    torch = _torch()
    batch = {"x": torch.zeros(2, 3), "items": [torch.ones(1)]}
    moved = move_batch_to_device(batch, torch.device("cpu"))
    assert moved["x"].device.type == "cpu"
    assert moved["items"][0].device.type == "cpu"


def test_forward_metadata_and_checkpoint_map_location_roundtrip(tmp_path: Path) -> None:
    dataset = tmp_path / "tiny.npz"
    runs_dir = tmp_path / "runs"
    train_cfg = tmp_path / "train.yaml"
    eval_cfg = tmp_path / "eval.yaml"
    _write_tiny_npz(dataset)
    _write_forward_config(train_cfg, dataset, runs_dir, device="cpu")

    train_run = run_forward_train(train_cfg)
    _write_forward_config(eval_cfg, dataset, runs_dir, device="cpu", checkpoint=train_run / "model.pt")
    eval_run = run_forward_eval(eval_cfg)

    train_metrics = json.loads((train_run / "metrics.json").read_text(encoding="utf-8"))
    eval_metrics = json.loads((eval_run / "metrics.json").read_text(encoding="utf-8"))
    run_meta = json.loads((train_run / "run.meta.json").read_text(encoding="utf-8"))
    for payload in (train_metrics, eval_metrics, run_meta):
        assert payload["requested_device"] == "cpu"
        assert payload["resolved_device"] == "cpu"
        assert payload["precision"] == "fp32"
        assert "peak_gpu_memory_mb" in payload
    assert eval_metrics["checkpoint_loaded"] is True


def test_inverse_sample_metadata_uses_resolved_device(tmp_path: Path) -> None:
    dataset = tmp_path / "tiny.npz"
    runs_dir = tmp_path / "runs"
    config = tmp_path / "inverse.yaml"
    _write_tiny_npz(dataset)
    _write_inverse_config(config, dataset, runs_dir, device="cpu")

    run_dir = run_inverse_sample(config)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "run.meta.json").read_text(encoding="utf-8"))
    assert metrics["resolved_device"] == "cpu"
    assert meta["resolved_device"] == "cpu"
    assert metrics["candidate_sampling_time_s"] >= 0.0
    assert metrics["surrogate_ranking_time_s"] >= 0.0


def test_forward_gpu_integration_smoke(tmp_path: Path) -> None:
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dataset = tmp_path / "tiny.npz"
    config = tmp_path / "forward_gpu.yaml"
    _write_tiny_npz(dataset)
    _write_forward_config(config, dataset, tmp_path / "runs", device="auto", require=True)

    run_dir = run_forward_train(config)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["resolved_device"].startswith("cuda")


def test_inverse_gpu_integration_smoke(tmp_path: Path) -> None:
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dataset = tmp_path / "tiny.npz"
    config = tmp_path / "inverse_gpu.yaml"
    _write_tiny_npz(dataset)
    _write_inverse_config(config, dataset, tmp_path / "runs", device="auto", require=True)

    run_dir = run_inverse_sample(config)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["resolved_device"].startswith("cuda")
