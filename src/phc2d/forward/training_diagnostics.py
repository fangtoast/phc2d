"""Opt-in parameter, phase-boundary and channel-fusion evidence.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations
import importlib.metadata
import json
import shutil
from pathlib import Path
from typing import Any
from phc2d.forward.budget_confirmation import file_digest, write_json


def parameter_record(model: Any) -> dict:
    params = list(model.parameters())
    return dict(parameter_elements=sum(p.numel() for p in params),
                real_degrees_of_freedom=sum(p.numel() * (2 if p.is_complex() else 1) for p in params),
                parameter_bytes=sum(p.numel() * p.element_size() for p in params))


def gate_record(model: Any) -> dict:
    records = {}
    for name, module in model.named_modules():
        if hasattr(module, "channel_logits"):
            weights = module.channel_logits.detach().softmax(dim=0).cpu()
            records[name] = dict(weights=weights.tolist(), mean=weights.mean(dim=1).tolist(),
                                 minimum=weights.min(dim=1).values.tolist(), maximum=weights.max(dim=1).values.tolist(),
                                 std=weights.std(dim=1, unbiased=False).tolist())
    return records


def save_training_diagnostics(out: Path, model: Any, payload: dict, torch: Any) -> None:
    epoch = int(payload["epoch"])
    if not (out / "parameter_audit.json").exists():
        versions = {"torch": str(torch.__version__)}
        for name in ("neuraloperator", "tensorly", "tensorly-torch", "opt-einsum"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        write_json(out / "parameter_audit.json", dict(**parameter_record(model), versions=versions))
    gates = gate_record(model)
    if gates:
        with (out / "channel_gate_history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(epoch=epoch, gates=gates)) + "\n")
    if epoch == 300:
        torch.save({**payload, "checkpoint_role": "phase300_terminal_full_state"}, out / "epoch300.pt")
        shutil.copyfile(out / "model.pt", out / "phase300_best.pt")
        write_json(out / "phase300_evidence.json", dict(
            terminal_sha256=file_digest(out / "epoch300.pt"),
            selected_sha256=file_digest(out / "phase300_best.pt"),
            selection_metric="validation_bands_2_10_mre", training_loss="normalized_full_matrix_mse"))
