"""Source compatibility and terminal evidence for adaptive Full-Wh runs.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from phc2d.models import describe_model_identity, get_model_spec


def initialize_adaptive_phase(model: Any, optimizer: Any, generator: Any, out: Path, seed: int, torch: Any) -> None:
    """Uniformly restart every architecture from its selected epoch-300-budget weights."""
    payload = torch.load(out / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"])
    optimizer.state.clear()
    generator.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import json
    (out / "adaptive_phase_start.json").write_text(json.dumps(dict(
        cumulative_epoch=301, selected_weights_epoch=payload["epoch"],
        optimizer_reset=True, random_states_reset=True,
        source_role="validation_best_within_first_300_epochs"), indent=2) + "\n", encoding="utf-8")


def verify_warm_start_contract(warm_start: dict, resolved: dict, contract: dict, torch: Any) -> None:
    """Reject changed architecture, split, preprocessing or train-only normalizers."""
    source = yaml.safe_load((Path(warm_start["source_run"]) / "config.resolved.yaml").read_text(encoding="utf-8"))
    if source["model"] != resolved["model"] or source["representation"] != resolved["representation"]:
        raise ValueError("warm-start architecture or representation changed")
    if source["dataset"] != resolved["dataset"]:
        raise ValueError("warm-start dataset or split contract changed")
    allowed = {"max_epochs", "required_stop_epoch", "convergence_window", "convergence_metric",
               "convergence_relative_threshold", "require_terminal_checkpoint", "max_train_batches"}
    for key, value in source["training"].items():
        if key not in allowed and resolved["training"].get(key) != value:
            raise ValueError(f"warm-start training contract changed: {key}")
    if warm_start["source_result"]["train_subset_hash"] != contract["train_subset_hash"]:
        raise ValueError("warm-start selected training subset changed")
    payload = warm_start["terminal_payload"]
    for name in ("target_mean", "target_std"):
        if not np.array_equal(payload[name].cpu().numpy(), contract[name]):
            raise ValueError(f"warm-start train-only normalizer changed: {name}")
    if payload["k_points_sha256"] != contract["k_points_sha256"]:
        raise ValueError("warm-start k-point identity changed")
    warm_start["source_scientific_contract_verified"] = True


def save_adaptive_terminal(out: Path, model: Any, resolved: dict, contract: dict, epoch: int, convergence: dict, torch: Any) -> Path:
    """Store terminal weights separately from validation-selected inference weights."""
    path = out / f"epoch{epoch}.pt"
    temporary = path.with_suffix(".pt.tmp")
    torch.save(dict(state_dict=model.state_dict(), epoch=epoch, checkpoint_role="terminal_epoch",
                    model_name=resolved["model"]["name"], model_params=resolved["model"]["params"],
                    model_identity=describe_model_identity(model, resolved["model"]["name"],
                        get_model_spec(resolved["model"]["name"]), model_kwargs=resolved["model"]["params"]),
                    input_shape=contract["grid_shape"] if resolved["representation"]["layout"] == "image" else (int(np.prod(contract["grid_shape"])),),
                    output_shape=contract["target_shape"],
                    resolved_config=resolved, convergence=convergence,
                    target_mean=torch.as_tensor(contract["target_mean"]),
                    target_std=torch.as_tensor(contract["target_std"]),
                    k_points_sha256=contract["k_points_sha256"]), temporary)
    os.replace(temporary, path)
    return path
