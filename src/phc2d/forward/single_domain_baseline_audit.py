"""Verify historical single-domain Full checkpoints against frozen source data.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import yaml

from phc2d.forward.budget_confirmation import file_digest, inspect_dataset


def audit(manifest: dict, source: dict, protocol: dict) -> dict:
    """Audit bytes, phase provenance, model and train-only data without test labels."""
    import torch
    from phc2d.models import build_model

    rows = manifest["candidates"]
    identities = {(r["domain"], r["kernel"], r["seed"]) for r in rows}
    expected = {(d, k, s) for d in manifest["domains"] for k in (3, 7) for s in (0, 1, 2)}
    if len(rows) != len(expected) or identities != expected:
        raise ValueError("baseline candidate identities are incomplete or duplicated")
    data = {}
    records = []
    for row in rows:
        domain, kernel, seed = row["domain"], row["kernel"], row["seed"]
        if domain not in data:
            spec = source["datasets"][domain]
            data[domain] = inspect_dataset(dict(path=manifest["dataset_paths"][domain], sha256=spec["sha256"],
                                                grid_field=spec["grid_field"]), 8000, "exp-fwd-021-domain-ranking-v1")
        run = Path(row["run_dir"])
        cfg = yaml.safe_load((run / "config.resolved.yaml").read_text(encoding="utf-8"))
        result = json.loads((run / "results.json").read_text(encoding="utf-8"))
        with (run / "history.csv").open(encoding="utf-8", newline="") as stream:
            history = list(csv.DictReader(stream))
        digest = file_digest(run / "model.pt")
        if digest != row["checkpoint_sha256"] or digest != result["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {run}")
        if (result["model"] != "global_local_band_operator_v1" or int(result["seed"]) != seed
                or result["selection_split"] != "validation" or result["checkpoint_role"] != "best_validation"
                or result["test_executed"] or result["status"] not in {"continuation_training_success", "formal_training_success", "training_complete_convergence_not_met"}):
            raise ValueError(f"source checkpoint identity or status changed: {run}")
        stop = int(result["stop_epoch"])
        if not 400 <= stop <= 1000 or [int(r["epoch"]) for r in history] != list(range(1, stop + 1)):
            raise ValueError(f"incomplete training history: {run}")
        if not all(math.isfinite(float(r[key])) for r in history for key in ("train_loss", "val_bands_1_10_mre_pct", "val_bands_2_10_mre_pct")):
            raise ValueError(f"nonfinite historical training: {run}")
        t = cfg["training"]
        for key, value in {"train_size": 8000, "train_subset_salt": "exp-fwd-021-domain-ranking-v1",
                           "micro_batch_size": 64, "effective_batch_size": 64, "optimizer": "adamw",
                           "learning_rate": 0.0005, "weight_decay": 0.00001,
                           "loss": "normalized_full_matrix_mse", "checkpoint_selection": "validation_bands_2_10_mre"}.items():
            if t.get(key) != value:
                raise ValueError(f"training contract {key} changed: {run}")
        if cfg["adaptive_optimization"] != protocol:
            raise ValueError(f"adaptive protocol changed: {run}")
        if (cfg["dataset"]["expected_dataset_sha256"] != data[domain]["sha256"]
                or cfg["dataset"]["grid_field"] != data[domain]["grid_field"]
                or row["train_subset_hash"] != data[domain]["train_subset_hash"]):
            raise ValueError(f"dataset or subset identity changed: {run}")
        params = cfg["model"]["params"]
        expected_params = {**source["models"]["global_local_band_operator_v1"], "local_kernel": kernel}
        if any(params.get(k) != v for k, v in expected_params.items()):
            raise ValueError(f"architecture changed: {run}")
        payload = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
        normalizer = hashlib.sha256(payload["target_mean"].numpy().tobytes() + payload["target_std"].numpy().tobytes()).hexdigest()
        if normalizer != data[domain]["normalizer_hash"] or payload["k_points_sha256"] != data[domain]["k_points_hash"]:
            raise ValueError(f"train-only normalizer or k points differ: {run}")
        model = build_model("global_local_band_operator_v1", input_shape=payload["input_shape"],
                            output_shape=payload["output_shape"], **payload["model_params"])
        model.load_state_dict(payload["state_dict"], strict=True)
        if kernel == 3:
            warm = result["warm_start"]
            if (not warm["source_scientific_contract_verified"] or warm["optimizer_state_restored"]
                    or warm["source_epoch"] != 300 or warm["initial_weights_role"] != "best_validation"):
                raise ValueError(f"warm-start phase contract invalid: {run}")
            for path, sha in ((warm["source_best_checkpoint"], warm["source_best_checkpoint_sha256"]),
                              (warm["source_terminal_checkpoint"], warm["source_terminal_checkpoint_sha256"])):
                if file_digest(Path(path)) != sha:
                    raise ValueError(f"source phase checkpoint changed: {path}")
            if file_digest(Path(warm["source_run"]) / "config.resolved.yaml") != warm["source_config_sha256"]:
                raise ValueError("source phase config changed")
            source_run = Path(warm["source_run"])
            source_result = json.loads((source_run / "results.json").read_text(encoding="utf-8"))
            with (source_run / "history.csv").open(encoding="utf-8", newline="") as stream:
                source_history = list(csv.DictReader(stream))
            metrics = ("train_loss", "val_bands_1_10_mre_pct", "val_bands_2_10_mre_pct")
            if (source_result["stop_epoch"] != 300 or source_result["best_epoch"] != warm["initial_weights_epoch"]
                    or [int(r["epoch"]) for r in source_history] != list(range(1, 301))
                    or any(float(a[key]) != float(b[key]) for a, b in zip(source_history, history[:300]) for key in metrics)):
                raise ValueError(f"warm-start history or phase identity changed: {run}")
        else:
            phase = json.loads((run / "phase300_evidence.json").read_text())
            phase_start = json.loads((run / "adaptive_phase_start.json").read_text())
            if (file_digest(run / "phase300_best.pt") != phase["selected_checkpoint_sha256"]
                    or file_digest(run / "epoch300.pt") != phase["terminal_checkpoint_sha256"]
                    or phase_start["source_checkpoint_sha256"] != phase["selected_checkpoint_sha256"]
                    or not phase_start["optimizer_reset"] or not phase_start["random_states_reset"]
                    or phase_start["cumulative_epoch"] != 301):
                raise ValueError(f"phase300 evidence changed: {run}")
        records.append(dict(**row, actual_checkpoint_sha256=digest, normalizer_hash=normalizer,
                            data_sha256=data[domain]["sha256"], status=result["status"],
                            stop_epoch=stop, config_sha256=file_digest(run / "config.resolved.yaml")))
    return dict(status="baseline_audit_pass", records=records, datasets=data, test_targets_loaded=False)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    for name in ("manifest", "source-profile", "adaptive-profile", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source = yaml.safe_load(args.source_profile.read_text(encoding="utf-8"))
    protocol = yaml.safe_load(args.adaptive_profile.read_text(encoding="utf-8"))["protocol"]
    result = audit(manifest, source, protocol)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"], candidate_count=len(result["records"]))))


if __name__ == "__main__":
    main()
