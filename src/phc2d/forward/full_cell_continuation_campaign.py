"""Control plane for EXP-FWD-009 weight-only convergence continuation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


TASKS = (
    ("convnext_band_v1", "B", "formal_12_convnext_band_v1_B_s0", "full_cell_k5_forward_009_binary_seed0_v1.yaml"),
    ("convnext_band_v1", "G", "formal_13_convnext_band_v1_G_s0", "full_cell_k5_forward_009_graded_seed0_v1.yaml"),
    ("tfno2d_band_v3", "B", "formal_14_tfno2d_band_v3_B_s0", "full_cell_k5_forward_009_binary_seed0_v1.yaml"),
    ("tfno2d_band_v3", "G", "formal_15_tfno2d_band_v3_G_s0", "full_cell_k5_forward_009_graded_seed0_v1.yaml"),
    ("dual_lane_band_operator_v1", "J", "formal_16_dual_lane_band_operator_v1_J_s0", "full_cell_k5_forward_009_dual_lane_seed0_v1.yaml"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def freeze_campaign(source_campaign: str | Path, campaign_dir: str | Path, *, repository_root: str | Path = ".") -> Path:
    root = Path(repository_root).resolve()
    source = Path(source_campaign).resolve()
    campaign = Path(campaign_dir).resolve()
    campaign.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for index, (model, lane, source_task, config_name) in enumerate(TASKS):
        source_run = source / source_task
        config = root / "configs" / "research" / config_name
        for required in (source_run / "history.csv", source_run / "results.json", source_run / "model.pt", source_run / "epoch200.pt", config):
            if not required.is_file():
                raise RuntimeError(f"missing EXP-FWD-009 freeze source: {required}")
        result = json.loads((source_run / "results.json").read_text(encoding="utf-8"))
        if result.get("model") != model or int(result.get("stop_epoch", -1)) != 200:
            raise RuntimeError(f"invalid source identity for {source_task}")
        rows.append(
            {
                "task_index": index,
                "task_id": f"continue_{index:02d}_{model}_{lane}_s0",
                "model": model,
                "lane": lane,
                "seed": 0,
                "head": "dense",
                "config": config.relative_to(root).as_posix(),
                "config_sha256": _sha256(config),
                "source_run": str(source_run),
                "source_checkpoint_sha256": _sha256(source_run / "model.pt"),
                "source_terminal_checkpoint_sha256": _sha256(source_run / "epoch200.pt"),
                "run_dir": str(campaign / f"continue_{index:02d}_{model}_{lane}_s0"),
            }
        )
    manifest = campaign / "task_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_json(
        campaign / "freeze.json",
        {
            "status": "frozen",
            "experiment_id": "EXP-FWD-009",
            "task_count": len(rows),
            "source_campaign": str(source),
            "continuation_mode": "epoch200_weights_optimizer_reset",
            "optimizer_state_restored": False,
            "source_epoch": 200,
            "max_total_epoch": 600,
            "check_interval": 20,
            "relative_threshold": 0.005,
        },
    )
    return campaign


def preflight_task(manifest: str | Path, task_index: int, *, repository_root: str | Path = ".") -> Path:
    root = Path(repository_root).resolve()
    row = _read_manifest(Path(manifest))[int(task_index)]
    import torch

    from phc2d.forward.full_cell_warm_start import load_weight_only_warm_start
    from phc2d.models import build_model, describe_model_identity, get_model_spec

    warm = load_weight_only_warm_start(
        row["source_run"],
        expected_model=row["model"],
        expected_seed=int(row["seed"]),
        torch_module=torch,
    )
    payload = warm["terminal_payload"]
    spec = get_model_spec(row["model"])
    kwargs = dict(payload.get("model_params") or {})
    if spec.accepts_k_points:
        kwargs["k_points"] = payload.get("k_points")
    model = build_model(
        row["model"],
        input_shape=tuple(payload["input_shape"]),
        output_shape=tuple(payload["output_shape"]),
        **kwargs,
    ).to("cuda:0")
    model.load_state_dict(payload["state_dict"], strict=True)
    identity = describe_model_identity(model, row["model"], spec, model_kwargs=payload.get("model_params") or {})
    source_identity = warm["source_result"].get("model_identity") or {}
    if identity["architecture_fingerprint"] != source_identity.get("architecture_fingerprint"):
        raise RuntimeError("warm-start architecture fingerprint mismatch")
    output = Path(manifest).resolve().parent / f"preflight_{int(task_index):02d}.json"
    _write_json(
        output,
        {
            "status": "pass",
            "task_id": row["task_id"],
            "model": row["model"],
            "lane": row["lane"],
            "source_terminal_checkpoint_sha256": warm["source_terminal_checkpoint_sha256"],
            "architecture_fingerprint": identity["architecture_fingerprint"],
            "optimizer_state_restored": False,
            "device": torch.cuda.get_device_name(0),
        },
    )
    return output


def run_task(manifest: str | Path, task_index: int, *, repository_root: str | Path = ".") -> Path:
    root = Path(repository_root).resolve()
    row = _read_manifest(Path(manifest))[int(task_index)]
    if not (Path(manifest).resolve().parent / f"preflight_{int(task_index):02d}.json").is_file():
        raise RuntimeError("training requires the matching GPU preflight artifact")
    module = "phc2d.forward.full_cell_dual_lane" if row["lane"] == "J" else "phc2d.forward.full_cell"
    command = ["python", "-m", module, "--config", row["config"], "--seed", row["seed"], "--run-dir", row["run_dir"], "--warm-start-run", row["source_run"]]
    if row["lane"] == "J":
        command += ["--head-kind", row["head"]]
    else:
        command += ["--model", row["model"]]
        if row["model"] in {"convnext_band_v1", "tfno2d_band_v3"}:
            command += ["--head-kind", row["head"]]
    subprocess.run(command, cwd=root, check=True)
    return Path(row["run_dir"])


def finalize_evaluation_inventory(
    source_campaign: str | Path,
    continuation_campaign: str | Path,
    *,
    output_path: str | Path | None = None,
) -> Path:
    """Freeze the final 17-checkpoint K5 evaluation inventory.

    The twelve untouched identities retain their EXP-FWD-008 best-validation
    checkpoints. The five identities listed in ``TASKS`` are replaced by the
    converged EXP-FWD-009 best-validation checkpoints.
    """

    source = Path(source_campaign).resolve()
    continuation = Path(continuation_campaign).resolve()
    base_rows = _read_manifest(source / "task_manifest.csv")
    continuation_rows = _read_manifest(continuation / "task_manifest.csv")
    replacements = {(row["model"], row["lane"], int(row["seed"])): row for row in continuation_rows}
    expected_replacements = {(model, lane, 0) for model, lane, _task, _config in TASKS}
    if len(base_rows) != 17 or set(replacements) != expected_replacements:
        raise RuntimeError("EXP-FWD-009 evaluation inventory source matrix mismatch")

    inventory: list[dict[str, Any]] = []
    for base in base_rows:
        identity_key = (base["model"], base["train_lane"], int(base["seed"]))
        replacement = replacements.get(identity_key)
        run = Path(replacement["run_dir"] if replacement else base["run_dir"])
        result_path, checkpoint = run / "results.json", run / "model.pt"
        if not result_path.is_file() or not checkpoint.is_file():
            raise RuntimeError(f"missing evaluation checkpoint evidence: {run}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if replacement:
            if result.get("status") != "continuation_training_success":
                raise RuntimeError(f"continuation checkpoint has not converged: {run}")
        elif result.get("status") not in {
            "training_success",
            "dual_lane_training_success",
            "training_complete_convergence_not_met",
        }:
            raise RuntimeError(f"invalid EXP-FWD-008 checkpoint status: {run}")
        model_identity = result.get("model_identity")
        required_identity = {
            "canonical_model_name",
            "factory_function_identity",
            "state_dict_schema_hash",
            "architecture_fingerprint",
            "parameter_count",
        }
        if not isinstance(model_identity, Mapping) or not required_identity.issubset(model_identity):
            raise RuntimeError(f"incomplete checkpoint identity: {run}")
        inventory.append(
            {
                **base,
                "checkpoint_path": checkpoint.as_posix(),
                "checkpoint_sha256": _sha256(checkpoint),
                "checkpoint_role": "best_validation",
                "selection_split": "validation",
                "evaluation_checkpoint_source_experiment": "EXP-FWD-009" if replacement else "EXP-FWD-008",
                "evaluation_checkpoint_source_status": result["status"],
                "evaluation_checkpoint_stop_epoch": result.get("stopped_total_epoch", result.get("stop_epoch")),
                **{key: model_identity[key] for key in required_identity},
            }
        )
    identities = {(row["model"], row["train_lane"], int(row["seed"])) for row in inventory}
    if len(inventory) != 17 or len(identities) != 17:
        raise RuntimeError("final K5 evaluation checkpoint identity collision")
    output = Path(output_path).resolve() if output_path else continuation / "evaluation_checkpoint_inventory.csv"
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite frozen evaluation inventory: {output}")
    _write_csv(output, inventory)
    _write_json(
        output.with_suffix(".json"),
        {
            "status": "evaluation_checkpoint_inventory_frozen",
            "checkpoint_count": 17,
            "exp_fwd_008_checkpoint_count": 12,
            "exp_fwd_009_checkpoint_count": 5,
            "source_campaign": str(source),
            "continuation_campaign": str(continuation),
            "inventory_sha256": _sha256(output),
        },
    )
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze"); freeze.add_argument("--source-campaign", required=True); freeze.add_argument("--campaign-dir", required=True); freeze.add_argument("--repository-root", default=".")
    preflight = sub.add_parser("preflight-task"); preflight.add_argument("--manifest", required=True); preflight.add_argument("--task-index", type=int, required=True); preflight.add_argument("--repository-root", default=".")
    run = sub.add_parser("run-task"); run.add_argument("--manifest", required=True); run.add_argument("--task-index", type=int, required=True); run.add_argument("--repository-root", default=".")
    inventory = sub.add_parser("finalize-evaluation-inventory"); inventory.add_argument("--source-campaign", required=True); inventory.add_argument("--continuation-campaign", required=True); inventory.add_argument("--output-path")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        print(freeze_campaign(args.source_campaign, args.campaign_dir, repository_root=args.repository_root))
    elif args.command == "preflight-task":
        print(preflight_task(args.manifest, args.task_index, repository_root=args.repository_root))
    elif args.command == "run-task":
        print(run_task(args.manifest, args.task_index, repository_root=args.repository_root))
    else:
        print(finalize_evaluation_inventory(args.source_campaign, args.continuation_campaign, output_path=args.output_path))


if __name__ == "__main__":
    main()


__all__ = ["finalize_evaluation_inventory", "freeze_campaign", "preflight_task", "run_task"]
