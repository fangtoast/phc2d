"""Acceptance suite driver for the phc2d repository.

This script intentionally focuses on "local non-MPB by default" validation and
provides optional flags for slow checks, field checks, and WSL MPB replay pilot.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from phc2d.runs import RUN_FIGURES_DIR, make_run_dir, validate_run_artifacts


@dataclass(frozen=True)
class AcceptanceContext:
    runs_root: Path
    include_slow: bool
    include_field: bool
    include_wsl_mpb: bool
    dry_run: bool
    suite_run_dir: Path
    scratch_dir: Path


@dataclass(frozen=True)
class LayerResult:
    key: str
    name: str
    status: str
    commands: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LayerSpec:
    key: str
    name: str
    runner: Any
    requires_slow: bool = False
    requires_field: bool = False
    requires_wsl_mpb: bool = False


def _run_shell(command: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        check=False,
        capture_output=True,
        encoding="utf-8",
        env=os.environ.copy(),
        timeout=timeout,
    )


def _is_dependency_available(name: str) -> bool:
    try:
        return __import__(name) is not None
    except Exception:
        return False


def _safe_read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _build_temp_config(
    source: Path,
    *,
    context: AcceptanceContext,
    layer_key: str,
    experiment_id: str,
    runs_subdir: str,
    run_dir: Path | None = None,
) -> Path:
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Config payload is not a mapping: {source}")
    experiment = payload.get("experiment")
    if not isinstance(experiment, dict):
        experiment = {}
        payload["experiment"] = experiment
    base_experiment = str(experiment.get("experiment_id", source.stem))
    experiment["experiment_id"] = f"{base_experiment}_{layer_key}_{experiment_id}"

    output = payload.get("output")
    if not isinstance(output, dict):
        output = {}
        payload["output"] = output
    output["runs_dir"] = str(context.runs_root / runs_subdir)
    if run_dir is not None:
        output["run_dir"] = str(run_dir)
    elif "run_dir" in output:
        output.pop("run_dir", None)

    out = context.scratch_dir / f"{source.stem}_{layer_key}_{experiment_id}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def _layer_skip(context: AcceptanceContext, layer: LayerSpec, note: str) -> LayerResult:
    if context.dry_run:
        return LayerResult(layer.key, layer.name, "skipped", notes=["dry-run: not executed"])
    return LayerResult(layer.key, layer.name, "skipped", notes=[note])


def _require_fast_only(context: AcceptanceContext, layer: LayerSpec, *, field: bool = False) -> LayerResult | None:
    if context.dry_run:
        return _layer_skip(context, layer, "dry-run")
    if layer.requires_slow and not context.include_slow:
        return _layer_skip(context, layer, "Skipped by default (enable --include-slow).")
    if layer.requires_field and not context.include_field:
        return _layer_skip(context, layer, "Skipped by default (enable --include-field).")
    if layer.requires_wsl_mpb and not context.include_wsl_mpb:
        return _layer_skip(context, layer, "Skipped by default (enable --include-wsl-mpb).")
    return None


def run_layer0_code_health(context: AcceptanceContext, layer: LayerSpec) -> LayerResult:
    skip = _require_fast_only(context, layer)
    if skip is not None:
        return skip

    commands = [
        f"{sys.executable} -m compileall src tests scripts tools",
    ]
    proc = _run_shell([sys.executable, "-m", "compileall", "src", "tests", "scripts", "tools"], timeout=180)
    if proc.returncode != 0:
        return LayerResult(
            layer.key,
            layer.name,
            "fail",
            commands=commands,
            notes=[proc.stderr.strip() or proc.stdout.strip()],
        )
    return LayerResult(
        layer.key,
        layer.name,
        "pass",
        commands=commands,
    )


def _require_torch_or_skip(context: AcceptanceContext, layer: LayerSpec) -> LayerResult | None:
    if not _is_dependency_available("torch"):
        return _layer_skip(context, layer, "Dependency not installed: torch.")
    return None


def run_layer1_data_schema(context: AcceptanceContext, layer: LayerSpec) -> LayerResult:
    skip = _require_fast_only(context, layer)
    if skip is not None:
        return skip

    outputs: list[str] = []
    notes: list[str] = []

    commands: list[str] = [
        f"{sys.executable} -m phc2d.cli data inspect --path data/tiny.npz",
    ]
    proc = _run_shell([sys.executable, "-m", "phc2d.cli", "data", "inspect", "--path", "data/tiny.npz"])
    if proc.returncode != 0:
        return LayerResult(layer.key, layer.name, "fail", commands=commands, notes=[proc.stderr.strip() or proc.stdout.strip()])
    try:
        payload = json.loads(proc.stdout)
        if "n_samples" not in payload:
            raise ValueError("missing key n_samples")
    except Exception as exc:
        return LayerResult(layer.key, layer.name, "fail", commands=commands, notes=[str(exc)])
    outputs.append("data/tiny.npz")

    if _is_dependency_available("h5py"):
        field_path = Path("data/field_tiny_mpb.h5")
        if field_path.exists():
            commands.append(f"{sys.executable} -m phc2d.cli data inspect --path {field_path}")
            proc = _run_shell(
                [sys.executable, "-m", "phc2d.cli", "data", "inspect", "--path", str(field_path)],
                timeout=60,
            )
            if proc.returncode != 0:
                notes.append(f"h5 field inspect failed: {proc.stderr.strip() or proc.stdout.strip()}")
            else:
                outputs.append(str(field_path))
        else:
            notes.append("Optional field dataset data/field_tiny_mpb.h5 is absent, skipped.")
    else:
        notes.append("Optional dependency h5py not installed, skipped field inspect.")

    status = "pass"
    if any(note.startswith("h5 field inspect failed") for note in notes):
        status = "fail"
    return LayerResult(layer.key, layer.name, status, commands=commands, outputs=outputs, notes=notes)


def run_layer2_registry_and_shape(context: AcceptanceContext, layer: LayerSpec) -> LayerResult:
    skip = _require_fast_only(context, layer)
    if skip is not None:
        return skip

    commands = [f"{sys.executable} -m phc2d.cli models list"]
    proc = _run_shell([sys.executable, "-m", "phc2d.cli", "models", "list"])
    if proc.returncode != 0:
        return LayerResult(layer.key, layer.name, "fail", commands=commands, notes=[proc.stderr.strip() or proc.stdout.strip()])
    try:
        payload = json.loads(proc.stdout)
    except Exception as exc:
        return LayerResult(layer.key, layer.name, "fail", commands=commands, notes=[str(exc)])

    names = {item.get("name") for item in payload if isinstance(item, dict)}
    expected = {"res_mlp", "lifted_cnn", "fno_lite", "fno2d_band"}
    missing = sorted(expected - names)
    if missing:
        return LayerResult(layer.key, layer.name, "fail", commands=commands, notes=[f"missing registry specs: {', '.join(missing)}"])

    no_torch = _require_torch_or_skip(context, layer)
    if no_torch is not None:
        commands.append("torch import check")
        return no_torch

    from phc2d.models import build_model, get_model_spec

    spec = get_model_spec("res_mlp")
    input_shape = tuple(int(v) for v in spec.defaults["input_shape"])
    output_shape = tuple(int(v) for v in spec.defaults["output_shape"])
    import torch

    model = build_model("res_mlp", **spec.defaults)
    x = torch.ones((1, *input_shape), dtype=torch.float32)
    y = model(x)
    if tuple(int(v) for v in y.shape[1:]) != output_shape:
        return LayerResult(
            layer.key,
            layer.name,
            "fail",
            commands=commands,
            notes=[f"Shape mismatch: got {tuple(y.shape)}, expect (*, {output_shape})"],
        )
    return LayerResult(layer.key, layer.name, "pass", commands=commands + ["torch shape smoke"])


def run_layer3_forward_smoke(context: AcceptanceContext, layer: LayerSpec) -> LayerResult:
    skip = _require_fast_only(context, layer)
    if skip is not None:
        return skip

    no_torch = _require_torch_or_skip(context, layer)
    if no_torch is not None:
        return no_torch

    source = Path("configs/smoke/forward_tiny_fno_lite.yaml")
    if not source.exists():
        return LayerResult(layer.key, layer.name, "fail", notes=[f"Config missing: {source}"])

    from phc2d.forward import run_forward_train, run_forward_eval
    token = datetime.now().strftime("%H%M%S")
    train_cfg = _build_temp_config(
        source,
        context=context,
        layer_key=layer.key,
        experiment_id=f"{token}_forward",
        runs_subdir="layer3_forward",
    )
    eval_cfg = _build_temp_config(
        source,
        context=context,
        layer_key=layer.key,
        experiment_id=f"{token}_forward",
        runs_subdir="layer3_forward",
    )

    commands = [
        f"{sys.executable} -m phc2d.cli forward train --config {train_cfg}",
        f"{sys.executable} -m phc2d.cli forward eval --config {eval_cfg}",
    ]
    try:
        train_run = run_forward_train(train_cfg)
        missing = validate_run_artifacts(train_run)
        if missing:
            return LayerResult(
                layer.key,
                layer.name,
                "fail",
                commands=commands,
                notes=[f"Missing artifacts: {', '.join(missing)}"],
            )
        # Keep same experiment id for checkpoint continuity.
        eval_cfg_payload = yaml.safe_load(eval_cfg.read_text(encoding="utf-8")) or {}
        if isinstance(eval_cfg_payload.get("output"), dict):
            eval_cfg_payload["output"]["run_dir"] = str(train_run / "eval")
        if isinstance(eval_cfg_payload.get("model"), dict):
            eval_cfg_payload["model"]["checkpoint_path"] = str(train_run / "model.pt")
        else:
            eval_cfg_payload["model"] = {"checkpoint_path": str(train_run / "model.pt")}
        eval_cfg.write_text(yaml.safe_dump(eval_cfg_payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
        eval_run = run_forward_eval(eval_cfg)
        missing = validate_run_artifacts(eval_run)
        if missing:
            return LayerResult(
                layer.key,
                layer.name,
                "fail",
                commands=commands,
                notes=[f"Missing artifacts: {', '.join(missing)}"],
            )
    except Exception as exc:
        return LayerResult(layer.key, layer.name, "fail", commands=commands, notes=[str(exc)])

    train_metrics = _safe_read_json(train_run / "metrics.json")
    eval_metrics = _safe_read_json(eval_run / "metrics.json") if eval_run.exists() else {}
    if "train_time_s" not in train_metrics:
        return LayerResult(layer.key, layer.name, "fail", commands=commands, notes=["Missing train_time_s"])
    if "eval_time_s" not in eval_metrics:
        return LayerResult(layer.key, layer.name, "fail", commands=commands, notes=["Missing eval_time_s"])

    return LayerResult(
        layer.key,
        layer.name,
        "pass",
        commands=commands,
        outputs=[str(train_run), str(eval_run)],
    )


def run_layer4_representation_benchmark(context: AcceptanceContext, layer: LayerSpec) -> LayerResult:
    skip = _require_fast_only(context, layer)
    if skip is not None:
        return skip

    no_torch = _require_torch_or_skip(context, layer)
    if no_torch is not None:
        return no_torch

    source = Path("configs/research/representation_ablation.yaml")
    if not source.exists():
        return LayerResult(layer.key, layer.name, "fail", notes=[f"Config missing: {source}"])
    bench_cfg = _build_temp_config(
        source,
        context=context,
        layer_key=layer.key,
        experiment_id="smoke",
        runs_subdir="layer4_benchmark",
    )

    from phc2d.forward import run_forward_benchmark

    commands = [f"{sys.executable} -m phc2d.cli forward benchmark --config {bench_cfg}"]
    try:
        run_dir = run_forward_benchmark(bench_cfg)
    except Exception as exc:
        return LayerResult(layer.key, layer.name, "fail", commands=commands, notes=[str(exc)])

    if not (run_dir / "leaderboard.csv").is_file():
        return LayerResult(layer.key, layer.name, "fail", commands=commands, notes=["leaderboard.csv missing"])
    if not (run_dir / "per_band_metrics.csv").is_file():
        return LayerResult(layer.key, layer.name, "fail", commands=commands, notes=["per_band_metrics.csv missing"])
    return LayerResult(
        layer.key,
        layer.name,
        "pass",
        commands=commands,
        outputs=[str(run_dir / "leaderboard.csv"), str(run_dir / "per_band_metrics.csv")],
    )


def run_layer5_near_field(context: AcceptanceContext, layer: LayerSpec) -> LayerResult:
    skip = _require_fast_only(context, layer)
    if skip is not None:
        return skip
    if not _is_dependency_available("h5py"):
        return _layer_skip(context, layer, "Dependency not installed: h5py.")

    source = Path("configs/smoke/near_field_tiny_unet.yaml")
    field_source = Path("data/field_tiny_mpb.h5")
    if not source.exists():
        return LayerResult(layer.key, layer.name, "fail", notes=[f"Config missing: {source}"])
    if not field_source.exists():
        return LayerResult(layer.key, layer.name, "fail", notes=[f"Dataset missing: {field_source}"])

    layer_cfg = _build_temp_config(
        source,
        context=context,
        layer_key=layer.key,
        experiment_id="field",
        runs_subdir="layer5_field",
    )
    from phc2d.forward import run_forward_train

    commands = [f"{sys.executable} -m phc2d.cli forward train --config {layer_cfg}"]
    try:
        run_dir = run_forward_train(layer_cfg)
    except Exception as exc:
        return LayerResult(layer.key, layer.name, "fail", commands=commands, notes=[str(exc)])
    if validate_run_artifacts(run_dir):
        return LayerResult(
            layer.key,
            layer.name,
            "fail",
            commands=commands,
            notes=[f"Missing artifacts: {', '.join(validate_run_artifacts(run_dir))}"],
        )
    metrics = _safe_read_json(run_dir / "metrics.json")
    if "field_inference_time_s" not in metrics:
        return LayerResult(layer.key, layer.name, "fail", commands=commands, notes=["missing field_inference_time_s"])
    return LayerResult(layer.key, layer.name, "pass", commands=commands, outputs=[str(run_dir)])


def run_layer6_generative_inverse(context: AcceptanceContext, layer: LayerSpec) -> LayerResult:
    skip = _require_fast_only(context, layer)
    if skip is not None:
        return skip

    source = Path("configs/smoke/generative_inverse_tiny_cvae.yaml")
    if not source.exists():
        return LayerResult(layer.key, layer.name, "fail", notes=[f"Config missing: {source}"])

    inverse_cfg = _build_temp_config(
        source,
        context=context,
        layer_key=layer.key,
        experiment_id="cvae",
        runs_subdir="layer6_inverse",
    )
    from phc2d.inverse import run_inverse_sample, run_inverse_eval

    commands = [
        f"{sys.executable} -m phc2d.cli inverse sample --config {inverse_cfg}",
        f"{sys.executable} -m phc2d.cli inverse eval --config {inverse_cfg}",
    ]
    try:
        sample_run = run_inverse_sample(inverse_cfg)
        if validate_run_artifacts(sample_run):
            return LayerResult(
                layer.key,
                layer.name,
                "fail",
                commands=commands,
                notes=[f"Missing artifacts: {', '.join(validate_run_artifacts(sample_run))}"],
            )
        eval_run = run_inverse_eval(inverse_cfg)
        if validate_run_artifacts(eval_run):
            return LayerResult(
                layer.key,
                layer.name,
                "fail",
                commands=commands,
                notes=[f"Missing artifacts: {', '.join(validate_run_artifacts(eval_run))}"],
            )
    except Exception as exc:
        return LayerResult(layer.key, layer.name, "fail", commands=commands, notes=[str(exc)])

    metrics = _safe_read_json(eval_run / "metrics.json")
    if "best_of_k_surrogate_mre" not in metrics or "surrogate_replay_gap" not in metrics:
        return LayerResult(layer.key, layer.name, "fail", commands=commands, notes=["missing inverse replay metrics"])
    return LayerResult(
        layer.key,
        layer.name,
        "pass",
        commands=commands,
        outputs=[str(sample_run), str(eval_run)],
    )


def run_layer7_mpb_replay(context: AcceptanceContext, layer: LayerSpec) -> LayerResult:
    skip = _require_fast_only(context, layer)
    if skip is not None:
        return skip

    if shutil.which("wsl") is None:
        return LayerResult(layer.key, layer.name, "blocked", notes=["wsl command not available"])

    config = Path("configs/smoke/generative_inverse_tiny_cvae.yaml")
    if not config.exists():
        return LayerResult(layer.key, layer.name, "fail", notes=[f"Config missing: {config}"])

    command = [sys.executable, "-m", "phc2d.cli", "replay", "mpb", "--config", str(config)]
    proc = _run_shell(command, timeout=120)
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip()
        if "not been migrated" in msg:
            return LayerResult(layer.key, layer.name, "blocked", commands=[" ".join(command)], notes=[msg])
        return LayerResult(layer.key, layer.name, "fail", commands=[" ".join(command)], notes=[msg])
    return LayerResult(layer.key, layer.name, "pass", commands=[" ".join(command)], notes=["Replay pilot executed"])


def collect_layer_specs(include_slow: bool, include_field: bool, include_wsl_mpb: bool) -> list[LayerSpec]:
    _ = (include_slow, include_field, include_wsl_mpb)
    return [
        LayerSpec("layer_0", "Layer 0 code health", run_layer0_code_health),
        LayerSpec("layer_1", "Layer 1 data/schema inspection", run_layer1_data_schema),
        LayerSpec("layer_2", "Layer 2 model registry and shape smoke", run_layer2_registry_and_shape),
        LayerSpec("layer_3", "Layer 3 forward train/eval smoke", run_layer3_forward_smoke),
        LayerSpec("layer_4", "Layer 4 representation benchmark pilot", run_layer4_representation_benchmark, requires_slow=True),
        LayerSpec("layer_5", "Layer 5 near-field feasibility", run_layer5_near_field, requires_field=True),
        LayerSpec("layer_6", "Layer 6 generative inverse smoke", run_layer6_generative_inverse),
        LayerSpec("layer_7", "Layer 7 WSL MPB replay pilot", run_layer7_mpb_replay, requires_wsl_mpb=True),
    ]


def collect_layers(context: AcceptanceContext) -> list[LayerSpec]:
    return collect_layer_specs(context.include_slow, context.include_field, context.include_wsl_mpb)


def _to_summary(results: list[LayerResult], *, context: AcceptanceContext) -> dict[str, Any]:
    totals = {"pass": 0, "fail": 0, "skipped": 0, "blocked": 0}
    for result in results:
        totals[result.status] = totals.get(result.status, 0) + 1
    return {
        "suite": "acceptance_suite",
        "generated_at": datetime.now().isoformat(),
        "runs_root": str(context.runs_root),
        "suite_run_dir": str(context.suite_run_dir),
        "flags": {
            "include_slow": context.include_slow,
            "include_field": context.include_field,
            "include_wsl_mpb": context.include_wsl_mpb,
            "dry_run": context.dry_run,
        },
        "totals": totals,
        "layers": [
            {
                "key": item.key,
                "name": item.name,
                "status": item.status,
                "commands": item.commands,
                "outputs": item.outputs,
                "notes": item.notes,
                "details": item.details,
            }
            for item in results
        ],
    }


def _write_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# phc2d Acceptance Suite Summary",
        "",
        f"- generated_at: {summary['generated_at']}",
        f"- suite_run_dir: {summary['suite_run_dir']}",
        f"- runs_root: {summary['runs_root']}",
        "",
        f"- include_slow: {summary['flags']['include_slow']}",
        f"- include_field: {summary['flags']['include_field']}",
        f"- include_wsl_mpb: {summary['flags']['include_wsl_mpb']}",
        f"- dry_run: {summary['flags']['dry_run']}",
        "",
        f"- pass: {summary['totals']['pass']}",
        f"- fail: {summary['totals']['fail']}",
        f"- blocked: {summary['totals']['blocked']}",
        f"- skipped: {summary['totals']['skipped']}",
        "",
        "## Layer results",
        "",
    ]
    for layer in summary["layers"]:
        status = layer["status"].upper()
        lines.append(f"### {layer['name']} [{status}]")
        if layer["commands"]:
            lines.append("- commands:")
            for cmd in layer["commands"]:
                lines.append(f"  - `{cmd}`")
        if layer["outputs"]:
            lines.append("- outputs:")
            for item in layer["outputs"]:
                lines.append(f"  - `{item}`")
        if layer["notes"]:
            for note in layer["notes"]:
                lines.append(f"- {note}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_suite(context: AcceptanceContext) -> dict[str, Any]:
    layers = collect_layers(context)
    results: list[LayerResult] = []
    for layer in layers:
        result = layer.runner(context, layer)
        results.append(result)
    summary = _to_summary(results, context=context)
    run_json = context.suite_run_dir / "summary.json"
    run_md = context.suite_run_dir / "summary.md"
    run_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_markdown(summary, run_md)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phc2d acceptance suite.")
    parser.add_argument("--dry-run", action="store_true", help="Render the suite without execution.")
    parser.add_argument("--include-slow", action="store_true", help="Enable slow/pilot layers.")
    parser.add_argument("--include-field", action="store_true", help="Enable field-oriented layer.")
    parser.add_argument("--include-wsl-mpb", action="store_true", help="Enable WSL MPB replay pilot.")
    parser.add_argument("--runs-root", default="runs", help="Run root directory.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, return_summary: bool = False) -> dict[str, Any] | None:
    args = parse_args(argv)
    runs_root = Path(args.runs_root)
    now = datetime.now()
    suite_run_dir = make_run_dir(
        "acceptance_suite",
        runs_dir=runs_root,
        timestamp=now,
    )
    context = AcceptanceContext(
        runs_root=runs_root,
        include_slow=args.include_slow,
        include_field=args.include_field,
        include_wsl_mpb=args.include_wsl_mpb,
        dry_run=args.dry_run,
        suite_run_dir=suite_run_dir,
        scratch_dir=suite_run_dir / "scratch",
    )
    summary = run_suite(context)
    if return_summary:
        return summary
    print(json.dumps(summary, ensure_ascii=False))
    return None


if __name__ == "__main__":
    main()
