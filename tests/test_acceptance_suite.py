"""Acceptance suite tests for phc2d validation entrypoint."""

from __future__ import annotations

from pathlib import Path

import pytest

from phc2d.runs import validate_run_artifacts

from scripts import run_acceptance_suite as suite


def _build_context(*, tmp_path: Path, dry_run: bool = False) -> suite.AcceptanceContext:
    return suite.AcceptanceContext(
        runs_root=tmp_path / "runs",
        include_slow=False,
        include_field=False,
        include_wsl_mpb=False,
        dry_run=dry_run,
        suite_run_dir=tmp_path / "runs" / "20260617" / "acceptance_suite_0000000000",
        scratch_dir=tmp_path / "runs" / "20260617" / "acceptance_suite_0000000000" / "scratch",
    )


def test_run_acceptance_suite_dry_run(tmp_path: Path) -> None:
    summary = suite.main(
        [
            "--dry-run",
            "--runs-root",
            str(tmp_path / "runs"),
        ],
        return_summary=True,
    )
    assert summary is not None
    assert summary["totals"]["skipped"] == len(summary["layers"])
    assert summary["totals"]["fail"] == 0
    suite_run_dir = Path(summary["suite_run_dir"])
    assert (suite_run_dir / "summary.json").is_file()
    assert (suite_run_dir / "summary.md").is_file()
    assert all(item["status"] == "skipped" for item in summary["layers"])
    assert all("dry-run" in " ".join(item.get("notes", [])) for item in summary["layers"])


def test_collect_layer_commands_and_metadata() -> None:
    context = _build_context(tmp_path=Path("tmp"), dry_run=True)
    specs = suite.collect_layers(context)
    keys = [spec.key for spec in specs]
    assert keys == [
        "layer_0",
        "layer_1",
        "layer_2",
        "layer_3",
        "layer_4",
        "layer_5",
        "layer_6",
        "layer_7",
    ]
    assert specs[4].requires_slow is True
    assert specs[5].requires_field is True
    assert specs[7].requires_wsl_mpb is True


def test_summary_schema(tmp_path: Path) -> None:
    summary = suite.main(["--dry-run", "--runs-root", str(tmp_path / "runs")], return_summary=True)
    assert summary["suite"] == "acceptance_suite"
    assert isinstance(summary["generated_at"], str)
    assert "runs_root" in summary
    assert "suite_run_dir" in summary
    assert set(summary["flags"].keys()) == {"include_slow", "include_field", "include_wsl_mpb", "dry_run"}
    assert set(summary["totals"]) == {"pass", "fail", "skipped", "blocked"}
    assert summary["totals"]["skipped"] == len(summary["layers"])
    for item in summary["layers"]:
        assert "key" in item
        assert "name" in item
        assert "status" in item
        assert item["status"] in {"pass", "fail", "skipped", "blocked"}
        assert "commands" in item and isinstance(item["commands"], list)
        assert "outputs" in item and isinstance(item["outputs"], list)
        assert "notes" in item and isinstance(item["notes"], list)


def test_run_layer_skip_if_optional_dependency_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    context = _build_context(tmp_path=tmp_path)
    context = suite.AcceptanceContext(
        runs_root=context.runs_root,
        include_slow=False,
        include_field=True,
        include_wsl_mpb=False,
        dry_run=False,
        suite_run_dir=context.suite_run_dir,
        scratch_dir=context.scratch_dir,
    )
    layer = suite.LayerSpec("layer_5", "Layer 5 near-field feasibility", suite.run_layer5_near_field, requires_field=True)

    original = suite._is_dependency_available

    monkeypatch.setattr(
        suite,
        "_is_dependency_available",
        lambda name: False if name == "h5py" else original(name),
    )
    result = suite.run_layer5_near_field(context, layer)
    assert result.status == "skipped"
    assert any("h5py" in item.lower() for item in result.notes)


def test_validate_run_artifacts_checker(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir(parents=True)
    for name in ["run.log", "run.meta.json", "config.resolved.yaml", "results.json", "metrics.json", "history.csv"]:
        (root / name).write_text("{}", encoding="utf-8")
    (root / "figures").mkdir()
    assert validate_run_artifacts(root) == []

    (root / "run.meta.json").unlink()
    missing = validate_run_artifacts(root)
    assert "run.meta.json" in missing
