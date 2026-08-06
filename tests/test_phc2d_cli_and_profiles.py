"""Smoke tests for the unified phc2d CLI and experiment profile contracts.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from phc2d.cli import build_parser
from phc2d.cli import main as cli_main
from phc2d.experiments import ExperimentSpec
from phc2d.replay import WSL_MEEP_ENVIRONMENT, build_wsl_meep_command
from phc2d.runs import ensure_figures_dir, write_metrics, write_resolved_config


def test_cli_help_parses_without_backend_imports(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["forward", "train", "--help"])
    assert exc.value.code == 0
    assert "Train a registered forward surrogate" in capsys.readouterr().out


def test_new_audit_and_smoke_help_do_not_import_backends(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()
    for argv, expected in (
        (["data", "make-smoke", "--help"], "synthetic dataset"),
        (["report", "docs-audit", "--help"], "project documentation"),
    ):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(argv)
        assert exc.value.code == 0
        assert expected in capsys.readouterr().out


def test_data_inspect_cli_reports_schema(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    import numpy as np

    path = tmp_path / "tiny.npz"
    np.savez(path, rho=np.zeros((2, 36), dtype=np.float32), Wh=np.ones((2, 3, 4), dtype=np.float32))

    cli_main(["data", "inspect", "--path", str(path)])
    out = capsys.readouterr().out
    assert '"n_samples": 2' in out
    assert '"k_points": 3' in out


def test_models_list_cli_reports_registry_without_torch(capsys: pytest.CaptureFixture[str]) -> None:
    cli_main(["models", "list"])
    out = capsys.readouterr().out
    assert '"name": "res_mlp"' in out
    assert '"name": "fno2d_band"' in out
    assert '"input_repr": "rho_wedge"' in out
    assert '"output_target": "Wh"' in out


def test_experiment_spec_to_dict() -> None:
    spec = ExperimentSpec(
        experiment_id="wang2026_forward_smoke",
        family="reproduction",
        stage="forward_train",
        dataset={"path": "data/tiny.npz"},
        model={"name": "pod_deeponet"},
        solver={"backend": "mpb"},
        output={"runs_dir": "runs"},
    )
    payload = spec.to_dict()
    assert payload["family"] == "reproduction"
    assert payload["stage"] == "forward_train"
    assert payload["model"]["name"] == "pod_deeponet"


def test_wsl_meep_command_shape() -> None:
    cmd = build_wsl_meep_command(
        repo_path_wsl="/mnt/d/project/article_reproduction/POD-DeepONET-2d(codex)",
        python_args=["-m", "pytest", "-q", "tests", "-m", "slow"],
    )
    assert cmd[:5] == ["wsl", "-d", "Ubuntu", "-e", "bash"]
    assert WSL_MEEP_ENVIRONMENT == "wsl_ubuntu_conda_meep_env"
    assert "conda activate meep_env" in cmd[-1]


def test_run_standard_outputs(tmp_path) -> None:
    write_metrics(tmp_path, {"mre": 0.1})
    write_resolved_config(tmp_path, "experiment_id: smoke\n")
    figures = ensure_figures_dir(tmp_path)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "config.resolved.yaml").read_text(encoding="utf-8")
    assert figures.name == "figures"


def test_replay_mpb_dispatch(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("experiment:\n  experiment_id: replay_cli\n  family: smoke\n  stage: inverse_sample\n", encoding="utf-8")

    captured: dict[str, Path] = {}

    def _fake_run(path):
        captured["path"] = Path(path)
        return tmp_path / "run"

    monkeypatch.setattr("phc2d.replay.run_replay_mpb", _fake_run)
    cli_main(["replay", "mpb", "--config", str(cfg)])
    assert captured["path"] == cfg
