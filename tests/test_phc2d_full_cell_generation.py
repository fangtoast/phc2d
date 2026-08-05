"""Tests for the evidence boundary of full-cell dataset generation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
import phc2d.data.full_cell_generation as full_cell_generation

from phc2d.cli import build_parser
from phc2d.data.cluster_deployment import (
    prepare_benchmark_sample_plan,
    recommend_slurm_resources,
    select_cluster_worker_count,
    should_run_extended_worker_benchmarks,
)
from phc2d.data.full_cell_acceptance import run_cluster_wsl_acceptance, run_xingyi_binary_wsl_acceptance
from phc2d.data.full_cell import FullCellStructureSpec, load_full_cell_hdf5
from phc2d.data.full_cell_generation import run_full_cell_generation, write_combined_full_cell_manifest
from phc2d.data.xingyi_deployment import select_xingyi_worker_count


def deterministic_fake_solver(structure, *, k_points, config):
    assert isinstance(structure, FullCellStructureSpec)
    base = 0.1 + (int(structure.geometry_hash[:4], 16) % 100) / 10000.0
    bands = base + np.arange(int(config.num_bands), dtype=np.float32)[None, :] * 0.05
    bands = np.repeat(bands, len(k_points), axis=0)
    return bands, {"solver_backend": "deterministic_fake"}


def failing_full_cell_solver(structure, *, k_points, config):
    raise RuntimeError(f"intentional failure for {structure.sample_id}")


def _write_config(tmp_path: Path) -> Path:
    payload = {
        "experiment": {
            "experiment_id": "full_cell_unit",
            "family": "research",
            "stage": "dataset_generation",
        },
        "dataset": {
            "protocol": "full_cell_v1",
            "lane": "graded_medium",
            "output_path": str(tmp_path / "dataset.h5"),
            "sample_count": 4,
            "split": {"train": 2, "val": 1, "test": 1},
            "seed": 17,
            "raster_resolutions": [32, 64, 128],
        },
        "generator": {
            "epsilon_air": 1.0,
            "epsilon_material": 8.9,
            "audit_resolution": 64,
            "fourier_cutoff": 2,
            "max_attempts": 128,
        },
        "solver": {
            "backend": "mpb",
            "environment": "wsl_meep_env",
            "use_proxy": False,
            "epsilon_air": 1.0,
            "epsilon_material": 8.9,
            "resolution": 64,
            "mesh_size": 3,
            "polarization": "TE",
            "num_bands": 10,
            "band_guard_count": 4,
            "band_order_policy": "frequency_ascending",
            "eigensolver_tolerance": 1.0e-7,
        },
        "k_grid": {
            "family": "square_full_bz",
            "n_side": 9,
            "coordinate_system": "mpb_reduced_reciprocal_coordinates",
            "endpoint_policy": "centered_unique_no_boundary_duplicates",
        },
        "workers": {
            "workers": 2,
            "min_workers": 1,
            "max_workers": 2,
            "worker_backend": "thread",
            "fail_fast": False,
            "suppress_solver_output": True,
        },
        "output": {"runs_dir": str(tmp_path / "runs")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_injected_solver_writes_complete_but_unaccepted_smoke_run(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    pytest.importorskip("matplotlib")
    config = _write_config(tmp_path)
    run_dir = tmp_path / "run"
    result = run_full_cell_generation(
        config,
        run_dir=run_dir,
        skip_environment_check=True,
        solver_fn=deterministic_fake_solver,
    )

    assert result == run_dir
    for name in (
        "run.log",
        "run.meta.json",
        "config.resolved.yaml",
        "results.json",
        "metrics.json",
        "history.csv",
        "dataset_manifest.yaml",
        "solver_provenance.json",
        "sample_metadata.jsonl",
        "failed_case_ledger.jsonl",
    ):
        assert (run_dir / name).is_file(), name
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "solver_provenance.json").read_text(encoding="utf-8"))
    dataset = load_full_cell_hdf5(tmp_path / "dataset.h5")

    assert results["status"] == "test_solver_smoke_only"
    assert provenance["acceptance_eligible"] is False
    assert dataset.Wh.shape == (4, 81, 10)
    assert sorted(np.concatenate([dataset.idx_train, dataset.idx_val, dataset.idx_test]).tolist()) == [0, 1, 2, 3]


def test_solver_failure_is_preserved_and_dataset_is_not_written(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    run_dir = tmp_path / "failed_run"
    run_full_cell_generation(
        config,
        run_dir=run_dir,
        skip_environment_check=True,
        solver_fn=failing_full_cell_solver,
    )

    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    ledger = (run_dir / "failed_case_ledger.jsonl").read_text(encoding="utf-8")
    assert results["status"] == "solver_execution_failed"
    assert results["dataset_written"] is False
    assert "intentional failure" in ledger
    assert not (tmp_path / "dataset.h5").exists()


def test_resume_reconstructs_frozen_sample_plan_without_resampling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("h5py")
    pytest.importorskip("matplotlib")
    config = _write_config(tmp_path)
    run_dir = tmp_path / "resumed_run"
    run_full_cell_generation(
        config,
        run_dir=run_dir,
        skip_environment_check=True,
        solver_fn=deterministic_fake_solver,
    )
    frozen_plan = (run_dir / "sample_plan.jsonl").read_bytes()

    def fail_if_resampled(*args, **kwargs):
        raise AssertionError("resume must not resample frozen geometry")

    monkeypatch.setattr(full_cell_generation, "_generate_structures", fail_if_resampled)
    run_full_cell_generation(
        config,
        run_dir=run_dir,
        workers=1,
        skip_environment_check=True,
        solver_fn=deterministic_fake_solver,
    )

    assert (run_dir / "sample_plan.jsonl").read_bytes() == frozen_plan
    assert "structure_generation,resumed" in (run_dir / "history.csv").read_text(encoding="utf-8")


def test_data_generate_cli_parser_does_not_import_meep() -> None:
    before = set(sys.modules)
    parser = build_parser()
    namespace = parser.parse_args(["data", "generate", "--config", "pilot.yaml", "--workers", "auto"])

    assert namespace.command == "data generate"
    assert namespace.workers == "auto"
    assert "meep" not in set(sys.modules) - before


def test_native_linux_boundary_requires_slurm_and_records_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mpb = SimpleNamespace(__version__=(1, 12, 0))
    fake_meep = SimpleNamespace(__version__="1.33.0-beta", mpb=fake_mpb)
    monkeypatch.setitem(sys.modules, "meep", fake_meep)
    monkeypatch.setattr(full_cell_generation.platform, "system", lambda: "Linux")
    monkeypatch.setattr(full_cell_generation.platform, "release", lambda: "5.15.0-xingyi")
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.setenv("CONDA_PREFIX", "/envs/meep_env_xingyi")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
    solver = {
        "environment": "native_linux_conda_meep_env",
        "expected_meep_version": "1.33.0-beta",
        "expected_mpb_version": "1.12.0",
    }

    problems, profile = full_cell_generation._validate_execution_boundary(solver)
    assert any("Slurm allocation" in problem for problem in problems)

    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "cnode2451")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "amd_m9_768")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "128")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "655360")
    monkeypatch.setenv("PHC2D_CLUSTER_ID", "paracloud_bscc_m9")
    problems, profile = full_cell_generation._validate_execution_boundary(solver)
    assert problems == []
    assert profile["meep_version"] == "1.33.0-beta"
    assert profile["mpb_version"] == "1.12.0"
    assert profile["cluster_id"] == "paracloud_bscc_m9"
    assert profile["slurm_job_partition"] == "amd_m9_768"
    assert profile["slurm_cpus_per_task"] == "128"
    assert profile["slurm_mem_per_node"] == "655360"
    assert profile["formal_acceptance_eligible"] is False
    assert profile["validation_boundary"] == "native_linux_real_mpb_generation_pending_wsl_acceptance"


def test_native_cluster_generation_is_pending_wsl_acceptance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("h5py")
    pytest.importorskip("matplotlib")
    config = _write_config(tmp_path)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["solver"]["environment"] = "native_linux_conda_meep_env"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    profile = {
        "configured_environment": "native_linux_conda_meep_env",
        "validation_boundary": "native_linux_real_mpb_generation_pending_wsl_acceptance",
        "formal_acceptance_eligible": False,
        "meep_version": "1.33.0-beta",
        "mpb_version": "1.12.0",
        "slurm_job_id": "1234",
        "slurm_job_nodelist": "cnode2451",
    }
    monkeypatch.setattr(full_cell_generation, "_validate_execution_boundary", lambda solver: ([], profile))
    monkeypatch.setattr(full_cell_generation, "solve_bands_for_full_cell_mpb", deterministic_fake_solver)

    run_dir = tmp_path / "native_run"
    run_full_cell_generation(config, run_dir=run_dir, solver_fn=deterministic_fake_solver)
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "solver_provenance.json").read_text(encoding="utf-8"))
    assert results["status"] == "cluster_generated_pending_wsl_acceptance"
    assert provenance["real_mpb_generation_eligible"] is True
    assert provenance["acceptance_eligible"] is False
    assert provenance["execution_profile"]["slurm_job_id"] == "1234"


def test_xingyi_configs_preserve_physics_and_raise_only_worker_capacity() -> None:
    root = Path(__file__).resolve().parents[1]
    pairs = (
        (
            root / "configs/research/full_cell_binary_bz9_10000_v1.yaml",
            root / "configs/research/full_cell_binary_bz9_10000_xingyi_v1.yaml",
        ),
        (
            root / "configs/research/full_cell_graded_bz9_10000_v1.yaml",
            root / "configs/research/full_cell_graded_bz9_10000_xingyi_v1.yaml",
        ),
    )
    for base_path, xingyi_path in pairs:
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        xingyi = yaml.safe_load(xingyi_path.read_text(encoding="utf-8"))
        assert xingyi["experiment"] == base["experiment"]
        assert xingyi["generator"] == base["generator"]
        assert xingyi["k_grid"] == base["k_grid"]
        for key in (
            "backend",
            "use_proxy",
            "lattice",
            "lattice_constant",
            "epsilon_air",
            "epsilon_material",
            "resolution",
            "mesh_size",
            "polarization",
            "num_bands",
            "band_guard_count",
            "band_order_policy",
            "eigensolver_tolerance",
        ):
            assert xingyi["solver"][key] == base["solver"][key]
        assert xingyi["solver"]["environment"] == "native_linux_conda_meep_env"
        assert xingyi["workers"]["max_workers"] == 64
        assert xingyi["workers"]["worker_thread_env"] == base["workers"]["worker_thread_env"]


def test_graded_xingyi_job_is_fail_closed_without_authorization() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "jobs/xingyi_graded_10000_prepared_only.sh").read_text(encoding="utf-8")
    assert 'PHC2D_AUTHORIZE_GRADED:-NO' in script
    assert '!= "YES"' in script


def test_xingyi_common_requires_deployed_commit_provenance() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "jobs/xingyi_common.sh").read_text(encoding="utf-8")
    assert "DEPLOYED_GIT_COMMIT" in script
    assert "PHC2D_DEPLOYED_GIT_COMMIT" in script
    assert "PHC2D_CLUSTER_ID=xingyi" in script
    assert 'set +u\nsource "$PHC2D_ENV/bin/activate"\nset -u' in script


def test_xingyi_shell_jobs_have_batch_shebangs() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "jobs").glob("xingyi_*.sh")):
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        assert first_line == "#!/usr/bin/env bash", path.name


def test_xingyi_batch_jobs_source_common_from_stable_project_root() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in (
        "xingyi_preflight.sh",
        "xingyi_binary_benchmark.sh",
        "xingyi_binary_10000.sh",
        "xingyi_graded_10000_prepared_only.sh",
    ):
        script = (root / "jobs" / name).read_text(encoding="utf-8")
        assert 'source "$PHC2D_ROOT/jobs/xingyi_common.sh"' in script
        assert "BASH_SOURCE" not in script


def test_paracloud_m9_config_preserves_graded_physics_and_allows_256_workers() -> None:
    root = Path(__file__).resolve().parents[1]
    base = yaml.safe_load((root / "configs/research/full_cell_graded_bz9_10000_v1.yaml").read_text(encoding="utf-8"))
    m9 = yaml.safe_load(
        (root / "configs/research/full_cell_graded_bz9_10000_paracloud_m9_v1.yaml").read_text(encoding="utf-8")
    )
    assert m9["experiment"] == base["experiment"]
    assert m9["generator"] == base["generator"]
    assert m9["k_grid"] == base["k_grid"]
    for key in (
        "backend",
        "use_proxy",
        "lattice",
        "lattice_constant",
        "epsilon_air",
        "epsilon_material",
        "resolution",
        "mesh_size",
        "polarization",
        "num_bands",
        "band_guard_count",
        "band_order_policy",
        "eigensolver_tolerance",
    ):
        assert m9["solver"][key] == base["solver"][key]
    assert m9["solver"]["environment"] == "native_linux_conda_meep_env"
    assert m9["workers"]["max_workers"] == 256
    assert m9["workers"]["worker_thread_env"] == base["workers"]["worker_thread_env"]


def test_paracloud_m9_formal_job_is_fail_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "jobs/paracloud_m9_graded_10000.sh").read_text(encoding="utf-8")
    common = (root / "jobs/paracloud_m9_common.sh").read_text(encoding="utf-8")
    assert 'PHC2D_AUTHORIZE_GRADED:-NO' in script
    assert "5a046251e18bd0964ea8c70f4d52fa3bc48bff1b7e8997d4a3f17855be267ec2" in script
    assert "PHC2D_EXPECTED_GIT_COMMIT" in script
    assert "PHC2D_PREFLIGHT_RUN_DIR" in script
    assert "PHC2D_ALLOW_RESUME" in script
    assert "SLURM_TMPDIR" in common
    assert "meep_env_paracloud_m9.tar.gz" in common
    assert "b2fd8f0cf039d27d7f203bab16ae1428b9f8cd470777475c92169e6833611117" in common
    assert "conda-unpack" in common
    assert common.index('export PATH="$PHC2D_ENV/bin:$PATH"') < common.index(
        '"$PHC2D_ENV/bin/conda-unpack"'
    )
    assert "set +u\nsource \"$PHC2D_ENV/bin/activate\"\nset -u" in common
    assert "mpb_version = '.'.join(map(str, mpb.__version__))" in common
    for path in (root / "jobs").glob("paracloud_m9_*.sh"):
        assert path.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash\n")
    environment_preflight = (
        root / "jobs/paracloud_m9_environment_preflight.sh"
    ).read_text(encoding="utf-8")
    assert "ldd --version 2>&1 | sed -n '1p'" in environment_preflight
    for name in (
        "paracloud_m9_environment_preflight.sh",
        "paracloud_m9_solver_preflight.sh",
        "paracloud_m9_graded_benchmark.sh",
        "paracloud_m9_graded_10000.sh",
    ):
        entrypoint = (root / "jobs" / name).read_text(encoding="utf-8")
        assert 'source "$PHC2D_ROOT/jobs/paracloud_m9_common.sh"' in entrypoint
        assert "BASH_SOURCE" not in entrypoint


def test_xingyi_binary_acceptance_replays_fixed_strata_and_k_points(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    pytest.importorskip("matplotlib")
    generation_config = _write_config(tmp_path)
    payload = yaml.safe_load(generation_config.read_text(encoding="utf-8"))
    payload["dataset"].update(
        {
            "lane": "binary_air_holes",
            "sample_count": 8,
            "split": {"train": 6, "val": 1, "test": 1},
        }
    )
    payload["workers"].update({"workers": 1, "max_workers": 1})
    generation_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    run_full_cell_generation(
        generation_config,
        run_dir=tmp_path / "generation",
        skip_environment_check=True,
        solver_fn=deterministic_fake_solver,
    )

    acceptance_config = tmp_path / "acceptance.yaml"
    acceptance_config.write_text(
        yaml.safe_dump(
            {
                "experiment": {"experiment_id": "acceptance_unit"},
                "acceptance": {
                    "structure_count": 8,
                    "workers": 1,
                    "median_mre_max": 0.005,
                    "p95_mre_max": 0.01,
                },
                "solver": payload["solver"],
                "output": {"runs_dir": str(tmp_path / "runs")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    out = run_xingyi_binary_wsl_acceptance(
        tmp_path / "dataset.h5",
        acceptance_config,
        run_dir=tmp_path / "acceptance",
        solver_fn=deterministic_fake_solver,
        skip_environment_check=True,
    )
    evidence = json.loads((out / "cross_environment_acceptance.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "test_only_not_acceptance_eligible"
    assert set(evidence["selected_strata"]) == {"polar_hole_1", "polar_hole_2", "spectral_threshold"}
    assert len(evidence["selected_sample_ids"]) == 8
    assert len(evidence["selected_k_grid_indices"]) == 9
    assert evidence["bands_compared"] == list(range(2, 11))
    assert evidence["failure_count"] == 0
    assert evidence["median_mre"] == pytest.approx(0.0)
    assert evidence["p95_mre"] == pytest.approx(0.0)


def test_graded_cluster_acceptance_covers_both_core_strata(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    pytest.importorskip("matplotlib")
    generation_config = _write_config(tmp_path)
    payload = yaml.safe_load(generation_config.read_text(encoding="utf-8"))
    payload["dataset"].update({"sample_count": 8, "split": {"train": 6, "val": 1, "test": 1}})
    payload["workers"].update({"workers": 1, "max_workers": 1})
    generation_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    run_full_cell_generation(
        generation_config,
        run_dir=tmp_path / "generation",
        skip_environment_check=True,
        solver_fn=deterministic_fake_solver,
    )
    acceptance_config = tmp_path / "graded_acceptance.yaml"
    acceptance_config.write_text(
        yaml.safe_dump(
            {
                "experiment": {"experiment_id": "graded_acceptance_unit"},
                "acceptance": {
                    "lane": "graded_medium",
                    "structure_count": 8,
                    "workers": 1,
                    "median_mre_max": 0.005,
                    "p95_mre_max": 0.01,
                },
                "solver": payload["solver"],
                "output": {"runs_dir": str(tmp_path / "runs")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    out = run_cluster_wsl_acceptance(
        tmp_path / "dataset.h5",
        acceptance_config,
        run_dir=tmp_path / "acceptance",
        solver_fn=deterministic_fake_solver,
        skip_environment_check=True,
    )
    evidence = json.loads((out / "cross_environment_acceptance.json").read_text(encoding="utf-8"))
    assert evidence["lane"] == "graded_medium"
    assert set(evidence["selected_strata"]) == {"graded_core_1", "graded_core_2"}
    assert evidence["failure_count"] == 0
    assert evidence["finite_and_band_ordered"] is True


def test_accept_cluster_cli_parser_does_not_import_meep() -> None:
    before = set(sys.modules)
    parser = build_parser()
    namespace = parser.parse_args(
        ["data", "accept-cluster", "--dataset", "binary.h5", "--config", "acceptance.yaml"]
    )

    assert namespace.command == "data accept-cluster"
    assert "meep" not in set(sys.modules) - before


def test_xingyi_worker_selection_uses_lower_count_within_ten_percent() -> None:
    rows = [
        {"workers": 16, "throughput_samples_per_s": 1.0, "failed_count": 0, "peak_rss_gib": 80.0},
        {"workers": 32, "throughput_samples_per_s": 1.85, "failed_count": 0, "peak_rss_gib": 150.0},
        {"workers": 48, "throughput_samples_per_s": 1.95, "failed_count": 0, "peak_rss_gib": 220.0},
        {"workers": 64, "throughput_samples_per_s": 2.0, "failed_count": 0, "peak_rss_gib": 310.0},
    ]

    decision = select_xingyi_worker_count(rows)
    assert decision["fastest_workers"] == 64
    assert decision["selected_workers"] == 32


def test_xingyi_worker_selection_rejects_failures_oom_and_high_memory() -> None:
    rows = [
        {"workers": 16, "throughput_samples_per_s": 1.0, "failed_count": 0, "peak_rss_gib": 100.0},
        {"workers": 32, "throughput_samples_per_s": 2.0, "failed_count": 1, "peak_rss_gib": 150.0},
        {"workers": 48, "throughput_samples_per_s": 3.0, "failed_count": 0, "peak_rss_gib": 450.0},
        {"workers": 64, "throughput_samples_per_s": 4.0, "failed_count": 0, "peak_rss_gib": 300.0, "oom": True},
    ]

    assert select_xingyi_worker_count(rows)["selected_workers"] == 16


def test_cluster_worker_selection_extension_and_resource_recommendation() -> None:
    rows = [
        {"workers": 32, "throughput_samples_per_s": 1.0, "failed_count": 0, "peak_rss_gib": 80.0},
        {"workers": 64, "throughput_samples_per_s": 1.8, "failed_count": 0, "peak_rss_gib": 140.0},
        {"workers": 96, "throughput_samples_per_s": 2.0, "failed_count": 0, "peak_rss_gib": 220.0},
        {"workers": 128, "throughput_samples_per_s": 2.2, "failed_count": 0, "peak_rss_gib": 300.0},
    ]
    assert should_run_extended_worker_benchmarks(rows) is True
    decision = select_cluster_worker_count(rows)
    assert decision["selected_workers"] == 96
    resources = recommend_slurm_resources(decision)
    assert resources == {
        "cpus_per_task": 96,
        "memory_gib": 280,
        "time_limit_hours": 24,
        "sample_count": 10000,
        "time_factor": 1.5,
        "memory_factor": 1.25,
    }


def test_benchmark_plan_preserves_geometry_and_changes_only_subset_splits(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    pytest.importorskip("matplotlib")
    config = _write_config(tmp_path)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["dataset"].update({"sample_count": 8, "split": {"train": 6, "val": 1, "test": 1}})
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    generation = tmp_path / "generation"
    run_full_cell_generation(
        config,
        run_dir=generation,
        skip_environment_check=True,
        solver_fn=deterministic_fake_solver,
    )
    source = generation / "sample_plan.jsonl"
    output = tmp_path / "benchmark" / "sample_plan.jsonl"
    prepare_benchmark_sample_plan(source, output, config, count=4)
    source_rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    output_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(output_rows) == 4
    for source_row, output_row in zip(source_rows[:4], output_rows):
        for key in ("sample_index", "sample_id", "generator_family", "stratum", "parameters", "geometry_hash"):
            assert output_row[key] == source_row[key]
    manifest = json.loads(output.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "benchmark_plan_derived"
    assert manifest["changed_fields"] == ["split"]


def test_combined_manifest_refuses_promotion_without_convergence(tmp_path: Path) -> None:
    lane = {
        "status": "pilot_generated_pending_convergence",
        "dataset_path": "data/lane.h5",
        "dataset_sha256": "abc",
    }
    graded = tmp_path / "graded.yaml"
    binary = tmp_path / "binary.yaml"
    graded.write_text(yaml.safe_dump(lane), encoding="utf-8")
    binary.write_text(yaml.safe_dump(lane), encoding="utf-8")

    with pytest.raises(ValueError, match="resolution_convergence_pass"):
        write_combined_full_cell_manifest(
            tmp_path / "combined.yaml",
            graded_manifest=graded,
            binary_manifest=binary,
            status="pilot_pass",
        )

    convergence = tmp_path / "convergence.json"
    convergence.write_text(json.dumps({"status": "resolution_convergence_pass"}), encoding="utf-8")
    result = write_combined_full_cell_manifest(
        tmp_path / "combined.yaml",
        graded_manifest=graded,
        binary_manifest=binary,
        convergence_evidence=convergence,
        status="pilot_pass",
    )
    payload = yaml.safe_load(result.read_text(encoding="utf-8"))
    assert payload["status"] == "pilot_pass"
    assert payload["merge_arrays"] is False
    assert payload["resolution_convergence"]["status"] == "resolution_convergence_pass"
