"""Contract tests for EXP-DAT-006 paired K8/K10/K15 datasets.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from phc2d.data.full_cell_fourier import independent_half_plane_modes
from phc2d.data.full_cell_generation import (
    _validate_config,
    load_full_cell_generation_config,
    prepare_full_cell_sample_plan,
)
from phc2d.data.full_cell_pairing import audit_fourier_pairing_plans
from phc2d.generators.full_cell import (
    generate_unified_fourier_binary,
    generate_unified_fourier_graded,
)


ROOT = Path(__file__).resolve().parents[1]
CUTOFFS = (8, 10, 15)


def _generator_config(lane: str, cutoff: int) -> dict[str, object]:
    config: dict[str, object] = {
        "fourier_cutoff": cutoff,
        "audit_resolution": 32,
        "paired_cohort_seed": 551 if lane == "binary_fourier" else 552,
        "pairing_contract": "cutoff_paired_cohort_v1",
        "binary_target_material_fraction_range": [0.50, 0.90],
        "graded_target_material_fraction_range": [0.60, 0.78],
        "graded_temperature_range": [0.08, 0.16],
        "target_material_fraction_tolerance": 0.01,
        "minimum_gray_fraction": 0.01,
        "bias_bisection_iterations": 80,
    }
    return config


def _generate(lane: str, cutoff: int, index: int):
    prefix = "EXP-DAT-006-B" if lane == "binary_fourier" else "EXP-DAT-006-G"
    sample_id = f"{prefix}_{index:06d}"
    config = _generator_config(lane, cutoff)
    if lane == "binary_fourier":
        return generate_unified_fourier_binary(
            sample_id,
            master_seed=101,
            material_fraction_bin=index % 4,
            config=config,
            pairing_key=sample_id,
        )
    return generate_unified_fourier_graded(
        sample_id,
        master_seed=102,
        material_fraction_bin=index % 4,
        temperature_bin=index % 4,
        config=config,
        pairing_key=sample_id,
    )


def _mode_map(structure) -> dict[tuple[int, int], tuple[float, float]]:
    return {
        (int(item["mx"]), int(item["my"])): (float(item["cos"]), float(item["sin"]))
        for item in structure.parameters["modes"]
    }


def test_exp_dat_006_configs_and_registry_freeze_the_contract() -> None:
    registry = yaml.safe_load((ROOT / "experiments/registry.yaml").read_text(encoding="utf-8"))
    entry = next(item for item in registry["experiments"] if item["id"] == "EXP-DAT-006")
    assert entry["status"] == "planned"
    assert entry["decision"]["state"] == "config_ready"
    assert len(entry["config"]) == 12

    for lane, seed, prefix, cohort_seed in (
        ("binary", 2026091501, "EXP-DAT-006-B", 2026091551),
        ("graded", 2026091502, "EXP-DAT-006-G", 2026091552),
    ):
        for cutoff in CUTOFFS:
            path = ROOT / (
                f"configs/research/full_cell_{lane}_fourier_k{cutoff}_bz9_10000_xingyi_v1.yaml"
            )
            config = load_full_cell_generation_config(path)
            assert _validate_config(config) == []
            assert config["dataset"]["sample_count"] == 10_000
            assert config["dataset"]["split"] == {"train": 8000, "val": 1000, "test": 1000}
            assert config["dataset"]["seed"] == seed
            assert config["dataset"]["sample_id_prefix"] == prefix
            assert config["dataset"]["raster_resolutions"] == [32, 64, 128, 256]
            assert config["generator"]["fourier_cutoff"] == cutoff
            assert config["generator"]["paired_cohort_seed"] == cohort_seed
            assert config["generator"]["pairing_contract"] == "cutoff_paired_cohort_v1"
            assert "paired_coefficient_seed" not in config["generator"]
            assert config["solver"]["resolution"] == 64
            assert config["solver"]["num_bands"] == 10
            assert config["solver"]["band_guard_count"] == 4
            assert config["k_grid"]["n_side"] == 9
            assert config["workers"]["workers"] == 64

            acceptance_path = ROOT / (
                f"configs/research/full_cell_{lane}_fourier_k{cutoff}_"
                "bz9_10000_wsl_acceptance_v1.yaml"
            )
            acceptance = yaml.safe_load(acceptance_path.read_text(encoding="utf-8"))
            assert acceptance["acceptance"]["expected_fourier_cutoff"] == cutoff
            assert acceptance["acceptance"]["expected_sample_id_prefix"] == prefix
            assert acceptance["acceptance"]["expected_paired_cohort_seed"] == cohort_seed
            assert acceptance["acceptance"]["structure_count"] == 8
            assert acceptance["acceptance"]["median_mre_max"] == 0.005
            assert acceptance["acceptance"]["p95_mre_max"] == 0.01


@pytest.mark.parametrize("lane", ["binary_fourier", "graded_fourier"])
def test_bulk_pairing_shares_targets_and_existing_modes_but_not_samples(lane: str) -> None:
    paired = [_generate(lane, cutoff, 0) for cutoff in CUTOFFS]
    assert [len(item.parameters["modes"]) for item in paired] == [98, 158, 354]
    assert [len(independent_half_plane_modes(cutoff)) for cutoff in CUTOFFS] == [98, 158, 354]
    assert len({item.parameters["target_material_fraction"] for item in paired}) == 1
    if lane == "graded_fourier":
        assert len({item.parameters["temperature"] for item in paired}) == 1

    mode_maps = [_mode_map(item) for item in paired]
    assert set(mode_maps[0]) < set(mode_maps[1]) < set(mode_maps[2])
    assert all(mode_maps[1][key] == value for key, value in mode_maps[0].items())
    assert all(mode_maps[2][key] == value for key, value in mode_maps[1].items())
    other = _generate(lane, 8, 1)
    assert _mode_map(other) != mode_maps[0]


def test_bulk_pairing_is_optional_and_rejects_the_legacy_pair_seed_combination() -> None:
    legacy = generate_unified_fourier_binary(
        "legacy_000000",
        master_seed=17,
        material_fraction_bin=0,
        config={
            "fourier_cutoff": 3,
            "audit_resolution": 32,
            "binary_target_material_fraction_range": [0.50, 0.90],
            "target_material_fraction_tolerance": 0.01,
        },
    )
    assert "paired_cohort_seed" not in legacy.parameters
    assert "pairing" not in legacy.metadata

    with pytest.raises(ValueError, match="mutually exclusive"):
        generate_unified_fourier_binary(
            "invalid",
            master_seed=17,
            material_fraction_bin=0,
            pairing_key="invalid",
            config={
                "fourier_cutoff": 3,
                "audit_resolution": 32,
                "paired_coefficient_seed": 1,
                "paired_cohort_seed": 2,
                "binary_target_material_fraction_range": [0.50, 0.90],
            },
        )


def test_sample_plan_persists_pairing_identity_and_prefix(tmp_path: Path) -> None:
    source = ROOT / "configs/research/full_cell_binary_fourier_k8_bz9_10000_xingyi_v1.yaml"
    config = load_full_cell_generation_config(source)
    config["experiment"]["experiment_id"] = "exp_dat_006_pairing_plan_test"
    config["dataset"]["sample_count"] = 4
    config["dataset"]["split"] = {"train": 4, "val": 0, "test": 0}
    config["dataset"]["output_path"] = str(tmp_path / "unused.h5")
    config["generator"]["audit_resolution"] = 32
    config["generator"]["plan_workers"] = 1
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    run_dir = prepare_full_cell_sample_plan(config_path, run_dir=tmp_path / "plan")
    rows = [
        json.loads(line)
        for line in (run_dir / "sample_plan.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads((run_dir / "sample_plan_manifest.json").read_text(encoding="utf-8"))
    assert [row["sample_id"] for row in rows] == [
        f"EXP-DAT-006-B_{index:06d}" for index in range(4)
    ]
    assert all(row["pairing"]["pairing_key"] == row["sample_id"] for row in rows)
    assert all(row["parameters"]["paired_cohort_seed"] == 2026091551 for row in rows)
    assert manifest["pairing"] == {
        "contract": "cutoff_paired_cohort_v1",
        "sample_id_prefix": "EXP-DAT-006-B",
        "paired_cohort_seed": 2026091551,
    }


def test_streaming_cross_k_pairing_audit_passes_and_detects_drift(tmp_path: Path) -> None:
    run_dirs = []
    for lane in ("binary_fourier", "graded_fourier"):
        prefix = "EXP-DAT-006-B" if lane == "binary_fourier" else "EXP-DAT-006-G"
        seed = 101 if lane == "binary_fourier" else 102
        cohort_seed = 551 if lane == "binary_fourier" else 552
        for cutoff in CUTOFFS:
            run_dir = tmp_path / f"{lane}_k{cutoff}"
            run_dir.mkdir()
            config = {
                "dataset": {
                    "protocol": "full_cell_fourier_v2",
                    "lane": lane,
                    "sample_count": 2,
                    "split": {"train": 2, "val": 0, "test": 0},
                    "seed": seed,
                    "sample_id_prefix": prefix,
                },
                "generator": {
                    "pairing_contract": "cutoff_paired_cohort_v1",
                    "paired_cohort_seed": cohort_seed,
                    "fourier_cutoff": cutoff,
                },
            }
            (run_dir / "config.resolved.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            rows = []
            for index in range(2):
                structure = _generate(lane, cutoff, index)
                rows.append(
                    {
                        "sample_id": structure.sample_id,
                        "sample_index": index,
                        "split": "train",
                        "stratum": f"q{index}",
                        "parameters": structure.parameters,
                        "pairing": structure.metadata["pairing"],
                    }
                )
            (run_dir / "sample_plan.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            run_dirs.append(run_dir)

    output = tmp_path / "audit.json"
    audit = audit_fourier_pairing_plans(
        run_dirs,
        expected_cutoffs=list(CUTOFFS),
        output_path=output,
    )
    assert audit["status"] == "passed"
    assert audit["plan_count"] == 6
    assert audit["lanes"]["binary_fourier"]["mode_counts"] == {
        "8": 98,
        "10": 158,
        "15": 354,
    }
    assert json.loads(output.read_text(encoding="utf-8")) == audit

    drift_path = tmp_path / "binary_fourier_k10" / "sample_plan.jsonl"
    drift_rows = [json.loads(line) for line in drift_path.read_text(encoding="utf-8").splitlines()]
    drift_rows[0]["parameters"]["modes"][0]["cos"] += 1.0
    drift_path.write_text("".join(json.dumps(row) + "\n" for row in drift_rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Shared Fourier coefficients differ"):
        audit_fourier_pairing_plans(run_dirs, expected_cutoffs=list(CUTOFFS))


def test_xingyi_scheduler_is_staged_bounded_and_dry_run_first() -> None:
    submit = (ROOT / "jobs/xingyi_submit_exp_dat_006.sh").read_text(encoding="utf-8")
    smoke = (ROOT / "jobs/xingyi_exp_dat_006_smoke.sh").read_text(encoding="utf-8")
    assert 'PHC2D_SUBMIT_MODE="${PHC2D_SUBMIT_MODE:-dry-run}"' in submit
    assert "--array=0-7" in submit
    assert "PHC2D_AUTHORIZE_EXP_DAT_006_FORMAL" in submit
    assert "PHC2D_EXP_DAT_006_AUTHORIZATION_ID" in submit
    assert "PHC2D_AUTHORIZE_FORMAL=$formal_worker_auth" in submit
    assert "smoke/preflight.env" in submit
    assert "wave_audit.env" in submit
    assert "submission_ledger.tsv" in submit
    assert "-t 06:00:00" in submit
    assert "-t 02:00:00" in submit
    assert "-t 01:00:00" in submit
    solve_block = submit.split("  solve)", 1)[1].split("  finalize)", 1)[0]
    assert "PHC2D_STAGE=finalize" not in solve_block
    assert "for lane in binary graded" in solve_block
    assert "--workers 8" in smoke
    assert "--max-samples 4" in smoke
    assert '--dataset-output "$run_dir/data/smoke.h5"' in smoke
    assert "no automatic retry" in smoke
