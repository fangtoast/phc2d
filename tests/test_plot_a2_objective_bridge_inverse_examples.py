"""Tests for A2 objective-bridge inverse example plotting.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scripts import plot_a2_objective_bridge_inverse_examples as plotter


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def synthetic_case(method: str, target_index: int, replay: float, surrogate: float) -> dict[str, object]:
    return {
        "method": method,
        "target_index": target_index,
        "replay_mre_pct": replay,
        "surrogate_mre_pct": surrogate,
    }


def test_paired_selection_is_unique_reproducible_and_metric_driven() -> None:
    single_scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 14.0, 11.0]
    mixture_scores = [1.2, 2.2, 3.2, 4.2, 5.2, 9.5, 3.0, 8.2, 9.2, 10.2, 11.0, 16.0]
    cases = {
        plotter.SINGLE_METHOD: [
            synthetic_case(plotter.SINGLE_METHOD, index, score, score - (4.0 if index == 9 else 0.2))
            for index, score in enumerate(single_scores)
        ],
        plotter.MIXTURE_METHOD: [
            synthetic_case(plotter.MIXTURE_METHOD, index, score, score - (5.0 if index == 10 else 0.1))
            for index, score in enumerate(mixture_scores)
        ],
    }
    first = plotter.select_paired_targets(cases, seed=20260728)
    second = plotter.select_paired_targets(cases, seed=20260728)
    selected_targets = [int(first[name]["target_index"]) for name in plotter.PAIRED_SELECTION_ORDER]
    assert selected_targets == [
        int(second[name]["target_index"]) for name in plotter.PAIRED_SELECTION_ORDER
    ]
    assert len(set(selected_targets)) == len(plotter.PAIRED_SELECTION_ORDER)
    rows = plotter.paired_case_rows(cases)
    assert first["joint_best"]["target_index"] == min(rows, key=lambda row: row["joint_score"])["target_index"]
    assert first["largest_disagreement"]["target_index"] == max(
        rows, key=lambda row: row["disagreement"]
    )["target_index"]
    already_used = {
        first["joint_best"]["target_index"],
        first["joint_median"]["target_index"],
        first["largest_disagreement"]["target_index"],
    }
    expected_joint_worst = next(
        row
        for row in sorted(rows, key=lambda row: (-row["joint_score"], row["target_index"]))
        if row["target_index"] not in already_used
    )
    assert first["joint_worst"]["target_index"] == expected_joint_worst["target_index"]


def test_method_extremes_are_strictly_sorted_and_disjoint() -> None:
    cases = [
        synthetic_case(plotter.SINGLE_METHOD, index, float(score), float(score))
        for index, score in enumerate([4, 1, 8, 3, 9, 2, 7, 6, 5, 10])
    ]
    best, worst = plotter.select_method_extremes(cases, count=3)
    assert [case["replay_mre_pct"] for case in best] == [1.0, 2.0, 3.0]
    assert [case["replay_mre_pct"] for case in worst] == [10.0, 9.0, 8.0]
    assert {case["target_index"] for case in best}.isdisjoint(
        {case["target_index"] for case in worst}
    )


def test_plot_objective_bridge_inverse_examples(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(plotter, "repo_root", lambda: tmp_path)

    def fake_save_figure(fig, output_base, formats, dpi):
        del dpi
        plotter.plt.close(fig)
        paths = []
        output_base.parent.mkdir(parents=True, exist_ok=True)
        for suffix in formats:
            path = output_base.with_suffix(f".{suffix}")
            path.write_bytes(b"rendered")
            paths.append(path)
        return paths

    monkeypatch.setattr(plotter, "save_figure", fake_save_figure)
    run_dir = tmp_path / "runs/test"
    dataset_path = tmp_path / "data/fake.npz"
    dataset_path.parent.mkdir(parents=True)
    target_count = 12
    grid_shape = (6, 6)
    dataset_rho = np.zeros((target_count, *grid_shape), dtype=np.float32)
    for target_index in range(target_count):
        dataset_rho[target_index].flat[target_index % dataset_rho[target_index].size] = 1.0
    np.savez(dataset_path, rho=dataset_rho)

    records = []
    replay_rows = []
    cache = run_dir / "replay/structure_cache"
    cache.mkdir(parents=True, exist_ok=True)
    for method_index, method in enumerate(plotter.METHOD_LABELS):
        for target_index in range(target_count):
            target = np.linspace(0.2, 1.1, 30, dtype=np.float32).reshape(5, 6)
            candidates = []
            predictions = []
            for candidate_index in range(2):
                candidate = np.zeros(grid_shape, dtype=np.float32)
                serial = method_index * target_count * 2 + target_index * 2 + candidate_index + 1
                for bit in range(min(candidate.size, serial.bit_length())):
                    candidate.flat[bit] = float((serial >> bit) & 1)
                candidates.append(candidate.tolist())
                predictions.append((target * (1.0 + 0.002 * (target_index + 1))).tolist())
            records.append(
                {
                    "method": method,
                    "target_index": target_index,
                    "candidates": candidates,
                    "surrogate_predictions": predictions,
                    "condition": {"Wh": target.tolist()},
                }
            )
            for candidate_index, candidate in enumerate(candidates):
                candidate_vector = np.asarray(candidate, dtype=np.float32).reshape(-1)
                sha = plotter.hash_array(candidate_vector)
                method_offset = 0.001 * method_index * (target_count - target_index)
                factor = 0.01 + 0.001 * target_index + method_offset + 0.02 * candidate_index
                replay = target * (1.0 + factor)
                np.save(cache / f"{sha}.npy", replay)
                replay_rows.append(
                    {
                        "method": method,
                        "target_index": target_index,
                        "candidate_index": candidate_index,
                        "rank_position": candidate_index,
                        "structure_sha256": sha,
                        "status": "computed",
                        "mpb_replay_mre": factor,
                    }
                )
    manifest = run_dir / "candidates/candidate_manifest.frozen.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"records": records}), encoding="utf-8")
    write_csv(run_dir / "replay/replay_candidates.csv", replay_rows)

    output = tmp_path / "output"
    rc = plotter.main(
        [
            "--run-dir",
            str(run_dir),
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(output),
            "--gallery-count",
            "4",
            "--extreme-count",
            "2",
            "--expected-target-count",
            str(target_count),
            "--formats",
            "png",
        ]
    )
    assert rc == 0
    assert (output / "paired_main4.png").exists()
    assert (output / "paired_supplement8.png").exists()
    for method in plotter.METHOD_LABELS:
        assert (output / f"{method}_best_median_worst_random.png").exists()
        assert (output / f"{method}_best2_worst2_gallery.png").exists()
        assert (output / f"{method}_random4_gallery.png").exists()

    selection_manifest = json.loads((output / "selection_manifest.json").read_text(encoding="utf-8"))
    assert selection_manifest["status"] == "frozen_for_visualization"
    assert len(selection_manifest["paired"]) == 8
    paired_targets = [row["target_index"] for row in selection_manifest["paired"]]
    assert len(set(paired_targets)) == 8
    for row in selection_manifest["paired"]:
        assert {
            row["methods"][plotter.SINGLE_METHOD]["target_index"],
            row["methods"][plotter.MIXTURE_METHOD]["target_index"],
        } == {row["target_index"]}

    random_targets = list(
        csv.DictReader((output / "random40_target_list.csv").open(newline="", encoding="utf-8"))
    )
    assert len(random_targets) == 4
    selection_rows = list(
        csv.DictReader((output / "selection_index.csv").open(newline="", encoding="utf-8"))
    )
    random_rows = [row for row in selection_rows if row["selection_group"] == "random40"]
    by_method = {
        method: [int(row["target_index"]) for row in random_rows if row["method"] == method]
        for method in plotter.METHOD_LABELS
    }
    assert by_method[plotter.SINGLE_METHOD] == by_method[plotter.MIXTURE_METHOD]
    assert list((output / "individual").glob("target_*.png"))
    assert (output / "figure_captions.md").exists()
    artifact_manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert artifact_manifest["artifacts"]
