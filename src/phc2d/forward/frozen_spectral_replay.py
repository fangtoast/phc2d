"""Replay frozen official checkpoints into new spectral metric sidecars.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from phc2d.forward.budget_confirmation import file_digest
from phc2d.forward.official_comparison_test import evaluate
from phc2d.forward.spectral_metric_evaluation import evaluate_manifest


def replay_manifest(manifest_path: Path, output: Path) -> dict:
    """Require every native replay to pass before beginning transfer replays.

    This function does not schedule work. It runs inside an explicitly authorized
    CUDA allocation. Any error stops the replay; no automatic retry is performed.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cells = manifest["cells"]
    identities = [(c["model"], c["source_domain"], c["target_domain"], c["seed"]) for c in cells]
    if not cells or len(identities) != len(set(identities)):
        raise ValueError("empty replay manifest or duplicate cells")
    native = [c for c in cells if c["comparison_role"] == "native"]
    if not native or len(native) != int(manifest["expected_native_cells"]):
        raise ValueError("native pilot coverage differs from frozen manifest")
    seed_sets = defaultdict(set)
    for cell in cells:
        if (cell["comparison_role"] == "native") != (cell["source_domain"] == cell["target_domain"]):
            raise ValueError("native role differs from source/target identity")
        if cell["comparison_role"] == "native":
            seed_sets[(cell["model"], cell["source_domain"])].add(int(cell["seed"]))
    if any(seeds != {0, 1, 2} for seeds in seed_sets.values()):
        raise ValueError("native pilot requires complete seed 0/1/2 groups")
    output.mkdir(parents=True, exist_ok=False)
    completed = []
    (output / "config.resolved.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    stages = [native, [c for c in cells if c["comparison_role"] != "native"]]
    for stage_index, stage in enumerate(stages):
        for cell in stage:
            cell_id = cell["cell_id"]
            if not cell_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in cell_id):
                raise ValueError("unsafe cell ID")
            original = Path(cell["original_result_dir"])
            result = json.loads((original / "results.json").read_text(encoding="utf-8"))
            baseline = json.loads((original / "metrics.json").read_text(encoding="utf-8"))
            if (result["checkpoint_sha256"] != cell["checkpoint_sha256"]
                    or result["target_sha256"] != cell["dataset_sha256"]
                    or result["target_split_sha256"] != cell["split_sha256"]):
                raise ValueError("original frozen test provenance differs from manifest")
            if abs(float(baseline["bands_1_10_mre_pct"]) - float(cell["expected_mre_pct"])) > 1e-5:
                raise ValueError("original MRE differs from frozen manifest")
            destination = output / "predictions" / cell_id
            try:
                evaluate(Path(cell["source_run"]), Path(cell["completed_marker"]),
                         Path(cell["dataset_path"]), cell["dataset_sha256"],
                         cell["grid_field"], destination,
                         external=bool(cell["external"]), save_predictions=True)
                export = json.loads((destination / "prediction_export.json").read_text(encoding="utf-8"))
                new_cell = {**cell, "k_points_sha256": export["k_points_sha256"],
                            "prediction_path": str(destination / "predictions.npy"),
                            "target_path": str(destination / "targets.npy"),
                            "indices_path": str(destination / "sample_indices.npy"),
                            "files_sha256": {str(destination / k): v for k, v in export["files_sha256"].items()}}
                if export["idx_hash"] != cell["split_sha256"]:
                    raise ValueError("replayed split differs from original")
                single_manifest = destination / "metric_manifest.json"
                single_manifest.write_text(json.dumps(dict(cells=[new_cell]), indent=2), encoding="utf-8")
                audit = evaluate_manifest(single_manifest, output / "metric_checks" / cell_id)
                if audit["computed_cells"] != 1:
                    raise ValueError("metric/MRE regression check failed; inspect cell audit")
                completed.append(new_cell)
                print(f"replayed {len(completed)}/{len(cells)} {cell_id}", flush=True)
            except Exception as error:
                (output / "failure.json").write_text(json.dumps(dict(cell_id=cell_id, error=str(error),
                    completed_cells=len(completed), status="blocked_no_automatic_retry"), indent=2), encoding="utf-8")
                raise
        if stage_index == 0:
            (output / "native_pilot_pass.json").write_text(json.dumps(dict(
                status="native_metric_regression_pass", cells=len(native))), encoding="utf-8")
    final_manifest = output / "prediction_manifest.json"
    final_manifest.write_text(json.dumps(dict(cells=completed), indent=2), encoding="utf-8")
    audit = evaluate_manifest(final_manifest, output / "multimetric")
    audit["replay_source_sha256"] = file_digest(Path(__file__))
    (output / "results.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay_manifest(args.manifest, args.output)))


if __name__ == "__main__":
    main()
