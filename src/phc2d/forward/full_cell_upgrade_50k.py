"""Freeze the preregistered 27-task Full-Wh upgrade 50k confirmation matrix.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


MODEL_NAMES = {
    "MM": "mlp_mixer_band_v1",
    "RM": "res_mlp_gated_v2",
    "CX": "convnext_band_v1",
    "TF": "tfno2d_band_v3",
    "DL": "dual_lane_band_operator_v1",
}
WAVES = {
    "50K-W1": (
        "TF:B:0", "TF:B:1", "TF:B:2", "TF:G:0", "TF:G:1", "TF:G:2", "DL:J:0", "DL:J:1",
    ),
    "50K-W2": (
        "DL:J:2", "MM:B:0", "MM:B:1", "MM:B:2", "MM:G:0", "MM:G:1", "MM:G:2", "RM:B:0",
    ),
    "50K-W3": (
        "RM:B:1", "RM:B:2", "RM:G:0", "RM:G:1", "RM:G:2", "CX:B:0", "CX:B:1", "CX:B:2",
    ),
    "50K-W4": ("CX:G:0", "CX:G:1", "CX:G:2"),
}
CONFIGS = {
    "B": "configs/research/full_cell_forward_upgrade_binary_50k_v1.yaml",
    "G": "configs/research/full_cell_forward_upgrade_graded_50k_v1.yaml",
    "J": "configs/research/full_cell_forward_upgrade_dual_lane_50k_v1.yaml",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_50k_manifest(
    head_lock: str | Path,
    admission_lock: str | Path,
    *,
    output_dir: str | Path,
    repository_root: str | Path = ".",
) -> Path:
    """Write all 27 frozen rows, preserving skipped rows when admission fails."""

    root = Path(repository_root).resolve()
    head_path = Path(head_lock)
    if not head_path.is_absolute():
        head_path = root / head_path
    admission_path = Path(admission_lock)
    if not admission_path.is_absolute():
        admission_path = root / admission_path
    head = _read_json(head_path)
    admission = _read_json(admission_path)
    if head.get("status") != "head_selection_frozen" or head.get("test_read_during_selection") is not False:
        raise RuntimeError("50k manifest requires a validation-only frozen head lock")
    if admission.get("status") != "50k_admission_frozen" or admission.get("test_influenced_head_selection") is not False:
        raise RuntimeError("50k manifest requires a post-freeze held-out-test admission lock")
    selected_heads = {
        (str(item["model"]), str(item["lane"])): str(item["selected_head"])
        for item in head["selections"]
    }
    admitted = {
        str(item["model"]): str(item["status"]) == "admitted"
        for item in admission["decisions"]
    }
    if set(admitted) != set(MODEL_NAMES.values()):
        raise RuntimeError("50k admission lock does not cover exactly the five planned upgrades")
    out = Path(output_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    global_index = 0
    for wave, specs in WAVES.items():
        for slot, spec in enumerate(specs):
            short, lane, seed_text = spec.split(":")
            model = MODEL_NAMES[short]
            head_lane = "J" if lane == "J" else lane
            head_name = selected_heads[(model, head_lane)]
            head_short = "D" if head_name == "dense" else "Q"
            action = "RUN" if admitted[model] else "SKIP_NOT_ADMITTED"
            task_id = f"{wave}_{short}_{head_short}_{lane}_s{seed_text}"
            rows.append(
                {
                    "task_index": global_index,
                    "wave": wave,
                    "slot": slot,
                    "task_id": task_id,
                    "action": action,
                    "model": model,
                    "head": head_name,
                    "lane": lane,
                    "seed": int(seed_text),
                    "config": CONFIGS[lane],
                    "run_dir": str((out / task_id).relative_to(root)),
                }
            )
            global_index += 1
    if len(rows) != 27 or len({row["task_id"] for row in rows}) != 27:
        raise RuntimeError("50k matrix must contain 27 unique rows")
    _write_csv(out / "task_manifest.csv", rows)
    inputs = {
        "schema_version": "phc2d.full_cell_upgrade_50k_matrix.v1",
        "status": "50k_task_matrix_frozen",
        "head_selection_lock": str(head_path.relative_to(root)),
        "head_selection_lock_sha256": _sha256(head_path),
        "admission_lock": str(admission_path.relative_to(root)),
        "admission_lock_sha256": _sha256(admission_path),
        "task_rows": 27,
        "run_rows": sum(row["action"] == "RUN" for row in rows),
        "skip_rows": sum(row["action"] != "RUN" for row in rows),
        "wave_counts": {wave: len(specs) for wave, specs in WAVES.items()},
        "head_selection_split": "validation",
        "admission_evaluation_split": "test_after_head_freeze",
    }
    (out / "lock_inputs.json").write_text(json.dumps(inputs, indent=2) + "\n", encoding="utf-8")
    (out / "results.json").write_text(
        json.dumps(
            {
                "status": "50k_dry_run_matrix_success",
                "formal_training_started": False,
                "task_manifest_sha256": _sha256(out / "task_manifest.csv"),
                **{key: inputs[key] for key in ("task_rows", "run_rows", "skip_rows")},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-lock", required=True)
    parser.add_argument("--admission-lock", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-root", default=".")
    args = parser.parse_args(argv)
    print(
        build_50k_manifest(
            args.head_lock,
            args.admission_lock,
            output_dir=args.output_dir,
            repository_root=args.repository_root,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["WAVES", "build_50k_manifest"]
