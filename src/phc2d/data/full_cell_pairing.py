"""Streaming audit for paired full-cell Fourier sample plans.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import yaml

from phc2d.data.full_cell_fourier import independent_half_plane_modes


PAIRING_CONTRACT = "cutoff_paired_cohort_v1"


@dataclass(frozen=True)
class FourierPlanDescriptor:
    """Identity fields required to compare one frozen Fourier sample plan."""

    run_dir: Path
    plan_path: Path
    lane: str
    cutoff: int
    sample_count: int
    split: dict[str, int]
    dataset_seed: int
    sample_id_prefix: str
    paired_cohort_seed: int


def _load_descriptor(run_dir: str | Path) -> FourierPlanDescriptor:
    root = Path(run_dir).resolve()
    config_path = root / "config.resolved.yaml"
    plan_path = root / "sample_plan.jsonl"
    if not config_path.is_file() or not plan_path.is_file():
        raise FileNotFoundError(
            f"Pairing audit requires config.resolved.yaml and sample_plan.jsonl: {root}"
        )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    dataset = dict(payload.get("dataset") or {})
    generator = dict(payload.get("generator") or {})
    if dataset.get("protocol") != "full_cell_fourier_v2":
        raise ValueError(f"Pairing audit requires full_cell_fourier_v2: {root}")
    if generator.get("paired_coefficient_seed") is not None:
        raise ValueError(f"Bulk pairing audit rejects paired_coefficient_seed: {root}")
    if generator.get("pairing_contract") != PAIRING_CONTRACT:
        raise ValueError(f"Unsupported pairing contract in {root}")
    required = {
        "sample_id_prefix": dataset.get("sample_id_prefix"),
        "paired_cohort_seed": generator.get("paired_cohort_seed"),
        "fourier_cutoff": generator.get("fourier_cutoff"),
    }
    missing = [key for key, value in required.items() if value is None or str(value) == ""]
    if missing:
        raise ValueError(f"Missing paired-plan fields {missing}: {root}")
    return FourierPlanDescriptor(
        run_dir=root,
        plan_path=plan_path,
        lane=str(dataset["lane"]),
        cutoff=int(generator["fourier_cutoff"]),
        sample_count=int(dataset["sample_count"]),
        split={name: int(dict(dataset["split"])[name]) for name in ("train", "val", "test")},
        dataset_seed=int(dataset["seed"]),
        sample_id_prefix=str(dataset["sample_id_prefix"]),
        paired_cohort_seed=int(generator["paired_cohort_seed"]),
    )


def _read_row(handle: TextIO, *, descriptor: FourierPlanDescriptor, index: int) -> dict[str, Any]:
    line = handle.readline()
    if not line:
        raise ValueError(
            f"Sample plan ended at row {index}; expected {descriptor.sample_count}: "
            f"{descriptor.plan_path}"
        )
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise TypeError(f"Sample plan row must be a mapping: {descriptor.plan_path}:{index + 1}")
    return payload


def _mode_map(
    row: dict[str, Any], descriptor: FourierPlanDescriptor
) -> dict[tuple[int, int], tuple[float, float]]:
    parameters = dict(row.get("parameters") or {})
    modes = {
        (int(item["mx"]), int(item["my"])): (float(item["cos"]), float(item["sin"]))
        for item in parameters.get("modes", [])
    }
    expected = set(independent_half_plane_modes(descriptor.cutoff))
    if set(modes) != expected:
        raise ValueError(
            f"Fourier mode set mismatch for {descriptor.lane} K={descriptor.cutoff} "
            f"sample {row.get('sample_id')}"
        )
    return modes


def _audit_lane(descriptors: list[FourierPlanDescriptor]) -> dict[str, Any]:
    ordered = sorted(descriptors, key=lambda item: item.cutoff)
    if len(ordered) < 2:
        raise ValueError(f"Pairing audit requires at least two cutoffs for {ordered[0].lane}")
    if len({item.cutoff for item in ordered}) != len(ordered):
        raise ValueError(f"Duplicate Fourier cutoff in pairing audit for {ordered[0].lane}")
    reference = ordered[0]
    for item in ordered[1:]:
        comparable = (
            item.sample_count == reference.sample_count
            and item.split == reference.split
            and item.dataset_seed == reference.dataset_seed
            and item.sample_id_prefix == reference.sample_id_prefix
            and item.paired_cohort_seed == reference.paired_cohort_seed
        )
        if not comparable:
            raise ValueError(
                f"Paired-plan identity differs between K={reference.cutoff} and K={item.cutoff} "
                f"for {reference.lane}"
            )

    digest = hashlib.sha256()
    seen_sample_ids: set[str] = set()
    with ExitStack() as stack:
        handles = [
            stack.enter_context(item.plan_path.open("r", encoding="utf-8")) for item in ordered
        ]
        for index in range(reference.sample_count):
            rows = [
                _read_row(handle, descriptor=item, index=index)
                for handle, item in zip(handles, ordered)
            ]
            sample_id = str(rows[0].get("sample_id"))
            if sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate pairing sample_id: {sample_id}")
            seen_sample_ids.add(sample_id)
            expected_id = f"{reference.sample_id_prefix}_{index:06d}"
            if sample_id != expected_id or int(rows[0].get("sample_index", -1)) != index:
                raise ValueError(f"Paired-plan identity mismatch at {reference.lane} row {index}")
            for row, item in zip(rows, ordered):
                if (
                    str(row.get("sample_id")) != sample_id
                    or int(row.get("sample_index", -1)) != index
                    or row.get("split") != rows[0].get("split")
                    or row.get("stratum") != rows[0].get("stratum")
                ):
                    raise ValueError(
                        f"Pair identity/split/stratum differs at {reference.lane} row {index}, "
                        f"K={item.cutoff}"
                    )
                pairing = dict(row.get("pairing") or {})
                parameters = dict(row.get("parameters") or {})
                if (
                    pairing.get("contract") != PAIRING_CONTRACT
                    or pairing.get("pairing_key") != sample_id
                    or int(pairing.get("paired_cohort_seed", -1)) != reference.paired_cohort_seed
                    or parameters.get("pairing_key") != sample_id
                    or int(parameters.get("paired_cohort_seed", -1)) != reference.paired_cohort_seed
                ):
                    raise ValueError(
                        f"Pairing metadata mismatch at {reference.lane} row {index}, "
                        f"K={item.cutoff}"
                    )
            parameter_rows = [dict(row.get("parameters") or {}) for row in rows]
            for key in ("target_material_fraction", "temperature"):
                present = [item[key] for item in parameter_rows if key in item]
                if present and (
                    len(present) != len(parameter_rows)
                    or any(value != present[0] for value in present[1:])
                ):
                    raise ValueError(f"Paired {key} differs at {reference.lane} row {index}")
            mode_maps = [_mode_map(row, item) for row, item in zip(rows, ordered)]
            for lower, upper, lower_desc, upper_desc in zip(
                mode_maps,
                mode_maps[1:],
                ordered,
                ordered[1:],
            ):
                if not set(lower) < set(upper):
                    raise ValueError(
                        f"K={lower_desc.cutoff} modes are not a strict subset of "
                        f"K={upper_desc.cutoff} for {reference.lane} row {index}"
                    )
                if any(upper[key] != value for key, value in lower.items()):
                    raise ValueError(
                        f"Shared Fourier coefficients differ for {reference.lane} row {index}, "
                        f"K={lower_desc.cutoff}->{upper_desc.cutoff}"
                    )
            digest.update(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "split": rows[0].get("split"),
                        "stratum": rows[0].get("stratum"),
                        "target_material_fraction": parameter_rows[0].get(
                            "target_material_fraction"
                        ),
                        "temperature": parameter_rows[0].get("temperature"),
                        "mode_hashes": [
                            hashlib.sha256(
                                json.dumps(
                                    sorted(mode_map.items()),
                                    separators=(",", ":"),
                                ).encode("utf-8")
                            ).hexdigest()
                            for mode_map in mode_maps
                        ],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        for handle, item in zip(handles, ordered):
            if handle.readline():
                raise ValueError(
                    f"Sample plan contains more than {item.sample_count} rows: {item.plan_path}"
                )

    return {
        "lane": reference.lane,
        "sample_count": reference.sample_count,
        "split": reference.split,
        "cutoffs": [item.cutoff for item in ordered],
        "mode_counts": {
            str(item.cutoff): len(independent_half_plane_modes(item.cutoff)) for item in ordered
        },
        "dataset_seed": reference.dataset_seed,
        "sample_id_prefix": reference.sample_id_prefix,
        "paired_cohort_seed": reference.paired_cohort_seed,
        "pairing_digest_sha256": digest.hexdigest(),
        "shared_mode_coefficients_exact": True,
        "target_and_split_identity_exact": True,
    }


def audit_fourier_pairing_plans(
    run_dirs: list[str | Path],
    *,
    expected_cutoffs: list[int] | tuple[int, ...] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit paired sample plans without loading all plan rows into memory."""

    descriptors = [_load_descriptor(path) for path in run_dirs]
    grouped: dict[str, list[FourierPlanDescriptor]] = {}
    for item in descriptors:
        grouped.setdefault(item.lane, []).append(item)
    if set(grouped) != {"binary_fourier", "graded_fourier"}:
        raise ValueError("Pairing audit requires binary_fourier and graded_fourier plan groups")
    cutoff_sets = {lane: {item.cutoff for item in items} for lane, items in grouped.items()}
    if cutoff_sets["binary_fourier"] != cutoff_sets["graded_fourier"]:
        raise ValueError("Binary and Graded pairing cohorts contain different cutoffs")
    if expected_cutoffs is not None:
        expected = {int(item) for item in expected_cutoffs}
        if len(expected) != len(expected_cutoffs):
            raise ValueError("Expected cutoffs must be unique")
        if cutoff_sets["binary_fourier"] != expected:
            raise ValueError("Pairing cohort cutoffs do not match the explicitly expected cutoffs")
    summaries = {lane: _audit_lane(items) for lane, items in sorted(grouped.items())}
    payload = {
        "status": "passed",
        "pairing_contract": PAIRING_CONTRACT,
        "plan_count": len(descriptors),
        "cutoffs": sorted(cutoff_sets["binary_fourier"]),
        "lanes": summaries,
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    return payload


__all__ = ["PAIRING_CONTRACT", "audit_fourier_pairing_plans"]
