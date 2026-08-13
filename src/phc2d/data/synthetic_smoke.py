"""Generate deterministic non-physical datasets for clean-clone smoke tests.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from phc2d.data.forward_dataset import ForwardDatasetBundle, save_forward_dataset


SYNTHETIC_SMOKE_STATUS = "synthetic_nonphysical_smoke"


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _positive_int(value: Any, *, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_indices(sample_count: int, split_cfg: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    counts = {
        name: _positive_int(split_cfg.get(name), name=f"split.{name}")
        for name in ("train", "val", "test")
    }
    if sum(counts.values()) != sample_count:
        raise ValueError(
            "split.train + split.val + split.test must equal dataset.sample_count"
        )
    train_end = counts["train"]
    val_end = train_end + counts["val"]
    return (
        np.arange(0, train_end, dtype=np.int64),
        np.arange(train_end, val_end, dtype=np.int64),
        np.arange(val_end, sample_count, dtype=np.int64),
    )


def _synthetic_arrays(
    *,
    sample_count: int,
    wedge_length: int,
    k_point_count: int,
    band_count: int,
    seed: int,
    fill_probability: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < fill_probability < 1.0:
        raise ValueError("generation.fill_probability must be between 0 and 1")
    rng = np.random.default_rng(seed)
    rho = rng.binomial(
        1,
        fill_probability,
        size=(sample_count, wedge_length),
    ).astype(np.float32)

    weights = rng.normal(0.0, 1.0, size=(wedge_length, band_count))
    projection = (rho @ weights) / max(float(np.sqrt(wedge_length)), 1.0)
    density = np.mean(rho, axis=1, keepdims=True)
    k_axis = np.linspace(0.0, 1.0, k_point_count, dtype=np.float64)
    band_axis = np.arange(band_count, dtype=np.float64)
    base = 0.18 + 0.075 * band_axis
    dispersion = 0.018 * np.sin(np.pi * k_axis)[:, None]
    slope = 0.004 * k_axis[:, None] * (band_axis[None, :] + 1.0)
    Wh = (
        base[None, None, :]
        + dispersion[None, :, :]
        + slope[None, :, :]
        + 0.012 * projection[:, None, :]
        + 0.025 * density[:, :, None]
    )
    Wh = np.sort(Wh, axis=-1).astype(np.float32)
    return rho, Wh


def make_synthetic_smoke_dataset(
    config_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate one deterministic dataset that is explicitly not physical evidence."""

    source = Path(config_path)
    raw_text = source.read_text(encoding="utf-8")
    raw = _mapping(yaml.safe_load(raw_text) or {}, name="config")
    if str(raw.get("schema_version", "")) != "phc2d.synthetic_smoke.v1":
        raise ValueError("schema_version must be 'phc2d.synthetic_smoke.v1'")

    dataset_cfg = _mapping(raw.get("dataset"), name="dataset")
    target_cfg = _mapping(raw.get("target"), name="target")
    split_cfg = _mapping(raw.get("split"), name="split")
    generation_cfg = _mapping(raw.get("generation"), name="generation")

    output_value = str(dataset_cfg.get("output_path", "")).strip()
    if not output_value:
        raise ValueError("dataset.output_path is required")
    output = Path(output_value)
    manifest = output.with_suffix(".manifest.yaml")
    if not overwrite and (output.exists() or manifest.exists()):
        existing = output if output.exists() else manifest
        raise FileExistsError(f"Refusing to overwrite existing smoke artifact: {existing}")

    sample_count = _positive_int(dataset_cfg.get("sample_count"), name="dataset.sample_count")
    wedge_length = _positive_int(
        dataset_cfg.get("rho_wedge_length"),
        name="dataset.rho_wedge_length",
    )
    k_point_count = _positive_int(target_cfg.get("k_point_count"), name="target.k_point_count")
    band_count = _positive_int(target_cfg.get("band_count"), name="target.band_count")
    seed = int(generation_cfg.get("seed", 0))
    fill_probability = float(generation_cfg.get("fill_probability", 0.35))
    idx_train, idx_val, idx_test = _split_indices(sample_count, split_cfg)
    rho, Wh = _synthetic_arrays(
        sample_count=sample_count,
        wedge_length=wedge_length,
        k_point_count=k_point_count,
        band_count=band_count,
        seed=seed,
        fill_probability=fill_probability,
    )

    split_by_index = {
        int(index): split
        for split, indices in (
            ("train", idx_train),
            ("val", idx_val),
            ("test", idx_test),
        )
        for index in indices
    }
    metadata = [
        {
            "sample_id": f"synthetic-smoke-{index:04d}",
            "generator_name": "phc2d.synthetic_smoke.v1",
            "evidence_status": SYNTHETIC_SMOKE_STATUS,
            "physical_verification_status": "not_applicable",
            "seed": seed,
            "split": split_by_index[index],
        }
        for index in range(sample_count)
    ]
    save_forward_dataset(
        output,
        ForwardDatasetBundle(
            rho=rho,
            Wh=Wh,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            config_yaml=raw_text,
            metadata=metadata,
            dataset_path=output,
        ),
    )
    manifest_payload = {
        "schema_version": "phc2d.synthetic_smoke_manifest.v1",
        "dataset_path": output.as_posix(),
        "dataset_sha256": _sha256(output),
        "evidence_status": SYNTHETIC_SMOKE_STATUS,
        "physical_verification_status": "not_applicable",
        "scientific_claims_allowed": False,
        "seed": seed,
        "rho_shape": list(rho.shape),
        "Wh_shape": list(Wh.shape),
        "split_counts": {
            "train": int(len(idx_train)),
            "val": int(len(idx_val)),
            "test": int(len(idx_test)),
        },
        "source_config": source.as_posix(),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        yaml.safe_dump(manifest_payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return {
        "status": "created",
        "evidence_status": SYNTHETIC_SMOKE_STATUS,
        "dataset_path": output.as_posix(),
        "manifest_path": manifest.as_posix(),
        "rho_shape": list(rho.shape),
        "Wh_shape": list(Wh.shape),
    }
