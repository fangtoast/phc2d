"""Dataset schema inspection and representation transforms.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from phc2d.data.k_path import attach_k_path_metadata, load_dataset_k_path_spec, validate_k_path_spec
from phc2d.data.lift import epsilon_from_rho, lift_rho_to_grid, wedge_indices
from phc2d.data.representations import DataRepresentationSpec


def _as_mapping(sample: Any) -> Mapping[str, Any]:
    if isinstance(sample, Mapping):
        return sample
    raise TypeError("sample must be a mapping for representation transforms")


def _array_info(arr: Any) -> dict[str, Any]:
    return {"shape": tuple(int(x) for x in arr.shape), "dtype": str(arr.dtype)}


def _shape_of(schema: Mapping[str, Any], field_name: str) -> tuple[int, ...] | None:
    fields = schema.get("fields", {})
    info = fields.get(field_name)
    if isinstance(info, Mapping) and "shape" in info:
        return tuple(int(x) for x in info["shape"])
    for key, value in fields.items():
        if str(key).split("/")[-1] == field_name and isinstance(value, Mapping) and "shape" in value:
            return tuple(int(x) for x in value["shape"])
    return None


def _field_name_set(fields: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    for key in fields:
        text = str(key)
        names.add(text)
        names.add(text.split("/")[-1])
    return names


def _has_any_field(fields: Mapping[str, Any], names: set[str]) -> bool:
    return bool(_field_name_set(fields) & set(names))


def _parse_metadata_text(raw: Any) -> Any:
    if raw is None:
        return None
    if hasattr(raw, "item"):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str) or not raw.strip():
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def infer_dataset_schema(path: str | Path) -> dict[str, Any]:
    """Inspect NPZ or HDF5 datasets without assuming fixed task shapes."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    schema: dict[str, Any] = {"path": str(p), "suffix": p.suffix.lower(), "fields": {}}
    if p.suffix.lower() == ".npz":
        with np.load(p, allow_pickle=False) as data:
            for key in data.files:
                schema["fields"][key] = _array_info(data[key])
            if "rho" in data:
                schema["n_samples"] = int(data["rho"].shape[0])
                schema["rho_shape"] = tuple(int(x) for x in data["rho"].shape)
                if data["rho"].ndim >= 2:
                    schema["rho_wedge_length"] = int(data["rho"].shape[-1])
            if "Wh" in data:
                schema["wh_shape"] = tuple(int(x) for x in data["Wh"].shape)
                if data["Wh"].ndim >= 3:
                    schema["k_points"] = int(data["Wh"].shape[1])
                    schema["n_bands"] = int(data["Wh"].shape[2])
            for split_name in ("idx_train", "idx_val", "idx_test", "train_idx", "val_idx", "test_idx"):
                if split_name in data:
                    schema.setdefault("splits", {})[split_name] = int(data[split_name].shape[0])
            if "config_yaml" in data:
                schema["has_config_yaml"] = True
            if "metadata_jsonl" in data:
                schema["has_metadata_jsonl"] = True
        k_spec = load_dataset_k_path_spec(p)
        if k_spec is not None:
            schema["k_path"] = attach_k_path_metadata(p, k_spec)
            schema["k_path"]["validation_problems"] = validate_k_path_spec(k_spec, expected_nk=schema.get("k_points"))
            schema["k_path_available"] = True
            schema["physical_k_coordinates_available"] = True
            schema["k_path_verified"] = bool(schema["k_path"].get("k_path_verified"))
            schema["k_path_id"] = schema["k_path"].get("k_path_id")
            schema["k_path_hash"] = schema["k_path"].get("k_path_hash")
            schema["coordinate_system"] = schema["k_path"].get("coordinate_system")
            schema["segment_count"] = schema["k_path"].get("segment_count")
        else:
            schema["k_path_available"] = False
            schema["physical_k_coordinates_available"] = False
            schema["k_path_verified"] = False
        schema["kind"] = "npz_forward"
        return schema

    if p.suffix.lower() in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise ImportError("h5py is required to inspect HDF5 field datasets") from exc

        with h5py.File(p, "r") as handle:
            def visit(name: str, obj: Any) -> None:
                if hasattr(obj, "shape"):
                    schema["fields"][name] = _array_info(obj)

            handle.visititems(visit)
            for attr_key, attr_value in handle.attrs.items():
                schema.setdefault("attrs", {})[str(attr_key)] = _parse_metadata_text(attr_value)
            metadata_raw = handle.attrs.get("metadata_json")
            parsed = _parse_metadata_text(metadata_raw)
            if parsed is not None:
                schema["metadata"] = parsed
        for name in ("rho", "epsilon", "eps", "efield", "hfield", "freqs", "k_points"):
            shape = _shape_of(schema, name)
            if shape:
                schema.setdefault("field_shapes", {})[name] = shape
        rho_shape = _shape_of(schema, "rho")
        if rho_shape:
            schema["n_samples"] = int(rho_shape[0])
            if len(rho_shape) >= 2:
                schema["rho_shape"] = rho_shape
                schema["rho_wedge_length"] = int(rho_shape[-1])
        for grid_name in ("epsilon", "eps"):
            grid_shape = _shape_of(schema, grid_name)
            if grid_shape and len(grid_shape) >= 3:
                schema.setdefault("n_samples", int(grid_shape[0]))
                schema["grid_shape"] = tuple(int(x) for x in grid_shape[-2:])
                break
        for field_name in ("efield", "hfield"):
            field_shape = _shape_of(schema, field_name)
            if field_shape:
                schema.setdefault("n_samples", int(field_shape[0]) if field_shape else None)
                schema["field_shape"] = field_shape
                break
        k_shape = _shape_of(schema, "k_points")
        if k_shape:
            schema["k_path_shape"] = k_shape
            if len(k_shape) >= 2:
                schema["k_points"] = int(k_shape[-2])
        freqs_shape = _shape_of(schema, "freqs")
        if freqs_shape:
            schema["freqs_shape"] = freqs_shape
            if len(freqs_shape) >= 3:
                schema["k_points"] = int(freqs_shape[-2])
                schema["n_bands"] = int(freqs_shape[-1])
        schema["kind"] = "hdf5_field_or_ibz"
        return schema

    schema["kind"] = "unknown"
    return schema


def make_input(sample: Any, spec: DataRepresentationSpec) -> Any:
    """Derive a model input representation from one raw sample mapping."""

    row = _as_mapping(sample)
    if spec.input_repr == "rho_wedge":
        return np.asarray(row["rho"], dtype=np.float32)
    if spec.input_repr == "rho_grid":
        if "rho_grid" in row:
            return np.asarray(row["rho_grid"], dtype=np.float32)
        if "binary_grid" in row:
            return np.asarray(row["binary_grid"], dtype=np.float32)
        return lift_rho_to_grid(np.asarray(row["rho"], dtype=np.float32), grid_shape=spec.grid_shape).astype(np.float32)
    if spec.input_repr == "epsilon_grid":
        if "epsilon" in row:
            return np.asarray(row["epsilon"], dtype=np.float32)
        if "eps" in row:
            return np.asarray(row["eps"], dtype=np.float32)
        return epsilon_from_rho(np.asarray(row["rho"], dtype=np.float32), grid_shape=spec.grid_shape).astype(np.float32)
    if spec.input_repr == "image_binary":
        grid = make_input(row, DataRepresentationSpec("rho_grid", spec.output_target, grid_shape=spec.grid_shape))
        return np.asarray(grid, dtype=np.float32)[..., None]
    if spec.input_repr == "image_epsilon":
        eps = make_input(row, DataRepresentationSpec("epsilon_grid", spec.output_target, grid_shape=spec.grid_shape))
        return np.asarray(eps, dtype=np.float32)[..., None]
    if spec.input_repr == "param_vector":
        return np.asarray(row["params"], dtype=np.float32)
    if spec.input_repr == "multimodal":
        return {
            "rho": row.get("rho"),
            "epsilon": row.get("epsilon", row.get("eps")),
            "metadata": row.get("metadata", {}),
            "k_points": row.get("k_points"),
            "source": row.get("source"),
        }
    if spec.input_repr == "field_conditioned":
        epsilon = row.get("epsilon", row.get("eps"))
        if epsilon is None and "rho" in row:
            epsilon = epsilon_from_rho(np.asarray(row["rho"], dtype=np.float32), grid_shape=spec.grid_shape).astype(np.float32)
        return {
            "epsilon": epsilon,
            "source": row.get("source"),
            "frequency": row.get("frequency", row.get("freqs")),
            "k_points": row.get("k_points"),
            "polarization": spec.polarization or row.get("polarization"),
        }
    raise ValueError(f"Unsupported input representation: {spec.input_repr}")


def make_target(sample: Any, spec: DataRepresentationSpec) -> Any:
    """Derive a training/evaluation target from one raw sample mapping."""

    row = _as_mapping(sample)
    if spec.output_target == "Wh":
        return np.asarray(row["Wh"], dtype=np.float32)
    if spec.output_target == "band_query":
        if "frequency" in row:
            return np.asarray(row["frequency"], dtype=np.float32)
        if "freqs" in row:
            return np.asarray(row["freqs"], dtype=np.float32)
        return np.asarray(row["Wh"], dtype=np.float32)
    if spec.output_target == "gap_descriptors":
        if "gap_descriptors" in row:
            return dict(row.get("gap_descriptors", {}))
        return {
            key: row[key]
            for key in ("gap_width", "gap_midgap_ratio", "gap_center", "gap_band_lower", "gap_band_upper")
            if key in row
        }
    if spec.output_target == "near_field_te":
        field = row.get("te_field", row.get("hfield"))
        return None if field is None else np.asarray(field, dtype=np.float32)
    if spec.output_target == "near_field_tm":
        field = row.get("tm_field", row.get("efield"))
        return None if field is None else np.asarray(field, dtype=np.float32)
    if spec.output_target == "near_field_te_tm":
        te_field = row.get("te_field", row.get("hfield"))
        tm_field = row.get("tm_field", row.get("efield"))
        return {
            "te": None if te_field is None else np.asarray(te_field, dtype=np.float32),
            "tm": None if tm_field is None else np.asarray(tm_field, dtype=np.float32),
        }
    if spec.output_target == "scalar_fom":
        return np.asarray(row["scalar_fom"], dtype=np.float32)
    if spec.output_target == "multi_task":
        return dict(row.get("targets", {}))
    raise ValueError(f"Unsupported output target: {spec.output_target}")


def validate_representation_compatibility(
    dataset_schema: Mapping[str, Any],
    spec: DataRepresentationSpec,
) -> list[str]:
    """Return human-readable incompatibilities for a schema/spec pair."""

    fields = dataset_schema.get("fields", {})
    problems: list[str] = []
    field_names = _field_name_set(fields)
    if spec.input_repr == "rho_wedge" and "rho" not in field_names:
        problems.append("input_repr=rho_wedge requires field 'rho'")
    if spec.input_repr in {"rho_grid", "image_binary"} and not _has_any_field(fields, {"rho", "rho_grid", "binary_grid"}):
        problems.append("rho_grid/image_binary requires 'rho', 'rho_grid', or 'binary_grid'")
    if spec.input_repr in {"epsilon_grid", "image_epsilon", "field_conditioned"} and not _has_any_field(fields, {"epsilon", "eps", "rho"}):
        problems.append("epsilon input requires 'epsilon', 'eps', or 'rho'")
    if spec.output_target == "Wh" and "Wh" not in field_names:
        problems.append("output_target=Wh requires field 'Wh'")
    if spec.output_target == "band_query" and not _has_any_field(fields, {"Wh", "freqs", "frequency"}):
        problems.append("band_query requires 'Wh', 'freqs', or 'frequency'")
    if spec.output_target in {"near_field_te", "near_field_tm", "near_field_te_tm"} and not _has_any_field(fields, {"efield", "hfield", "te_field", "tm_field"}):
        problems.append("near-field targets require field arrays such as 'efield'/'hfield'")
    rho_shape = _shape_of(dataset_schema, "rho")
    if spec.grid_shape is not None and rho_shape and len(rho_shape) >= 1:
        expected_wedge = len(wedge_indices(tuple(int(x) for x in spec.grid_shape)))
        if int(rho_shape[-1]) != expected_wedge:
            problems.append(
                f"grid_shape={spec.grid_shape} expects rho wedge length {expected_wedge}, "
                f"got {int(rho_shape[-1])}"
            )
    wh_shape = _shape_of(dataset_schema, "Wh")
    if spec.output_target == "Wh" and wh_shape and len(wh_shape) >= 3:
        if spec.k_points is not None and int(spec.k_points) != int(wh_shape[-2]):
            problems.append(f"spec.k_points={spec.k_points} does not match Wh k_points={int(wh_shape[-2])}")
        if spec.n_bands is not None and int(spec.n_bands) != int(wh_shape[-1]):
            problems.append(f"spec.n_bands={spec.n_bands} does not match Wh n_bands={int(wh_shape[-1])}")
    return problems


__all__ = [
    "infer_dataset_schema",
    "make_input",
    "make_target",
    "validate_representation_compatibility",
]
