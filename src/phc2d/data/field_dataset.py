"""Near-field dataset inspection helpers.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from phc2d.data.transforms import infer_dataset_schema


CORE_FIELD_DATASETS = ("epsilon", "rho", "k_points", "freqs", "efield", "hfield")
OPERATOR_CONTEXT_FIELDS = ("source", "boundary_condition", "grid_spacing", "domain_metadata")


@dataclass(frozen=True)
class FieldFeasibility:
    """Readiness summary for one near-field/operator task family."""

    task: str
    ready_for_schema_smoke: bool
    ready_for_full_claim: bool
    present: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class FieldDatasetSchema:
    """Compact schema summary for HDF5 near-field style datasets."""

    path: Path
    fields: dict[str, dict[str, Any]]
    metadata: dict[str, Any]
    attrs: dict[str, Any] = field(default_factory=dict)
    n_samples: int | None = None
    grid_shape: tuple[int, int] | None = None
    k_path_shape: tuple[int, ...] | None = None
    field_shape: tuple[int, ...] | None = None
    field_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    polarization: str | None = None
    polarization_metadata: dict[str, Any] = field(default_factory=dict)
    source_metadata: Any = None
    boundary_condition: Any = None
    grid_spacing: Any = None
    domain_metadata: Any = None

    @property
    def has_vector_fields(self) -> bool:
        return self.has_any_field(("efield", "hfield"))

    @property
    def dataset_names(self) -> tuple[str, ...]:
        """Return canonical leaf dataset names discovered in the HDF5 file."""

        return tuple(sorted({str(name).split("/")[-1] for name in self.fields}))

    @property
    def missing_core_datasets(self) -> tuple[str, ...]:
        """Core arrays expected by the near-field feasibility skeleton."""

        return tuple(name for name in CORE_FIELD_DATASETS if not self.has_field(name))

    @property
    def missing_operator_context(self) -> tuple[str, ...]:
        """Metadata still needed for source-conditioned or residual solvers."""

        values = {
            "source": self.source_metadata,
            "boundary_condition": self.boundary_condition,
            "grid_spacing": self.grid_spacing,
            "domain_metadata": self.domain_metadata,
        }
        return tuple(name for name, value in values.items() if value is None)

    @property
    def has_frequency_context(self) -> bool:
        """Whether frequency data is present for field-conditioned tasks."""

        return self.has_any_field(("freqs", "frequency", "wavelength"))

    @property
    def has_k_context(self) -> bool:
        """Whether k/query coordinate data is present for field-conditioned tasks."""

        return self.has_any_field(("k_points", "k_point", "k"))

    def has_field(self, name: str) -> bool:
        """Return whether a dataset exists by full path or leaf name."""

        needle = str(name)
        return any(key == needle or str(key).split("/")[-1] == needle for key in self.fields)

    def has_any_field(self, names: tuple[str, ...]) -> bool:
        """Return whether any dataset exists by full path or leaf name."""

        return any(self.has_field(name) for name in names)

    def feasibility(self) -> dict[str, FieldFeasibility]:
        """Summarize what the schema can support without overstating readiness."""

        common_present = tuple(
            name
            for name in ("epsilon", "rho", "k_points", "freqs", "efield", "hfield")
            if self.has_field(name)
        )
        field_missing = []
        if not self.has_any_field(("epsilon", "eps")):
            field_missing.append("epsilon")
        if not self.has_vector_fields:
            field_missing.append("efield_or_hfield")
        if not self.has_frequency_context:
            field_missing.append("freqs_or_frequency")
        if not self.has_k_context:
            field_missing.append("k_points")

        fno_unet_notes = (
            "Schema supports only a tiny supervised field smoke when all required arrays exist.",
            "It is not a full operator benchmark without source/domain coverage and more samples.",
        )
        neurolight_missing = tuple(dict.fromkeys(field_missing + list(self.missing_operator_context)))
        maxwell_missing = tuple(dict.fromkeys(field_missing + list(self.missing_operator_context)))

        return {
            "fno_field_unet_field": FieldFeasibility(
                task="fno_field_unet_field",
                ready_for_schema_smoke=not field_missing,
                ready_for_full_claim=False,
                present=common_present,
                missing=tuple(field_missing),
                notes=fno_unet_notes,
            ),
            "neurolight_adapted_field": FieldFeasibility(
                task="neurolight_adapted_field",
                ready_for_schema_smoke=not field_missing,
                ready_for_full_claim=False,
                present=common_present,
                missing=neurolight_missing,
                notes=(
                    "NeurOLight must remain source/frequency-conditioned field/operator work.",
                    "Do not treat this as a rho-to-band predictor or a plain Wh surrogate.",
                ),
            ),
            "maxwellnet_residual": FieldFeasibility(
                task="maxwellnet_residual",
                ready_for_schema_smoke=not field_missing,
                ready_for_full_claim=False,
                present=common_present,
                missing=maxwell_missing,
                notes=(
                    "MaxwellNet requires a residual definition with source, boundary, grid, and domain metadata.",
                    "Use feasibility reporting only until the residual context is present.",
                ),
            ),
        }


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if value is not None:
        return {"metadata": value}
    return {}


def _lookup_nested(mapping: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    stack: list[Any] = [mapping]
    while stack:
        current = stack.pop()
        if not isinstance(current, Mapping):
            continue
        for key, value in current.items():
            key_text = str(key).lower()
            if key_text in names:
                return value
            if isinstance(value, Mapping):
                stack.append(value)
    return None


def _extract_polarization(
    fields: Mapping[str, Any],
    metadata: Mapping[str, Any],
    attrs: Mapping[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    payload: dict[str, Any] = {}
    for source_name, source in (("metadata", metadata), ("attrs", attrs)):
        for key in ("polarization", "polarizations", "mode", "modes"):
            value = _lookup_nested(source, (key,))
            if value is not None:
                payload[f"{source_name}.{key}"] = value
    field_tokens = {
        token.lower()
        for name in fields
        for token in str(name).replace("\\", "/").replace("_", "/").split("/")
    }
    inferred = sorted(token.upper() for token in field_tokens if token.lower() in {"te", "tm"})
    if inferred:
        payload["inferred_from_field_paths"] = tuple(inferred)
    direct = payload.get("metadata.polarization") or payload.get("attrs.polarization")
    if isinstance(direct, bytes):
        direct = direct.decode("utf-8")
    if direct is not None:
        return str(direct), payload
    if inferred:
        return ",".join(inferred), payload
    return None, payload


def _metadata_value(metadata: Mapping[str, Any], attrs: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    value = _lookup_nested(metadata, names)
    if value is not None:
        return value
    return _lookup_nested(attrs, names)


def _field_shapes(schema: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    raw = schema.get("field_shapes", {})
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if isinstance(value, tuple):
                shapes[str(key)] = value
            elif isinstance(value, list):
                shapes[str(key)] = tuple(int(item) for item in value)
    return shapes


def inspect_field_dataset(path: str | Path) -> FieldDatasetSchema:
    """Inspect a field dataset and return only field-relevant metadata."""

    schema = infer_dataset_schema(path)
    metadata = _as_dict(schema.get("metadata", {}))
    attrs = _as_dict(schema.get("attrs", {}))
    fields = dict(schema.get("fields", {}))
    polarization, polarization_metadata = _extract_polarization(fields, metadata, attrs)
    return FieldDatasetSchema(
        path=Path(path),
        fields=fields,
        metadata=metadata,
        attrs=attrs,
        n_samples=schema.get("n_samples"),
        grid_shape=schema.get("grid_shape"),
        k_path_shape=schema.get("k_path_shape"),
        field_shape=schema.get("field_shape"),
        field_shapes=_field_shapes(schema),
        polarization=polarization,
        polarization_metadata=polarization_metadata,
        source_metadata=_metadata_value(metadata, attrs, ("source", "source_repr", "source_metadata")),
        boundary_condition=_metadata_value(
            metadata,
            attrs,
            ("boundary_condition", "boundary_conditions", "bc", "pml"),
        ),
        grid_spacing=_metadata_value(metadata, attrs, ("grid_spacing", "dx", "dy")),
        domain_metadata=_metadata_value(
            metadata,
            attrs,
            ("domain_metadata", "domain", "cell_size", "lattice"),
        ),
    )


__all__ = [
    "CORE_FIELD_DATASETS",
    "OPERATOR_CONTEXT_FIELDS",
    "FieldDatasetSchema",
    "FieldFeasibility",
    "inspect_field_dataset",
]
