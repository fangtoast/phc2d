"""Train-split analytic-parameter codecs for full-cell inverse design.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from phc2d.data.full_cell import FULL_CELL_SCHEMA_VERSION, FullCellStructureSpec


FULL_CELL_ANALYTIC_CODEC_VERSION = "phc2d.full_cell_analytic_codec.v1"
FULL_CELL_ANALYTIC_OUTPUT_REPR = "full_cell_analytic_params"
FULL_CELL_VARIANTS = (
    "binary_polar_holes_1",
    "binary_polar_holes_2",
    "binary_spectral_threshold",
    "graded_rbf_fourier_1",
    "graded_rbf_fourier_2",
)
PRODUCTION_PHYSICAL_WIDTHS = {
    "binary_polar_holes_1": 11,
    "binary_polar_holes_2": 22,
    "binary_spectral_threshold": 289,
    "graded_rbf_fourier_1": 57,
    "graded_rbf_fourier_2": 63,
}

_AUXILIARY_PARAMETER_NAMES = frozenset(
    {
        "cutoff_scale",
        "target_air_fraction",
        "target_material_fraction",
        "threshold_quantile",
    }
)
_PATH_TOKEN = re.compile(r"([^.[\]]+)|\[(\d+)\]")


def full_cell_variant_id(structure: FullCellStructureSpec) -> str:
    """Return the inverse-training variant represented by one structure."""

    family = structure.generator_family
    if family == "binary_polar_holes":
        count = int(structure.parameters.get("hole_count", len(structure.parameters.get("holes", ()))))
        variant = f"binary_polar_holes_{count}"
    elif family == "graded_rbf_fourier":
        count = int(structure.metadata.get("core_count", len(structure.parameters.get("blobs", ()))))
        variant = f"graded_rbf_fourier_{count}"
    elif family == "binary_spectral_threshold":
        variant = family
    else:  # pragma: no cover - FullCellStructureSpec rejects unsupported families
        raise ValueError(f"Unsupported full-cell generator family: {family}")
    if variant not in FULL_CELL_VARIANTS:
        raise ValueError(f"Unsupported full-cell analytic variant: {variant}")
    return variant


def _parameter_mapping(structure: FullCellStructureSpec) -> dict[str, float]:
    return {
        str(name): float(value)
        for name, value, active in zip(
            structure.parameter_names,
            structure.parameter_vector,
            structure.parameter_mask,
        )
        if bool(active)
    }


def _parameter_role(name: str) -> str:
    if name in _AUXILIARY_PARAMETER_NAMES:
        return "auxiliary"
    if name == "hole_count" or name.endswith((".mx", ".my", ".order")):
        return "fixed"
    return "learned"


def _transform_kind(name: str) -> str:
    leaf = name.rsplit(".", 1)[-1]
    if leaf in {"center_x", "center_y"}:
        return "periodic"
    if leaf == "angle":
        return "angle"
    if leaf in {"radius", "sigma_x", "sigma_y", "temperature", "fourier_strength"}:
        return "log_robust"
    return "robust"


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    center = float(np.median(array))
    q05, q95 = np.quantile(array, [0.05, 0.95])
    scale = float(0.5 * (q95 - q05))
    if not np.isfinite(scale) or scale <= 1.0e-12:
        scale = float(np.std(array))
    if not np.isfinite(scale) or scale <= 1.0e-12:
        scale = 1.0
    return center, scale


def _path_tokens(path: str) -> list[str | int]:
    tokens: list[str | int] = []
    for match in _PATH_TOKEN.finditer(path):
        key, index = match.groups()
        tokens.append(int(index) if index is not None else str(key))
    if not tokens:
        raise ValueError(f"Invalid analytic parameter path: {path}")
    return tokens


def _get_path(payload: Mapping[str, Any], path: str) -> float:
    current: Any = payload
    for token in _path_tokens(path):
        current = current[token]
    return float(current)


def _set_path(payload: dict[str, Any], path: str, value: float) -> None:
    tokens = _path_tokens(path)
    current: Any = payload
    for token in tokens[:-1]:
        current = current[token]
    current[tokens[-1]] = float(value)


def _decode_text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


@dataclass(frozen=True)
class FullCellAnalyticCodec:
    """A fitted, invertible codec for one full-cell analytic variant."""

    variant_id: str
    learned_parameter_names: tuple[str, ...]
    transform_kinds: tuple[str, ...]
    locations: tuple[float, ...]
    scales: tuple[float, ...]
    fixed_parameter_names: tuple[str, ...]
    auxiliary_parameter_names: tuple[str, ...]
    template_mapping: dict[str, Any]
    train_sample_count: int
    schema_version: str = FULL_CELL_ANALYTIC_CODEC_VERSION

    def __post_init__(self) -> None:
        if self.variant_id not in FULL_CELL_VARIANTS:
            raise ValueError(f"Unsupported full-cell analytic variant: {self.variant_id}")
        width = len(self.learned_parameter_names)
        if not (
            len(self.transform_kinds) == width
            and len(self.locations) == width
            and len(self.scales) == width
        ):
            raise ValueError("learned names, transforms, locations, and scales must have equal length")
        if int(self.train_sample_count) <= 0:
            raise ValueError("train_sample_count must be positive")
        if any(kind not in {"periodic", "angle", "log_robust", "robust"} for kind in self.transform_kinds):
            raise ValueError("codec contains an unsupported transform kind")
        if any(not np.isfinite(scale) or float(scale) <= 0.0 for scale in self.scales):
            raise ValueError("codec scales must be finite and positive")
        template = FullCellStructureSpec.from_mapping(self.template_mapping)
        if full_cell_variant_id(template) != self.variant_id:
            raise ValueError("codec template does not match variant_id")

    @property
    def physical_width(self) -> int:
        return len(self.learned_parameter_names)

    @property
    def encoded_width(self) -> int:
        return sum(2 if kind in {"periodic", "angle"} else 1 for kind in self.transform_kinds)

    @property
    def encoded_feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for name, kind in zip(self.learned_parameter_names, self.transform_kinds):
            if kind in {"periodic", "angle"}:
                names.extend((f"{name}.sin", f"{name}.cos"))
            else:
                names.append(name)
        return tuple(names)

    @classmethod
    def fit(cls, structures: Sequence[FullCellStructureSpec]) -> "FullCellAnalyticCodec":
        """Fit transforms from structures that must all belong to one train split variant."""

        items = tuple(structures)
        if not items:
            raise ValueError("Cannot fit a full-cell analytic codec from no structures")
        variant_id = full_cell_variant_id(items[0])
        if any(full_cell_variant_id(item) != variant_id for item in items):
            raise ValueError("A full-cell analytic codec can only fit one variant at a time")

        mappings = [_parameter_mapping(item) for item in items]
        parameter_names = tuple(mappings[0])
        if any(tuple(row) != parameter_names for row in mappings[1:]):
            raise ValueError(f"Parameter schema drift inside variant {variant_id}")
        learned = tuple(name for name in parameter_names if _parameter_role(name) == "learned")
        fixed = tuple(name for name in parameter_names if _parameter_role(name) == "fixed")
        auxiliary = tuple(name for name in parameter_names if _parameter_role(name) == "auxiliary")
        if not learned:
            raise ValueError(f"Variant {variant_id} has no learned analytic parameters")

        for name in fixed:
            reference = mappings[0][name]
            if any(row[name] != reference for row in mappings[1:]):
                raise ValueError(f"Fixed parameter {name} changes inside variant {variant_id}")

        kinds = tuple(_transform_kind(name) for name in learned)
        values = np.asarray([[row[name] for name in learned] for row in mappings], dtype=np.float64)
        locations: list[float] = []
        scales: list[float] = []
        for column, kind in enumerate(kinds):
            raw = values[:, column]
            if not np.all(np.isfinite(raw)):
                raise ValueError(f"Non-finite values in analytic parameter {learned[column]}")
            if kind in {"periodic", "angle"}:
                locations.append(0.0)
                scales.append(1.0)
                continue
            transformed = raw
            if kind == "log_robust":
                if np.any(raw <= 0.0):
                    raise ValueError(f"Positive analytic parameter {learned[column]} contains non-positive values")
                transformed = np.log(raw)
            center, scale = _robust_location_scale(transformed)
            locations.append(center)
            scales.append(scale)

        template = items[0].to_mapping()
        template["sample_id"] = f"codec_template_{variant_id}"
        template["metadata"] = {
            "codec_template": True,
            "source_sample_id": items[0].sample_id,
        }
        return cls(
            variant_id=variant_id,
            learned_parameter_names=learned,
            transform_kinds=kinds,
            locations=tuple(locations),
            scales=tuple(scales),
            fixed_parameter_names=fixed,
            auxiliary_parameter_names=auxiliary,
            template_mapping=template,
            train_sample_count=len(items),
        )

    def assert_production_contract(self) -> None:
        """Fail closed when a fitted codec does not match the frozen 10k protocol."""

        expected = PRODUCTION_PHYSICAL_WIDTHS[self.variant_id]
        if self.physical_width != expected:
            raise ValueError(
                f"Variant {self.variant_id} has {self.physical_width} learned parameters; "
                f"the production contract requires {expected}"
            )

    def encode_structure(self, structure: FullCellStructureSpec) -> np.ndarray:
        """Encode one structure into a finite network target vector."""

        if full_cell_variant_id(structure) != self.variant_id:
            raise ValueError(f"Expected variant {self.variant_id}, got {full_cell_variant_id(structure)}")
        mapping = _parameter_mapping(structure)
        if any(name not in mapping for name in self.learned_parameter_names):
            raise ValueError(f"Structure {structure.sample_id} does not match the fitted parameter schema")
        encoded: list[float] = []
        for name, kind, location, scale in zip(
            self.learned_parameter_names,
            self.transform_kinds,
            self.locations,
            self.scales,
        ):
            value = float(mapping[name])
            if kind == "periodic":
                phase = 2.0 * np.pi * value
                encoded.extend((float(np.sin(phase)), float(np.cos(phase))))
            elif kind == "angle":
                encoded.extend((float(np.sin(value)), float(np.cos(value))))
            else:
                transformed = float(np.log(value)) if kind == "log_robust" else value
                encoded.append((transformed - float(location)) / float(scale))
        result = np.asarray(encoded, dtype=np.float64)
        if result.shape != (self.encoded_width,) or not np.all(np.isfinite(result)):
            raise ValueError(f"Codec produced invalid output for {structure.sample_id}")
        return result

    def encode_many(self, structures: Sequence[FullCellStructureSpec]) -> np.ndarray:
        """Encode a homogeneous sequence of structures."""

        rows = [self.encode_structure(item) for item in structures]
        if not rows:
            return np.empty((0, self.encoded_width), dtype=np.float64)
        return np.stack(rows, axis=0)

    def _decode_parameter_tensors(self, encoded: Any, *, torch: Any) -> dict[str, Any]:
        if int(encoded.shape[-1]) != self.encoded_width:
            raise ValueError(f"Encoded tensor must have trailing width {self.encoded_width}")
        parameters: dict[str, Any] = {}
        cursor = 0
        for name, kind, location, scale in zip(
            self.learned_parameter_names,
            self.transform_kinds,
            self.locations,
            self.scales,
        ):
            if kind in {"periodic", "angle"}:
                phase = torch.atan2(encoded[..., cursor], encoded[..., cursor + 1])
                value = phase / (2.0 * np.pi) if kind == "periodic" else torch.remainder(phase, 2.0 * np.pi)
                if kind == "periodic":
                    value = torch.remainder(value + 0.5, 1.0) - 0.5
                cursor += 2
            else:
                transformed = encoded[..., cursor] * float(scale) + float(location)
                value = torch.exp(transformed) if kind == "log_robust" else transformed
                cursor += 1
            parameters[name] = value
        return parameters

    def decode_soft_raster_tensor(
        self,
        encoded: Any,
        *,
        grid_shape: Sequence[int],
        binary_temperature: float = 0.01,
        torch: Any | None = None,
    ) -> Any:
        """Differentiably decode encoded candidates to material-weight rasters."""

        if torch is None:
            try:
                import torch as torch_module
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("torch is required for differentiable full-cell decoding") from exc
            torch = torch_module
        if len(grid_shape) != 2 or any(int(value) <= 0 for value in grid_shape):
            raise ValueError("grid_shape must contain two positive dimensions")
        if not np.isfinite(binary_temperature) or float(binary_temperature) <= 0.0:
            raise ValueError("binary_temperature must be finite and positive")

        values = encoded
        if not bool(getattr(values, "is_floating_point", lambda: False)()):
            values = values.to(dtype=torch.float32)
        squeeze = int(values.ndim) == 1
        if squeeze:
            values = values.unsqueeze(0)
        if int(values.ndim) != 2:
            raise ValueError("Encoded tensor must have shape [encoded_width] or [batch, encoded_width]")
        parameters = self._decode_parameter_tensors(values, torch=torch)
        batch = int(values.shape[0])
        rows, cols = int(grid_shape[0]), int(grid_shape[1])
        x_axis = -0.5 + (torch.arange(cols, device=values.device, dtype=values.dtype) + 0.5) / float(cols)
        y_axis = -0.5 + (torch.arange(rows, device=values.device, dtype=values.dtype) + 0.5) / float(rows)
        y, x = torch.meshgrid(y_axis, x_axis, indexing="ij")
        x = x.unsqueeze(0).expand(batch, -1, -1)
        y = y.unsqueeze(0).expand(batch, -1, -1)
        template = FullCellStructureSpec.from_mapping(self.template_mapping)

        def active_or_template(path: str) -> Any:
            value = parameters.get(path)
            if value is not None:
                return value
            return torch.full(
                (batch,),
                _get_path(template.parameters, path),
                device=values.device,
                dtype=values.dtype,
            )

        def fourier_field(modes: Sequence[Mapping[str, Any]]) -> Any:
            field = torch.zeros((batch, rows, cols), device=values.device, dtype=values.dtype)
            for index, mode in enumerate(modes):
                phase = 2.0 * np.pi * (int(mode["mx"]) * x + int(mode["my"]) * y)
                cosine = active_or_template(f"modes[{index}].cos")[:, None, None]
                sine = active_or_template(f"modes[{index}].sin")[:, None, None]
                field = field + cosine * torch.cos(phase) + sine * torch.sin(phase)
            return field

        if self.variant_id.startswith("graded_rbf_fourier_"):
            field = torch.zeros((batch, rows, cols), device=values.device, dtype=values.dtype)
            for index, _blob in enumerate(template.parameters.get("blobs", ())):
                center_x = active_or_template(f"blobs[{index}].center_x")[:, None, None]
                center_y = active_or_template(f"blobs[{index}].center_y")[:, None, None]
                dx = torch.remainder(x - center_x + 0.5, 1.0) - 0.5
                dy = torch.remainder(y - center_y + 0.5, 1.0) - 0.5
                angle = active_or_template(f"blobs[{index}].angle")[:, None, None]
                cos_a, sin_a = torch.cos(angle), torch.sin(angle)
                xr = cos_a * dx + sin_a * dy
                yr = -sin_a * dx + cos_a * dy
                sigma_x = active_or_template(f"blobs[{index}].sigma_x")[:, None, None]
                sigma_y = active_or_template(f"blobs[{index}].sigma_y")[:, None, None]
                amplitude = active_or_template(f"blobs[{index}].amplitude")[:, None, None]
                field = field + amplitude * torch.exp(
                    -0.5 * (torch.square(xr / sigma_x) + torch.square(yr / sigma_y))
                )
            strength = active_or_template("fourier_strength")[:, None, None]
            field = field + strength * fourier_field(template.parameters.get("modes", ()))
            bias = active_or_template("bias")[:, None, None]
            temperature = active_or_template("temperature")[:, None, None]
            raster = torch.sigmoid((field - bias) / temperature)
        elif self.variant_id.startswith("binary_polar_holes_"):
            raster = torch.ones((batch, rows, cols), device=values.device, dtype=values.dtype)
            for index, hole in enumerate(template.parameters.get("holes", ())):
                center_x = active_or_template(f"holes[{index}].center_x")[:, None, None]
                center_y = active_or_template(f"holes[{index}].center_y")[:, None, None]
                dx = torch.remainder(x - center_x + 0.5, 1.0) - 0.5
                dy = torch.remainder(y - center_y + 0.5, 1.0) - 0.5
                theta = torch.atan2(dy, dx)
                perturbation = torch.zeros_like(theta)
                for mode_index, mode in enumerate(hole.get("modes", ())):
                    order = int(mode["order"])
                    cosine = active_or_template(
                        f"holes[{index}].modes[{mode_index}].cos"
                    )[:, None, None]
                    sine = active_or_template(
                        f"holes[{index}].modes[{mode_index}].sin"
                    )[:, None, None]
                    perturbation = (
                        perturbation
                        + cosine * torch.cos(order * theta)
                        + sine * torch.sin(order * theta)
                    )
                radius = active_or_template(f"holes[{index}].radius")[:, None, None]
                boundary = radius * torch.exp(perturbation)
                distance = torch.sqrt(torch.square(dx) + torch.square(dy) + 1.0e-16)
                raster = raster * torch.sigmoid((distance - boundary) / float(binary_temperature))
        else:
            field = fourier_field(template.parameters.get("modes", ()))
            threshold = active_or_template("threshold")[:, None, None]
            raster = torch.sigmoid((field - threshold) / float(binary_temperature))
        return raster[0] if squeeze else raster

    def decode_structure(
        self,
        encoded: Sequence[float] | np.ndarray,
        *,
        template: FullCellStructureSpec | None = None,
        sample_id: str | None = None,
        snap_tolerance: float = 1.0e-12,
    ) -> FullCellStructureSpec:
        """Decode one network vector into a hard analytic structure specification."""

        base = template or FullCellStructureSpec.from_mapping(self.template_mapping)
        if full_cell_variant_id(base) != self.variant_id:
            raise ValueError(f"Decode template must belong to {self.variant_id}")
        values = np.asarray(encoded, dtype=np.float64).reshape(-1)
        if values.shape != (self.encoded_width,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Encoded vector must be finite with width {self.encoded_width}")

        parameters = copy.deepcopy(base.parameters)
        cursor = 0
        for name, kind, location, scale in zip(
            self.learned_parameter_names,
            self.transform_kinds,
            self.locations,
            self.scales,
        ):
            if kind in {"periodic", "angle"}:
                sin_value, cos_value = float(values[cursor]), float(values[cursor + 1])
                norm = float(np.hypot(sin_value, cos_value))
                if norm <= 1.0e-12:
                    raise ValueError(f"Degenerate sin/cos pair for {name}")
                phase = float(np.arctan2(sin_value / norm, cos_value / norm))
                decoded = phase / (2.0 * np.pi) if kind == "periodic" else phase % (2.0 * np.pi)
                if kind == "periodic":
                    decoded = float((decoded + 0.5) % 1.0 - 0.5)
                cursor += 2
            else:
                transformed = float(values[cursor]) * float(scale) + float(location)
                decoded = float(np.exp(transformed)) if kind == "log_robust" else transformed
                cursor += 1
            template_value = _get_path(base.parameters, name)
            if abs(decoded - template_value) <= float(snap_tolerance):
                decoded = template_value
            _set_path(parameters, name, decoded)

        parameter_vector = tuple(_get_path(parameters, name) for name in base.parameter_names)
        metadata = dict(base.metadata)
        metadata.update(
            {
                "analytic_codec_version": self.schema_version,
                "analytic_variant_id": self.variant_id,
            }
        )
        return FullCellStructureSpec(
            sample_id=str(sample_id or base.sample_id),
            generator_family=base.generator_family,
            material_kind=base.material_kind,
            parameters=parameters,
            parameter_names=base.parameter_names,
            parameter_vector=parameter_vector,
            parameter_mask=base.parameter_mask,
            epsilon_air=base.epsilon_air,
            epsilon_material=base.epsilon_material,
            schema_version=base.schema_version,
            metadata=metadata,
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-serializable fitted codec manifest."""

        return {
            "schema_version": self.schema_version,
            "output_repr": FULL_CELL_ANALYTIC_OUTPUT_REPR,
            "variant_id": self.variant_id,
            "physical_width": self.physical_width,
            "encoded_width": self.encoded_width,
            "encoded_feature_names": list(self.encoded_feature_names),
            "learned_parameter_names": list(self.learned_parameter_names),
            "transform_kinds": list(self.transform_kinds),
            "locations": list(self.locations),
            "scales": list(self.scales),
            "fixed_parameter_names": list(self.fixed_parameter_names),
            "auxiliary_parameter_names": list(self.auxiliary_parameter_names),
            "train_sample_count": int(self.train_sample_count),
            "template_mapping": copy.deepcopy(self.template_mapping),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FullCellAnalyticCodec":
        """Restore a fitted codec from its manifest mapping."""

        if str(value.get("schema_version")) != FULL_CELL_ANALYTIC_CODEC_VERSION:
            raise ValueError("Unsupported full-cell analytic codec schema")
        if str(value.get("output_repr")) != FULL_CELL_ANALYTIC_OUTPUT_REPR:
            raise ValueError("Codec manifest has the wrong output representation")
        codec = cls(
            schema_version=str(value["schema_version"]),
            variant_id=str(value["variant_id"]),
            learned_parameter_names=tuple(str(item) for item in value["learned_parameter_names"]),
            transform_kinds=tuple(str(item) for item in value["transform_kinds"]),
            locations=tuple(float(item) for item in value["locations"]),
            scales=tuple(float(item) for item in value["scales"]),
            fixed_parameter_names=tuple(str(item) for item in value.get("fixed_parameter_names", ())),
            auxiliary_parameter_names=tuple(str(item) for item in value.get("auxiliary_parameter_names", ())),
            template_mapping=dict(value["template_mapping"]),
            train_sample_count=int(value["train_sample_count"]),
        )
        if int(value.get("physical_width", codec.physical_width)) != codec.physical_width:
            raise ValueError("Codec manifest physical_width is inconsistent")
        if int(value.get("encoded_width", codec.encoded_width)) != codec.encoded_width:
            raise ValueError("Codec manifest encoded_width is inconsistent")
        return codec


def fit_full_cell_analytic_codecs(
    structures: Sequence[FullCellStructureSpec],
    *,
    require_all_variants: bool = True,
    require_production_contract: bool = True,
) -> dict[str, FullCellAnalyticCodec]:
    """Fit one codec per variant from an already selected train split."""

    grouped: dict[str, list[FullCellStructureSpec]] = {variant: [] for variant in FULL_CELL_VARIANTS}
    for structure in structures:
        grouped[full_cell_variant_id(structure)].append(structure)
    if require_all_variants:
        missing = [variant for variant, rows in grouped.items() if not rows]
        if missing:
            raise ValueError(f"Train split is missing full-cell variants: {', '.join(missing)}")
    codecs: dict[str, FullCellAnalyticCodec] = {}
    for variant, rows in grouped.items():
        if not rows:
            continue
        codec = FullCellAnalyticCodec.fit(rows)
        if require_production_contract:
            codec.assert_production_contract()
        codecs[variant] = codec
    return codecs


def fit_full_cell_analytic_codecs_from_hdf5(
    path: str | Path,
    *,
    require_all_variants: bool = True,
    require_production_contract: bool = True,
) -> dict[str, FullCellAnalyticCodec]:
    """Fit codecs from HDF5 ``idx_train`` without loading raster datasets."""

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("h5py is required to fit full-cell analytic codecs") from exc

    source = Path(path)
    with h5py.File(source, "r") as handle:
        if str(handle.attrs.get("schema_version", "")) != FULL_CELL_SCHEMA_VERSION:
            raise ValueError(f"Not a phc2d full-cell v1 HDF5 dataset: {source}")
        if "idx_train" not in handle or "structure_json" not in handle:
            raise ValueError(f"Full-cell dataset lacks idx_train or structure_json: {source}")
        indices = np.asarray(handle["idx_train"], dtype=np.int64).reshape(-1)
        if indices.size == 0:
            raise ValueError(f"Full-cell dataset has an empty train split: {source}")
        structures = [
            FullCellStructureSpec.from_mapping(json.loads(_decode_text(handle["structure_json"][int(index)])))
            for index in indices
        ]
    return fit_full_cell_analytic_codecs(
        structures,
        require_all_variants=require_all_variants,
        require_production_contract=require_production_contract,
    )


def save_full_cell_analytic_codecs(
    codecs: Mapping[str, FullCellAnalyticCodec],
    path: str | Path,
) -> Path:
    """Write a deterministic multi-variant codec manifest."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": FULL_CELL_ANALYTIC_CODEC_VERSION,
        "output_repr": FULL_CELL_ANALYTIC_OUTPUT_REPR,
        "variants": {key: codecs[key].to_mapping() for key in sorted(codecs)},
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def load_full_cell_analytic_codecs(path: str | Path) -> dict[str, FullCellAnalyticCodec]:
    """Load and validate a multi-variant codec manifest."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if str(payload.get("schema_version")) != FULL_CELL_ANALYTIC_CODEC_VERSION:
        raise ValueError("Unsupported full-cell analytic codec bundle schema")
    variants = payload.get("variants")
    if not isinstance(variants, Mapping) or not variants:
        raise ValueError("Codec bundle contains no variants")
    codecs = {str(key): FullCellAnalyticCodec.from_mapping(value) for key, value in variants.items()}
    if any(key != codec.variant_id for key, codec in codecs.items()):
        raise ValueError("Codec bundle variant keys do not match their manifests")
    return codecs


__all__ = [
    "FULL_CELL_ANALYTIC_CODEC_VERSION",
    "FULL_CELL_ANALYTIC_OUTPUT_REPR",
    "FULL_CELL_VARIANTS",
    "PRODUCTION_PHYSICAL_WIDTHS",
    "FullCellAnalyticCodec",
    "fit_full_cell_analytic_codecs",
    "fit_full_cell_analytic_codecs_from_hdf5",
    "full_cell_variant_id",
    "load_full_cell_analytic_codecs",
    "save_full_cell_analytic_codecs",
]
