"""Tests for multi-representation data contracts and inverse candidates."""

from __future__ import annotations

import numpy as np

from phc2d.data import (
    DataRepresentationSpec,
    infer_dataset_schema,
    make_input,
    make_target,
    validate_representation_compatibility,
    wedge_indices,
)
from phc2d.inverse import binary_morphology_metrics, diversity_hamming, morphology_score, unique_rate
from phc2d.models import ModelSpec, build_model, get_model_spec, register_model


def test_forward_npz_schema_and_representation_transform(tmp_path):
    path = tmp_path / "tiny.npz"
    grid_shape = (8, 8)
    rho = np.zeros((2, len(wedge_indices(grid_shape))), dtype=np.float32)
    rho[0, 0] = 1.0
    wh = np.ones((2, 5, 3), dtype=np.float32)
    np.savez(path, rho=rho, Wh=wh, idx_train=np.array([0]), idx_val=np.array([], dtype=np.int64), idx_test=np.array([1]))

    schema = infer_dataset_schema(path)
    assert schema["n_samples"] == 2
    assert schema["k_points"] == 5
    assert schema["n_bands"] == 3

    spec = DataRepresentationSpec(input_repr="epsilon_grid", output_target="Wh", grid_shape=grid_shape)
    assert validate_representation_compatibility(schema, spec) == []
    sample = {"rho": rho[0], "Wh": wh[0]}
    assert make_input(sample, spec).shape == grid_shape
    assert make_target(sample, spec).shape == (5, 3)


def test_model_spec_registry_keeps_factory_api():
    name = "unit_test_model_spec"
    register_model(
        name,
        lambda scale=1.0: {"scale": scale},
        replace=True,
        spec=ModelSpec(
            name=name,
            family="mlp",
            input_repr="rho_wedge",
            output_target="Wh",
            requires=(),
        ),
    )
    assert build_model(name, scale=2.0)["scale"] == 2.0
    spec = get_model_spec(name)
    assert spec.input_repr == "rho_wedge"
    assert spec.output_target == "Wh"


def test_candidate_diversity_and_morphology_metrics():
    candidates = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    assert unique_rate(candidates) == 2 / 3
    assert diversity_hamming(candidates) > 0.0

    metrics = binary_morphology_metrics(np.array([[1, 0], [1, 1]], dtype=np.float32))
    assert metrics["fill_fraction"] == 0.75
    assert 0.0 <= morphology_score(metrics) <= 1.0
