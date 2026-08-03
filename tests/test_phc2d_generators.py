"""Tests for shared structure generators."""

from __future__ import annotations

import numpy as np

from phc2d.generators import build_structure_generator, collect_structure_metadata


def test_generator_registry_and_metadata_schema():
    generator = build_structure_generator("legacy_random")
    sample = generator.sample(rng=np.random.default_rng(1), dataset_cfg={"prob_high": 0.5, "n_wedge": 36})

    metadata = collect_structure_metadata(
        binary_grid=sample.binary_grid,
        seed=1,
        sample_size=16,
        generator_name=generator.name,
        symmetry=str(sample.metadata.get("symmetry", "none")),
        postprocess_steps=list(sample.postprocess_steps),
    )

    for key in [
        "generator_name",
        "seed",
        "grid_size",
        "fill_fraction",
        "symmetry",
        "num_connected_components",
        "largest_component_ratio",
        "num_holes",
        "min_feature_size",
        "postprocess_steps",
    ]:
        assert key in metadata
    assert sample.rho.shape == (36,)
    assert sample.binary_grid.shape == (16, 16)
