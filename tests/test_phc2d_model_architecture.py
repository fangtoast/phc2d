"""Architecture contracts for the modularized model package.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from phc2d.models import architecture_fingerprint, build_model
from phc2d.models import registry
from phc2d.models.band_cnn import build_lifted_cnn
from phc2d.models.band_mlp import build_mlp, build_pod_res_mlp, build_res_mlp
from phc2d.models.band_operators import build_fno2d_band, build_fno_lite
from phc2d.models.baselines import (
    build_fno2d_band as compatibility_build_fno2d_band,
)
from phc2d.models.baselines import build_fno_lite as compatibility_build_fno_lite
from phc2d.models.baselines import build_lifted_cnn as compatibility_build_lifted_cnn
from phc2d.models.baselines import build_mlp as compatibility_build_mlp
from phc2d.models.baselines import build_pod_res_mlp as compatibility_build_pod_res_mlp
from phc2d.models.baselines import build_res_mlp as compatibility_build_res_mlp
from phc2d.models.full_cell_convnext import build_convnext_band_v1
from phc2d.models.full_cell_mixer import build_mlp_mixer_band_v1
from phc2d.models.full_cell_operators import (
    build_dual_lane_band_operator_v1,
    build_tfno2d_band_v3,
)
from phc2d.models.full_cell_resmlp import build_res_mlp_gated_v2
from phc2d.models.full_cell_upgrades import (
    build_convnext_band_v1 as compatibility_build_convnext_band_v1,
)
from phc2d.models.full_cell_upgrades import (
    build_dual_lane_band_operator_v1 as compatibility_build_dual_lane_band_operator_v1,
)
from phc2d.models.full_cell_upgrades import (
    build_mlp_mixer_band_v1 as compatibility_build_mlp_mixer_band_v1,
)
from phc2d.models.full_cell_upgrades import (
    build_res_mlp_gated_v2 as compatibility_build_res_mlp_gated_v2,
)
from phc2d.models.full_cell_upgrades import (
    build_tfno2d_band_v3 as compatibility_build_tfno2d_band_v3,
)


def test_legacy_model_modules_are_thin_identity_preserving_facades() -> None:
    assert compatibility_build_mlp is build_mlp
    assert compatibility_build_pod_res_mlp is build_pod_res_mlp
    assert compatibility_build_res_mlp is build_res_mlp
    assert compatibility_build_lifted_cnn is build_lifted_cnn
    assert compatibility_build_fno_lite is build_fno_lite
    assert compatibility_build_fno2d_band is build_fno2d_band
    assert compatibility_build_mlp_mixer_band_v1 is build_mlp_mixer_band_v1
    assert compatibility_build_res_mlp_gated_v2 is build_res_mlp_gated_v2
    assert compatibility_build_convnext_band_v1 is build_convnext_band_v1
    assert compatibility_build_tfno2d_band_v3 is build_tfno2d_band_v3
    assert compatibility_build_dual_lane_band_operator_v1 is build_dual_lane_band_operator_v1

    model_dir = Path(inspect.getsourcefile(registry)).parent
    assert len((model_dir / "baselines.py").read_text(encoding="utf-8").splitlines()) <= 40
    assert len((model_dir / "field.py").read_text(encoding="utf-8").splitlines()) <= 40
    assert len((model_dir / "full_cell_upgrades.py").read_text(encoding="utf-8").splitlines()) <= 40


def test_registry_mechanism_is_separate_from_builtin_catalog() -> None:
    registry_source = inspect.getsource(registry)
    assert "ModelSpec(" not in registry_source
    assert len(registry_source.splitlines()) <= 300
    catalog_dir = Path(inspect.getsourcefile(registry)).parent / "catalog"
    catalog_sizes = {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in catalog_dir.glob("*.py")
    }
    assert catalog_sizes
    assert max(catalog_sizes.values()) <= 250


@pytest.mark.parametrize(
    ("name", "kwargs", "implementation_module", "expected_fingerprint"),
    [
        (
            "res_mlp",
            {"input_shape": (7,), "output_shape": (4, 2), "hidden_dim": 11, "depth": 2},
            "phc2d.models.band_mlp",
            "16cb15ca8729e6a33505f4d370fbcf4e2d0d21b1b0107f0ed09e925a4065df46",
        ),
        (
            "lifted_cnn",
            {"input_shape": (1, 11, 13), "output_shape": (4, 2), "channels": 6},
            "phc2d.models.band_cnn",
            "0f940e33efb2198f1e645055b10d0f5b921ced88ff9551afc409fa3a8d66da08",
        ),
        (
            "fno2d_band",
            {
                "input_shape": (1, 11, 13),
                "output_shape": (4, 2),
                "width": 8,
                "depth": 2,
                "modes": (3, 4),
                "band_modes": (3, 4),
                "fourier_rank": 4,
            },
            "phc2d.models.band_operators",
            "922e0b4b3f825b7616ee097ec131d6123263913c9fcea14004c1603757c51510",
        ),
        (
            "deep_onet_query",
            {
                "input_shape": (1, 6, 6),
                "output_shape": (3,),
                "k_dim": 1,
                "branch_width": 16,
                "trunk_width": 16,
                "latent_dim": 8,
                "depth": 2,
            },
            "phc2d.models.query",
            "afaf8f9b552cb337b10138e75c28ce5b2263dd9b82dae79786b776f084594c79",
        ),
        (
            "convnext_band_v1",
            {
                "input_shape": (32, 32),
                "output_shape": (5, 3),
                "stage_dims": (8, 16, 32),
                "stage_depths": (1, 1, 1),
            },
            "phc2d.models.full_cell_convnext",
            "1383f505714a8fb2c575fd339176a5ab20c949580edd241aa322e3dd58d19202",
        ),
        (
            "tfno2d_band_v3",
            {
                "input_shape": (32, 32),
                "output_shape": (5, 3),
                "width": 8,
                "depth": 2,
                "modes": (4, 4),
                "norm_groups": 2,
            },
            "phc2d.models.full_cell_operators",
            "0010e61b171a0ab51a969bf36b5f1f88397dff016a9d7eeee50e722d34211df7",
        ),
        (
            "unet_field_v2",
            {
                "grid_channels": 5,
                "context_dim": 3,
                "field_channels": 6,
                "width": 8,
                "depth": 2,
            },
            "phc2d.models.field_conditioned",
            "2253afb0ab538f86a461dce9aab20961e8933588f5fdc30d2f3ce35f9e53c386",
        ),
    ],
)
def test_module_split_preserves_historical_architecture_fingerprint(
    name: str,
    kwargs: dict,
    implementation_module: str,
    expected_fingerprint: str,
) -> None:
    pytest.importorskip("torch")
    model = build_model(name, **kwargs)
    assert type(model).__module__ == implementation_module
    assert architecture_fingerprint(model) == expected_fingerprint
