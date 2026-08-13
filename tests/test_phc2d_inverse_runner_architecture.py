"""Architecture contracts for the modular inverse-design runtime.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from phc2d.inverse import generator_models
from phc2d.inverse import generator_runtime
from phc2d.inverse import runner
from phc2d.inverse import surrogate_runtime


GENERATOR_ARCHITECTURE_NAMES = {
    "_ConditionalGenerator",
    "_DirectInverseGenerator",
    "_MixtureConditionalGenerator",
}


def test_runner_preserves_historical_generator_imports() -> None:
    for name in GENERATOR_ARCHITECTURE_NAMES | {"_generator_kind", "_make_generator_network"}:
        assert getattr(runner, name) is getattr(generator_models, name)

    for name in {"_MockConditionalGenerator", "_TrainedConditionalGenerator"}:
        assert getattr(runner, name) is getattr(generator_runtime, name)


def test_runner_preserves_historical_surrogate_imports() -> None:
    names = {
        "_ForwardFrozenSurrogate",
        "_MockSurrogate",
        "_QueryFrozenSurrogate",
        "_resolve_forward_surrogate_checkpoint",
        "_resolve_query_surrogate_checkpoint",
    }
    for name in names:
        assert getattr(runner, name) is getattr(surrogate_runtime, name)


def test_runner_no_longer_defines_generator_architectures() -> None:
    runner_path = Path(inspect.getsourcefile(runner) or "")
    tree = ast.parse(runner_path.read_text(encoding="utf-8-sig"))
    direct_classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}

    assert direct_classes == set()
    assert len(Path(inspect.getsourcefile(generator_models) or "").read_text(encoding="utf-8").splitlines()) <= 300
    assert len(Path(inspect.getsourcefile(generator_runtime) or "").read_text(encoding="utf-8").splitlines()) <= 250
    assert len(Path(inspect.getsourcefile(surrogate_runtime) or "").read_text(encoding="utf-8").splitlines()) <= 400


def test_generator_checkpoint_state_keys_remain_stable() -> None:
    torch = pytest.importorskip("torch")
    latent = generator_models._ConditionalGenerator(
        condition_dim=6,
        candidate_dim=4,
        latent_dim=3,
        hidden_dim=8,
        torch=torch,
    )
    direct = generator_models._DirectInverseGenerator(
        condition_dim=6,
        candidate_dim=4,
        hidden_dim=8,
        torch=torch,
    )
    mixture = generator_models._MixtureConditionalGenerator(
        condition_dim=6,
        candidate_dim=4,
        latent_dim=3,
        hidden_dim=8,
        depth=2,
        mode_count=3,
        torch=torch,
    )

    expected_mlp_keys = ("0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias")
    assert tuple(latent.state_dict()) == expected_mlp_keys
    assert tuple(direct.state_dict()) == expected_mlp_keys
    assert tuple(mixture.state_dict()) == ("condition_encoder", "mode_embedding", "decoder")
