"""Typed experiment profile metadata.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

ExperimentFamily = Literal["reproduction", "research"]
ExperimentStage = Literal[
    "data",
    "forward_train",
    "forward_eval",
    "inverse_train",
    "surrogate_eval",
    "mpb_replay",
    "report",
]


@dataclass(frozen=True)
class ExperimentSpec:
    """Minimal manifest shared by data, training, inverse, replay, and report jobs."""

    experiment_id: str
    family: ExperimentFamily
    stage: ExperimentStage
    dataset: Mapping[str, Any] = field(default_factory=dict)
    model: Mapping[str, Any] = field(default_factory=dict)
    solver: Mapping[str, Any] = field(default_factory=dict)
    output: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML friendly representation."""
        return {
            "experiment_id": self.experiment_id,
            "family": self.family,
            "stage": self.stage,
            "dataset": dict(self.dataset),
            "model": dict(self.model),
            "solver": dict(self.solver),
            "output": dict(self.output),
            "metadata": dict(self.metadata),
        }
