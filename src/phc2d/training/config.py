"""Common training configuration dataclasses.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ForwardTrainingConfig:
    """Small shared contract for forward surrogate train entrypoints."""

    dataset_path: Path
    model_name: str
    output_dir: Path
    epochs: int = 1
    batch_size: int = 64
    seed: int = 0
