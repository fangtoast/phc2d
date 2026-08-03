"""Protocol for frozen forward surrogates.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class ForwardSurrogateAdapter(Protocol):
    """Minimal interface required by inverse-design experiments."""

    nk: int
    nb: int
    input_repr: str
    mu: float
    sigma: float
    metadata: dict[str, Any]

    def predict(self, rho: Any) -> np.ndarray:
        """Predict physical band values with shape ``(N, nk, nb)``."""

    def predict_torch(self, rho: Any, *, standardized: bool = False, no_grad: bool = False) -> Any:
        """Predict bands as a torch tensor for differentiable inverse training."""
