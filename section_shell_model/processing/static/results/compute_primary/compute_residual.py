"""Primary results: equilibrium residual f - K d."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class ComputeResidual:
    def __init__(
        self,
        *,
        K: sp.spmatrix,
        f: np.ndarray,
        d_global: np.ndarray,
    ) -> None:
        self.K = K
        self.f = np.asarray(f, dtype=float).ravel()
        self.d_global = np.asarray(d_global, dtype=float).ravel()

    def compute(self) -> np.ndarray:
        return np.asarray(self.f - self.K @ self.d_global, dtype=float)


__all__ = ["ComputeResidual"]
