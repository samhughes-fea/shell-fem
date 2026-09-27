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
        Kd: np.ndarray | None = None,
    ) -> None:
        self.K = K
        self.f = np.asarray(f, dtype=float).ravel()
        self.d_global = np.asarray(d_global, dtype=float).ravel()
        self.Kd = None if Kd is None else np.asarray(Kd, dtype=float).ravel()

    def compute(self) -> np.ndarray:
        Kd = self.Kd if self.Kd is not None else np.asarray(self.K @ self.d_global, dtype=float)
        return np.asarray(self.f - Kd, dtype=float)


__all__ = ["ComputeResidual"]
