"""Primary results: masked support reactions at fixed DOFs."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class ComputeReactionForce:
    def __init__(
        self,
        *,
        K: sp.spmatrix,
        f: np.ndarray,
        d_global: np.ndarray,
        fixed: set[int],
    ) -> None:
        self.K = K
        self.f = np.asarray(f, dtype=float).ravel()
        self.d_global = np.asarray(d_global, dtype=float).ravel()
        self.fixed = fixed

    def compute(self) -> np.ndarray:
        r_full = np.asarray(self.K @ self.d_global - self.f, dtype=float)
        reactions = np.zeros_like(r_full)
        for d in self.fixed:
            if 0 <= d < r_full.size:
                reactions[d] = r_full[d]
        return reactions


__all__ = ["ComputeReactionForce"]
