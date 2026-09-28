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
        Kd: np.ndarray | None = None,
    ) -> None:
        self.K = K
        self.f = np.asarray(f, dtype=float).ravel()
        self.d_global = np.asarray(d_global, dtype=float).ravel()
        self.fixed = fixed
        self.Kd = None if Kd is None else np.asarray(Kd, dtype=float).ravel()

    def compute(self) -> np.ndarray:
        Kd = self.Kd if self.Kd is not None else np.asarray(self.K @ self.d_global, dtype=float)
        r_full = Kd - self.f
        reactions = np.zeros_like(r_full)
        for d in self.fixed:
            if 0 <= d < r_full.size:
                reactions[d] = r_full[d]
        return reactions


__all__ = ["ComputeReactionForce"]
