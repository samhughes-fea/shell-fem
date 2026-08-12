"""MITC4 bilinear shape-function operator (1d-fem utility template)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_XI_NODES = np.array([-1.0, 1.0, 1.0, -1.0], dtype=np.float64)
_ET_NODES = np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ShapeFunctionOperator:
    """Bilinear MITC4 parent-space shape functions on ``[-1,1]^2``."""

    L_s: float
    L_x: float

    @property
    def dxi_ds(self) -> float:
        return 2.0 / float(self.L_s)

    @property
    def deta_dx(self) -> float:
        return 2.0 / float(self.L_x)

    def natural_coordinate_form(
        self,
        xi: NDArray[np.float64],
        eta: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return ``N, dN_dxi, dN_deta`` with shape ``(n_gp, 4)``."""
        xi = np.asarray(xi, dtype=np.float64).ravel()
        eta = np.asarray(eta, dtype=np.float64).ravel()
        n_gp = int(xi.size)
        xi_b = xi[:, None]
        eta_b = eta[:, None]
        n = 0.25 * (1.0 + _XI_NODES[None, :] * xi_b) * (1.0 + _ET_NODES[None, :] * eta_b)
        dndxi = 0.25 * _XI_NODES[None, :] * (1.0 + _ET_NODES[None, :] * eta_b)
        dndet = 0.25 * _ET_NODES[None, :] * (1.0 + _XI_NODES[None, :] * xi_b)
        return n, dndxi, dndet
