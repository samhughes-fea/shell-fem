"""MITC4 strain-displacement operator (1d-fem utility template)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .shape_functions import ShapeFunctionOperator


@dataclass(frozen=True, slots=True)
class StrainDisplacementOperator:
    """Batched MITC4 B-matrices for membrane, bending, shear, and drilling rows."""

    shape: ShapeFunctionOperator
    ndof: int = 6

    def natural_coordinate_form(
        self,
        xi: NDArray[np.float64],
        eta: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
        """Return ``Bm, Bb, Bs, Bd`` with leading dimension ``n_gp``."""
        n, dndxi, dndet = self.shape.natural_coordinate_form(xi, eta)
        dnds = dndxi * self.shape.dxi_ds
        dndx = dndet * self.shape.deta_dx
        n_gp = int(xi.size)
        ndof = int(self.ndof)
        n_elem = 4 * ndof
        bm = np.zeros((n_gp, 3, n_elem), dtype=np.float64)
        bb = np.zeros((n_gp, 3, n_elem), dtype=np.float64)
        bs = np.zeros((n_gp, 2, n_elem), dtype=np.float64)
        for i in range(4):
            base = i * ndof
            bm[:, 0, base + 0] = dndx[:, i]
            bm[:, 1, base + 1] = dnds[:, i]
            bm[:, 2, base + 0] = dnds[:, i]
            bm[:, 2, base + 1] = dndx[:, i]
            bb[:, 0, base + 3] = dndx[:, i]
            bb[:, 2, base + 3] = dnds[:, i]
            bb[:, 1, base + 4] = -dnds[:, i]
            bb[:, 2, base + 4] = -dndx[:, i]
        for sign_eta, coeff in ((+1.0, 0.5), (-1.0, 0.5)):
            eta_tp = np.full(n_gp, sign_eta, dtype=np.float64)
            n_tp, dndxi_tp, dndet_tp = self.shape.natural_coordinate_form(xi, eta_tp)
            dndx_tp = dndet_tp * self.shape.deta_dx
            coeff_arr = coeff * (1.0 + eta) if sign_eta > 0 else coeff * (1.0 - eta)
            for i in range(4):
                base = i * ndof
                bs[:, 0, base + 2] += coeff_arr * dndx_tp[:, i]
                bs[:, 0, base + 3] -= coeff_arr * n_tp[:, i]
        for sign_xi, coeff in ((+1.0, 0.5), (-1.0, 0.5)):
            xi_tp = np.full(n_gp, sign_xi, dtype=np.float64)
            n_tp, dndxi_tp, _ = self.shape.natural_coordinate_form(xi_tp, eta)
            dnds_tp = dndxi_tp * self.shape.dxi_ds
            coeff_arr = coeff * (1.0 + xi) if sign_xi > 0 else coeff * (1.0 - xi)
            for i in range(4):
                base = i * ndof
                bs[:, 1, base + 2] += coeff_arr * dnds_tp[:, i]
                bs[:, 1, base + 4] += coeff_arr * n_tp[:, i]
        bd: NDArray[np.float64] | None = None
        if ndof == 6:
            bd = np.zeros((n_gp, 1, 24), dtype=np.float64)
            for i in range(4):
                base = i * 6
                bd[:, 0, base + 0] = -dnds[:, i]
                bd[:, 0, base + 1] = dndx[:, i]
                bd[:, 0, base + 5] = -n[:, i]
        return bm, bb, bs, bd
