"""Reissner--Mindlin strain-displacement and MITC4 assumed-shear operators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .shape_functions import ShapeFunctionOperator


@dataclass(frozen=True, slots=True)
class StrainDisplacementOperator:
    """Build physical ``B`` blocks in canonical ABDAs generalized ordering.

    The public compatibility degrees of freedom are
    ``[u1, u2, w, beta_s, beta_x, beta_z]`` and are mapped once to canonical
    rotations by ``theta1=beta_s`` and ``theta2=-beta_x``.
    """

    shape: ShapeFunctionOperator
    ndof: int = 6

    def natural_coordinate_form(
        self,
        xi: NDArray[np.float64],
        eta: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
        """Return ``B_m, B_b, B_s_mitc, B_d`` at parent-space points."""
        phi, gradients, _, _ = self.shape.physical_coordinate_form(xi, eta)
        n_gp = int(xi.size)
        ndof = int(self.ndof)
        if ndof not in (5, 6):
            raise ValueError(f"ndof must be 5 or 6, got {ndof}")
        n_elem = 4 * ndof
        bm = np.zeros((n_gp, 3, n_elem), dtype=np.float64)
        bb = np.zeros((n_gp, 3, n_elem), dtype=np.float64)
        for i in range(4):
            base = i * ndof
            d1 = gradients[:, i, 0]
            d2 = gradients[:, i, 1]
            # epsilon0 = [u1,1, u2,2, u1,2 + u2,1]
            bm[:, 0, base + 0] = d1
            bm[:, 1, base + 1] = d2
            bm[:, 2, base + 0] = d2
            bm[:, 2, base + 1] = d1
            # kappa = -L_m theta, theta1=beta_s, theta2=-beta_x.
            bb[:, 0, base + 3] = -d1
            bb[:, 1, base + 4] = d2
            bb[:, 2, base + 3] = -d2
            bb[:, 2, base + 4] = d1
        bs = self.mitc_shear(xi, eta)
        bd: NDArray[np.float64] | None = None
        if ndof == 6:
            bd = np.zeros((n_gp, 1, 24), dtype=np.float64)
            for i in range(4):
                base = i * 6
                bd[:, 0, base + 0] = -gradients[:, i, 1]
                bd[:, 0, base + 1] = gradients[:, i, 0]
                bd[:, 0, base + 5] = -phi[:, i]
        return bm, bb, bs, bd

    def compatible_shear(
        self,
        xi: NDArray[np.float64],
        eta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return compatible ``gamma_s=[w,1-theta1,w,2-theta2]``."""
        phi, gradients, _, _ = self.shape.physical_coordinate_form(xi, eta)
        ndof = int(self.ndof)
        bs = np.zeros((phi.shape[0], 2, 4 * ndof), dtype=np.float64)
        for node in range(4):
            base = node * ndof
            bs[:, 0, base + 2] = gradients[:, node, 0]
            bs[:, 0, base + 3] = -phi[:, node]
            bs[:, 1, base + 2] = gradients[:, node, 1]
            bs[:, 1, base + 4] = phi[:, node]
        return bs

    def mitc_shear(
        self,
        xi: NDArray[np.float64],
        eta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return covariantly tied MITC4 shear in physical ``(x1,x2)`` axes."""
        xi_v = np.asarray(xi, dtype=np.float64).ravel()
        eta_v = np.asarray(eta, dtype=np.float64).ravel()
        if xi_v.shape != eta_v.shape:
            raise ValueError("xi and eta must contain the same number of points")
        n_gp = int(xi_v.size)
        n_elem = 4 * int(self.ndof)

        # Standard edge-midpoint tying points:
        # A=(0,-1), C=(0,+1) for gamma_xi;
        # D=(-1,0), B=(+1,0) for gamma_eta.
        tying = {
            "A": (0.0, -1.0),
            "C": (0.0, +1.0),
            "D": (-1.0, 0.0),
            "B": (+1.0, 0.0),
        }
        covariant: dict[str, NDArray[np.float64]] = {}
        for name, (xi_t, eta_t) in tying.items():
            xi_a = np.array([xi_t], dtype=np.float64)
            eta_a = np.array([eta_t], dtype=np.float64)
            b_physical = self.compatible_shear(xi_a, eta_a)[0]
            _, _, _, jacobian = self.shape.physical_coordinate_form(xi_a, eta_a)
            component = 0 if name in ("A", "C") else 1
            tangent = jacobian[0, :, component]
            covariant[name] = tangent @ b_physical

        out = np.zeros((n_gp, 2, n_elem), dtype=np.float64)
        _, _, _, jacobians = self.shape.physical_coordinate_form(xi_v, eta_v)
        for gp in range(n_gp):
            b_xi = 0.5 * (
                (1.0 - eta_v[gp]) * covariant["A"]
                + (1.0 + eta_v[gp]) * covariant["C"]
            )
            b_eta = 0.5 * (
                (1.0 - xi_v[gp]) * covariant["D"]
                + (1.0 + xi_v[gp]) * covariant["B"]
            )
            b_covariant = np.vstack((b_xi, b_eta))
            out[gp] = np.linalg.solve(jacobians[gp].T, b_covariant)
        return out

    def generalized(
        self,
        xi: NDArray[np.float64],
        eta: NDArray[np.float64],
        *,
        use_mitc: bool = True,
    ) -> NDArray[np.float64]:
        """Return assembled physical ``B_g=[B_m;B_b;B_s]``."""
        bm, bb, bs_mitc, _ = self.natural_coordinate_form(xi, eta)
        bs = bs_mitc if use_mitc else self.compatible_shear(xi, eta)
        return np.concatenate((bm, bb, bs), axis=1)
