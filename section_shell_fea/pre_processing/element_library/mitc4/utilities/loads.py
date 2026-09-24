"""Work-conjugate consistent loads for the flat Reissner--Mindlin Q4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .B_matrix import StrainDisplacementOperator
from .shape_functions import ShapeFunctionOperator

EdgeName = Literal["xi_minus", "xi_plus", "eta_minus", "eta_plus"]


def _gauss_rule(order: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if int(order) == 2:
        point = 1.0 / np.sqrt(3.0)
        return np.array([-point, point]), np.ones(2)
    if int(order) == 3:
        point = np.sqrt(3.0 / 5.0)
        return np.array([-point, 0.0, point]), np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    raise ValueError(f"quadrature order must be 2 or 3, got {order}")


@dataclass(frozen=True, slots=True)
class ElementLoadOperator:
    """Integrate loads against ``Phi_u`` or physical ``B_g``."""

    shape: ShapeFunctionOperator
    ndof: int = 6

    def area_pressure(self, pressure: float, *, quadrature_order: int = 2) -> NDArray[np.float64]:
        """Return consistent nodal load for constant positive ``x3`` pressure."""
        points, weights = _gauss_rule(quadrature_order)
        force = np.zeros(4 * int(self.ndof), dtype=np.float64)
        for i, xi in enumerate(points):
            for j, eta in enumerate(points):
                phi, _, determinant, _ = self.shape.physical_coordinate_form(
                    np.array([xi]), np.array([eta])
                )
                weight = weights[i] * weights[j] * determinant[0]
                for node in range(4):
                    force[node * int(self.ndof) + 2] += (
                        float(pressure) * phi[0, node] * weight
                    )
        return force

    def edge_resultants(
        self,
        edge: EdgeName,
        *,
        membrane_traction: NDArray[np.float64] | tuple[float, float] = (0.0, 0.0),
        edge_moment: NDArray[np.float64] | tuple[float, float] = (0.0, 0.0),
        transverse_shear: float = 0.0,
        quadrature_order: int = 2,
    ) -> NDArray[np.float64]:
        """Integrate boundary pairs ``(u0,t)``, ``(theta,m)`` and ``(w,q)``."""
        traction = np.asarray(membrane_traction, dtype=np.float64)
        moment = np.asarray(edge_moment, dtype=np.float64)
        if traction.shape != (2,) or moment.shape != (2,):
            raise ValueError("membrane_traction and edge_moment must each have shape (2,)")
        points, weights = _gauss_rule(quadrature_order)
        force = np.zeros(4 * int(self.ndof), dtype=np.float64)
        for point, weight in zip(points, weights):
            if edge == "xi_minus":
                xi, eta, tangent_column = -1.0, float(point), 1
            elif edge == "xi_plus":
                xi, eta, tangent_column = 1.0, float(point), 1
            elif edge == "eta_minus":
                xi, eta, tangent_column = float(point), -1.0, 0
            elif edge == "eta_plus":
                xi, eta, tangent_column = float(point), 1.0, 0
            else:
                raise ValueError(f"unknown Q4 edge {edge!r}")
            phi, _, _, jacobian = self.shape.physical_coordinate_form(
                np.array([xi]), np.array([eta])
            )
            line_jacobian = float(np.linalg.norm(jacobian[0, :, tangent_column]))
            for node in range(4):
                base = node * int(self.ndof)
                scale = phi[0, node] * float(weight) * line_jacobian
                force[base + 0] += traction[0] * scale
                force[base + 1] += traction[1] * scale
                force[base + 2] += float(transverse_shear) * scale
                force[base + 3] += moment[0] * scale
                # theta2=-beta_x: beta_x virtual work has the opposite sign.
                force[base + 4] -= moment[1] * scale
        return force

    def initial_resultants(
        self,
        resultants: NDArray[np.float64],
        *,
        quadrature_order: int = 3,
    ) -> NDArray[np.float64]:
        """Return ``integral(B_g.T r0 dA)`` for constant generalized resultants."""
        resultant_vector = np.asarray(resultants, dtype=np.float64)
        if resultant_vector.shape != (8,) or not np.all(np.isfinite(resultant_vector)):
            raise ValueError("resultants must be a finite canonical vector with shape (8,)")
        points, weights = _gauss_rule(quadrature_order)
        force = np.zeros(4 * int(self.ndof), dtype=np.float64)
        for i, xi in enumerate(points):
            for j, eta in enumerate(points):
                b_generalized = StrainDisplacementOperator(
                    self.shape, int(self.ndof)
                ).generalized(np.array([xi]), np.array([eta]))[0]
                _, _, determinant, _ = self.shape.physical_coordinate_form(
                    np.array([xi]), np.array([eta])
                )
                force += (
                    b_generalized.T
                    @ resultant_vector
                    * weights[i]
                    * weights[j]
                    * determinant[0]
                )
        return force

    def legacy_strip_load(
        self,
        *,
        nx: float,
        nxy: float,
        kappa: float | None = None,
        donnell_area_ratio: float = 1.0,
    ) -> NDArray[np.float64]:
        """Adapt existing outboard-edge resultants and optional Donnell pressure."""
        force = self.edge_resultants(
            "eta_plus", membrane_traction=(float(nx), float(nxy))
        )
        if kappa is not None:
            force += self.area_pressure(
                float(nx) * float(kappa) * float(donnell_area_ratio)
            )
        return force


__all__ = ["EdgeName", "ElementLoadOperator"]
