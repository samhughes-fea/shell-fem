"""Validated flat four-node isoparametric geometry for MITC4."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class FlatQuad4Geometry:
    """Flat Q4 geometry in local plate coordinates ``(x1, x2)``.

    Node order follows parent corners ``(-1,-1), (+1,-1), (+1,+1),
    (-1,+1)``.  The coordinate columns are physical ``x1`` and ``x2``.
    """

    local_coordinates: NDArray[np.float64]
    jacobian_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.local_coordinates, dtype=np.float64)
        if coordinates.shape != (4, 2):
            raise ValueError(
                "FlatQuad4Geometry.local_coordinates must have shape (4, 2), "
                f"got {coordinates.shape}"
            )
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("FlatQuad4Geometry.local_coordinates must be finite")
        object.__setattr__(self, "local_coordinates", coordinates.copy())
        # Validate the four corners, the four MITC tying points and the centre.
        xi = np.array([-1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0])
        eta = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0])
        self.evaluate(xi, eta)

    @classmethod
    def rectangle(cls, *, L_s: float, L_x: float) -> "FlatQuad4Geometry":
        """Build the existing strip rectangle using ``x1=x`` and ``x2=s``."""
        ls = float(L_s)
        lx = float(L_x)
        if not np.isfinite(ls) or not np.isfinite(lx) or ls <= 0.0 or lx <= 0.0:
            raise ValueError(f"L_s and L_x must be finite and positive, got {L_s}, {L_x}")
        return cls(
            np.array(
                [
                    [-0.5 * lx, -0.5 * ls],
                    [-0.5 * lx, +0.5 * ls],
                    [+0.5 * lx, +0.5 * ls],
                    [+0.5 * lx, -0.5 * ls],
                ],
                dtype=np.float64,
            )
        )

    def evaluate(
        self,
        xi: NDArray[np.float64],
        eta: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return Jacobians, determinants and physical basis gradients.

        Returns
        -------
        jacobian
            ``(n, 2, 2)`` with columns ``a_xi = dx/dxi`` and
            ``a_eta = dx/deta``.
        determinant
            ``(n,)`` positive area-map determinants.
        gradients
            ``(n, 4, 2)`` rows ``[dphi/dx1, dphi/dx2]``.
        """
        xi_v = np.asarray(xi, dtype=np.float64).ravel()
        eta_v = np.asarray(eta, dtype=np.float64).ravel()
        if xi_v.shape != eta_v.shape:
            raise ValueError("xi and eta must contain the same number of points")
        xi_nodes = np.array([-1.0, 1.0, 1.0, -1.0])
        eta_nodes = np.array([-1.0, -1.0, 1.0, 1.0])
        dphi_dxi = 0.25 * xi_nodes[None, :] * (
            1.0 + eta_nodes[None, :] * eta_v[:, None]
        )
        dphi_deta = 0.25 * eta_nodes[None, :] * (
            1.0 + xi_nodes[None, :] * xi_v[:, None]
        )
        parent_gradients = np.stack([dphi_dxi, dphi_deta], axis=2)
        jacobian = np.einsum(
            "ia,gip->gap", self.local_coordinates, parent_gradients, optimize=True
        )
        signed_determinant = np.linalg.det(jacobian)
        if np.any(~np.isfinite(signed_determinant)) or np.any(
            np.abs(signed_determinant) <= float(self.jacobian_tolerance)
        ):
            raise ValueError("Q4 Jacobian is degenerate")
        signs = np.sign(signed_determinant)
        if np.any(signs != signs[0]):
            raise ValueError("Q4 geometry is folded or changes orientation")
        determinant = np.abs(signed_determinant)
        inverse = np.linalg.inv(jacobian)
        gradients = np.einsum(
            "gip,gpa->gia", parent_gradients, inverse, optimize=True
        )
        return jacobian, determinant, gradients

    @property
    def area(self) -> float:
        """Element area from exact 2×2 Gauss integration."""
        p = 1.0 / np.sqrt(3.0)
        xi, eta = np.meshgrid(np.array([-p, p]), np.array([-p, p]), indexing="ij")
        _, determinant, _ = self.evaluate(xi.ravel(), eta.ravel())
        return float(np.sum(determinant))

    @property
    def quality_metrics(self) -> dict[str, float]:
        """Return deterministic distortion diagnostics at 2×2 Gauss points."""
        point = 1.0 / np.sqrt(3.0)
        xi, eta = np.meshgrid(
            np.array([-point, point]), np.array([-point, point]), indexing="ij"
        )
        jacobian, determinant, _ = self.evaluate(xi.ravel(), eta.ravel())
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        condition = singular_values[:, 0] / singular_values[:, -1]
        return {
            "area": self.area,
            "det_j_min": float(np.min(determinant)),
            "det_j_max": float(np.max(determinant)),
            "jacobian_ratio": float(np.min(determinant) / np.max(determinant)),
            "condition_max": float(np.max(condition)),
        }


__all__ = ["FlatQuad4Geometry"]
