"""MITC4 bilinear basis and displacement-interpolation operator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .geometry import FlatQuad4Geometry

_XI_NODES = np.array([-1.0, 1.0, 1.0, -1.0], dtype=np.float64)
_ET_NODES = np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ShapeFunctionOperator:
    """Bilinear Q4 basis ``phi`` and interpolation ``Phi_u``.

    ``N`` is deliberately not used for interpolation because it is reserved for
    membrane resultants throughout the shell-result cascade.
    """

    L_s: float | None = None
    L_x: float | None = None
    geometry: FlatQuad4Geometry | None = None

    def __post_init__(self) -> None:
        geometry = self.geometry
        if geometry is None:
            if self.L_s is None or self.L_x is None:
                raise ValueError("Provide geometry or both L_s and L_x")
            geometry = FlatQuad4Geometry.rectangle(L_s=float(self.L_s), L_x=float(self.L_x))
        object.__setattr__(self, "geometry", geometry)

    @classmethod
    def from_local_coordinates(
        cls,
        local_coordinates: NDArray[np.float64],
    ) -> "ShapeFunctionOperator":
        """Construct from arbitrary valid flat-Q4 ``(x1,x2)`` coordinates."""
        return cls(geometry=FlatQuad4Geometry(local_coordinates))

    @property
    def dxi_ds(self) -> float:
        """Legacy affine derivative factor; unavailable for a general Q4."""
        if self.L_s is None:
            raise AttributeError("dxi_ds is defined only for the L_s/L_x rectangle")
        return 2.0 / float(self.L_s)

    @property
    def deta_dx(self) -> float:
        """Legacy affine derivative factor; unavailable for a general Q4."""
        if self.L_x is None:
            raise AttributeError("deta_dx is defined only for the L_s/L_x rectangle")
        return 2.0 / float(self.L_x)

    def natural_coordinate_form(
        self,
        xi: NDArray[np.float64],
        eta: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return ``phi, dphi_dxi, dphi_deta`` with shape ``(n_gp, 4)``."""
        xi = np.asarray(xi, dtype=np.float64).ravel()
        eta = np.asarray(eta, dtype=np.float64).ravel()
        if xi.shape != eta.shape:
            raise ValueError("xi and eta must contain the same number of points")
        n_gp = int(xi.size)
        xi_b = xi[:, None]
        eta_b = eta[:, None]
        phi = 0.25 * (1.0 + _XI_NODES[None, :] * xi_b) * (
            1.0 + _ET_NODES[None, :] * eta_b
        )
        dphi_dxi = 0.25 * _XI_NODES[None, :] * (
            1.0 + _ET_NODES[None, :] * eta_b
        )
        dphi_deta = 0.25 * _ET_NODES[None, :] * (
            1.0 + _XI_NODES[None, :] * xi_b
        )
        return phi, dphi_dxi, dphi_deta

    def physical_coordinate_form(
        self,
        xi: NDArray[np.float64],
        eta: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Return ``phi, grad_x(phi), detJ, J`` at parent-space points."""
        phi, _, _ = self.natural_coordinate_form(xi, eta)
        assert self.geometry is not None
        jacobian, determinant, gradients = self.geometry.evaluate(xi, eta)
        return phi, gradients, determinant, jacobian

    def displacement_interpolation(
        self,
        xi: NDArray[np.float64],
        eta: NDArray[np.float64],
        *,
        ndof: int = 6,
    ) -> NDArray[np.float64]:
        """Return canonical ``Phi_u`` with shape ``(n_gp, n_field, 4*ndof)``.

        Canonical fields are ``[u1, u2, w, theta1, theta2]``.  For the external
        compatibility basis, ``theta1=beta_s`` and ``theta2=-beta_x``.  A sixth
        row interpolates the optional drilling rotation.
        """
        if int(ndof) not in (5, 6):
            raise ValueError(f"ndof must be 5 or 6, got {ndof}")
        phi, _, _ = self.natural_coordinate_form(xi, eta)
        n_gp = int(phi.shape[0])
        n_field = 6 if int(ndof) == 6 else 5
        out = np.zeros((n_gp, n_field, 4 * int(ndof)), dtype=np.float64)
        for node in range(4):
            base = node * int(ndof)
            out[:, 0, base + 0] = phi[:, node]
            out[:, 1, base + 1] = phi[:, node]
            out[:, 2, base + 2] = phi[:, node]
            out[:, 3, base + 3] = phi[:, node]
            out[:, 4, base + 4] = -phi[:, node]
            if int(ndof) == 6:
                out[:, 5, base + 5] = phi[:, node]
        return out
