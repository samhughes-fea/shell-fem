"""Formulation physics contracts for the generalised section BVP.

Production is linear static: formulations supply constant ``K`` / ``f`` in the
global DOF basis consumed by
:class:`~blade_precompute.section_shell_model.engine.periodic_reduced_bvp.PeriodicReducedBVP`.

``ResidualTangentPhysics`` is a **reserved contract** (not a scheduled milestone).
``ShellAnalysisRegime.NONLINEAR`` stays empty; do not wire a Newton runner unless
explicitly scoped outside this package's current production path.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp


@runtime_checkable
class LinearFormulationPhysics(Protocol):
    """Linear static contribution in the global DOF basis."""

    def global_tangent(self) -> sp.spmatrix:
        """Return constant tangent stiffness ``K``."""
        ...

    def global_external_force(self) -> NDArray[np.float64]:
        """Return assembled external force ``f`` (or zero)."""
        ...


@runtime_checkable
class ResidualTangentPhysics(Protocol):
    """Reserved residual / tangent interface in the global DOF basis.

    Not registered for production and **not a scheduled milestone**. Kept so the
    generalised section BVP (one free set, named load channels) does not hard-wire
    linear-only method signatures into the reduction layer.
    """

    def internal_residual(
        self,
        u_global: NDArray[np.float64],
        state: Any | None = None,
    ) -> NDArray[np.float64]:
        """Return ``R_int(u)`` (same length as ``u_global``)."""
        ...

    def tangent_stiffness(
        self,
        u_global: NDArray[np.float64],
        state: Any | None = None,
    ) -> sp.spmatrix:
        """Return ``K_t(u)`` in the same global DOF ordering as linear ``K``."""
        ...


__all__ = ["LinearFormulationPhysics", "ResidualTangentPhysics"]
