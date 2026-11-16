"""What geometric nonlinearity exists today, and what post-buckling still needs.

This package solves linear statics (one factorisation of undeformed ``K``,
``K u = F``) and linear buckling (``(K + λ K_g) φ = 0`` after that linear
prestress). ``K_g`` is already assembled for shells (membrane resultants) and,
from the solver-finish stack, for layered solids (Cauchy stress). Pressure is
integrated along the undeformed element normal. None of that is a Newton loop
or a follower load.

Post-buckling of blade panels and shear webs, and large deflection of plates,
need residual iteration on the current configuration plus a pressure that
follows the deformed surface. Those stages are named here so the gap is
testable; they are not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass


class GeometricNonlinearityNotImplemented(NotImplementedError):
    """Raised when a geometrically nonlinear stage or runner is called.

    Linear statics and linear buckling remain the solvers. See
    ``geometric_nonlinearity_scope``.
    """


@dataclass(frozen=True)
class GeometricNonlinearityScope:
    """Inventory of geometric nonlinearity: present operators versus the gap.

    ``exists`` is what the linear static and linear buckling paths already do.
    ``missing`` is what a post-buckling / large-deflection solve would still
    have to add. ``intended_uses`` is the product scope of that future work,
    not a claim that it runs today.
    """

    exists: tuple[str, ...]
    missing: tuple[str, ...]
    intended_uses: tuple[str, ...]


_EXISTS = (
    "linear static: one factorisation of undeformed K, K u = F",
    "linear buckling: (K + λ Kg) φ = 0 after a linear prestress",
    "shell Kg of membrane prestress Nx, Ny, Nxy (bending and shear do not enter)",
    "solid Kg of Cauchy stress (layered Hex8/Hex20)",
    "dead pressure along the undeformed element +e3",
)

_MISSING = (
    "Newton (or any residual iteration) on the current configuration",
    "material stiffness recomputed on the deformed mesh",
    "Kg assembled into the static tangent K_T = K + Kg(u)",
    "follower pressure (load stiffness; force along the current normal)",
    "updated geometry / finite rotation kinematics",
    "path following (arc-length) for post-buckling of panels and webs",
)

_INTENDED_USES = (
    "post-buckling of blade panels and shear webs",
    "large deflection of plates (deflection on the order of thickness and beyond)",
)


def geometric_nonlinearity_scope() -> GeometricNonlinearityScope:
    """The labelled gap: linear ``K`` / ``K_g`` exist; Newton and follower pressure do not."""
    return GeometricNonlinearityScope(exists=_EXISTS, missing=_MISSING, intended_uses=_INTENDED_USES)
