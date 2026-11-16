"""Post-buckling / large-deflection runner. Orchestration shell only.

Intended sequence, once a solver exists (not implemented here):

    optional linear buckling mode as an imperfection seed
    for each load increment:
        AssembleFollowerPressure on the current surface
        NewtonEquilibrium: tangent K_T = K(u) + Kg(u), residual, line search
        or arc-length for limit points
        update geometry

``K_g`` already exists for linear buckling (membrane prestress of shells;
Cauchy-stress geometric stiffness of layered solids). It is not assembled into
a static tangent, the mesh is not updated, and pressure stays dead.

Scope: post-buckling of blade panels and shear webs, and large deflection of
plates. Not a geometrically exact shell, not contact, not material
nonlinearity. ``execute`` raises; linear statics and linear buckling are
unchanged.
"""

from __future__ import annotations

from shell_fem.pre_processing.containers.boundary_conditions import BoundaryConditions
from shell_fem.pre_processing.containers.mesh import FEAMesh
from shell_fem.pre_processing.load_library.load_case import LoadCase
from shell_fem.processing.nonlinear.scope import (
    GeometricNonlinearityNotImplemented,
    geometric_nonlinearity_scope,
)


class GeometricNonlinearRunner:
    """Post-buckling and large-deflection analysis (not implemented).

    >>> GeometricNonlinearRunner(mesh, bcs, load).execute()  # raises

    The constructor stores the same objects a linear runner would take so the
    future call shape is visible. ``execute`` always raises
    ``GeometricNonlinearityNotImplemented``. Use ``LinearStaticRunner`` or
    ``LinearBucklingRunner`` instead.
    """

    def __init__(self, mesh: FEAMesh, bcs: BoundaryConditions, load_case: LoadCase):
        self.mesh = mesh
        self.bcs = bcs
        self.load_case = load_case

    def execute(self) -> "GeometricNonlinearRunner":
        scope = geometric_nonlinearity_scope()
        missing = "; ".join(scope.missing)
        uses = "; ".join(scope.intended_uses)
        raise GeometricNonlinearityNotImplemented(
            "Geometric nonlinearity is scoped, not implemented. "
            f"Intended uses: {uses}. Still missing: {missing}. "
            "LinearStaticRunner and LinearBucklingRunner remain the solvers."
        )

    def get_results(self):
        raise RuntimeError("Call execute() before get_results(); execute() is not implemented")
