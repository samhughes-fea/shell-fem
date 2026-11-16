"""Geometric nonlinearity: scoped, not implemented.

Linear statics and linear buckling exist. Newton equilibrium, follower pressure
and large-deflection kinematics do not. This package documents that gap for
post-buckling of panels and webs and for large plate deflection; it is not a
solver.
"""

from shell_fem.processing.nonlinear.operations import AssembleFollowerPressure, NewtonEquilibrium
from shell_fem.processing.nonlinear.scope import (
    GeometricNonlinearityNotImplemented,
    GeometricNonlinearityScope,
    geometric_nonlinearity_scope,
)

__all__ = [
    "AssembleFollowerPressure",
    "GeometricNonlinearityNotImplemented",
    "GeometricNonlinearityScope",
    "NewtonEquilibrium",
    "geometric_nonlinearity_scope",
]
