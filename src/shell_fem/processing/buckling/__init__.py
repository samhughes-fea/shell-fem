"""Linear buckling stages: geometric stiffness of a static prestress, the eigenproblem and the modes.

``geometric.AssembleGeometricStiffness``, ``eigen.SolveBucklingEigenproblem`` and
``results.ComputeBucklingModes`` are the stages; ``LinearBucklingRunner`` sequences them
and ``linear_buckling`` runs that runner. The load factors multiply the reference load.

The earlier stage classes of ``linear_buckling`` (``operations.AssembleGeometricStiffness``
and ``operations.SolveBucklingEigenproblem``, which this package used to export) still
import from here with a DeprecationWarning until 0.3.0.
"""

import warnings as _warnings

_DEPRECATED = ("AssembleGeometricStiffness", "SolveBucklingEigenproblem")


def __getattr__(name):
    if name in _DEPRECATED:
        _warnings.warn(
            f"shell_fem.processing.buckling.{name} is the old linear_buckling stage and goes in 0.3.0; "
            f"use shell_fem.processing.buckling.{'geometric' if name.startswith('Assemble') else 'eigen'}.{name}",
            DeprecationWarning, stacklevel=2)
        from shell_fem.processing.buckling import operations

        return getattr(operations, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
