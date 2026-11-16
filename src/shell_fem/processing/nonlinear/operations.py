"""Named stages of a geometrically nonlinear solve. None of them run.

The linear static stages factorise undeformed ``K`` once. Linear buckling
assembles ``K_g`` of that prestress and solves the eigenproblem; it does not
put ``K_g`` into a static tangent or iterate a residual. Follower pressure
would replace today's dead ``pressure_load_vector`` (undeformed ``+e3``).

Calling ``run`` records that those stages are still missing.
"""

from __future__ import annotations

from shell_fem.processing.nonlinear.scope import GeometricNonlinearityNotImplemented, geometric_nonlinearity_scope


def _raise(stage: str) -> None:
    scope = geometric_nonlinearity_scope()
    missing = "; ".join(scope.missing)
    raise GeometricNonlinearityNotImplemented(
        f"{stage} is not implemented. Linear statics and linear buckling remain "
        f"the solvers. Still missing: {missing}."
    )


class NewtonEquilibrium:
    """Residual iteration ``r(u) = F_int(u) - F_ext(u) = 0``. Not implemented.

    Today's static path is ``SolveReducedSystem``: one factorisation of
    undeformed ``K``, then ``K_r q = T^T (F - K g)`` per load case. There is
    no residual check, line search, or load increment.
    """

    def run(self, *args, **kwargs):
        _raise("Newton equilibrium")


class AssembleFollowerPressure:
    """Pressure along the current normal, plus load stiffness. Not implemented.

    ``ShellElement.pressure_load_vector`` integrates ``p`` along undeformed
    ``+e3`` and does not take a displacement. That is a dead pressure. A
    follower load would rebuild the normal (and the surface area) on the
    displaced mesh and add the corresponding load-stiffness contribution to
    the tangent.
    """

    def run(self, *args, **kwargs):
        _raise("Follower pressure")
