"""Static solve stages, one class per step (see ``solve_states`` for the sequence)."""

from shell_fem.processing.static.operations.assembly import AssembleGlobalSystem, AssembleLoadVector, GlobalAssembler
from shell_fem.processing.static.operations.constraints import ConstrainGlobalSystem, ConstraintEliminator
from shell_fem.processing.static.operations.linear_solvers import (
    HAVE_SCIPY, DenseLinearSolver, LinearSystemSolver, SingularStiffnessError, SparseLinearSolver, default_linear_solver)
from shell_fem.processing.static.operations.reconstruction import ReconstructGlobalField
from shell_fem.processing.static.operations.reduction import ReduceConstrainedSystem
from shell_fem.processing.static.operations.solution import SolveReducedSystem
from shell_fem.processing.static.operations.solve_states import (
    ConstrainedSystem, FactorisedSystem, GlobalSystem, ReducedSystem)

__all__ = [
    "AssembleGlobalSystem", "AssembleLoadVector", "ConstrainGlobalSystem", "ConstrainedSystem",
    "ConstraintEliminator", "DenseLinearSolver", "FactorisedSystem", "GlobalAssembler", "GlobalSystem",
    "HAVE_SCIPY", "LinearSystemSolver", "ReconstructGlobalField", "ReduceConstrainedSystem", "ReducedSystem",
    "SingularStiffnessError", "SolveReducedSystem", "SparseLinearSolver", "default_linear_solver",
]
