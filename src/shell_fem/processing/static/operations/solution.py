"""Solution stage: factorise K_r once, then solve for each load vector."""

from __future__ import annotations

from typing import Optional

import numpy as np

from shell_fem.processing.static.operations.linear_solvers import LinearSystemSolver, default_linear_solver
from shell_fem.processing.static.operations.solve_states import FactorisedSystem, ReducedSystem


class SolveReducedSystem:
    """Stage: factorise the reduced stiffness and solve K_r q = T^T (F - K g)."""

    def __init__(self, linear_solver: Optional[LinearSystemSolver] = None):
        self.linear_solver = linear_solver or default_linear_solver()

    def factorise(self, system: ReducedSystem) -> FactorisedSystem:
        return FactorisedSystem(reduced=system, solve=self.linear_solver.factorize(system.K_r),
                                solver_name=self.linear_solver.name)

    @staticmethod
    def solve(system: FactorisedSystem, F: np.ndarray) -> np.ndarray:
        """Reduced solution q for the global load vector F."""
        return np.asarray(system.solve(system.reduced.reduce_load(F))).ravel()
