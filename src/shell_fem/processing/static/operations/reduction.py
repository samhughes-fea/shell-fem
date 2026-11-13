"""Reduction stage: K_r = T^T K T on the independent DOFs."""

from __future__ import annotations

import numpy as np

from shell_fem.processing.static.operations.solve_states import ConstrainedSystem, ReducedSystem


class ReduceConstrainedSystem:
    """Stage: project the constrained system onto its independent DOFs."""

    def run(self, system: ConstrainedSystem) -> ReducedSystem:
        K_r = system.T.T @ system.K @ system.T
        K_g = np.asarray(system.K @ system.g).ravel()
        return ReducedSystem(constrained=system, K_r=K_r, K_g=K_g)
