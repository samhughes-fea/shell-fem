"""Buckling modes: expand each eigenvector to every global DOF and scale it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from shell_fem.pre_processing.containers.mesh import FEAMesh
from shell_fem.processing.buckling.operations.buckling_states import BucklingEigenState
from shell_fem.processing.static.operations.solve_states import ReducedSystem
from shell_fem.processing.static.results.containers import SolutionResult


@dataclass(frozen=True)
class BucklingMode:
    """One buckling mode: the load factor on the reference load and the mode shape.

    ``displacements`` is in global DOFs, scaled so that the largest translation is 1
    (and positive); prescribed DOFs are zero in a mode whatever their static value.
    """

    index: int
    load_factor: float
    mesh: FEAMesh
    displacements: np.ndarray

    def node_displacement(self, node_id: int) -> np.ndarray:
        base = self.mesh.dof_index(node_id, 0)
        return self.displacements[base : base + 6].copy()

    def displacement_array(self) -> np.ndarray:
        """(n_nodes, 6) mode shape in mesh node order."""
        return self.mesh.standard_array(self.displacements).copy()

    def as_solution(self) -> SolutionResult:
        """The mode as a ``SolutionResult`` (no loads or reactions), for plotting and VTK export."""
        zeros = np.zeros_like(self.displacements)
        return SolutionResult(self.mesh, self.displacements.copy(), zeros, zeros.copy(), 0.0, None,
                              {"buckling_mode": self.index, "load_factor": self.load_factor})


class ComputeBucklingModes:
    """Stage: modes U = T phi on every global DOF, each scaled to a unit largest translation."""

    def __init__(self, mesh: FEAMesh):
        self.mesh = mesh

    def run(self, reduced: ReducedSystem, state: BucklingEigenState) -> List[BucklingMode]:
        T = reduced.constrained.T
        modes = []
        for k in range(state.n_modes):
            U = np.asarray(T @ state.modes[:, k]).ravel()
            translations = self.mesh.standard_array(U)[:, :3]
            peak = translations.flat[np.argmax(np.abs(translations))] if translations.size else 0.0
            if peak != 0.0:
                U = U / peak
            modes.append(BucklingMode(k + 1, float(state.load_factors[k]), self.mesh, U))
        return modes
