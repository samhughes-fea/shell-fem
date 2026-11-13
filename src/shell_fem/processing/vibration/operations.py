"""Vibration stages: mass matrix, then the smallest natural frequencies."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from shell_fem.pre_processing.containers.boundary_conditions import BoundaryConditions
from shell_fem.pre_processing.containers.mesh import FEAMesh
from shell_fem.processing.eigenpairs import expand_modes, reduce_matrices, vibration_eigenpairs
from shell_fem.processing.static.operations.assembly import GlobalAssembler


class AssembleMassMatrix:
    """Stage: consistent mass of every shell element."""

    def __init__(self, mesh: FEAMesh, assembler: Optional[GlobalAssembler] = None):
        self.mesh = mesh
        self.assembler = assembler or GlobalAssembler()

    def run(self):
        return self.assembler.scatter_element_matrices(self.mesh, lambda element: element.mass_matrix())


class SolveVibrationEigenproblem:
    """Stage: K φ = ω² M φ on the free degrees of freedom.

    Returns angular frequencies (rad/s) and full-system mode shapes
    (n_dof, n_modes), mass-normalised and ordered by increasing frequency.
    """

    def __init__(self, mesh: FEAMesh, bcs: BoundaryConditions, n_modes: int = 6):
        self.mesh = mesh
        self.bcs = bcs
        self.n_modes = n_modes

    def run(self, K, M) -> Tuple[np.ndarray, np.ndarray]:
        T, (K_r, M_r), _ = reduce_matrices(self.mesh, self.bcs, K, M)
        omega, reduced = vibration_eigenpairs(K_r, M_r, self.n_modes)
        return omega, expand_modes(T, reduced)
