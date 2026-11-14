"""Buckling stages: geometric stiffness from a static solution, then the load factors."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from laminate_constitutive_model.laminate_hygrothermal import NULL_STATE
from shell_fem.pre_processing.containers.boundary_conditions import BoundaryConditions
from shell_fem.pre_processing.containers.mesh import FEAMesh
from shell_fem.processing.eigenpairs import buckling_eigenpairs, expand_modes, reduce_matrices
from shell_fem.processing.static.operations.assembly import GlobalAssembler
from shell_fem.processing.static.results.containers import SolutionResult


class AssembleGeometricStiffness:
    """Stage: geometric stiffness Kg of the membrane forces in a static solution."""

    def __init__(self, mesh: FEAMesh, solution: SolutionResult, assembler: Optional[GlobalAssembler] = None):
        self.mesh = mesh
        self.solution = solution
        self.assembler = assembler or GlobalAssembler()

    def run(self):
        load_case = self.solution.load_case

        def matrix(element):
            state = load_case.state_for(element.element_id) if load_case is not None else NULL_STATE
            return element.geometric_stiffness_matrix(self.solution.element_displacements(element.element_id), state)

        return self.assembler.scatter_element_matrices(self.mesh, matrix)


class SolveBucklingEigenproblem:
    """Stage: (K + λ Kg) φ = 0 on the free degrees of freedom.

    Returns positive load factors, ascending, and full-system modes
    (n_dof, n_modes) scaled so the largest absolute component of each mode is 1.
    """

    def __init__(self, mesh: FEAMesh, bcs: BoundaryConditions, n_modes: int = 1):
        self.mesh = mesh
        self.bcs = bcs
        self.n_modes = n_modes

    def run(self, K, Kg) -> Tuple[np.ndarray, np.ndarray]:
        T, (K_r, Kg_r), _ = reduce_matrices(self.mesh, self.bcs, K, Kg)
        factors, reduced = buckling_eigenpairs(K_r, Kg_r, self.n_modes)
        return factors, expand_modes(T, reduced)
