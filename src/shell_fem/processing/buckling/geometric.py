"""Geometric stiffness stage for LinearBucklingRunner.

``operations.AssembleGeometricStiffness`` is the stage ``linear_buckling`` calls. This
one returns a ``GeometricSystem`` and is what ``LinearBucklingRunner`` sequences.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from shell_fem.pre_processing.containers.mesh import FEAMesh
from shell_fem.processing.buckling.states import GeometricSystem
from shell_fem.processing.static.operations.linear_solvers import HAVE_SCIPY, _sp
from shell_fem.processing.static.results.containers import SolutionResult


class AssembleGeometricStiffness:
    """Stage: K_g of the prestress of a solved reference load case.

    Shells contribute the geometric stiffness of their membrane resultants;
    layered solids contribute the geometric stiffness of their Cauchy stress.
    The hygrothermal state of the reference load case is taken off the
    prestress, so a thermal load prestresses only where it is restrained.
    """

    def __init__(self, mesh: FEAMesh, sparse: Optional[bool] = None):
        self.mesh = mesh
        self.sparse = HAVE_SCIPY if sparse is None else bool(sparse)
        if self.sparse and not HAVE_SCIPY:
            raise ImportError("Sparse assembly requires SciPy")

    def run(self, reference: SolutionResult) -> GeometricSystem:
        mesh, n = self.mesh, self.mesh.n_dof
        load_case = reference.load_case
        rows, cols, vals = [], [], []
        for element in mesh.elements:
            dofs = mesh.element_dofs(element)
            state = load_case.state_for(element.element_id) if load_case is not None else None
            if state is not None and state.is_null:
                state = None
            K_e = element.geometric_stiffness_matrix(reference.displacements[dofs], state)
            rows.append(np.repeat(dofs, dofs.size))
            cols.append(np.tile(dofs, dofs.size))
            vals.append(K_e.ravel())
        if not rows:
            raise ValueError("Mesh has no elements")
        rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        if self.sparse:
            K_g = _sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        else:
            K_g = np.zeros((n, n))
            np.add.at(K_g, (rows, cols), vals)
        return GeometricSystem(K_g=K_g, n_dof=n)
