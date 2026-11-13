"""Assembly stages: scatter element stiffness and loads into global arrays."""

from __future__ import annotations

from typing import Optional

import numpy as np

from shell_fem.pre_processing.containers.boundary_conditions import BoundaryConditions
from shell_fem.pre_processing.containers.mesh import FEAMesh
from shell_fem.pre_processing.element_library.shell import ShellElement
from shell_fem.pre_processing.load_library.load_case import LoadCase
from shell_fem.processing.static.operations.linear_solvers import HAVE_SCIPY, _sp
from shell_fem.processing.static.operations.solve_states import GlobalSystem


class GlobalAssembler:
    """Scatter element matrices/vectors into global arrays via the DOF map."""

    def __init__(self, sparse: Optional[bool] = None):
        self.sparse = HAVE_SCIPY if sparse is None else bool(sparse)
        if self.sparse and not HAVE_SCIPY:
            raise ImportError("Sparse assembly requires SciPy")

    def assemble_stiffness(self, mesh: FEAMesh, bcs: Optional[BoundaryConditions] = None):
        n = mesh.n_dof
        rows, cols, vals = [], [], []
        for element in mesh.elements:
            dofs = mesh.element_dofs(element)
            K_e = element.stiffness_matrix()
            size = dofs.size
            rows.append(np.repeat(dofs, size))
            cols.append(np.tile(dofs, size))
            vals.append(K_e.ravel())
        if not rows:
            raise ValueError("Mesh has no elements")
        if bcs is not None and bcs.springs:
            idx = np.array([mesh.dof_index(nd, d) for nd, d in bcs.springs], dtype=int)
            rows.append(idx)
            cols.append(idx)
            vals.append(np.array(list(bcs.springs.values())))
        rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        if self.sparse:
            return _sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        K = np.zeros((n, n))
        np.add.at(K, (rows, cols), vals)
        return K

    def scatter_element_matrices(self, mesh: FEAMesh, element_matrix):
        """Assemble a global matrix from ``element_matrix(shell)`` in global DOFs."""
        n = mesh.n_dof
        rows, cols, vals = [], [], []
        for element in mesh.shell_elements:
            matrix = np.asarray(element_matrix(element), dtype=float)
            dofs = mesh.element_dofs(element)
            size = dofs.size
            if matrix.shape != (size, size):
                raise ValueError(f"Element {element.element_id} matrix has shape {matrix.shape}, expected {(size, size)}")
            rows.append(np.repeat(dofs, size))
            cols.append(np.tile(dofs, size))
            vals.append(matrix.ravel())
        if not rows:
            raise ValueError("Mesh has no shell elements")
        rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        if self.sparse:
            return _sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        M = np.zeros((n, n))
        np.add.at(M, (rows, cols), vals)
        return M

    def element_load_vector(self, element: ShellElement, load_case: LoadCase, exposed: bool = True) -> np.ndarray:
        """Distributed loads on one element (pressure, body, hygrothermal) in global DOFs.

        ``exposed=False`` marks a solid whose top face is covered by another solid: it
        takes only the pressures listed for it, not the mesh-wide pressure.
        """
        F = np.zeros(element.n_dof)
        pressure = load_case.pressure_on(element.element_id) if exposed else \
            load_case.element_pressures.get(element.element_id, 0.0)
        if callable(pressure) or pressure != 0.0:
            F += element.pressure_load_vector(pressure)
        for body in load_case.body_loads:
            if body.applies_to(element.element_id):
                F += element.body_load_vector(body.acceleration)
        state = load_case.state_for(element.element_id)
        if not state.is_null:
            F += element.hygrothermal_load_vector(state)
        return F

    def line_load_vector(self, mesh: FEAMesh, load_case: LoadCase) -> np.ndarray:
        """Consistent nodal loads of every line load; each mesh edge is loaded once."""
        F = np.zeros(mesh.n_dof)
        for load in load_case.line_loads:
            seen = set()
            for element in mesh.elements:
                for edge in element.edge_index(load.node_ids):
                    key = tuple(sorted(element.node_ids[i] for i in element.EDGES[edge]))
                    if key in seen:
                        continue
                    seen.add(key)
                    F[mesh.element_dofs(element)] += element.edge_load_vector(edge, load.force_per_length)
            if not seen:
                raise ValueError("A line load matches no element edge; list the nodes along the loaded edge")
        return F

    def assemble_loads(self, mesh: FEAMesh, load_case: LoadCase) -> np.ndarray:
        F = np.zeros(mesh.n_dof)
        for (node, dof), value in load_case.nodal_loads.items():
            F[mesh.dof_index(node, dof)] += value
        covered = mesh.covered_solids() if mesh.has_solids else set()
        for element in mesh.elements:
            F[mesh.element_dofs(element)] += self.element_load_vector(
                element, load_case, element.element_id not in covered)
        if load_case.line_loads:
            F += self.line_load_vector(mesh, load_case)
        return F


class AssembleGlobalSystem:
    """Stage: global stiffness K from every element, plus grounded springs."""

    def __init__(self, mesh: FEAMesh, bcs: Optional[BoundaryConditions] = None,
                 assembler: Optional[GlobalAssembler] = None):
        self.mesh = mesh
        self.bcs = bcs
        self.assembler = assembler or GlobalAssembler()

    def run(self) -> GlobalSystem:
        K = self.assembler.assemble_stiffness(self.mesh, self.bcs)
        return GlobalSystem(K=K, n_dof=self.mesh.n_dof)


class AssembleLoadVector:
    """Stage: global load vector F of one load case (nodal, pressure, line, body, hygrothermal)."""

    def __init__(self, mesh: FEAMesh, assembler: Optional[GlobalAssembler] = None):
        self.mesh = mesh
        self.assembler = assembler or GlobalAssembler()

    def run(self, load_case: LoadCase) -> np.ndarray:
        return self.assembler.assemble_loads(self.mesh, load_case)
