"""Constraint stage: supports, prescribed values and multi-point constraints as U = T q + g."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from shell_fem.pre_processing.containers.boundary_conditions import BoundaryConditions
from shell_fem.pre_processing.containers.mesh import FEAMesh
from shell_fem.processing.static.operations.linear_solvers import HAVE_SCIPY, _sp
from shell_fem.processing.static.operations.solve_states import ConstrainedSystem, GlobalSystem


class ConstraintEliminator:
    """Builds U = T U_r + g from prescribed values and linear constraints."""

    def build(self, mesh: FEAMesh, bcs: BoundaryConditions):
        n = mesh.n_dof
        prescribed: Dict[int, float] = {}
        for (node, dof), value in bcs.prescribed.items():
            for i in mesh.expand_dof(node, dof):
                prescribed[i] = value
        slaves = {}
        for c in bcs.constraints:
            s = mesh.dof_index(*c.slave)
            if s in prescribed:
                raise ValueError(f"DOF {c.slave} is both prescribed and a constraint slave")
            if s in slaves:
                raise ValueError(f"DOF {c.slave} is the slave of more than one constraint")
            slaves[s] = c
        for node in mesh.solid_only_nodes():  # rotations without stiffness
            for d in (3, 4, 5):
                i = mesh.dof_index(node, d)
                if i not in slaves:
                    prescribed.setdefault(i, 0.0)
        free_nodes = mesh.element_free_nodes()
        if free_nodes:  # reference nodes: hold DOFs that no constraint or spring uses
            used = {mesh.dof_index(*k) for c in bcs.constraints for k, _ in c.masters}
            used |= {mesh.dof_index(*k) for k in bcs.springs}
            for node in free_nodes:
                for d in range(mesh.node_dof_count(node)):
                    i = mesh.dof_index(node, d)
                    if i not in slaves and i not in used:
                        prescribed.setdefault(i, 0.0)
        free = [i for i in range(n) if i not in prescribed and i not in slaves]
        column = {dof: j for j, dof in enumerate(free)}
        g = np.zeros(n)
        for i, v in prescribed.items():
            g[i] = v
        rows, cols, vals = list(free), list(range(len(free))), [1.0] * len(free)
        for s, c in slaves.items():
            g[s] += c.constant
            for key, coef in c.masters:
                m = mesh.dof_index(*key)
                if m in slaves:
                    raise ValueError(f"Master DOF {key} is itself a constraint slave (chains are not supported)")
                if m in prescribed:
                    g[s] += coef * prescribed[m]
                else:
                    rows.append(s)
                    cols.append(column[m])
                    vals.append(coef)
        if HAVE_SCIPY:
            T = _sp.csr_matrix((vals, (rows, cols)), shape=(n, len(free)))
        else:
            T = np.zeros((n, len(free)))
            np.add.at(T, (np.array(rows), np.array(cols)), np.array(vals))
        return T, g


class ConstrainGlobalSystem:
    """Stage: impose the boundary conditions on a global system by elimination."""

    def __init__(self, mesh: FEAMesh, bcs: BoundaryConditions, eliminator: Optional[ConstraintEliminator] = None):
        self.mesh = mesh
        self.bcs = bcs
        self.eliminator = eliminator or ConstraintEliminator()

    def run(self, system: GlobalSystem) -> ConstrainedSystem:
        T, g = self.eliminator.build(self.mesh, self.bcs)
        return ConstrainedSystem(K=system.K, T=T, g=g)
