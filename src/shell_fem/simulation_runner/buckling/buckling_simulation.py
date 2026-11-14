"""Linear buckling of a shell mesh under one reference load.

The static solve supplies the membrane forces. The load factors multiply that
reference load. The solve does not rebuild the mesh and does not assemble a rotor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shell_fem.pre_processing.containers.boundary_conditions import BoundaryConditions
from shell_fem.pre_processing.load_library.load_case import LoadCase
from shell_fem.processing.buckling.operations import AssembleGeometricStiffness, SolveBucklingEigenproblem
from shell_fem.processing.static.operations.assembly import AssembleGlobalSystem
from shell_fem.simulation_runner.shell_mesh import as_shell_mesh
from shell_fem.simulation_runner.static.linear_static_simulation import StructuralGlobalSolver
from shell_fem.stage_log import stage


@dataclass(frozen=True)
class BucklingResult:
    """Linear buckling of one reference load.

    ``load_factors`` are the positive multipliers of ``load_case``, ascending,
    so the critical load is the first factor times the reference load.
    ``modes`` has shape (n_dof, n_modes). Each column is a full displacement
    vector scaled so its largest absolute component is 1.
    """

    load_factors: np.ndarray
    modes: np.ndarray
    load_case: str

    def __len__(self) -> int:
        return int(self.load_factors.size)


def linear_buckling(mesh, bcs: BoundaryConditions, load_case: LoadCase, n_modes: int = 1) -> BucklingResult:
    """Buckling load factors of a shell mesh under ``load_case``.

    ``mesh`` is an ``FEAMesh`` or a blade mesh (its ``.mesh`` is used). The
    reference load is solved linearly, its membrane forces form the geometric
    stiffness, and (K + λ Kg) φ = 0 supplies the factors. ``n_modes`` is how
    many positive factors to return, lowest first.
    """
    with stage("shell_fem", "shell_buckling"):
        return _linear_buckling(mesh, bcs, load_case, n_modes)


def _linear_buckling(mesh, bcs: BoundaryConditions, load_case: LoadCase, n_modes: int) -> BucklingResult:
    shell = as_shell_mesh(mesh)
    solution = StructuralGlobalSolver().solve(shell, bcs, load_case)
    K = AssembleGlobalSystem(shell, bcs).run().K
    Kg = AssembleGeometricStiffness(shell, solution).run()
    factors, modes = SolveBucklingEigenproblem(shell, bcs, n_modes).run(K, Kg)
    return BucklingResult(factors, modes, load_case.name)
