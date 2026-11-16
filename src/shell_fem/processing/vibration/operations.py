"""Vibration stages: mass matrix, then the smallest natural frequencies."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from shell_fem.pre_processing.containers.boundary_conditions import BoundaryConditions
from shell_fem.pre_processing.containers.mesh import FEAMesh
from shell_fem.processing.eigenpairs import expand_modes, reduce_matrices, vibration_eigenpairs
from shell_fem.processing.static.operations.assembly import GlobalAssembler


def rigid_body_displacements(mesh: FEAMesh) -> np.ndarray:
    """(n_dof, 6) rigid translations and rotations about the origin.

    Columns are UX, UY, UZ, RX, RY, RZ. Enrichment amplitudes stay at zero.
    """
    r = np.zeros((mesh.n_dof, 6))
    idx = mesh.standard_dof_indices()
    xyz = mesh.coordinates_array()
    r[idx[:, 0], 0] = 1.0
    r[idx[:, 1], 1] = 1.0
    r[idx[:, 2], 2] = 1.0
    r[idx[:, 1], 3] = -xyz[:, 2]
    r[idx[:, 2], 3] = xyz[:, 1]
    r[idx[:, 3], 3] = 1.0
    r[idx[:, 0], 4] = xyz[:, 2]
    r[idx[:, 2], 4] = -xyz[:, 0]
    r[idx[:, 4], 4] = 1.0
    r[idx[:, 0], 5] = -xyz[:, 1]
    r[idx[:, 1], 5] = xyz[:, 0]
    r[idx[:, 5], 5] = 1.0
    return r


def modal_mass_metrics(mesh: FEAMesh, modes: np.ndarray, mass) -> Tuple[np.ndarray, np.ndarray]:
    """Generalised mass φ^T M φ and mass participation φ^T M r for each mode.

    ``participation`` has shape (n_modes, 6) in the order UX, UY, UZ, RX, RY, RZ
    about the origin. Modes that are mass-normalised have generalised mass 1.
    """
    modes = np.asarray(modes, dtype=float)
    generalised = np.einsum("ij,ij->j", modes, mass @ modes)
    participation = modes.T @ (mass @ rigid_body_displacements(mesh))
    return generalised, participation


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
