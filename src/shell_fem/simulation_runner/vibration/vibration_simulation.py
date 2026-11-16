"""Natural frequencies and mode shapes of a shell model.

This is the blade eigenproblem. The frequencies and shapes are what a caller
passes on; nothing here places those modes on a hub or a shaft.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from shell_fem.pre_processing.containers.boundary_conditions import BoundaryConditions
from shell_fem.pre_processing.element_library.added_mass import StripHeaveAddedMass
from shell_fem.processing.eigenpairs import element_mode_energy
from shell_fem.processing.static.operations.assembly import AssembleGlobalSystem
from shell_fem.processing.vibration.added_mass import AssembleAddedMassMatrix
from shell_fem.processing.vibration.operations import (
    AssembleMassMatrix, SolveVibrationEigenproblem, modal_mass_metrics,
)
from shell_fem.simulation_runner.shell_mesh import as_shell_mesh
from shell_fem.processing.stage_log import stage


@dataclass(frozen=True)
class VibrationModes:
    """Blade modes of one shell mesh.

    ``angular_frequencies_rad_s`` and ``frequencies_hz`` ascend together.
    ``modes`` has shape (n_dof, n_modes). Each column is a full displacement
    vector, mass-normalised (φ^T M φ = 1), with constrained degrees of freedom
    at zero. ``generalised_mass`` is φ^T M φ per mode (1 after that normalisation).
    ``participation`` is φ^T M r for the six rigid-body directions about the
    origin (UX, UY, UZ, RX, RY, RZ). ``element_strain_energy`` is 1/2 φ_e^T K_e φ_e
    per shell element per mode, in ``element_ids`` order. The shapes are the shell
    modes. They are not a rotor mode.
    """

    angular_frequencies_rad_s: np.ndarray
    frequencies_hz: np.ndarray
    modes: np.ndarray
    generalised_mass: np.ndarray
    participation: np.ndarray
    element_ids: np.ndarray
    element_strain_energy: np.ndarray

    def __len__(self) -> int:
        return int(self.angular_frequencies_rad_s.size)


def vibration_modes(
    mesh,
    bcs: BoundaryConditions,
    n_modes: int = 6,
    added_mass: Optional[StripHeaveAddedMass] = None,
) -> VibrationModes:
    """Natural frequencies and mass-normalised mode shapes of a shell mesh.

    ``mesh`` is an ``FEAMesh`` or a blade mesh (its ``.mesh`` is used). ``bcs``
    are the homogeneous supports, the same ones a static solve would use. The
    default six modes are the first six of that blade. Pass ``n_modes`` for a
    different count. The result also carries generalised mass, rigid-body mass
    participation and per-element modal strain energy. Rotor assembly is left
    to the caller.

    ``added_mass`` is off by default (structural ``M`` only). Pass a labelled
    ``StripHeaveAddedMass`` (Newman 1977 / Theodorsen 1935 strip-heave) to add
    consistent ``M_a`` on the local-normal translation of every shell.
    """
    with stage("shell_fem", "shell_vibration"):
        return _vibration_modes(mesh, bcs, n_modes, added_mass)


def _vibration_modes(
    mesh,
    bcs: BoundaryConditions,
    n_modes: int,
    added_mass: Optional[StripHeaveAddedMass],
) -> VibrationModes:
    shell = as_shell_mesh(mesh)
    if any(element.laminate.areal_density() <= 0.0 for element in shell.elements):
        raise ValueError("vibration needs a positive laminate density on every element")
    K = AssembleGlobalSystem(shell, bcs).run().K
    mass_stage = AssembleMassMatrix(shell)
    M = mass_stage.run()
    if added_mass is not None:
        M = M + AssembleAddedMassMatrix(shell, added_mass, mass_stage.assembler).run()
    with stage("shell_fem", "eigen_solve") as rec:
        omega, modes = SolveVibrationEigenproblem(shell, bcs, n_modes).run(K, M)
        rec.set("modes", int(np.asarray(omega).size))
    generalised, participation = modal_mass_metrics(shell, modes, M)
    ids, energy = element_mode_energy(shell, modes)
    return VibrationModes(omega, omega / (2.0 * np.pi), modes, generalised, participation, ids, energy)
