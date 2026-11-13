"""Natural frequencies and mode shapes of a shell model.

This is the blade eigenproblem. The frequencies and shapes are what a caller
passes on; nothing here places those modes on a hub or a shaft.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shell_fem.pre_processing.containers.boundary_conditions import BoundaryConditions
from shell_fem.processing.static.operations.assembly import AssembleGlobalSystem
from shell_fem.processing.vibration.operations import AssembleMassMatrix, SolveVibrationEigenproblem
from shell_fem.simulation_runner.shell_mesh import as_shell_mesh


@dataclass(frozen=True)
class VibrationModes:
    """Blade modes of one shell mesh.

    ``angular_frequencies_rad_s`` and ``frequencies_hz`` ascend together.
    ``modes`` has shape (n_dof, n_modes). Each column is a full displacement
    vector, mass-normalised (φ^T M φ = 1), with constrained degrees of freedom
    at zero. The shapes are the shell modes. They are not a rotor mode.
    """

    angular_frequencies_rad_s: np.ndarray
    frequencies_hz: np.ndarray
    modes: np.ndarray

    def __len__(self) -> int:
        return int(self.angular_frequencies_rad_s.size)


def vibration_modes(mesh, bcs: BoundaryConditions, n_modes: int = 6) -> VibrationModes:
    """Natural frequencies and mass-normalised mode shapes of a shell mesh.

    ``mesh`` is an ``FEAMesh`` or a blade mesh (its ``.mesh`` is used). ``bcs``
    are the homogeneous supports, the same ones a static solve would use. The
    default six modes are the first six of that blade. Pass ``n_modes`` for a
    different count. Rotor assembly is left to the caller.
    """
    shell = as_shell_mesh(mesh)
    if any(element.laminate.areal_density() <= 0.0 for element in shell.shell_elements):
        raise ValueError("vibration needs a positive laminate density on every shell element")
    K = AssembleGlobalSystem(shell, bcs).run().K
    M = AssembleMassMatrix(shell).run()
    omega, modes = SolveVibrationEigenproblem(shell, bcs, n_modes).run(K, M)
    return VibrationModes(omega, omega / (2.0 * np.pi), modes)
