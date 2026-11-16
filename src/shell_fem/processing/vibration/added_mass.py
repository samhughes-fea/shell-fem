"""Assemble labelled strip-heave added mass on shells.

Default vibration leaves this stage out. Pass a ``StripHeaveAddedMass`` to
``vibration_modes`` to add ``M_a`` after structural ``M``. Layered solids
contribute zeros: strip theory is a surface (shell) model.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from shell_fem.pre_processing.containers.mesh import FEAMesh
from shell_fem.pre_processing.element_library.added_mass import StripHeaveAddedMass, element_added_mass_matrix
from shell_fem.pre_processing.element_library.shell import ShellElement
from shell_fem.processing.static.operations.assembly import GlobalAssembler


class AssembleAddedMassMatrix:
    """Stage: consistent ``M_a`` of every shell from the labelled strip-heave model."""

    def __init__(
        self,
        mesh: FEAMesh,
        model: StripHeaveAddedMass,
        assembler: Optional[GlobalAssembler] = None,
    ):
        if not isinstance(model, StripHeaveAddedMass):
            raise TypeError(
                f"added_mass must be StripHeaveAddedMass (labelled strip_heave); got {type(model)!r}"
            )
        self.mesh = mesh
        self.model = model
        self.assembler = assembler or GlobalAssembler()

    def run(self):
        def matrix(element):
            if isinstance(element, ShellElement):
                return element_added_mass_matrix(element, self.model)
            return np.zeros((element.n_dof, element.n_dof))

        return self.assembler.scatter_element_matrices(self.mesh, matrix)
