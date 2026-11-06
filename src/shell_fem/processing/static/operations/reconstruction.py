"""Reconstruction stage: global displacements U = T q + g."""

from __future__ import annotations

import numpy as np

from shell_fem.processing.static.operations.solve_states import ReducedSystem


class ReconstructGlobalField:
    """Stage: expand the reduced solution back to every global DOF."""

    def run(self, system: ReducedSystem, q: np.ndarray) -> np.ndarray:
        c = system.constrained
        return np.asarray(c.T @ q).ravel() + c.g
