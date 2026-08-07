"""Abstract shell element formulation interface (mirror 1d-fem Element1DBase)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.containers.formulation_objects import (
    ShellForceObject,
    ShellStiffnessObject,
)


class ShellElementBase(ABC):
    """Live element: geometry/material/load slices with formulation methods."""

    element_id: int
    dof_per_node: int
    panel_index: int
    local_element_index: int

    @abstractmethod
    def element_stiffness_matrix(self) -> ShellStiffnessObject:
        """Integrate B/D at Gauss points and return Ke plus optional GP cache."""

    @abstractmethod
    def element_force_vector(self) -> ShellForceObject:
        """Equivalent nodal loads (membrane, Donnell, traction, …)."""

    def assemble_global_dof_indices(
        self,
        global_node_ids: NDArray[np.int64] | list[int],
    ) -> NDArray[np.int64]:
        """Map local element DOFs to global indices (4 nodes × dof_per_node)."""
        n = self.dof_per_node
        gids = np.asarray(global_node_ids, dtype=np.int64)
        g_dofs = np.empty(4 * n, dtype=np.int64)
        for i, gn in enumerate(gids):
            base = i * n
            g_dofs[base : base + n] = gn * n + np.arange(n, dtype=np.int64)
        return g_dofs


__all__ = ["ShellElementBase"]
