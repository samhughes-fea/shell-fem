"""DisassembleGlobalShellField — scatter d_global to element U_e and element residuals."""

from __future__ import annotations

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.containers.formulation_objects import ShellFormulationCache
from blade_structure_utilities.fem_utilities.section_shell_model.processing.static.operations.solve_states import DisassembledShellResults, GlobalShellSystem, ReconstructedShellField


class DisassembleGlobalShellField:
    def run(
        self,
        cache: ShellFormulationCache,
        global_sys: GlobalShellSystem,
        field: ReconstructedShellField,
    ) -> DisassembledShellResults:
        U_e: list[np.ndarray] = []
        residual_e: list[np.ndarray] = []
        norms: list[float] = []
        d = field.d_global
        for stiff, force in zip(cache.stiffness_objects, cache.force_objects):
            g_dofs = stiff.global_dof_indices
            if g_dofs is None:
                continue
            u_elem = np.asarray(d[g_dofs], dtype=float)
            r_elem = np.asarray(stiff.K_e @ u_elem - force.F_e, dtype=float)
            U_e.append(u_elem)
            residual_e.append(r_elem)
            norms.append(float(np.linalg.norm(r_elem)))
        return DisassembledShellResults(
            U_e=U_e,
            element_residual_e=residual_e,
            element_residual_norms=norms,
        )


__all__ = ["DisassembleGlobalShellField"]
