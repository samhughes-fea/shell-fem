"""PrepareLocalShellSystem — validate element Ke/Fe before assembly."""

from __future__ import annotations

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.containers.formulation_objects import ShellFormulationCache


class PrepareLocalShellSystem:
    def run(self, cache: ShellFormulationCache) -> ShellFormulationCache:
        n = cache.n_dof_node
        for stiff, force in zip(cache.stiffness_objects, cache.force_objects):
            if stiff.global_dof_indices is None:
                raise ValueError(f"element {stiff.element_id}: global_dof_indices not set")
            ke, fe = stiff.K_e, force.F_e
            if ke.ndim != 2 or ke.shape[0] != ke.shape[1]:
                raise ValueError(f"element {stiff.element_id}: invalid Ke shape {ke.shape}")
            if fe.shape != (4 * n,):
                raise ValueError(f"element {stiff.element_id}: invalid fe shape {fe.shape}")
            if not (np.all(np.isfinite(ke)) and np.all(np.isfinite(fe))):
                raise ValueError(f"element {stiff.element_id}: non-finite Ke or fe")
        return cache


__all__ = ["PrepareLocalShellSystem"]
