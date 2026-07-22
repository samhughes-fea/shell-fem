"""ReconstructGlobalShellField — expand reduced solution to full DOF vector."""

from __future__ import annotations

import numpy as np

from blade_precompute.section_shell_model.processing.operations.solve_states import ReconstructedShellField, ReducedShellSystem, SolvedReducedShellSystem
from ..results.containers.map_results import ShellMapEntry


class ReconstructGlobalShellField:
    def run(
        self,
        reduced: ReducedShellSystem,
        solved: SolvedReducedShellSystem,
        fixed: set[int],
        n_gdof: int,
        *,
        reduction_map: list[ShellMapEntry] | None = None,
    ) -> tuple[ReconstructedShellField, list[ShellMapEntry]]:
        d_global = np.zeros(n_gdof, dtype=float)
        if len(reduced.independent) > 0:
            d_global = np.asarray(reduced.T @ solved.q, dtype=float)
        nan_detected = bool(np.any(~np.isfinite(d_global)))
        fixed_leak = any(abs(d_global[d]) > 1e-9 for d in fixed if d < n_gdof)
        field = ReconstructedShellField(
            d_global=d_global,
            nan_detected=nan_detected,
            fixed_dof_leak=fixed_leak,
        )
        reconstruction_map = self.build_reconstruction_map(reduction_map, d_global, n_gdof)
        return field, reconstruction_map

    @staticmethod
    def build_reconstruction_map(
        reduction_map: list[ShellMapEntry] | None,
        d_global: np.ndarray,
        n_gdof: int,
    ) -> list[ShellMapEntry]:
        if reduction_map is None:
            reduction_map = [ShellMapEntry(global_dof=d) for d in range(n_gdof)]
        by_dof = {e.global_dof: e for e in reduction_map}
        out: list[ShellMapEntry] = []
        for dof in range(n_gdof):
            base = by_dof.get(dof, ShellMapEntry(global_dof=dof))
            out.append(
                ShellMapEntry(
                    global_dof=base.global_dof,
                    fixed=base.fixed,
                    independent=base.independent,
                    slave=base.slave,
                    masters=list(base.masters),
                    reduced_index=base.reduced_index,
                    reconstructed_value=float(d_global[dof]) if dof < d_global.size else 0.0,
                )
            )
        return out


__all__ = ["ReconstructGlobalShellField"]
