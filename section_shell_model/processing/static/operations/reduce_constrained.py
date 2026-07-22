"""ReduceConstrainedShellSystem — build T and form K_reduced, f_reduced."""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp

from blade_structure_utilities.section_shell_model.processing.static.operations.solve_states import ConstrainedShellSystem, GlobalShellSystem, ReducedShellSystem
from ..results.containers.map_results import ShellMapEntry

KrrModifier = Callable[[sp.csc_matrix, np.ndarray, sp.csr_matrix], tuple[sp.csc_matrix, np.ndarray]]


class ReduceConstrainedShellSystem:
    def run(
        self,
        global_sys: GlobalShellSystem,
        constrained: ConstrainedShellSystem,
        n_gdof: int,
        *,
        k_rr_modifier: KrrModifier | None = None,
        constraint_map: list[ShellMapEntry] | None = None,
    ) -> tuple[ReducedShellSystem, list[ShellMapEntry]]:
        fixed = constrained.fixed
        constraints = constrained.constraints
        constrained_slaves = set(constraints.keys())
        independent = np.array(
            [i for i in range(n_gdof) if i not in fixed and i not in constrained_slaves],
            dtype=int,
        )
        red_pos = {dof: i for i, dof in enumerate(independent.tolist())}
        t_rows: list[int] = []
        t_cols: list[int] = []
        t_vals: list[float] = []
        for dof in independent.tolist():
            t_rows.append(dof)
            t_cols.append(red_pos[dof])
            t_vals.append(1.0)
        for slave, terms in constraints.items():
            if slave in fixed:
                continue
            for master, coeff in terms:
                if master in red_pos:
                    t_rows.append(slave)
                    t_cols.append(red_pos[master])
                    t_vals.append(float(coeff))
        T = sp.coo_matrix((t_vals, (t_rows, t_cols)), shape=(n_gdof, len(independent))).tocsr()
        K_rr = (T.T @ global_sys.K @ T).tocsc()
        f_r = np.asarray(T.T @ global_sys.f, dtype=float)
        if k_rr_modifier is not None:
            K_rr, f_r = k_rr_modifier(K_rr, f_r, T)
        reduced = ReducedShellSystem(T=T, K_reduced=K_rr, f_reduced=f_r, independent=independent)
        reduction_map = self.build_reduction_map(constraint_map, independent, red_pos, n_gdof)
        return reduced, reduction_map

    @staticmethod
    def build_reduction_map(
        constraint_map: list[ShellMapEntry] | None,
        independent: np.ndarray,
        red_pos: dict[int, int],
        n_gdof: int,
    ) -> list[ShellMapEntry]:
        if constraint_map is None:
            constraint_map = [ShellMapEntry(global_dof=d) for d in range(n_gdof)]
        by_dof = {e.global_dof: e for e in constraint_map}
        out: list[ShellMapEntry] = []
        for dof in range(n_gdof):
            base = by_dof.get(dof, ShellMapEntry(global_dof=dof))
            reduced_index = red_pos.get(dof)
            out.append(
                ShellMapEntry(
                    global_dof=base.global_dof,
                    fixed=base.fixed,
                    independent=base.independent,
                    slave=base.slave,
                    masters=list(base.masters),
                    reduced_index=reduced_index,
                    reconstructed_value=base.reconstructed_value,
                )
            )
        return out


__all__ = ["KrrModifier", "ReduceConstrainedShellSystem"]
