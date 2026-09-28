"""ReduceConstrainedShellSystem — static ``ReductionOperator`` + per-trial apply."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable

import numpy as np
import scipy.sparse as sp

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.contracts.solve_states import (
    ConstrainedShellSystem,
    GlobalShellSystem,
    ReducedShellSystem,
)
from ...results.containers.map_results import ShellMapEntry

KrrModifier = Callable[[sp.csc_matrix, np.ndarray, sp.csr_matrix], tuple[sp.csc_matrix, np.ndarray]]


def _normalized_mpc_fingerprint(
    constraints: dict[int, list[tuple[int, float]]],
    fixed: set[int],
) -> tuple[tuple[int, tuple[tuple[int, float], ...]], ...]:
    """Immutable, order-normalized MPC fingerprint (slaves in fixed skipped)."""
    items: list[tuple[int, tuple[tuple[int, float], ...]]] = []
    for slave in sorted(constraints):
        if slave in fixed:
            continue
        terms = tuple(sorted((int(m), float(c)) for m, c in constraints[slave]))
        items.append((int(slave), terms))
    return tuple(items)


@dataclass(frozen=True)
class ReductionIdentity:
    """Immutable identity for safe ``ReductionOperator`` reuse across trials.

    Includes global DOF count, fixed DOFs, normalized MPC equations, and optional
    mode tags that change reduction semantics (BC / cap-tie / interface / modifier).
    """

    n_gdof: int
    fixed: frozenset[int]
    mpc: tuple[tuple[int, tuple[tuple[int, float], ...]], ...]
    bc_mode: str = ""
    cap_tie_mode: str = ""
    interface_constraint_mode: str = ""
    modifier_key: Hashable = None

    @classmethod
    def from_constrained(
        cls,
        constrained: ConstrainedShellSystem,
        n_gdof: int,
        *,
        bc_mode: str = "",
        cap_tie_mode: str = "",
        interface_constraint_mode: str = "",
        modifier_key: Hashable = None,
    ) -> ReductionIdentity:
        fixed = frozenset(int(d) for d in constrained.fixed)
        return cls(
            n_gdof=int(n_gdof),
            fixed=fixed,
            mpc=_normalized_mpc_fingerprint(constrained.constraints, set(fixed)),
            bc_mode=str(bc_mode or ""),
            cap_tie_mode=str(cap_tie_mode or ""),
            interface_constraint_mode=str(interface_constraint_mode or ""),
            modifier_key=modifier_key,
        )


@dataclass
class ReductionOperator:
    """Static reduction operator: ``independent``, ``red_pos``, sparse ``T``.

    Build once from constraints + DOF ordering; apply per trial to form
    ``K_rr = Tᵀ K T`` and ``f_r = Tᵀ f``. Never reuse ``K_rr`` across different
    global ``K`` merely because ``T`` matches.
    """

    identity: ReductionIdentity
    independent: np.ndarray
    red_pos: dict[int, int]
    T: sp.csr_matrix

    @classmethod
    def build(
        cls,
        constrained: ConstrainedShellSystem,
        n_gdof: int,
        *,
        bc_mode: str = "",
        cap_tie_mode: str = "",
        interface_constraint_mode: str = "",
        modifier_key: Hashable = None,
    ) -> ReductionOperator:
        identity = ReductionIdentity.from_constrained(
            constrained,
            n_gdof,
            bc_mode=bc_mode,
            cap_tie_mode=cap_tie_mode,
            interface_constraint_mode=interface_constraint_mode,
            modifier_key=modifier_key,
        )
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
        return cls(identity=identity, independent=independent, red_pos=red_pos, T=T)

    def apply(
        self,
        global_sys: GlobalShellSystem,
        *,
        k_rr_modifier: KrrModifier | None = None,
    ) -> ReducedShellSystem:
        K_rr = (self.T.T @ global_sys.K @ self.T).tocsc()
        f_r = np.asarray(self.T.T @ global_sys.f, dtype=float)
        if k_rr_modifier is not None:
            K_rr, f_r = k_rr_modifier(K_rr, f_r, self.T)
        return ReducedShellSystem(
            T=self.T,
            K_reduced=K_rr,
            f_reduced=f_r,
            independent=self.independent,
        )


class ReduceConstrainedShellSystem:
    def __init__(self) -> None:
        self.last_operator: ReductionOperator | None = None

    def run(
        self,
        global_sys: GlobalShellSystem,
        constrained: ConstrainedShellSystem,
        n_gdof: int,
        *,
        k_rr_modifier: KrrModifier | None = None,
        constraint_map: list[ShellMapEntry] | None = None,
        operator: ReductionOperator | None = None,
        bc_mode: str = "",
        cap_tie_mode: str = "",
        interface_constraint_mode: str = "",
        modifier_key: Hashable = None,
        collect_maps: bool = True,
    ) -> tuple[ReducedShellSystem, list[ShellMapEntry]]:
        if operator is None:
            operator = ReductionOperator.build(
                constrained,
                n_gdof,
                bc_mode=bc_mode,
                cap_tie_mode=cap_tie_mode,
                interface_constraint_mode=interface_constraint_mode,
                modifier_key=modifier_key,
            )
        else:
            expected = ReductionIdentity.from_constrained(
                constrained,
                n_gdof,
                bc_mode=bc_mode,
                cap_tie_mode=cap_tie_mode,
                interface_constraint_mode=interface_constraint_mode,
                modifier_key=modifier_key,
            )
            if operator.identity != expected:
                raise ValueError(
                    "ReductionOperator identity mismatch; rebuild required "
                    f"(cached={operator.identity!r}, expected={expected!r})"
                )
        reduced = operator.apply(global_sys, k_rr_modifier=k_rr_modifier)
        self.last_operator = operator
        if not collect_maps:
            return reduced, []
        reduction_map = self.build_reduction_map(
            constraint_map, operator.independent, operator.red_pos, n_gdof
        )
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


__all__ = [
    "KrrModifier",
    "ReduceConstrainedShellSystem",
    "ReductionIdentity",
    "ReductionOperator",
]
