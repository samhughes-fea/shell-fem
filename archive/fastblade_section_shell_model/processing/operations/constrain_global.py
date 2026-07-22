"""ConstrainGlobalShellSystem — propagate fixity and build DOF provenance."""

from __future__ import annotations

from blade_precompute.section_shell_model.processing.operations.solve_states import ConstrainedShellSystem
from ..results.containers.map_results import ShellMapEntry


class ConstrainGlobalShellSystem:
    def run(
        self,
        constraints: dict[int, list[tuple[int, float]]],
        fixed: set[int],
        n_gdof: int,
    ) -> tuple[ConstrainedShellSystem, list[ShellMapEntry]]:
        propagated = True
        fixed = set(fixed)
        while propagated:
            propagated = False
            for slave, terms in constraints.items():
                if slave in fixed:
                    continue
                if terms and all(m in fixed for m, _ in terms):
                    fixed.add(slave)
                    propagated = True
        constraint_map = self.build_constraint_map(constraints, fixed, n_gdof)
        constrained = ConstrainedShellSystem(
            constraints=constraints,
            fixed=fixed,
            dof_provenance=[],
        )
        return constrained, constraint_map

    @staticmethod
    def build_constraint_map(
        constraints: dict[int, list[tuple[int, float]]],
        fixed: set[int],
        n_gdof: int,
    ) -> list[ShellMapEntry]:
        slaves = set(constraints.keys())
        entries: list[ShellMapEntry] = []
        for dof in range(n_gdof):
            if dof in fixed:
                entries.append(ShellMapEntry(global_dof=dof, fixed=True))
            elif dof in slaves:
                entries.append(
                    ShellMapEntry(
                        global_dof=dof,
                        slave=True,
                        masters=list(constraints[dof]),
                    )
                )
            else:
                entries.append(ShellMapEntry(global_dof=dof, independent=True))
        return entries


__all__ = ["ConstrainGlobalShellSystem"]
