"""Functional constrained-solve kernel (internal; not a production lifecycle).

Accepts a frozen ``GlobalShellSystem`` plus ``SectionConstraintSet`` and returns
the constrained/reduced systems, ``T``, solution, and reconstructed field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.types import (
    SectionConstraintSet,
    constraint_equations_to_solver_dict,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.reduction.constrain_global import (
    ConstrainGlobalShellSystem,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.recovery.reconstruct_field import (
    ReconstructGlobalShellField,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.reduction.reduce_constrained import (
    ReduceConstrainedShellSystem,
    ReductionOperator,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.solution.solve_reduced import (
    SolveReducedShellSystem,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.contracts.solve_states import (
    ConstrainedShellSystem,
    GlobalShellSystem,
    ReconstructedShellField,
    ReducedShellSystem,
    SolvedReducedShellSystem,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.containers.map_results import (
    ShellMapEntry,
)

KrrModifier = Callable[[sp.csc_matrix, np.ndarray, sp.csr_matrix], tuple[sp.csc_matrix, np.ndarray]]


@dataclass
class ConstrainedSolveResult:
    constrained: ConstrainedShellSystem
    reduced: ReducedShellSystem
    solved: SolvedReducedShellSystem
    field: ReconstructedShellField
    constraint_map: list[ShellMapEntry]
    reduction_map: list[ShellMapEntry]
    reconstruction_map: list[ShellMapEntry]
    T: sp.csr_matrix
    factor: spla.SuperLU | None
    d_global: NDArray[np.float64]
    residual_rel: float | None = None
    pivot_ratio: float | None = None
    quality_ok: bool = True
    quality_reason: str = ""
    operator: ReductionOperator | None = None


def solve_constrained_system(
    global_sys: GlobalShellSystem,
    constraint_set: SectionConstraintSet,
    n_gdof: int,
    *,
    k_rr_modifier: KrrModifier | None = None,
    factor: spla.SuperLU | None = None,
    operator: ReductionOperator | None = None,
    collect_maps: bool = True,
    bc_mode: str = "",
    cap_tie_mode: str = "",
    interface_constraint_mode: str = "",
) -> ConstrainedSolveResult:
    """Constrain → reduce → solve → reconstruct. No station policy inference."""
    constraints = constraint_equations_to_solver_dict(constraint_set.equations)
    fixed = set(constraint_set.fixed_dofs)
    constrained, constraint_map = ConstrainGlobalShellSystem().run(
        constraints, fixed, int(n_gdof), collect_maps=collect_maps
    )
    reducer = ReduceConstrainedShellSystem()
    reduced, reduction_map = reducer.run(
        global_sys,
        constrained,
        int(n_gdof),
        k_rr_modifier=k_rr_modifier,
        constraint_map=constraint_map or None,
        operator=operator,
        bc_mode=bc_mode,
        cap_tie_mode=cap_tie_mode,
        interface_constraint_mode=interface_constraint_mode,
        collect_maps=collect_maps,
    )
    # New K_rr from this reduction: only reuse factor when identity still matches.
    solver = SolveReducedShellSystem(reduced, factor=factor, strict_factor_identity=False)
    solved = solver.solve()
    field, reconstruction_map = ReconstructGlobalShellField().run(
        reduced,
        solved,
        set(constrained.fixed),
        int(n_gdof),
        reduction_map=reduction_map or None,
        collect_maps=collect_maps,
    )
    return ConstrainedSolveResult(
        constrained=constrained,
        reduced=reduced,
        solved=solved,
        field=field,
        constraint_map=constraint_map,
        reduction_map=reduction_map,
        reconstruction_map=reconstruction_map,
        T=reduced.T,
        factor=solved.factor,
        d_global=np.asarray(field.d_global, dtype=float),
        residual_rel=solver.last_residual_rel,
        pivot_ratio=solver.last_pivot_ratio,
        quality_ok=bool(solver.last_quality_ok),
        quality_reason=str(solver.last_quality_reason or ""),
        operator=reducer.last_operator,
    )


__all__ = ["ConstrainedSolveResult", "solve_constrained_system"]
