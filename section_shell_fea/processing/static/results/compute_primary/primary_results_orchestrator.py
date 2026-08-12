"""Primary results orchestrator: reactions, residual, legacy r_full = K d - f, and
element disassembly (U_e / element residuals) — returns a fully populated
``ShellPrimaryResultSet`` (global + elemental), per
PROCESSING_ARCHITECTURE_PROPOSAL.md tier 1/2 contract tightening.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.solve_states import (
    DisassembledShellResults,
    GlobalShellSystem,
    PrimaryShellResults,
    ReconstructedShellField,
)
from ...operations.disassemble_field import DisassembleGlobalShellField
from ..containers.container_hopper import ShellPrimaryResultSet
from ..containers.elemental_results import ShellElementalResults
from ..containers.global_results import ShellGlobalResults
from .compute_reaction_force import ComputeReactionForce
from .compute_residual import ComputeResidual

if TYPE_CHECKING:
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.formulation_objects import ShellFormulationCache


class ShellPrimaryResultsOrchestrator:
    """Reactions/residual + (when a formulation cache is available) element
    disassembly, assembled into a ``ShellPrimaryResultSet``.

    ``cache``/``global_sys``/``field`` are optional: without them (e.g. a bare
    ``from_global`` runner with no per-element formulation objects) the result
    set carries an empty ``ShellElementalResults`` — matching the pre-tightening
    runner behaviour exactly.
    """

    def __init__(
        self,
        *,
        K: sp.spmatrix,
        f: np.ndarray,
        d_global: np.ndarray,
        fixed: set[int],
        cache: "ShellFormulationCache | None" = None,
        global_sys: GlobalShellSystem | None = None,
        field: ReconstructedShellField | None = None,
        T: sp.csr_matrix | None = None,
        K_rr_factor: spla.SuperLU | None = None,
    ) -> None:
        self.K = K
        self.f = f
        self.d_global = d_global
        self.fixed = fixed
        self.cache = cache
        self.global_sys = global_sys
        self.field = field
        self.T = T
        self.K_rr_factor = K_rr_factor
        self.primary: PrimaryShellResults | None = None
        self.disassembled: DisassembledShellResults | None = None

    def compute(self) -> ShellPrimaryResultSet:
        r_full = np.asarray(self.K @ self.d_global - self.f, dtype=float)
        reactions = ComputeReactionForce(
            K=self.K, f=self.f, d_global=self.d_global, fixed=self.fixed,
        ).compute()
        residual = ComputeResidual(
            K=self.K, f=self.f, d_global=self.d_global,
        ).compute()
        self.primary = PrimaryShellResults(reactions=reactions, residual=residual, r_full=r_full)

        if self.cache is not None and self.global_sys is not None and self.field is not None:
            self.disassembled = DisassembleGlobalShellField().run(
                self.cache, self.global_sys, self.field,
            )
            elemental = ShellElementalResults(
                U_e=self.disassembled.U_e,
                element_residual_e=self.disassembled.element_residual_e,
                element_residual_norms=self.disassembled.element_residual_norms,
            )
        else:
            self.disassembled = None
            elemental = ShellElementalResults()

        return ShellPrimaryResultSet(
            global_results=ShellGlobalResults(
                K=self.K,
                f=self.f,
                T=self.T,
                d_global=self.d_global,
                r_full=r_full,
                reactions=reactions,
                residual=residual,
                K_rr_factor=self.K_rr_factor,
            ),
            elemental_results=elemental,
        )


__all__ = ["ShellPrimaryResultsOrchestrator"]
