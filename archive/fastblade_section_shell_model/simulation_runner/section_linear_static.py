"""Section linear static simulation runner (mirror 1d-fem LinearStaticSimulationRunner)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from blade_precompute.section_shell_model.pre_processing.containers.formulation_objects import ShellFormulationCache
from blade_precompute.section_shell_model.processing.operations.solve_states import (
    ConstrainedShellSystem,
    DisassembledShellResults,
    GlobalShellSystem,
    PrimaryShellResults,
    ReconstructedShellField,
    ReducedShellSystem,
    SolvedReducedShellSystem,
)
from ..processing.diagnostics.dof_provenance import dump_index_maps
from ..processing.operations.assemble_global import AssembleGlobalShellSystem
from ..processing.operations.constrain_global import ConstrainGlobalShellSystem
from ..processing.operations.prepare_local import PrepareLocalShellSystem
from ..processing.operations.reconstruct_field import ReconstructGlobalShellField
from ..processing.operations.reduce_constrained import KrrModifier, ReduceConstrainedShellSystem
from ..processing.operations.solve_reduced import SolveReducedShellSystem
from ..processing.results.compute_primary import ShellPrimaryResultsOrchestrator
from ..processing.results.compute_quaternary import ShellQuaternaryResultsOrchestrator
from ..processing.results.compute_secondary import ShellSecondaryResultsOrchestrator
from ..processing.results.compute_tertiary import ShellTertiaryResultsOrchestrator
from ..processing.results.containers.container_hopper import (
    ShellIndexMapSet,
    ShellPrimaryResultSet,
    ShellQuaternaryResultSet,
    ShellSecondaryResultSet,
    ShellTertiaryResultSet,
)
from ..processing.results.containers.map_results import ShellMapEntry


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def section_solve_legacy_dict(result: "SectionLinearStaticResult") -> dict[str, object]:
    """Return a dict mirroring the former ``reduce_propagate_solve`` key set.

    Kept for diagnostics / external callers; prefer typed
    :class:`SectionLinearStaticResult` fields.
    """
    return {
        "d_global": result.d_global,
        "r_full": result.r_full,
        "T": result.T,
        "independent": result.independent,
        "fixed_sorted": result.fixed_sorted,
        "constrained_slaves": result.constrained_slaves,
        "constrained_masters": result.constrained_masters,
        "K_rr_factor": result.K_rr_factor,
    }


@dataclass
class SectionLinearStaticResult:
    """Full stage pipeline output for one linear static solve."""

    d_global: np.ndarray
    r_full: np.ndarray
    T: sp.csr_matrix
    independent: np.ndarray
    fixed_sorted: list[int]
    constrained_slaves: set[int]
    constrained_masters: set[int]
    K_rr_factor: spla.SuperLU | None
    index_maps: ShellIndexMapSet = field(default_factory=ShellIndexMapSet)
    primary_results: ShellPrimaryResultSet | None = None
    secondary_results: ShellSecondaryResultSet | None = None
    tertiary_results: ShellTertiaryResultSet | None = None
    quaternary_results: ShellQuaternaryResultSet | None = None
    cache: ShellFormulationCache | None = None
    global_sys: GlobalShellSystem | None = None
    constrained: ConstrainedShellSystem | None = None
    reduced: ReducedShellSystem | None = None
    solved: SolvedReducedShellSystem | None = None
    field: ReconstructedShellField | None = None
    primary: PrimaryShellResults | None = None
    disassembled: DisassembledShellResults | None = None

    def legacy_dict(self) -> dict[str, object]:
        return section_solve_legacy_dict(self)


class SectionLinearStaticRunner:
    """Orchestrates the section static solve pipeline (1d-fem LinearStaticSimulationRunner analogue)."""

    def __init__(self) -> None:
        self.cache: ShellFormulationCache | None = None
        self.K: sp.spmatrix | None = None
        self.f: np.ndarray | None = None
        self.global_sys: GlobalShellSystem | None = None
        self.constrained: ConstrainedShellSystem | None = None
        self.reduced: ReducedShellSystem | None = None
        self.solved: SolvedReducedShellSystem | None = None
        self.field: ReconstructedShellField | None = None
        self.primary: PrimaryShellResults | None = None
        self.disassembled: DisassembledShellResults | None = None
        self.primary_results: ShellPrimaryResultSet | None = None
        self.secondary_results: ShellSecondaryResultSet | None = None
        self.tertiary_results: ShellTertiaryResultSet | None = None
        self.quaternary_results: ShellQuaternaryResultSet | None = None
        self.index_maps: ShellIndexMapSet = ShellIndexMapSet()
        self.T: sp.csr_matrix | None = None
        self.fixed: set[int] = set()
        self.constraints: dict[int, list[tuple[int, float]]] = {}
        self.n_gdof: int = 0
        self.factor: spla.SuperLU | None = None
        self._solver: SolveReducedShellSystem | None = None
        self._k_rr_modifier: KrrModifier | None = None
        self._last: SectionLinearStaticResult | None = None

    @classmethod
    def from_global(
        cls,
        K_global: sp.spmatrix,
        f_global: np.ndarray,
        constraints: dict[int, list[tuple[int, float]]],
        fixed: set[int],
        n_gdof: int,
        *,
        K_rr_factor: spla.SuperLU | None = None,
    ) -> SectionLinearStaticRunner:
        runner = cls()
        runner.K = K_global
        runner.f = np.asarray(f_global, dtype=float)
        runner.global_sys = GlobalShellSystem(K=K_global, f=runner.f)
        runner.constraints = constraints
        runner.fixed = set(fixed)
        runner.n_gdof = n_gdof
        runner.factor = K_rr_factor
        return runner

    @classmethod
    def from_cache(
        cls,
        cache: ShellFormulationCache,
        n_gdof: int,
        constraints: dict[int, list[tuple[int, float]]],
        fixed: set[int],
    ) -> SectionLinearStaticRunner:
        runner = cls()
        runner.cache = cache
        runner.constraints = constraints
        runner.fixed = set(fixed)
        runner.n_gdof = n_gdof
        return runner

    def prepare_local_system(self) -> ShellFormulationCache:
        if self.cache is None:
            raise RuntimeError("prepare_local_system requires from_cache()")
        self.cache = PrepareLocalShellSystem().run(self.cache)
        return self.cache

    def assemble_global_system(self) -> GlobalShellSystem:
        if self.cache is not None:
            self.global_sys, self.index_maps.assembly_map = AssembleGlobalShellSystem().run(
                self.cache, self.n_gdof,
            )
        elif self.global_sys is not None:
            self.index_maps.assembly_map = [
                ShellMapEntry(global_dof=d, independent=True) for d in range(self.n_gdof)
            ]
        else:
            raise RuntimeError("assemble_global_system requires cache or pre-set global_sys")
        return self.global_sys

    def constrain_global_system(self) -> ConstrainedShellSystem:
        self.constrained, self.index_maps.constraint_map = ConstrainGlobalShellSystem().run(
            self.constraints, self.fixed, self.n_gdof,
        )
        self.fixed = set(self.constrained.fixed)
        return self.constrained

    def reduce_constrained_system(self) -> ReducedShellSystem:
        if self.global_sys is None or self.constrained is None:
            raise RuntimeError("reduce_constrained_system requires assembled + constrained system")
        self.reduced, self.index_maps.reduction_map = ReduceConstrainedShellSystem().run(
            self.global_sys,
            self.constrained,
            self.n_gdof,
            k_rr_modifier=self._k_rr_modifier,
            constraint_map=self.index_maps.constraint_map,
        )
        self.T = self.reduced.T
        return self.reduced

    def solve_reduced_system(self, *, f_reduced: np.ndarray | None = None) -> SolvedReducedShellSystem:
        if self.reduced is None:
            raise RuntimeError("solve_reduced_system requires reduced system")
        self._solver = SolveReducedShellSystem(self.reduced, factor=self.factor)
        self.solved = self._solver.solve(f_reduced)
        self.factor = self.solved.factor
        return self.solved

    def reconstruct_global_field(self) -> ReconstructedShellField:
        if self.reduced is None or self.solved is None or self.constrained is None:
            raise RuntimeError("reconstruct_global_field requires reduced + solved + constrained")
        self.field, self.index_maps.reconstruction_map = ReconstructGlobalShellField().run(
            self.reduced,
            self.solved,
            self.constrained.fixed,
            self.n_gdof,
            reduction_map=self.index_maps.reduction_map,
        )
        return self.field

    def compute_primary_results(self) -> ShellPrimaryResultSet:
        if self.global_sys is None or self.field is None or self.constrained is None:
            raise RuntimeError("compute_primary_results requires global_sys + field + constrained")
        orchestrator = ShellPrimaryResultsOrchestrator(
            K=self.global_sys.K,
            f=self.global_sys.f,
            d_global=self.field.d_global,
            fixed=self.constrained.fixed,
            cache=self.cache,
            global_sys=self.global_sys,
            field=self.field,
            T=self.reduced.T if self.reduced else None,
            K_rr_factor=self.factor,
        )
        self.primary_results = orchestrator.compute()
        self.primary = orchestrator.primary
        self.disassembled = orchestrator.disassembled
        return self.primary_results

    def compute_secondary_results(
        self,
        *,
        L_x: float = 1.0,
        theory: str | None = None,
        panel_meta: list[dict[str, Any]] | None = None,
        panel_maps: list[Any] | None = None,
        panels: list[Any] | None = None,
        provenance_tag: str = "processing.compute_secondary",
        compute_gaussian: bool = False,
        compute_nodal: bool = False,
    ) -> ShellSecondaryResultSet:
        """Tier 2: shell resultants per element from tier 1's disassembled U_e.

        Opt-in — not part of the default ``solve()`` flow (matches today's
        opt-in status of secondary/tertiary in the RESULTS_DESIGN wired-computers
        table); callers needing resultants call this explicitly after ``solve()``.

        ``compute_gaussian`` / ``compute_nodal`` (R10) populate secondary
        Gaussian / nodal resolution containers; both default off.
        """
        if self.primary_results is None or self.cache is None:
            raise RuntimeError("compute_secondary_results requires primary_results + cache")
        self.secondary_results = ShellSecondaryResultsOrchestrator(
            primary=self.primary_results,
            cache=self.cache,
            L_x=L_x,
            theory=theory,
            panel_meta=panel_meta,
            panel_maps=panel_maps,
            panels=panels,
            provenance_tag=provenance_tag,
            compute_gaussian=compute_gaussian,
            compute_nodal=compute_nodal,
        ).compute()
        return self.secondary_results

    def compute_tertiary_results(self) -> ShellTertiaryResultSet:
        """Tier 3: CLT ply-stress recovery from tier 2 resultants + cache.panel_laminates."""
        if self.secondary_results is None or self.cache is None:
            raise RuntimeError("compute_tertiary_results requires secondary_results + cache")
        self.tertiary_results = ShellTertiaryResultsOrchestrator(
            secondary=self.secondary_results, cache=self.cache,
        ).compute()
        return self.tertiary_results

    def compute_quaternary_results(self) -> ShellQuaternaryResultSet:
        """Tier 4: failure indices from tier 3 ply stresses + cache.panel_laminates strengths."""
        if self.tertiary_results is None or self.cache is None:
            raise RuntimeError("compute_quaternary_results requires tertiary_results + cache")
        self.quaternary_results = ShellQuaternaryResultsOrchestrator(
            tertiary=self.tertiary_results, cache=self.cache,
        ).compute()
        return self.quaternary_results

    def solve_linear_system_only(
        self,
        *,
        f_override: np.ndarray | None = None,
        k_rr_modifier: KrrModifier | None = None,
    ) -> np.ndarray:
        """Top-level solve flow (1d-fem solve_linear_system_only analogue)."""
        self._k_rr_modifier = k_rr_modifier
        if self.cache is not None:
            self.prepare_local_system()
            self.assemble_global_system()
        elif self.global_sys is None and self.K is not None and self.f is not None:
            f_vec = self.f if f_override is None else np.asarray(f_override, dtype=float)
            self.global_sys = GlobalShellSystem(K=self.K, f=f_vec)
        elif f_override is not None and self.global_sys is not None:
            self.global_sys = GlobalShellSystem(
                K=self.global_sys.K,
                f=np.asarray(f_override, dtype=float),
                diagnostics=self.global_sys.diagnostics,
            )
        self.constrain_global_system()
        self.reduce_constrained_system()
        self.solve_reduced_system()
        self.reconstruct_global_field()
        assert self.field is not None
        return self.field.d_global

    def solve(
        self,
        *,
        f_override: np.ndarray | None = None,
        k_rr_modifier: KrrModifier | None = None,
    ) -> SectionLinearStaticResult:
        self.solve_linear_system_only(f_override=f_override, k_rr_modifier=k_rr_modifier)
        self.compute_primary_results()
        assert self.primary is not None
        assert self.reduced is not None
        assert self.field is not None
        constrained_slaves = set(self.constraints.keys())
        constrained_masters = {m for terms in self.constraints.values() for m, _ in terms}
        if _env_truthy("SHELL_DOF_PROVENANCE_DUMP"):
            self._dump_index_maps()
        result = SectionLinearStaticResult(
            d_global=np.asarray(self.field.d_global, dtype=float),
            r_full=np.asarray(self.primary.r_full, dtype=float),
            T=self.reduced.T,
            independent=self.reduced.independent,
            fixed_sorted=sorted(self.constrained.fixed if self.constrained else self.fixed),
            constrained_slaves=constrained_slaves,
            constrained_masters=constrained_masters,
            K_rr_factor=self.factor,
            index_maps=self.index_maps,
            primary_results=self.primary_results,
            secondary_results=self.secondary_results,
            tertiary_results=self.tertiary_results,
            quaternary_results=self.quaternary_results,
            cache=self.cache,
            global_sys=self.global_sys,
            constrained=self.constrained,
            reduced=self.reduced,
            solved=self.solved,
            field=self.field,
            primary=self.primary,
            disassembled=self.disassembled,
        )
        self._last = result
        self.fixed.clear()
        self.fixed.update(result.fixed_sorted)
        return result

    def solve_many(
        self,
        f_vectors: list[np.ndarray],
        *,
        k_rr_modifier: KrrModifier | None = None,
    ) -> list[np.ndarray]:
        """Factor once; solve multiple RHS (unit-load superposition)."""
        first = self.solve(f_override=f_vectors[0] if f_vectors else None, k_rr_modifier=k_rr_modifier)
        out = [first.d_global]
        if self.factor is None or self.reduced is None or self._solver is None:
            return [np.zeros(self.n_gdof)] * len(f_vectors)
        for f_vec in f_vectors[1:]:
            if self.global_sys is None:
                break
            f_r = np.asarray(self.reduced.T.T @ f_vec, dtype=float)
            solved = self._solver.solve(f_r)
            d = np.asarray(self.reduced.T @ solved.q, dtype=float)
            out.append(d)
        return out

    def _dump_index_maps(self) -> None:
        """Opt-in DOF provenance dump; see :mod:`..processing.diagnostics.dof_provenance`."""
        dump_index_maps(self.index_maps)


__all__ = [
    "SectionLinearStaticResult",
    "SectionLinearStaticRunner",
    "section_solve_legacy_dict",
]
