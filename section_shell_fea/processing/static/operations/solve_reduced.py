"""SolveReducedShellSystem — factor K_reduced once; solve many RHS."""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.solve_quality import (
    CatastrophicSolveError,
    assess_solve_quality,
    pivot_ratio_from_factor,
    residual_rel_reduced,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.solve_states import (
    ReducedShellSystem,
    SolvedReducedShellSystem,
)


class SolveReducedShellSystem:
    def __init__(self, reduced: ReducedShellSystem, *, factor: spla.SuperLU | None = None) -> None:
        self._reduced = reduced
        self._factor = factor
        self.last_residual_rel: float | None = None
        self.last_pivot_ratio: float | None = None
        self.last_quality_ok: bool = True
        self.last_quality_reason: str = ""

    def prepare(self) -> spla.SuperLU | None:
        if len(self._reduced.independent) == 0:
            self._factor = None
            return None
        if self._factor is None:
            try:
                self._factor = spla.splu(self._reduced.K_reduced)
            except RuntimeError as exc:
                raise CatastrophicSolveError(f"SuperLU factorization failed: {exc}") from exc
        return self._factor

    def solve(self, f_reduced: np.ndarray | None = None) -> SolvedReducedShellSystem:
        f_r = self._reduced.f_reduced if f_reduced is None else np.asarray(f_reduced, dtype=float)
        if len(self._reduced.independent) == 0:
            self.last_residual_rel = 0.0
            self.last_pivot_ratio = None
            self.last_quality_ok = True
            self.last_quality_reason = ""
            return SolvedReducedShellSystem(q=np.array([]), factor=None)
        try:
            factor = self.prepare()
        except CatastrophicSolveError:
            raise
        assert factor is not None
        try:
            q = factor.solve(f_r)
        except Exception as exc:
            raise CatastrophicSolveError(f"SuperLU solve failed: {exc}") from exc
        q_a = np.asarray(q, dtype=float).ravel()
        self.last_residual_rel = residual_rel_reduced(self._reduced.K_reduced, q_a, f_r)
        self.last_pivot_ratio = pivot_ratio_from_factor(factor)
        report = assess_solve_quality(
            residual_rel=self.last_residual_rel,
            pivot_ratio=self.last_pivot_ratio,
            q=q_a,
        )
        self.last_quality_ok = bool(report.ok)
        self.last_quality_reason = str(report.reason)
        if report.catastrophic:
            raise CatastrophicSolveError(report.reason or "catastrophic reduced solve")
        return SolvedReducedShellSystem(q=q_a, factor=factor)


__all__ = ["SolveReducedShellSystem"]
