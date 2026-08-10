"""SolveReducedShellSystem — factor K_reduced once; solve many RHS."""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla

from blade_structure_utilities.fem_utilities.section_shell_model.processing.static.operations.solve_states import ReducedShellSystem, SolvedReducedShellSystem


class SolveReducedShellSystem:
    def __init__(self, reduced: ReducedShellSystem, *, factor: spla.SuperLU | None = None) -> None:
        self._reduced = reduced
        self._factor = factor

    def prepare(self) -> spla.SuperLU | None:
        if len(self._reduced.independent) == 0:
            self._factor = None
            return None
        if self._factor is None:
            self._factor = spla.splu(self._reduced.K_reduced)
        return self._factor

    def solve(self, f_reduced: np.ndarray | None = None) -> SolvedReducedShellSystem:
        f_r = self._reduced.f_reduced if f_reduced is None else np.asarray(f_reduced, dtype=float)
        if len(self._reduced.independent) == 0:
            return SolvedReducedShellSystem(q=np.array([]), factor=None)
        factor = self.prepare()
        assert factor is not None
        q = factor.solve(f_r)
        return SolvedReducedShellSystem(q=q, factor=factor)


__all__ = ["SolveReducedShellSystem"]
