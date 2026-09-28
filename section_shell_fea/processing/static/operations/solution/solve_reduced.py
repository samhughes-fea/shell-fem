"""SolveReducedShellSystem — factor K_reduced once; solve many RHS."""

from __future__ import annotations

from typing import Hashable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.solve.solve_quality import (
    CatastrophicSolveError,
    assess_solve_quality,
    pivot_ratio_from_factor,
    residual_rel_reduced,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.contracts.solve_states import (
    ReducedShellSystem,
    SolvedReducedShellSystem,
)


def k_reduced_identity(K_reduced: sp.spmatrix) -> tuple[Hashable, ...]:
    """Exact identity fingerprint for a reduced stiffness matrix.

    Used to guard SuperLU factor reuse: same identity → reuse; changed matrix
    → mandatory refactor (never silently solve a new ``K`` with a stale factor).
    """
    K = K_reduced.tocsc()
    return (
        tuple(K.shape),
        int(K.nnz),
        hash(K.data.tobytes()) if K.nnz else 0,
        hash(K.indices.tobytes()) if K.nnz else 0,
        hash(K.indptr.tobytes()),
    )


class SolveReducedShellSystem:
    """Factor-once reduced solver with fail-closed SuperLU identity ownership.

    An injected ``factor`` must carry ``factor_k_identity`` of the matrix that was
    actually factored. Missing identity is never filled from the *current*
    ``K_reduced`` (that would silently trust a stale factor). Policies:

    * ``strict_factor_identity=True`` (default): raise on missing/mismatched identity.
    * ``strict_factor_identity=False``: discard the factor and refactor.
    """

    def __init__(
        self,
        reduced: ReducedShellSystem,
        *,
        factor: spla.SuperLU | None = None,
        factor_k_identity: tuple[Hashable, ...] | None = None,
        strict_factor_identity: bool = True,
    ) -> None:
        self._reduced = reduced
        self._factor = factor
        self._factor_k_identity = factor_k_identity
        self._factor_was_injected = factor is not None
        self._strict_factor_identity = bool(strict_factor_identity)
        self.last_residual_rel: float | None = None
        self.last_pivot_ratio: float | None = None
        self.last_quality_ok: bool = True
        self.last_quality_reason: str = ""
        if factor is not None and factor_k_identity is None:
            # Never invent an identity for an unverified injected factor.
            if self._strict_factor_identity:
                raise CatastrophicSolveError(
                    "injected SuperLU factor requires factor_k_identity of the "
                    "matrix that was factored (fail-closed)"
                )
            self._factor = None
            self._factor_k_identity = None
            self._factor_was_injected = False

    @property
    def factor_k_identity(self) -> tuple[Hashable, ...] | None:
        return self._factor_k_identity

    def prepare(self) -> spla.SuperLU | None:
        if len(self._reduced.independent) == 0:
            self._factor = None
            self._factor_k_identity = None
            return None
        current_id = k_reduced_identity(self._reduced.K_reduced)
        if self._factor is not None:
            if self._factor_k_identity == current_id:
                return self._factor
            if self._strict_factor_identity and self._factor_was_injected:
                raise CatastrophicSolveError(
                    "stale SuperLU factor does not match current K_reduced identity; "
                    "refactor required (fail-closed)"
                )
            # Non-injected / allow-refactor path: discard and rebuild.
            self._factor = None
            self._factor_k_identity = None
        try:
            self._factor = spla.splu(self._reduced.K_reduced)
        except RuntimeError as exc:
            raise CatastrophicSolveError(f"SuperLU factorization failed: {exc}") from exc
        self._factor_k_identity = current_id
        self._factor_was_injected = False
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

    def solve_many(self, f_reduced_list: list[np.ndarray]) -> list[SolvedReducedShellSystem]:
        """Factor once; solve each reduced RHS with the same SuperLU factor."""
        if not f_reduced_list:
            return []
        self.prepare()
        return [self.solve(f_r) for f_r in f_reduced_list]


__all__ = ["SolveReducedShellSystem", "k_reduced_identity"]
