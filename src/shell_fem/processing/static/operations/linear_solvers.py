"""Linear algebra strategies for the reduced (constrained) stiffness system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

try:  # SciPy is optional: sparse assembly and factorisation when available.
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spla

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover - exercised only without SciPy
    _sp = None
    _spla = None
    HAVE_SCIPY = False


class SingularStiffnessError(RuntimeError):
    """Raised when the constrained stiffness matrix cannot be factorised."""


class LinearSystemSolver(ABC):
    """Strategy for solving the constrained symmetric positive definite system.

    ``pivot_tolerance`` is the smallest admissible ratio between the smallest and the
    largest factorisation pivot; a mechanism (unrestrained rigid-body mode) drives
    this ratio to round-off level (~1e-17) while well-posed shell models stay many
    orders of magnitude above it.
    """

    name = "linear-solver"

    def __init__(self, pivot_tolerance: float = 1e-14):
        self.pivot_tolerance = float(pivot_tolerance)

    @abstractmethod
    def factorize(self, K) -> Callable[[np.ndarray], np.ndarray]:
        """Factorise K once and return a function solving K x = F for one or more right-hand sides."""

    def solve(self, K, F: np.ndarray) -> np.ndarray:
        """Solve K x = F for a symmetric positive definite K (dense or sparse)."""
        return self.factorize(K)(F)

    def _check_pivots(self, pivots: np.ndarray) -> None:
        pivots = np.abs(np.asarray(pivots, dtype=float))
        if pivots.size and (not np.all(np.isfinite(pivots)) or pivots.min() <= self.pivot_tolerance * pivots.max()):
            raise SingularStiffnessError(
                "Stiffness matrix is singular (a rigid-body mode is unrestrained); check the constraints"
            )


class DenseLinearSolver(LinearSystemSolver):
    """Dense Cholesky factorisation (NumPy only)."""

    name = "numpy-dense"

    def factorize(self, K):
        K = K.toarray() if hasattr(K, "toarray") else np.asarray(K, dtype=float)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError as exc:
            raise SingularStiffnessError("Stiffness matrix is not positive definite; check the constraints") from exc
        self._check_pivots(np.diag(L) ** 2)

        def solve(F):
            y = np.linalg.solve(L, np.asarray(F, dtype=float))
            return np.linalg.solve(L.T, y)

        return solve


class SparseLinearSolver(LinearSystemSolver):
    """Sparse LU factorisation (SuperLU via SciPy).

    ``ordering`` is SuperLU's column ordering. The default minimum-degree ordering of
    K^T K keeps the fill small for both shell and solid models (on layered solid
    models it factorises several times faster than COLAMD); pass "COLAMD" for
    SuperLU's own default.
    """

    name = "scipy-sparse"

    def __init__(self, pivot_tolerance: float = 1e-14, ordering: str = "MMD_ATA"):
        super().__init__(pivot_tolerance)
        self.ordering = ordering
        if not HAVE_SCIPY:
            raise ImportError("SparseLinearSolver requires SciPy")

    def factorize(self, K):
        K = _sp.csc_matrix(K)
        try:
            lu = _spla.splu(K, permc_spec=self.ordering)
        except RuntimeError as exc:  # exactly singular
            raise SingularStiffnessError("Stiffness matrix is singular; check the constraints") from exc
        self._check_pivots(lu.U.diagonal())
        return lambda F: lu.solve(np.asarray(F, dtype=float))


def default_linear_solver() -> LinearSystemSolver:
    return SparseLinearSolver() if HAVE_SCIPY else DenseLinearSolver()
