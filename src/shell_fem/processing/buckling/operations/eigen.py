"""Eigen stage: the buckling load factors and modes of the reduced system."""

from __future__ import annotations

import numpy as np

from shell_fem.processing.buckling.operations.buckling_states import BucklingEigenState, GeometricSystem
from shell_fem.processing.static.operations.linear_solvers import HAVE_SCIPY, _sp, _spla
from shell_fem.processing.static.operations.solve_states import FactorisedSystem


class SolveBucklingEigenproblem:
    """Stage: K_r phi = lambda (-K_g,r) phi, with K_g,r = T^T K_g T on the same constraints as the static solve.

    It is solved as -K_g,r phi = mu K_r phi with mu = 1 / lambda, since K_r is positive
    definite and K_g,r is not: the largest mu are the lowest buckling loads.

    ``n_modes``    how many modes to return.
    ``signs``      "positive" (default) keeps the load factors of the reference load as
                   applied; "both" also returns negative factors (the reversed load).
    ``method``     "auto" (dense below ``dense_limit`` independent DOFs or without
                   SciPy, Lanczos above), "dense" or "lanczos".
    """

    def __init__(self, n_modes: int = 5, signs: str = "positive", method: str = "auto", dense_limit: int = 1500,
                 tolerance: float = 0.0):
        if n_modes < 1:
            raise ValueError("n_modes must be at least 1")
        if signs not in ("positive", "both"):
            raise ValueError("signs must be 'positive' or 'both'")
        if method not in ("auto", "dense", "lanczos"):
            raise ValueError("method must be 'auto', 'dense' or 'lanczos'")
        if method == "lanczos" and not HAVE_SCIPY:
            raise ImportError("The Lanczos eigensolver requires SciPy")
        self.n_modes, self.signs, self.method = int(n_modes), signs, method
        self.dense_limit, self.tolerance = int(dense_limit), float(tolerance)

    def run(self, factorised: FactorisedSystem, geometric: GeometricSystem) -> BucklingEigenState:
        reduced = factorised.reduced
        T = reduced.constrained.T
        K_g = T.T @ geometric.K_g @ T
        n = int(T.shape[1])
        method = self.method
        if method == "auto":
            method = "dense" if (n <= self.dense_limit or not HAVE_SCIPY) else "lanczos"
        if method == "lanczos" and self.n_modes >= n - 1:
            method = "dense"
        if method == "dense":
            mu, phi = self._dense(reduced.K_r, K_g)
        else:
            mu, phi = self._lanczos(factorised, K_g, n)
        scale = float(np.max(np.abs(mu))) if mu.size else 0.0
        keep = np.abs(mu) > max(1e-12 * scale, 1e-300)
        if self.signs == "positive":
            keep &= mu > 0.0
        mu, phi = mu[keep], phi[:, keep]
        lam = 1.0 / mu
        order = np.argsort(np.abs(lam))[: self.n_modes]
        info = {"n_free": n, "n_requested": self.n_modes}
        if not order.size:
            info["reason"] = "the reference load puts nothing in compression"
        return BucklingEigenState(load_factors=lam[order], modes=phi[:, order], method=method, info=info)

    @staticmethod
    def _dense(K_r, K_g):
        K = K_r.toarray() if hasattr(K_r, "toarray") else np.asarray(K_r, dtype=float)
        G = K_g.toarray() if hasattr(K_g, "toarray") else np.asarray(K_g, dtype=float)
        L = np.linalg.cholesky(0.5 * (K + K.T))
        A = np.linalg.solve(L, np.linalg.solve(L, -0.5 * (G + G.T)).T)  # L^-1 (-K_g) L^-T
        mu, y = np.linalg.eigh(0.5 * (A + A.T))
        return mu, np.linalg.solve(L.T, y)

    def _lanczos(self, factorised: FactorisedSystem, K_g, n: int):
        K_r = _sp.csr_matrix(factorised.reduced.K_r)
        A = -_sp.csr_matrix(K_g)
        A = 0.5 * (A + A.T)
        Minv = _spla.LinearOperator((n, n), matvec=lambda x: factorised.solve(np.asarray(x).ravel()), dtype=float)
        k = min(n - 2, self.n_modes if self.signs == "positive" else 2 * self.n_modes)
        which = "LA" if self.signs == "positive" else "LM"
        mu, phi = _spla.eigsh(A, k=max(k, 1), M=K_r, Minv=Minv, which=which, tol=self.tolerance)
        return mu, phi
