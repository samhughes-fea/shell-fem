"""Generalised eigenproblems for shell vibration and linear buckling.

Vibration is K φ = ω² M φ with M positive definite. Linear buckling is
(K + λ Kg) φ = 0, solved as K φ = λ (-Kg) φ and reported as the positive load
factors λ. Small systems use a dense factorisation. Larger ones use ARPACK
when SciPy is installed. Neither solver assembles a rotor.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from shell_fem.pre_processing.containers.boundary_conditions import BoundaryConditions
from shell_fem.pre_processing.containers.mesh import FEAMesh
from shell_fem.processing.static.operations.constraints import ConstraintEliminator
from shell_fem.processing.static.operations.linear_solvers import HAVE_SCIPY, SingularStiffnessError, _sp

# Dense factorisation stays comfortable well past the verification meshes.
# Above this free-DOF count the solve switches to ARPACK.
DENSE_LIMIT = 800


def _symmetric(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def _dense(A) -> np.ndarray:
    if hasattr(A, "toarray"):
        A = A.toarray()
    return _symmetric(np.asarray(A, dtype=float))


def reduce_matrices(mesh: FEAMesh, bcs: BoundaryConditions, *matrices):
    """Project matrices onto the independent DOFs. Constraints must be homogeneous.

    Returns (T, reduced matrices, n_free) with U = T q.
    """
    T, g = ConstraintEliminator().build(mesh, bcs)
    if np.max(np.abs(np.asarray(g, dtype=float))) > 1e-12:
        raise ValueError("vibration and buckling need homogeneous constraints (prescribed values of zero)")
    reduced = [T.T @ M @ T for M in matrices]
    return T, reduced, int(T.shape[1])


def expand_modes(T, modes: np.ndarray) -> np.ndarray:
    """Map reduced modes (n_free, n_modes) to the full DOF vector."""
    full = T @ modes
    return np.asarray(full, dtype=float)


def _require_modes(n_modes: int, n_free: int) -> int:
    n_modes = int(n_modes)
    if n_modes < 1:
        raise ValueError("n_modes must be at least 1")
    if n_free < 1:
        raise ValueError("no free degrees of freedom; check the constraints")
    if n_modes >= n_free:
        raise ValueError(f"n_modes={n_modes} must be smaller than the {n_free} free degrees of freedom")
    return n_modes


def vibration_eigenpairs(K, M, n_modes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Smallest positive (ω, φ) of K φ = ω² M φ.

    φ columns are mass-normalised, φ^T M φ = 1, and ordered by increasing ω
    in radians per second. ω² is the eigenvalue.
    """
    n = int(K.shape[0])
    n_modes = _require_modes(n_modes, n)
    if n <= DENSE_LIMIT or not HAVE_SCIPY:
        if n > DENSE_LIMIT:
            raise ImportError("a shell model of this size needs SciPy for the vibration solve")
        return _dense_vibration(_dense(K), _dense(M), n_modes)
    return _sparse_vibration(K, M, n_modes)


def buckling_eigenpairs(K, Kg, n_modes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Smallest positive load factors of (K + λ Kg) φ = 0.

    Each mode is scaled so its largest absolute component is 1.
    """
    n = int(K.shape[0])
    n_modes = _require_modes(n_modes, n)
    if n <= DENSE_LIMIT or not HAVE_SCIPY:
        if n > DENSE_LIMIT:
            raise ImportError("a shell model of this size needs SciPy for the buckling solve")
        return _dense_buckling(_dense(K), _dense(Kg), n_modes)
    return _sparse_buckling(K, Kg, n_modes)


def _dense_vibration(K: np.ndarray, M: np.ndarray, n_modes: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "mass matrix is not positive definite; every free degree of freedom needs inertia "
            "and the laminate density must be positive"
        ) from exc
    try:
        np.linalg.cholesky(K)
    except np.linalg.LinAlgError as exc:
        raise SingularStiffnessError(
            "Stiffness matrix is not positive definite; check the constraints"
        ) from exc
    eye = np.eye(M.shape[0])
    Y = np.linalg.solve(L, eye)  # L^{-1}; φ = L^{-T} y and y^T y = φ^T M φ
    standard = _symmetric(Y @ K @ Y.T)
    evals, yvecs = np.linalg.eigh(standard)
    positive = np.flatnonzero(evals > 0.0)
    if positive.size < n_modes:
        raise ValueError(f"only {positive.size} positive frequencies; requested {n_modes}")
    pick = positive[:n_modes]
    omega = np.sqrt(evals[pick])
    modes = Y.T @ yvecs[:, pick]
    return omega, modes


def _dense_buckling(K: np.ndarray, Kg: np.ndarray, n_modes: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        np.linalg.cholesky(K)
    except np.linalg.LinAlgError as exc:
        raise SingularStiffnessError(
            "Stiffness matrix is not positive definite; check the constraints"
        ) from exc
    # K φ = μ^{-1} (-Kg) φ, so the largest positive μ are the smallest positive λ = 1/μ.
    operator = np.linalg.solve(K, -Kg)
    mu, vecs = np.linalg.eig(operator)
    return _factors_from_mu(mu, vecs, n_modes)


def _sparse_vibration(K, M, n_modes: int) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.sparse.linalg import eigsh

    Ks, Ms = _csr(K), _csr(M)
    try:
        evals, modes = eigsh(Ks, k=n_modes, M=Ms, sigma=0.0, which="LM")
    except Exception as exc:
        raise SingularStiffnessError(
            "Vibration eigenproblem could not be solved; check the constraints and the laminate density"
        ) from exc
    order = np.argsort(np.real(evals))
    omega2 = np.real(evals[order])
    modes = np.real(modes[:, order])
    if np.any(omega2 <= 0.0):
        raise ValueError("vibration solve returned a non-positive frequency; check the constraints")
    for i in range(modes.shape[1]):
        scale = float(modes[:, i] @ (Ms @ modes[:, i]))
        if scale <= 0.0:
            raise ValueError("vibration mode has no mass")
        modes[:, i] /= np.sqrt(scale)
    return np.sqrt(omega2), modes


def _sparse_buckling(K, Kg, n_modes: int) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.sparse.linalg import LinearOperator, eigs, splu

    Ks, Kgs = _csr(K), _csr(Kg)
    try:
        factor = splu(Ks.tocsc())
    except RuntimeError as exc:
        raise SingularStiffnessError(
            "Stiffness matrix is singular (a rigid-body mode is unrestrained); check the constraints"
        ) from exc
    n = Ks.shape[0]
    neg_kg = -Kgs

    def matvec(x):
        return factor.solve(neg_kg @ x)

    operator = LinearOperator((n, n), matvec=matvec, dtype=float)
    # Ask for a few extra values so a non-positive one can be dropped.
    k = min(n - 1, max(n_modes + 4, n_modes))
    try:
        mu, vecs = eigs(operator, k=k, which="LR")
    except Exception as exc:
        raise SingularStiffnessError(
            "Buckling eigenproblem could not be solved; check the constraints and the reference load"
        ) from exc
    return _factors_from_mu(mu, vecs, n_modes)


def _factors_from_mu(mu: np.ndarray, vecs: np.ndarray, n_modes: int) -> Tuple[np.ndarray, np.ndarray]:
    real = np.real(mu)
    imag = np.imag(mu)
    scale = np.maximum(1.0, np.abs(real))
    ok = np.isfinite(real) & (np.abs(imag) <= 1e-6 * scale) & (real > 1e-8)
    idx = np.flatnonzero(ok)
    if idx.size < n_modes:
        raise ValueError(f"only {idx.size} positive buckling factors; requested {n_modes}")
    # Largest μ first, which is the smallest positive load factor.
    chosen = idx[np.argsort(real[idx])[::-1]][:n_modes]
    factors = 1.0 / real[chosen]
    modes = np.real(vecs[:, chosen])
    for i, j in enumerate(chosen):
        imag_part = np.max(np.abs(np.imag(vecs[:, j])))
        peak = np.max(np.abs(modes[:, i]))
        if peak == 0.0 or imag_part > 1e-6 * max(1.0, peak):
            raise ValueError("buckling mode is not a real displacement")
        modes[:, i] /= peak
    order = np.argsort(factors)
    return factors[order], modes[:, order]


def _csr(A):
    if _sp.issparse(A):
        A = 0.5 * (A + A.T)
        return A.tocsr()
    return _sp.csr_matrix(_dense(A))
