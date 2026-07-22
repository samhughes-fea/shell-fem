"""Buckling helpers and a reserved nonlinear placeholder (out of scope).

The production periodic / assembly path is **linear static**: ``K u = f``.

This module exposes:
- :func:`nonlinear_shell_solve_not_implemented` — permanent stub; geometric
  nonlinear Newton is **out of scope** for ``section_shell_model`` (see
  ``docs/DEFERRED_RESEARCH.md``). The generalised section BVP
  (:mod:`..engine.periodic_reduced_bvp`) is independent of that choice.
- :func:`linear_buckling_smallest_eigenvalues` — dense generalized eigen solver for ``K φ = λ (-K_g) φ``,
  which is algebraically identical to `(K + λ K_g) φ = 0` whenever both matrices live in the same dof basis.

A full quadrilateral MITC4 **stress stiffness assembler** (“shell `K_G`”) belongs in assembly code once
stress fields are threaded through per Gaussian point; callers then pass the reduced pencils here.

See also orthotropic screening in :mod:`strip_section_solver.engine.panel_buckling`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from numpy.typing import NDArray


@dataclass(frozen=True)
class NonlinearShellSolveConfig:
    """Placeholder config retained for API stability; nonlinear solve is out of scope."""

    max_newton_iter: int = 25
    force_linear_solve_only: bool = True


def linear_buckling_smallest_eigenvalues(
    K: sp.spmatrix,
    K_g: sp.spmatrix,
    *,
    nev: int = 5,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Solve `(K + λ K_g) φ = 0` via the equivalent ``K φ = λ (-K_g) φ``.

    Uses dense ``scipy.linalg.eig`` for robustness at moderate ``n_dof`` (<~ few thousand).
    """
    Kd = np.asarray(K.toarray(), dtype=np.complex128)
    Md = np.asarray(-(K_g).toarray(), dtype=np.complex128)
    n = int(Kd.shape[0])
    if Md.shape != (n, n):
        raise ValueError(f"-K_g shape {Md.shape} must match K {(n, n)}")

    w, vr = la.eig(Kd, Md)
    order = np.argsort(np.abs(w))
    w = w[order][: min(int(nev), n)]
    modes = vr[:, order][:, : min(int(nev), n)]

    return np.asarray(np.real_if_close(w)), modes


def nonlinear_shell_solve_not_implemented(*_args: Any, **_kwargs: Any) -> None:
    """Out of scope for ``section_shell_model``; raises ``NotImplementedError``."""

    raise NotImplementedError(
        "Geometrically nonlinear shell incremental solve is out of scope for "
        "section_shell_model. Use run_section_periodic_bc for linear equilibrium "
        "on the generalised periodic BVP (PeriodicReducedBVP)."
    )
