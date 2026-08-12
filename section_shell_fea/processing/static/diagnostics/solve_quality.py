"""One-shot reduced-solve quality metrics for cascade trust (confidence Phase A)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla

# Soft gates (borderline → cascade_trust_ok=False, still produce cascade).
DEFAULT_RESIDUAL_REL_MAX = 1.0e-8
DEFAULT_PIVOT_RATIO_MAX = 1.0e12


@dataclass(frozen=True)
class SolveQualityReport:
    residual_rel: float | None
    pivot_ratio: float | None
    ok: bool
    reason: str = ""
    catastrophic: bool = False


def pivot_ratio_from_factor(factor: spla.SuperLU | None) -> float | None:
    """max|U_ii| / min|U_ii| on the SuperLU U factor diagonal (None if unavailable)."""
    if factor is None:
        return None
    try:
        u = factor.U
        diag = np.abs(np.asarray(u.diagonal(), dtype=float).ravel())
    except Exception:
        return None
    if diag.size == 0:
        return None
    dmax = float(np.max(diag))
    dmin = float(np.min(diag))
    if not np.isfinite(dmax) or not np.isfinite(dmin) or dmin <= 0.0:
        return float("inf")
    return dmax / dmin


def residual_rel_reduced(
    K_reduced,
    q: np.ndarray,
    f_reduced: np.ndarray,
) -> float:
    """‖K_rr q − f_r‖ / max(‖f_r‖, ε)."""
    q_a = np.asarray(q, dtype=float).ravel()
    f_a = np.asarray(f_reduced, dtype=float).ravel()
    if q_a.size == 0:
        return 0.0
    r = np.asarray(K_reduced @ q_a, dtype=float).ravel() - f_a
    num = float(np.linalg.norm(r))
    den = max(float(np.linalg.norm(f_a)), 1.0e-30)
    return num / den


def assess_solve_quality(
    *,
    residual_rel: float | None,
    pivot_ratio: float | None,
    q: np.ndarray | None = None,
    residual_rel_max: float = DEFAULT_RESIDUAL_REL_MAX,
    pivot_ratio_max: float = DEFAULT_PIVOT_RATIO_MAX,
) -> SolveQualityReport:
    """Classify reduced-solve quality (catastrophic vs soft fail vs ok)."""
    if q is not None:
        q_a = np.asarray(q, dtype=float).ravel()
        if q_a.size and not np.all(np.isfinite(q_a)):
            return SolveQualityReport(
                residual_rel=residual_rel,
                pivot_ratio=pivot_ratio,
                ok=False,
                reason="non_finite_q",
                catastrophic=True,
            )
    if residual_rel is not None and not np.isfinite(residual_rel):
        return SolveQualityReport(
            residual_rel=residual_rel,
            pivot_ratio=pivot_ratio,
            ok=False,
            reason="non_finite_residual",
            catastrophic=True,
        )
    reasons: list[str] = []
    if residual_rel is not None and residual_rel > float(residual_rel_max):
        reasons.append(f"residual_rel={residual_rel:.3e}>{residual_rel_max:.3e}")
    if pivot_ratio is not None and (
        not np.isfinite(pivot_ratio) or pivot_ratio > float(pivot_ratio_max)
    ):
        reasons.append(f"pivot_ratio={pivot_ratio}>{pivot_ratio_max:.3e}")
    if reasons:
        return SolveQualityReport(
            residual_rel=residual_rel,
            pivot_ratio=pivot_ratio,
            ok=False,
            reason=";".join(reasons),
            catastrophic=False,
        )
    return SolveQualityReport(
        residual_rel=residual_rel,
        pivot_ratio=pivot_ratio,
        ok=True,
        reason="",
        catastrophic=False,
    )


class CatastrophicSolveError(RuntimeError):
    """Raised when the reduced solve cannot produce a usable displacement field."""


__all__ = [
    "CatastrophicSolveError",
    "DEFAULT_PIVOT_RATIO_MAX",
    "DEFAULT_RESIDUAL_REL_MAX",
    "SolveQualityReport",
    "assess_solve_quality",
    "pivot_ratio_from_factor",
    "residual_rel_reduced",
]
