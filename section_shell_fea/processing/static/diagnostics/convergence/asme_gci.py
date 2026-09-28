"""ASME V&V 20 / Roache Grid Convergence Index helpers (pure numerics).

Three consecutive meshes ordered fine→coarse ``(h1,Q1), (h2,Q2), (h3,Q3)``.
Supports unequal refinement ratios (study ladder fracs are not constant-``r``).

No FE imports — safe for unit tests and CSV post-processing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

FS_THREE_LEVEL = 1.25
FS_TWO_LEVEL = 3.0
P_MIN = 0.5
P_MAX = 4.0
P_DEFAULT = 2.0


@dataclass(frozen=True)
class GciResult:
    """Fine-grid GCI package for one metric series."""

    ok: bool
    gci_fine: float
    p_obs: float
    q_ext: float
    r21: float
    r32: float
    fs: float
    asymptotic_ratio: float
    oscillatory: bool
    detail: str = ""


def richardson_extrapolate(
    h1: float,
    h2: float,
    q1: float,
    q2: float,
    *,
    p: float = P_DEFAULT,
) -> float:
    """Richardson: ``Q_ext = (r^p Q1 - Q2) / (r^p - 1)`` with ``r = h2/h1``."""
    h1f, h2f = float(h1), float(h2)
    q1f, q2f = float(q1), float(q2)
    if h1f <= 0.0 or h2f <= 0.0:
        raise ValueError(f"h must be positive, got h1={h1f}, h2={h2f}")
    r = h2f / h1f
    if abs(r - 1.0) < 1e-12:
        return q1f
    rp = r ** float(p)
    denom = rp - 1.0
    if abs(denom) < 1e-30:
        return q1f
    return (rp * q1f - q2f) / denom


def gci_fine(
    h1: float,
    h2: float,
    q1: float,
    q2: float,
    *,
    p: float = P_DEFAULT,
    fs: float = FS_THREE_LEVEL,
) -> float:
    """Fine-grid GCI: ``Fs |Q1-Q2| / ((r^p - 1) |Q1|)`` with ``r = h2/h1``."""
    h1f, h2f = float(h1), float(h2)
    q1f, q2f = float(q1), float(q2)
    if h1f <= 0.0 or h2f <= 0.0:
        raise ValueError(f"h must be positive, got h1={h1f}, h2={h2f}")
    if not np.isfinite(q1f) or abs(q1f) < 1e-30:
        return float("nan")
    r = h2f / h1f
    denom = (r ** float(p) - 1.0) * abs(q1f)
    if abs(denom) < 1e-30:
        return float("nan")
    return float(fs) * abs(q1f - q2f) / denom


def observed_order(
    h1: float,
    h2: float,
    h3: float,
    q1: float,
    q2: float,
    q3: float,
    *,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> tuple[float, bool]:
    """ASME observed order ``p`` for possibly unequal ``r21``, ``r32``.

    Returns ``(p, oscillatory)``. Oscillatory when ``(Q3-Q2)/(Q2-Q1) < 0``.
    """
    h1f, h2f, h3f = float(h1), float(h2), float(h3)
    q1f, q2f, q3f = float(q1), float(q2), float(q3)
    if min(h1f, h2f, h3f) <= 0.0:
        raise ValueError("all h must be positive")
    if not (h1f < h2f < h3f):
        raise ValueError(
            f"require fine→coarse h1 < h2 < h3, got {h1f}, {h2f}, {h3f}"
        )

    r21 = h2f / h1f
    r32 = h3f / h2f
    eps21 = q2f - q1f
    eps32 = q3f - q2f
    if abs(eps21) < 1e-30:
        return P_DEFAULT, False

    s = float(np.sign(eps32 / eps21))
    oscillatory = s < 0.0
    if abs(s) < 0.5:
        return P_DEFAULT, oscillatory

    # Constant-r closed form when ratios match.
    if abs(r21 - r32) < 1e-6:
        p = float(np.log(abs(eps32 / eps21)) / np.log(r21))
        return float(np.clip(p, P_MIN, P_MAX)), oscillatory

    # ASME iterative correction for unequal refinement ratios.
    p = P_DEFAULT
    ln_r21 = float(np.log(r21))
    for _ in range(int(max_iter)):
        num = r21**p - s
        den = r32**p - s
        if abs(den) < 1e-30 or num <= 0.0 or den <= 0.0:
            break
        q_corr = float(np.log(num / den))
        p_new = abs(float(np.log(abs(eps32 / eps21))) + q_corr) / ln_r21
        if abs(p_new - p) < float(tol):
            p = p_new
            break
        p = p_new
    return float(np.clip(p, P_MIN, P_MAX)), oscillatory


def gci_triplet(
    h: Sequence[float],
    q: Sequence[float],
    *,
    p_assumed: float = P_DEFAULT,
) -> GciResult:
    """Compute fine-grid GCI from three (or two) meshes.

    ``h`` / ``q`` may be fine→coarse or coarse→fine; values are sorted by ``h``.
    With three+ levels, the three finest are used (``Fs=1.25``). Two levels use
    ``Fs=3`` and ``p_assumed``.
    """
    hh = np.asarray(list(h), dtype=float).ravel()
    qq = np.asarray(list(q), dtype=float).ravel()
    if hh.size != qq.size:
        raise ValueError("h and q must have the same length")
    if hh.size < 2:
        return GciResult(
            ok=False,
            gci_fine=float("nan"),
            p_obs=float("nan"),
            q_ext=float("nan"),
            r21=float("nan"),
            r32=float("nan"),
            fs=float("nan"),
            asymptotic_ratio=float("nan"),
            oscillatory=False,
            detail="need_at_least_two_levels",
        )
    if not (np.all(np.isfinite(hh)) and np.all(np.isfinite(qq))):
        return GciResult(
            ok=False,
            gci_fine=float("nan"),
            p_obs=float("nan"),
            q_ext=float("nan"),
            r21=float("nan"),
            r32=float("nan"),
            fs=float("nan"),
            asymptotic_ratio=float("nan"),
            oscillatory=False,
            detail="non_finite_inputs",
        )
    if np.any(hh <= 0.0):
        return GciResult(
            ok=False,
            gci_fine=float("nan"),
            p_obs=float("nan"),
            q_ext=float("nan"),
            r21=float("nan"),
            r32=float("nan"),
            fs=float("nan"),
            asymptotic_ratio=float("nan"),
            oscillatory=False,
            detail="non_positive_h",
        )

    order = np.argsort(hh)  # fine → coarse
    hh = hh[order]
    qq = qq[order]

    if hh.size == 2:
        h1, h2 = float(hh[0]), float(hh[1])
        q1, q2 = float(qq[0]), float(qq[1])
        r21 = h2 / h1
        p = float(p_assumed)
        fs = FS_TWO_LEVEL
        q_ext = richardson_extrapolate(h1, h2, q1, q2, p=p)
        gci = gci_fine(h1, h2, q1, q2, p=p, fs=fs)
        return GciResult(
            ok=bool(np.isfinite(gci)),
            gci_fine=float(gci) if np.isfinite(gci) else float("nan"),
            p_obs=p,
            q_ext=float(q_ext),
            r21=r21,
            r32=float("nan"),
            fs=fs,
            asymptotic_ratio=float("nan"),
            oscillatory=False,
            detail="two_level_fs3",
        )

    h1, h2, h3 = float(hh[0]), float(hh[1]), float(hh[2])
    q1, q2, q3 = float(qq[0]), float(qq[1]), float(qq[2])
    r21 = h2 / h1
    r32 = h3 / h2

    p, oscillatory = observed_order(h1, h2, h3, q1, q2, q3)
    # Unreliable observed order → use theoretical p=2 for the GCI number.
    p_for_gci = float(p)
    if oscillatory or p <= P_MIN + 1e-12 or p >= P_MAX - 1e-12:
        p_for_gci = float(p_assumed)

    if oscillatory:
        # Still report a two-level Fs=3 estimate on the finest pair for CSV.
        fs = FS_TWO_LEVEL
        q_ext = richardson_extrapolate(h1, h2, q1, q2, p=p_for_gci)
        gci12 = gci_fine(h1, h2, q1, q2, p=p_for_gci, fs=fs)
        return GciResult(
            ok=False,
            gci_fine=float(gci12) if np.isfinite(gci12) else float("nan"),
            p_obs=float(p),
            q_ext=float(q_ext),
            r21=r21,
            r32=r32,
            fs=fs,
            asymptotic_ratio=float("nan"),
            oscillatory=True,
            detail="gci_non_monotonic",
        )

    fs = FS_THREE_LEVEL
    q_ext = richardson_extrapolate(h1, h2, q1, q2, p=p_for_gci)
    gci12 = gci_fine(h1, h2, q1, q2, p=p_for_gci, fs=fs)
    gci23 = gci_fine(h2, h3, q2, q3, p=p_for_gci, fs=fs)
    asym = float("nan")
    if np.isfinite(gci12) and np.isfinite(gci23) and gci12 > 0.0:
        denom = (r21**p_for_gci) * gci12
        if abs(denom) > 1e-30:
            asym = gci23 / denom

    ok = bool(np.isfinite(gci12))
    detail = ""
    if abs(p_for_gci - p) > 1e-12:
        detail = "p_clamped_assumed"
    if not ok:
        detail = "gci_undefined"
    return GciResult(
        ok=ok,
        gci_fine=float(gci12) if ok else float("nan"),
        p_obs=float(p),
        q_ext=float(q_ext),
        r21=r21,
        r32=r32,
        fs=fs,
        asymptotic_ratio=float(asym),
        oscillatory=False,
        detail=detail,
    )


__all__ = [
    "FS_THREE_LEVEL",
    "FS_TWO_LEVEL",
    "GciResult",
    "gci_fine",
    "gci_triplet",
    "observed_order",
    "richardson_extrapolate",
]
