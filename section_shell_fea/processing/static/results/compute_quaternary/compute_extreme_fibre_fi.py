"""Material-agnostic extreme-fibre FI at laminate free surfaces (±h/2).

Uses textbook ultimate strengths σ_uts, σ_ucs, τ_us. One path for every panel;
every element is evaluated. Does not read constitutive kind or Hashin ply strengths.

Blade-structure reading of the two free surfaces (same laminate ±h/2):
skin = fluid / void; spar cap = skin / void; web = fore (LE side) / aft (TE side).
See compute_tertiary/docs § panel face semantics.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ..containers.ply_results import FACE_INNER, FACE_OUTER


def extreme_fibre_stress_from_ply_mat(
    sig_ply_mat: NDArray[np.float64],
    n_ply_true: int | None = None,
) -> NDArray[np.float64]:
    """Laminate free-surface plane stress ``(n_elem, 2, 3)`` = [outer, inner].

    Outer = top-plate outer face; inner = bottom-plate inner face.
    """
    sig = np.asarray(sig_ply_mat, dtype=np.float64)
    if sig.ndim != 4 or sig.shape[-1] != 3 or sig.shape[-2] != 3:
        raise ValueError(
            f"sig_ply_mat must be (n_elem, n_ply, 3, 3), got shape {sig.shape}"
        )
    n_elem, n_ply = int(sig.shape[0]), int(sig.shape[1])
    n_use = int(n_ply_true) if n_ply_true is not None else n_ply
    n_use = max(1, min(n_use, n_ply))
    out = np.zeros((n_elem, 2, 3), dtype=np.float64)
    out[:, 0, :] = sig[:, n_use - 1, FACE_OUTER, :]
    out[:, 1, :] = sig[:, 0, FACE_INNER, :]
    return out


def extreme_fibre_fi(
    sig_extreme: NDArray[np.float64],
    sigma_uts: float,
    sigma_ucs: float,
    tau_us: float,
) -> NDArray[np.float64]:
    """Element envelope FI from outer/inner plane-stress vectors.

    Component ratios: |σ11|/S11, |σ22|/S22 with S = σ_uts if σ≥0 else σ_ucs;
    |τ12|/τ_us. Surface FI = max of three; element FI = max(outer, inner).
    """
    sig = np.asarray(sig_extreme, dtype=np.float64)
    if sig.ndim != 3 or sig.shape[-1] != 3:
        raise ValueError(f"sig_extreme must be (n_elem, 2, 3), got {sig.shape}")
    uts = max(float(sigma_uts), 1e-12)
    ucs = max(float(sigma_ucs), 1e-12)
    tus = max(float(tau_us), 1e-12)

    s11 = sig[..., 0]
    s22 = sig[..., 1]
    t12 = sig[..., 2]
    s11_den = np.where(s11 >= 0.0, uts, ucs)
    s22_den = np.where(s22 >= 0.0, uts, ucs)
    fi11 = np.abs(s11) / s11_den
    fi22 = np.abs(s22) / s22_den
    fi12 = np.abs(t12) / tus
    fi_surf = np.maximum(np.maximum(fi11, fi22), fi12)
    return np.max(fi_surf, axis=-1)


def panel_ultimates_finite(sigma_uts: float, sigma_ucs: float, tau_us: float) -> bool:
    """True when all three ultimate strengths are finite and positive."""
    return (
        math.isfinite(sigma_uts)
        and sigma_uts > 0.0
        and math.isfinite(sigma_ucs)
        and sigma_ucs > 0.0
        and math.isfinite(tau_us)
        and tau_us > 0.0
    )


__all__ = [
    "extreme_fibre_fi",
    "extreme_fibre_stress_from_ply_mat",
    "panel_ultimates_finite",
]
