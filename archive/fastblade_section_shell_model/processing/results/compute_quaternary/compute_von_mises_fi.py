"""Tier 4: Von Mises failure index — isotropic constituents only.

New in Phase 6 (quaternary scope tightening, see PROCESSING_ARCHITECTURE_PROPOSAL.md).
Wraps ``materials.failure_criteria.von_mises_plane_stress_fi``, replacing the inline
call previously in ``section_optimisation/engine/shell_fi_eval.py``.

**Scope (resolves open question #5 of the proposal):** Von Mises is strictly an
isotropic criterion. This module applies it only to isotropic constituents (e.g.
metal spar caps, adhesive layers) — composite laminate plies remain Hashin-only
(``compute_hashin_fi.py``). It is *not* applied as an informational metric on
rotated composite ply stress. Callers must supply the isotropic-equivalent
laminate-frame stress state (e.g. from a single-"ply" CLT recovery over an
isotropic material), not a material-frame composite ply stress array.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def von_mises_fi(
    sigma_iso: NDArray[np.float64],
    sigma_allow: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Batched Von Mises FI: ``sigma_iso[..., 3]`` = ``[σ11, σ22, τ12]`` (isotropic frame).

    Thin wrapper over ``materials.failure_criteria.von_mises_plane_stress_fi`` — no
    numerics change, only the call site moves here per the tier-4 criterion contract.
    """
    from blade_precompute.materials.failure_criteria import von_mises_plane_stress_fi

    return von_mises_plane_stress_fi(
        np.asarray(sigma_iso, dtype=float), np.asarray(sigma_allow, dtype=float)
    )


def von_mises_fi_from_row(
    sig3: NDArray[np.float64],
    sigma_allow: float,
) -> float:
    """Single-row facade: ``sig3`` = ``[σ11, σ22, τ12]`` (isotropic frame), scalar FI out."""
    fi = von_mises_fi(
        np.asarray(sig3, dtype=float).reshape(1, 3),
        np.array([sigma_allow], dtype=float),
    )
    return float(fi[0])


__all__ = ["von_mises_fi", "von_mises_fi_from_row"]
