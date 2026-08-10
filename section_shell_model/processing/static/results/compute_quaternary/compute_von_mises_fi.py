"""Tier 4: Von Mises failure index — isotropic constituents only.

Wraps ``materials.failure_criteria.von_mises_plane_stress_fi``.

**Scope:** Von Mises is strictly an isotropic criterion. This module applies it
only to isotropic constituents (e.g. metal spar caps, adhesive layers) —
composite laminate plies remain Hashin-only (``compute_hashin_fi.py``).
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
    from blade_structure_utilities.material_utilities.failure import von_mises_plane_stress_fi

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
