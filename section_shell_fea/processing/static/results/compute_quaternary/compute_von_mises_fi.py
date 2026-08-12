"""Tier 4: Von Mises failure index for isotropic shells (yield criterion).

``FI = σ_vm / σ_y``. Consumes mid-face tertiary stress selected by the orchestrator.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.material_utilities.failure import von_mises_plane_stress_fi


def von_mises_fi(
    sigma_iso: NDArray[np.float64],
    sigma_y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Batched plane-stress Von Mises FI; ``sigma_iso`` is ``(n, 3)`` mid-face stress."""
    return von_mises_plane_stress_fi(
        np.asarray(sigma_iso, dtype=float), np.asarray(sigma_y, dtype=float)
    )


def von_mises_fi_from_row(
    sigma_iso_row: NDArray[np.float64],
    sigma_y: float,
) -> float:
    """Single-row convenience wrapper."""
    fi = von_mises_fi(
        np.asarray(sigma_iso_row, dtype=float).reshape(1, 3),
        np.array([sigma_y], dtype=float),
    )
    return float(fi[0])


__all__ = ["von_mises_fi", "von_mises_fi_from_row"]
