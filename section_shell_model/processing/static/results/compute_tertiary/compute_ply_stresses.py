"""Tier 3: ply mid-surface strains/stresses, rotated to the ply material frame.

Wraps ``materials/clpt_recovery.ply_mid_stresses_laminate`` / ``ply_mid_strains_laminate``
(laminate axes) and ``rotate_plies_to_material`` (laminate axes -> ply material axes),
per PROCESSING_ARCHITECTURE_PROPOSAL.md tier 3.

Note: for zero-degree plies (laminate axes == material axes for that ply), the
rotation is the identity map, so this reproduces the ply stress values that today's
fused ``api/fi_recovery.hashin_fi_per_ply_from_rows_batch`` feeds into ``hashin_fi``
bit-for-bit. For off-axis plies this is now the *correct* material-frame stress
(the fused path never rotated before evaluating Hashin) — see
docs/architecture/DEPRECATION_REGISTER.md and PROCESSING_ARCHITECTURE_PROPOSAL.md
non-goals for the numerics-preservation rationale.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.material_utilities.laminate import (
    ply_mid_strains_laminate,
    ply_mid_stresses_laminate,
    rotate_plies_to_material,
)


def ply_strains_stresses_material_frame(
    lam: Any,
    eps0: NDArray[np.float64],
    kappa: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-ply strain/stress at one element's mid-surface, rotated to material axes.

    Returns
    -------
    eps_mat, sig_mat
        ``(n_ply, 3)`` each = ``[eps11, eps22, gamma12]`` / ``[sig11, sig22, tau12]``
        in ply material axes.
    """
    eps_lam = np.asarray(ply_mid_strains_laminate(lam, eps0, kappa), dtype=np.float64)
    sig_lam = np.asarray(ply_mid_stresses_laminate(lam, eps0, kappa), dtype=np.float64)
    # rotate_plies_to_material expects a leading "component" batch axis (n_comp, n_ply, 3);
    # a single element/laminate is n_comp=1, squeezed back out after the einsum.
    T_ply = np.asarray(lam.stiffness_bundle().T_ply, dtype=np.float64)[None, ...]
    eps_mat = rotate_plies_to_material(eps_lam[None, ...], T_ply)[0]
    sig_mat = rotate_plies_to_material(sig_lam[None, ...], T_ply)[0]
    return eps_mat, sig_mat


__all__ = ["ply_strains_stresses_material_frame"]
