r"""Tier 3: per-plate outer/mid/inner strains/stresses, rotated **E → M**.

CLPT models a laminate as a **stack of thin plates**. This module evaluates the
same plane-stress recovery on every plate at outer, mid, and inner faces in
element/laminate frame **E**, then rotates into ply material frame **M**.

Stress uses ``T_ply`` (\(T_\\sigma\)) and engineering strain uses ``T_ply_eps``
(\(T_\\varepsilon\)) — these are **different** matrices. See the coordinate_frames
README §"Laminate (E) and ply material (M)".
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.material_utilities.laminate import (
    ply_face_strains_laminate,
    ply_face_stresses_laminate,
    rotate_plies_to_material,
)

# Shared face-axis convention (matches material_utilities.laminate.FACE_*)
FACE_OUTER = 0
FACE_MID = 1
FACE_INNER = 2


def mid_face(sig_or_eps: NDArray[np.float64]) -> NDArray[np.float64]:
    """Slice mid face from ``(..., n_ply, 3_faces, 3)`` → ``(..., n_ply, 3)``."""
    return np.asarray(sig_or_eps, dtype=np.float64)[..., FACE_MID, :]


def ply_strains_stresses_material_frame(
    lam: Any,
    eps0: NDArray[np.float64],
    kappa: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-plate outer/mid/inner strain/stress for one element in **M**.

    Returns
    -------
    eps_mat, sig_mat
        ``(n_ply, 3, 3)`` each = plate × {outer, mid, inner} ×
        ``[ε11, ε22, γ12]`` / ``[σ11, σ22, τ12]`` (ply material 123).
    """
    eps_lam = np.asarray(ply_face_strains_laminate(lam, eps0, kappa), dtype=np.float64)
    sig_lam = np.asarray(ply_face_stresses_laminate(lam, eps0, kappa), dtype=np.float64)
    bundle = lam.stiffness_bundle()
    # Stress and engineering strain take *different* transforms: T_sigma carries
    # 2sc on the normal rows, T_epsilon carries 2sc on the shear row. Rotating
    # strain with T_sigma (as this did before) corrupts gamma12 and, whenever the
    # laminate carries shear, the normal strains too.
    T_sig = np.asarray(bundle.T_ply, dtype=np.float64)[None, ...]
    T_eps = np.asarray(bundle.T_ply_eps, dtype=np.float64)[None, ...]
    eps_mat = rotate_plies_to_material(eps_lam[None, ...], T_eps)[0]
    sig_mat = rotate_plies_to_material(sig_lam[None, ...], T_sig)[0]
    return eps_mat, sig_mat


__all__ = [
    "FACE_INNER",
    "FACE_MID",
    "FACE_OUTER",
    "mid_face",
    "ply_strains_stresses_material_frame",
]
