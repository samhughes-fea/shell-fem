"""Shell orientation utilities for CalculiX S4R quad elements."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def shell_basis_from_quad(coords_4x3: NDArray) -> tuple[NDArray, NDArray, NDArray, float, float]:
    """Return shell basis vectors and characteristic lengths from quad corners.

    Node ordering matches ``discretise_strip`` CCW layout:
    0: (s_min, x_min), 1: (s_max, x_min), 2: (s_max, x_max), 3: (s_min, x_max).

    Parameters
    ----------
    coords_4x3
        ``(4, 3)`` corner coordinates ``[x, y, z]`` in the assembly frame.

    Returns
    -------
    e1, e2, e3
        Unit shell basis vectors (contour, span, normal) as 3-vectors.
    L_s, L_x
        Edge lengths along e1 and e2 directions.
    """
    p = np.asarray(coords_4x3, dtype=float)
    if p.shape != (4, 3):
        raise ValueError(f"coords_4x3 must be (4, 3); got {p.shape}")

    v_s = p[1] - p[0]
    v_x = p[3] - p[0]
    L_s = float(np.linalg.norm(v_s))
    L_x = float(np.linalg.norm(v_x))
    if L_s < 1e-30:
        v_s = p[2] - p[3]
        L_s = float(np.linalg.norm(v_s))
    if L_x < 1e-30:
        v_x = p[2] - p[1]
        L_x = float(np.linalg.norm(v_x))

    e1 = v_s / max(L_s, 1e-30)
    e3 = np.cross(v_s, p[2] - p[0])
    n3 = float(np.linalg.norm(e3))
    if n3 < 1e-30:
        e3 = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        e3 = e3 / n3
    e2 = np.cross(e3, e1)
    e2 = e2 / max(float(np.linalg.norm(e2)), 1e-30)

    return e1, e2, e3, L_s, L_x


def local_to_global_transform(e1: NDArray, e2: NDArray, e3: NDArray) -> NDArray:
    """Build 6×6 block-diagonal transform from curvilinear **C** to CCX Cartesian DOFs.

    ``e1,e2,e3`` are **C** basis vectors expressed in the CCX/B-aligned mesh
    coordinates (not blade frame **R**).
    """

    r = np.vstack([e1, e2, e3])
    t3 = r.T
    t6 = np.zeros((6, 6))
    t6[:3, :3] = t3
    t6[3:, 3:] = t3
    return t6


def quad_coords_from_strip(
    yz_nodes: NDArray,
    local_indices: list[int],
    *,
    l_x: float = 1.0,
    n_contour: int | None = None,
) -> NDArray:
    """Build ``(4, 3)`` CCX/B-aligned coords for a strip quad from arc-length indices.

    ``yz_nodes`` has one ``(y, z)`` row per contour station (length ``n``) in the
    section plane. Strip indices ``0 … n-1`` are the bottom layer (``x=0``);
    ``n … 2n-1`` are the top layer (``x=l_x``) along curvilinear span.
    """

    yz = np.asarray(yz_nodes, dtype=float)
    n = int(n_contour) if n_contour is not None else len(yz)
    coords = np.zeros((4, 3))
    for k, li in enumerate(local_indices):
        si = li if li < n else li - n
        x_val = 0.0 if li < n else float(l_x)
        coords[k, 0] = x_val
        coords[k, 1] = yz[si, 0]
        coords[k, 2] = yz[si, 1]
    return coords


__all__ = [
    "local_to_global_transform",
    "quad_coords_from_strip",
    "shell_basis_from_quad",
]
