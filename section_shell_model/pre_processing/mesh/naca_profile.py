"""
Thin-wall NACA 4-digit profile helpers (LE→TE upper/lower stack).

``blade_structure_utilities.geometry_utilities.sdf.naca`` produces an outer TE→LE / LE→TE polygon and is
not interchangeable with the equal-row thin-wall contract used by sectorial
warping and station airfoil builders.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _pin_le_te_chord(x: np.ndarray) -> NDArray[np.float64]:
    out = np.asarray(x, dtype=np.float64).copy()
    out[0] = 0.0
    out[-1] = 1.0
    return out


def chordwise_stations(n: int, *, spacing: str = "nested_cosine") -> NDArray[np.float64]:
    """
    Chord-normalised stations ``x/c`` in ``[0, 1]``.

    The first station is exactly the leading edge (``x/c = 0``) and the last
    exactly the trailing edge (``x/c = 1``).
    """
    if n < 2:
        raise ValueError("chordwise_stations requires n >= 2")
    if spacing == "uniform":
        return np.linspace(0.0, 1.0, n)
    if spacing == "cosine":
        t = np.linspace(0.0, np.pi, n)
        x = 0.5 * (1.0 - np.cos(t))
        return _pin_le_te_chord(x)
    if spacing == "nested_cosine":
        u = np.linspace(0.0, np.pi, n)
        v = 0.5 * (1.0 - np.cos(u))
        x = 0.5 * (1.0 - np.cos(np.pi * v))
        return _pin_le_te_chord(x)
    if spacing == "nested_cosine_3":
        u = np.linspace(0.0, np.pi, n)
        v = 0.5 * (1.0 - np.cos(u))
        w = 0.5 * (1.0 - np.cos(np.pi * v))
        x = 0.5 * (1.0 - np.cos(np.pi * w))
        return _pin_le_te_chord(x)
    if spacing == "le_dense":
        u = np.linspace(0.0, 1.0, n)
        x = np.sin(0.5 * np.pi * u)
        return _pin_le_te_chord(x)
    raise ValueError(
        f"unknown chord spacing {spacing!r}; use 'cosine', 'nested_cosine', "
        f"'nested_cosine_3', 'le_dense', or 'uniform'"
    )


def naca_four_digit(
    m: float,
    p: float,
    t_c: float,
    n: int = 200,
    *,
    chord_spacing: str = "nested_cosine",
) -> NDArray[np.float64]:
    """
    NACA 4-digit cambered profile on the thin-wall airfoil contract.

    Returns ``(2n, 2)``: upper LE→TE, then lower LE→TE on identical chord
    stations. Trailing edge is forced sharp at ``(1, y_c(1))``.
    """
    x = chordwise_stations(n, spacing=chord_spacing)
    yt = 5 * t_c * (
        0.2969 * np.sqrt(np.maximum(x, 0.0))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )
    yc = np.zeros_like(x)
    if m > 1e-15 and p > 1e-9 and p < 1.0 - 1e-9:
        mask = x < p
        yc[mask] = (m / p**2) * (2 * p * x[mask] - x[mask] ** 2)
        yc[~mask] = (m / (1 - p) ** 2) * (
            (1 - 2 * p) + 2 * p * x[~mask] - x[~mask] ** 2
        )
    z_te = float(yc[-1])
    upper = np.column_stack([x, yc + yt])
    lower = np.column_stack([x, yc - yt])
    upper[-1, :] = (1.0, z_te)
    lower[-1, :] = (1.0, z_te)
    return np.vstack([upper, lower])


__all__ = ["chordwise_stations", "naca_four_digit"]
