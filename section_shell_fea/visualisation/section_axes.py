"""Section-plane plot contract for FE-worker figures.

Thin FE-side wrappers over the canonical helpers in
``geometry_utilities.coordinate_frames``. Section figures draw **horizontal =**
:math:`Y_B` (edgewise) and **vertical =** :math:`X_B` (flapwise), then invert the
horizontal axis so the leading edge reads on the right.

Plotting a canonical-B array as ``ax.plot(pts[:, 0], pts[:, 1])`` renders the
section **rotated 90°** — column 0 is the flapwise axis, not the chord. Route
coordinates through :func:`section_hv` (or the helpers below) and finish with
:func:`finish_section_axes`.

The inversion is display-only: stored arrays keep :math:`+Y_B` toward the TE
(``Y_LE < 0 < Y_TE``) and are never negated to fake the flip.

Chord-frame (NACA adapter) diagnostics such as the XFOIL running-load plots are
deliberately **not** section figures and do not use this contract.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.geometry_utilities.coordinate_frames import (
    SECTION_PLOT_XLABEL,
    SECTION_PLOT_YLABEL,
    fit_square_section_axes,
    section_plot_axes,
    setup_section_axes,
)


def section_hv(pts_B) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Canonical-B ``(N, 2)`` → ``(horizontal, vertical)`` = ``(Y_B, X_B)``."""
    return section_plot_axes(np.asarray(pts_B, dtype=np.float64).reshape(-1, 2))


def section_point_hv(x_B: float, y_B: float) -> tuple[float, float]:
    """One canonical-B point → ``(horizontal, vertical)`` for ``ax.text`` / markers."""
    return float(y_B), float(x_B)


def section_rings_hv(rings) -> list[NDArray[np.float64]]:
    """Remap canonical-B vertex rings for ``PolyCollection`` / ``fill``."""
    return [
        np.asarray(r, dtype=np.float64).reshape(-1, 2)[:, ::-1] for r in rings
    ]


def finish_section_axes(
    ax,
    *,
    square: bool = True,
    label_color: str | None = None,
    **kwargs,
) -> None:
    """Apply canon labels + LE-on-RHS invert, then square the aerofoil window.

    ``kwargs`` pass through to
    :func:`coordinate_frames.setup_section_axes` (``xlabel`` / ``ylabel`` /
    ``invert_for_le_on_rhs`` / ``equal_aspect``). ``label_color`` re-tints the
    axis labels for the dark-theme figures without duplicating the label strings.
    """
    setup_section_axes(ax, **kwargs)
    if label_color is not None:
        ax.xaxis.label.set_color(label_color)
        ax.yaxis.label.set_color(label_color)
    if square:
        fit_square_section_axes(ax)


__all__ = [
    "SECTION_PLOT_XLABEL",
    "SECTION_PLOT_YLABEL",
    "finish_section_axes",
    "section_hv",
    "section_point_hv",
    "section_rings_hv",
]
