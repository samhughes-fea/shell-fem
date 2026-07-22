"""Coordinate-frame transforms for section outline geometry."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from blade_utilities.point_to_surftraction.geometry import SectionOutlineGeometry
from blade_utilities.point_to_surftraction.tractions import TractionField


def bframe_to_sframe_rotation(twist_rad: float) -> NDArray[np.float64]:
    """Return the 2×2 matrix mapping B-frame row vectors to the S-frame."""
    c, s = float(np.cos(twist_rad)), float(np.sin(twist_rad))
    return np.array([[c, -s], [s, c]], dtype=float)


def rotate_traction_field(traction: TractionField, rotation: NDArray[np.float64]) -> TractionField:
    """Rotate all panel force vectors by ``rotation`` (B→S when built from ``bframe_to_sframe_rotation``)."""
    rot = np.asarray(rotation, dtype=float)
    return TractionField(
        pressure_forces=traction.pressure_forces @ rot,
        shear_forces=traction.shear_forces @ rot,
        total_forces=traction.total_forces @ rot,
    )


def bframe_outline_to_sframe_chord(
    geometry: SectionOutlineGeometry,
    twist_rad: float,
    *,
    te_first: bool = False,
) -> tuple[SectionOutlineGeometry, NDArray[np.float64]]:
    """Untwist and normalize a B-frame outline to unit-chord S-frame coordinates.

    Returns the transformed geometry and the B→S rotation matrix for force vectors.
    When ``te_first=True``, roll the ring so the trailing edge is the first vertex
    (XFOIL convention).
    """
    ring = np.asarray(geometry.coordinates[:-1], dtype=float)
    rot = bframe_to_sframe_rotation(twist_rad)
    pts = ring @ rot
    i_te = int(np.argmax(pts[:, 0]))
    i_le = int(np.argmin(pts[:, 0]))
    chord_x = float(pts[i_te, 0] - pts[i_le, 0])
    if chord_x <= 0.0:
        raise ValueError("Degenerate chord after untwisting section outline.")
    pts = (pts - pts[i_le]) / chord_x
    if te_first:
        pts = np.roll(pts, -i_te, axis=0)
    sframe = SectionOutlineGeometry.from_closed_contour(pts, ensure_clockwise=False)
    return sframe, rot


def xfoil_te_arc_offset(geometry: SectionOutlineGeometry, twist_rad: float) -> float:
    """Normalized arc position of the TE vertex on the original B-frame outline."""
    ring = np.asarray(geometry.coordinates[:-1], dtype=float)
    rot = bframe_to_sframe_rotation(twist_rad)
    pts = ring @ rot
    i_te = int(np.argmax(pts[:, 0]))
    edge_arc = np.concatenate([[0.0], np.cumsum(geometry.panel_lengths)])
    return float(edge_arc[i_te] / geometry.perimeter)
