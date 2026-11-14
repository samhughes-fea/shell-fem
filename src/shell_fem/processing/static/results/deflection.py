"""Deflection profile of a static solution along one mesh axis.

The static solve returns a displacement at every node. A limit-state check wants
that motion collected by station: the mean translation and rotation of each
cross-section, and the rigid twist of the section about the profile axis. For a
blade the axis is the span, z.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from shell_fem.processing.static.results.containers import SolutionResult

_AXIS = ("x", "y", "z")
_PLANE = ((1, 2), (2, 0), (0, 1))


@dataclass(frozen=True)
class DeflectionProfile:
    """Section motion along one coordinate of a static solution.

    ``stations`` are the coordinate values on ``axis`` (0=x, 1=y, 2=z).
    ``translation`` and ``rotation`` are the mean nodal motion at each station,
    in global axes, shape (n_stations, 3). ``twist`` is the rigid rotation of the
    section about ``axis``, in radians, from the translations about the section
    centroid.
    """

    axis: int
    stations: np.ndarray
    translation: np.ndarray
    rotation: np.ndarray
    twist: np.ndarray


def deflection_profile(
    solution: SolutionResult,
    axis: int = 2,
    stations: Optional[Sequence[float]] = None,
    tol: float = 1e-9,
) -> DeflectionProfile:
    """Section-mean deflection along ``axis`` of one static solution.

    Stations default to the distinct node coordinates on that axis. Pass the
    blade's span positions to read the mesh sections. ``axis`` 2 is the span.
    """
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    mesh = solution.mesh
    xyz = mesh.coordinates_array()
    motion = solution.displacement_array()
    coord = xyz[:, axis]
    if stations is None:
        ordered = np.sort(coord)
        if ordered.size == 0:
            raise ValueError("the mesh has no nodes")
        picked = [float(ordered[0])]
        for value in ordered[1:]:
            if float(value) - picked[-1] > tol:
                picked.append(float(value))
        station_values = np.asarray(picked, dtype=float)
    else:
        station_values = np.asarray(list(stations), dtype=float).ravel()
        if station_values.size == 0:
            raise ValueError("stations must be a non-empty sequence of coordinates")
    n = station_values.size
    translation = np.zeros((n, 3))
    rotation = np.zeros((n, 3))
    twist = np.zeros(n)
    a, b = _PLANE[axis]
    for i, station in enumerate(station_values):
        mask = np.abs(coord - station) <= tol
        if not np.any(mask):
            raise ValueError(f"no nodes at {_AXIS[axis]}={station}")
        u = motion[mask]
        x = xyz[mask]
        translation[i] = u[:, :3].mean(axis=0)
        rotation[i] = u[:, 3:6].mean(axis=0)
        r = np.column_stack([x[:, a] - x[:, a].mean(), x[:, b] - x[:, b].mean()])
        du = np.column_stack([u[:, a] - u[:, a].mean(), u[:, b] - u[:, b].mean()])
        denom = float(np.sum(r * r))
        twist[i] = float(np.sum(r[:, 0] * du[:, 1] - r[:, 1] * du[:, 0]) / denom) if denom > 0.0 else 0.0
    return DeflectionProfile(axis, station_values, translation, rotation, twist)
