"""Kernel-agnostic strip/panel mesh topology primitives (pre_processing/mesh layer).

These are pure element-count and geometry helpers with no ``engine`` or
``element_library`` dependencies, so that mesh construction
(:mod:`.section_strip_mesh`, :mod:`.shell_mesh_map`) does not have to import
from ``engine``.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Union

import numpy as np

# Per-panel element-count spec accepted by the mesh builder and global solvers:
# a scalar (applied to every panel), a mapping/sequence keyed by panel index, or
# ``None`` (use the caller default).
NElements = Union[int, Mapping[int, int], Sequence[int], None]


def panel_endpoints(nodes_yz: np.ndarray) -> dict[str, np.ndarray]:
    """Return the start/end node coordinates of a panel midline polyline."""
    return {
        "start": np.asarray(nodes_yz[0], dtype=float),
        "end": np.asarray(nodes_yz[-1], dtype=float),
    }


def cluster_points(
    points: list[tuple[int, str, np.ndarray]], tol: float
) -> list[list[tuple[int, str, np.ndarray]]]:
    """Group ``(panel_index, which, yz)`` endpoints into junction clusters by proximity."""
    clusters: list[list[tuple[int, str, np.ndarray]]] = []
    for p in points:
        placed = False
        for c in clusters:
            if float(np.linalg.norm(p[2] - c[0][2])) <= tol:
                c.append(p)
                placed = True
                break
        if not placed:
            clusters.append([p])
    return clusters


__all__ = ["NElements", "cluster_points", "panel_endpoints"]
