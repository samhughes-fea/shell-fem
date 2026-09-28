"""Skin node-pair helpers and Channel B apply refuse (FEA-L-003).

Channel B face traction is applied only via a labelled
``SectionForceVector`` → ``section_force_vector_to_nodal_load`` (FEA-L-001 /
SL-014). The former nearest-YZ ``traction_face_loads`` path is refused.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.analysis_dof_layout import (
    U_S,
    W,
    gdof,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.panel_node_assembly import normal_2d


def _skin_node_pairs(node_meta: Any) -> tuple[NDArray[np.float64], NDArray[np.float64], list[tuple[int, int]]]:
    """Collect physical node-pairs from ``node_meta`` by ``pair_group``.

    Returns
    -------
    locs : (n_pair, 2) midline (y, z) of each pair
    tangents : (n_pair, 2) unit contour tangent at each pair
    gn_pairs : list of (gn_bottom, gn_top) global node ids
    """
    yz = np.asarray(node_meta.yz, dtype=float)
    is_top = np.asarray(node_meta.is_top, dtype=bool)
    tan = np.asarray(node_meta.tangent_yz, dtype=float)
    pair_group = np.asarray(node_meta.pair_group, dtype=np.int64)

    groups: dict[int, dict[str, int]] = {}
    for gn in range(node_meta.n_nodes):
        gid = int(pair_group[gn])
        if gid < 0 or not np.isfinite(yz[gn, 0]):
            continue
        slot = "top" if bool(is_top[gn]) else "bottom"
        groups.setdefault(gid, {})[slot] = gn

    locs: list[list[float]] = []
    tangents: list[list[float]] = []
    gn_pairs: list[tuple[int, int]] = []
    for gid in sorted(groups):
        g = groups[gid]
        if "bottom" not in g or "top" not in g:
            continue
        gb = g["bottom"]
        t = tan[gb]
        if not np.all(np.isfinite(t)) or float(np.hypot(t[0], t[1])) < 1e-12:
            t = np.array([1.0, 0.0])
        else:
            t = t / float(np.hypot(t[0], t[1]))
        locs.append([float(yz[gb, 0]), float(yz[gb, 1])])
        tangents.append([float(t[0]), float(t[1])])
        gn_pairs.append((int(gb), int(g["top"])))

    return (
        np.asarray(locs, dtype=float).reshape(-1, 2),
        np.asarray(tangents, dtype=float).reshape(-1, 2),
        gn_pairs,
    )


def traction_face_loads(
    node_meta: Any,
    force_yz: NDArray[np.float64],
    point_yz: NDArray[np.float64],
    *,
    ndof: int,
    L_x: float = 1.0,
) -> NDArray[np.float64]:
    """Refuse proximity placement of unlabelled TractionField panels (FEA-L-003).

    Channel B must be shaped outside FE to a labelled ``SectionForceVector``
    (SL-014) and applied with ``section_force_vector_to_nodal_load`` (FEA-L-001).
    """
    raise ValueError(
        "traction_face_loads is refused (FEA-L-003): do not place unlabelled "
        "TractionField panels by nearest skin node-pair. Shape Channel B outside "
        "FE via section_loads.traction_field_to_section_force_vector / "
        "distribute_loads(channel='surface_traction') on a from_skin_midlines "
        "outline, then apply the labelled SectionForceVector with "
        "section_force_vector_to_nodal_load (FEA-L-001)."
    )


def applied_force_resultant_yz(
    node_meta: Any,
    f_global: NDArray[np.float64],
    *,
    ndof: int,
) -> NDArray[np.float64]:
    """Reconstruct the net applied (y, z) force from a nodal load vector.

    Inverse of the local ``(u_s, w)`` resolution used by FEA-L-001 consistent
    nodal loads — for static-consistency audits.
    """
    tan = np.asarray(node_meta.tangent_yz, dtype=float)
    total = np.zeros(2, dtype=float)
    for gn in range(node_meta.n_nodes):
        t = tan[gn]
        if not np.all(np.isfinite(t)) or float(np.hypot(t[0], t[1])) < 1e-12:
            continue
        t = t / float(np.hypot(t[0], t[1]))
        n = normal_2d(t)
        f_s = float(f_global[gdof(gn, U_S, ndof)])
        f_w = float(f_global[gdof(gn, W, ndof)])
        total += f_s * t + f_w * n
    return total


__all__ = ["traction_face_loads", "applied_force_resultant_yz"]
