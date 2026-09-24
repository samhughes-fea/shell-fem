"""Direct aerodynamic traction -> skin element face loads (Phase C1).

Maps a ``section_loads`` surface :class:`TractionField` (per-panel force
``(Fy, Fz)`` in N/m on the section outline, already scaled to the external
``q_y, q_z`` line loads via the LoadMatcher) **directly** onto the connected
shell mesh as consistent nodal loads — instead of collapsing it to an
integrated six-component resultant.

Each outline-panel force is placed on the nearest skin mesh node-pair
(bottom+top through-thickness layers share the same midline ``(y, z)`` in **B**),
split equally between the two layers (a net mid-surface translational load,
no artificial through-thickness moment), and resolved into the node's
curvilinear strip frame **C** (≡ **E** on each facet): the tangential
component drives ``u_s`` and the outward-normal component drives ``w``.
``u_x`` (spanwise) receives nothing — in-plane aero loads have no span
component on a unit slice.

Total applied force is conserved exactly: each panel force is assigned to one
node-pair, so ``sum(applied (y,z) nodal forces) == sum(total_forces)``.  This
is the static-consistency audit that ties the FEM load back to the
LoadMatcher-calibrated ``(q_y, q_z)``.

The resulting global force vector superposes on the resultant-driven
kinematic-BC state by linearity (the combined solve lives in Phase C3).
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
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.shell_mesh_map import normal_2d


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
    """Assemble the global nodal force vector from a surface traction field.

    Parameters
    ----------
    node_meta
        ``GlobalNodeMeta`` from the connected assembly (carries node ``yz``,
        ``tangent_yz``, ``is_top``, ``pair_group``).
    force_yz : (n, 2)
        Per-panel traction force ``(Fy, Fz)`` in N/m (e.g. ``TractionField.total_forces``).
    point_yz : (n, 2)
        Location of each panel force in the section (y, z) frame (panel centres).
    ndof
        Per-node DOF count of the assembled system (8 for HSDT, 6 for MITC4+).
    L_x
        Slice span length [m]; scales N/m -> N for the unit cell.

    Returns
    -------
    f_global : (n_nodes * ndof,) consistent nodal load vector.
    """
    force_yz = np.asarray(force_yz, dtype=float).reshape(-1, 2)
    point_yz = np.asarray(point_yz, dtype=float).reshape(-1, 2)
    if force_yz.shape[0] != point_yz.shape[0]:
        raise ValueError("force_yz and point_yz must have the same number of rows.")

    locs, tangents, gn_pairs = _skin_node_pairs(node_meta)
    n_gdof = node_meta.n_nodes * ndof
    f = np.zeros(n_gdof, dtype=float)
    if locs.shape[0] == 0:
        return f

    for F, c in zip(force_yz, point_yz):
        d2 = np.sum((locs - c[None, :]) ** 2, axis=1)
        k = int(np.argmin(d2))
        t = tangents[k]
        n = normal_2d(t)
        f_s = float(F[0] * t[0] + F[1] * t[1]) * L_x   # tangential -> u_s
        f_w = float(F[0] * n[0] + F[1] * n[1]) * L_x   # outward-normal -> w
        gb, gt = gn_pairs[k]
        for gn in (gb, gt):
            f[gdof(gn, U_S, ndof)] += 0.5 * f_s
            f[gdof(gn, W, ndof)] += 0.5 * f_w
    return f


def applied_force_resultant_yz(
    node_meta: Any,
    f_global: NDArray[np.float64],
    *,
    ndof: int,
) -> NDArray[np.float64]:
    """Reconstruct the net applied (y, z) force from a nodal load vector.

    Inverse of :func:`traction_face_loads`'s local resolution — used by the
    static-consistency audit to confirm the FEM load equals the input
    ``sum(total_forces)``.
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
