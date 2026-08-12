"""Theory-agnostic spine for connected multi-panel shell assembly.

These helpers are parameterised by ``ndof`` (DOFs per node) and operate purely
on :class:`PanelGlobalMap` / cluster topology, so they are shared by any
connected solver that uses the same first-six DOF layout (production MITC4
``ndof=6``).  The first six per-node DOFs
``[u_x, u_s, w, beta_s, beta_x, beta_z]`` are curvilinear strip frame **C**
(≡ element **E** on each facet; see ``coordinate_frames``) and have identical
meaning across theories; any extra DOFs are handled by passing ``clamp_dofs`` /
constraint ties from the caller.

Production MITC4 / periodic assembly folds onto these helpers via
``connected_panel_topology.build_connected_panel_maps`` and
``section_assembly.assemble_section_from_zones``.

Shared DOF indices (first six)::

    U_X, U_S, W, BETA_S, BETA_X, BETA_ZET = 0, 1, 2, 3, 4, 5
"""

from __future__ import annotations

import numpy as np

from ....pre_processing.mesh.shell_mesh_map import PanelGlobalMap

U_X, U_S, W, BETA_S, BETA_X, BETA_ZET = 0, 1, 2, 3, 4, 5

# Constraint format: slave_dof -> [(master_dof, coeff), ...]; q_slave = sum coeff*q_master.
Constraints = dict[int, list[tuple[int, float]]]

# Contour-end strip supports (panel contour endpoints).
DEFAULT_BC_MODE = "contour_support"


def canonicalize_bc_mode(bc_mode: str) -> str:
    """Return the public ``bc_mode`` string (identity; kept for cache-key callers)."""
    return str(bc_mode)


def normalize_bc_mode(bc_mode: str) -> str:
    """Normalize public ``bc_mode`` strings (identity validation entry point)."""
    return str(bc_mode)


def is_contour_support(bc_mode: str) -> bool:
    """True when contour-end strip supports (both bottom+top layers) are requested."""
    return canonicalize_bc_mode(bc_mode) == "contour_support"


def gdof(node_idx: int, local: int, ndof: int) -> int:
    """Global DOF index for ``local`` DOF of node ``node_idx``."""
    return node_idx * ndof + local


def assign_global_nodes_shared(
    panel_maps: list[PanelGlobalMap],
    clusters: list[list[tuple[int, str, np.ndarray]]],
) -> int:
    """Shared-node numbering: coincident panel endpoints share global nodes.

    Each junction cluster gets one bottom + one top global node shared by all
    panels meeting there, giving automatic kinematic compatibility (no MPCs).
    Interior nodes are panel-unique.  Mutates ``panel_maps[*].global_nodes``.

    Returns the total number of global nodes allocated.
    """
    next_node = 0
    endpoint_global_node: dict[tuple[int, str, str], int] = {}
    for c in clusters:
        g_bottom = next_node
        g_top = next_node + 1
        next_node += 2
        for pi, which, _ in c:
            endpoint_global_node[(pi, which, "bottom")] = g_bottom
            endpoint_global_node[(pi, which, "top")] = g_top
    for pm in panel_maps:
        if len(pm.s_nodes) == 0:
            continue
        n = len(pm.s_nodes)
        pm.global_nodes[0] = endpoint_global_node[(pm.panel_index, "start", "bottom")]
        pm.global_nodes[n - 1] = endpoint_global_node[(pm.panel_index, "end", "bottom")]
        pm.global_nodes[n] = endpoint_global_node[(pm.panel_index, "start", "top")]
        pm.global_nodes[2 * n - 1] = endpoint_global_node[(pm.panel_index, "end", "top")]
        for li in range(1, n - 1):
            pm.global_nodes[li] = next_node
            next_node += 1
        for li in range(n + 1, 2 * n - 1):
            pm.global_nodes[li] = next_node
            next_node += 1
    return next_node


def assign_global_nodes_independent(panel_maps: list[PanelGlobalMap]) -> int:
    """Unshared numbering: every panel node is unique (no junction coupling)."""
    next_node = 0
    for pm in panel_maps:
        if len(pm.s_nodes) == 0:
            continue
        for li in range(2 * len(pm.s_nodes)):
            pm.global_nodes[li] = next_node
            next_node += 1
    return next_node


def assign_global_nodes_cluster_basis(
    panel_maps: list[PanelGlobalMap],
    clusters: list[list[tuple[int, str, np.ndarray]]],
) -> dict[str, object]:
    """Cluster-basis (transformed_basis) numbering for non-collinear junctions.

    Each junction cluster-layer gets a **main** reference node and a **rot** node
    (the rot node carries the normal-direction rotation only); panel endpoint nodes
    stay unique and are tied to the cluster reference DOFs by MPCs
    (:func:`build_cluster_basis_constraints`). Mutates ``panel_maps`` global ids.

    Returns dict with ``next_node`` and the cluster/endpoint node maps.
    """
    cluster_layer_node: dict[tuple[int, str], int] = {}
    cluster_layer_rot_node: dict[tuple[int, str], int] = {}
    endpoint_cluster_node: dict[tuple[int, str, str], int] = {}
    endpoint_cluster_rot_node: dict[tuple[int, str, str], int] = {}
    next_node = 0
    for cid, c in enumerate(clusters):
        for layer in ("bottom", "top"):
            cluster_layer_node[(cid, layer)] = next_node
            next_node += 1
            cluster_layer_rot_node[(cid, layer)] = next_node
            next_node += 1
        for pi, which, _ in c:
            for layer in ("bottom", "top"):
                endpoint_cluster_node[(pi, which, layer)] = cluster_layer_node[(cid, layer)]
                endpoint_cluster_rot_node[(pi, which, layer)] = cluster_layer_rot_node[(cid, layer)]
    for pm in panel_maps:
        if len(pm.s_nodes) == 0:
            continue
        for li in range(2 * len(pm.s_nodes)):
            pm.global_nodes[li] = next_node
            next_node += 1
    return {
        "next_node": next_node,
        "cluster_layer_node": cluster_layer_node,
        "cluster_layer_rot_node": cluster_layer_rot_node,
        "endpoint_cluster_node": endpoint_cluster_node,
        "endpoint_cluster_rot_node": endpoint_cluster_rot_node,
        "endpoint_cluster_index": {
            (pi, which): cid for cid, c in enumerate(clusters) for pi, which, _ in c
        },
    }


def build_cluster_basis_constraints(
    panel_maps: list[PanelGlobalMap],
    zones: list,
    clusters: list[list[tuple[int, str, np.ndarray]]],
    endpoint_cluster_node: dict[tuple[int, str, str], int],
    endpoint_cluster_rot_node: dict[tuple[int, str, str], int],
    cluster_layer_node: dict[tuple[int, str], int],
    cluster_layer_rot_node: dict[tuple[int, str], int],
    *,
    ndof: int,
    identity_extra_dofs: tuple[int, ...] = (),
) -> Constraints:
    """2-D kinematic compatibility MPCs against cluster reference DOFs.

    Ports the MITC4 cluster-basis constraints (``global_shell_assembly``) to any
    ``ndof``: u_x / beta_x / beta_z (and ``identity_extra_dofs``, e.g. HSDT
    gamma_x/gamma_s) are tied identity to the cluster **main** node; the in-plane
    (u_s, w) and tangential rotation beta_s are mapped by the 2-D rotation between
    each panel tangent and the cluster reference tangent.
    """
    from ....pre_processing.mesh.shell_mesh_map import normal_2d, panel_end_tangent

    constraints: Constraints = {}
    for cluster in clusters:
        if len(cluster) < 1:
            continue
        t_mats: list[np.ndarray] = []
        for pi, which, _ in cluster:
            t = panel_end_tangent(np.asarray(panels[pi].nodes, dtype=float), which)
            t_mats.append(np.outer(t, t))
        M = np.sum(np.stack(t_mats, axis=0), axis=0)
        evals, evecs = np.linalg.eigh(M)
        t_ref = np.asarray(evecs[:, int(np.argmax(evals))], dtype=float)
        if float(np.linalg.norm(t_ref)) < 1e-12:
            t_ref = np.array([1.0, 0.0])
        if float(t_ref[0]) < 0.0:
            t_ref = -t_ref
        t_ref = t_ref / max(float(np.linalg.norm(t_ref)), 1e-12)
        n_ref = normal_2d(t_ref)

        for pi, end, _ in cluster:
            pm = panel_maps[pi]
            n = len(pm.s_nodes)
            if n == 0:
                continue
            row = 0 if end == "start" else n - 1
            t_p = panel_end_tangent(np.asarray(panels[pi].nodes, dtype=float), end)
            n_p = normal_2d(t_p)
            ts, tn = float(np.dot(t_p, t_ref)), float(np.dot(t_p, n_ref))
            ns, nn = float(np.dot(n_p, t_ref)), float(np.dot(n_p, n_ref))
            for off, layer in ((row, "bottom"), (row + n, "top")):
                gn_p = pm.global_nodes[off]
                gn_m = endpoint_cluster_node[(pi, end, layer)]
                gn_r = endpoint_cluster_rot_node[(pi, end, layer)]
                constraints[gdof(gn_p, U_X, ndof)] = [(gdof(gn_m, U_X, ndof), 1.0)]
                us = [(gdof(gn_m, U_S, ndof), ts)]
                if abs(tn) > 1e-12:
                    us.append((gdof(gn_m, W, ndof), tn))
                constraints[gdof(gn_p, U_S, ndof)] = us
                w = [(gdof(gn_m, U_S, ndof), ns)]
                if abs(nn) > 1e-12:
                    w.append((gdof(gn_m, W, ndof), nn))
                constraints[gdof(gn_p, W, ndof)] = w
                constraints[gdof(gn_p, BETA_X, ndof)] = [(gdof(gn_m, BETA_X, ndof), 1.0)]
                bs = [(gdof(gn_m, BETA_S, ndof), ts)]
                if abs(tn) > 1e-12:
                    bs.append((gdof(gn_r, BETA_S, ndof), tn))
                constraints[gdof(gn_p, BETA_S, ndof)] = bs
                constraints[gdof(gn_p, BETA_ZET, ndof)] = [(gdof(gn_m, BETA_ZET, ndof), 1.0)]
                for d in identity_extra_dofs:
                    constraints[gdof(gn_p, d, ndof)] = [(gdof(gn_m, d, ndof), 1.0)]

    # Tie each rot node's non-BETA_S DOFs to its main node (rot carries beta_s(n_ref) only).
    for key, gn_rot in cluster_layer_rot_node.items():
        gn_main = cluster_layer_node[key]
        for d in (U_X, U_S, W, BETA_X, BETA_ZET, *identity_extra_dofs):
            constraints[gdof(gn_rot, d, ndof)] = [(gdof(gn_main, d, ndof), 1.0)]
    return constraints


def build_cluster_basis_supports(
    panel_maps: list[PanelGlobalMap],
    endpoint_cluster_index: dict[tuple[int, str], int],
    cluster_layer_node: dict[tuple[int, str], int],
    cluster_layer_rot_node: dict[tuple[int, str], int],
    *,
    ndof: int,
    bc_mode: str = DEFAULT_BC_MODE,
    clamp_dofs: tuple[int, ...] = (W, BETA_S, BETA_X),
) -> set[int]:
    """Fixed-DOF set for cluster-basis numbering: RBM pin + clamped cluster refs."""
    mode = normalize_bc_mode(bc_mode)
    fixed: set[int] = set()
    fixed.add(gdof(0, U_X, ndof))
    fixed.add(gdof(0, U_S, ndof))
    fixed.add(gdof(1, U_X, ndof))
    done: set[tuple[int, str]] = set()
    for pm in panel_maps:
        if len(pm.s_nodes) == 0:
            continue
        n = len(pm.s_nodes)
        support_local = (0, n - 1, n, 2 * n - 1) if mode == "contour_support" else (0, n - 1)
        for ln in support_local:
            which = "start" if ln in (0, n) else "end"
            layer = "bottom" if ln in (0, n - 1) else "top"
            cid = endpoint_cluster_index.get((pm.panel_index, which), -1)
            if cid < 0 or (cid, layer) in done:
                continue
            done.add((cid, layer))
            gn_m = cluster_layer_node[(cid, layer)]
            gn_r = cluster_layer_rot_node[(cid, layer)]
            for d in clamp_dofs:
                fixed.add(gdof(gn_m, d, ndof))
            fixed.add(gdof(gn_r, BETA_S, ndof))
    return fixed


def attach_cluster_node_meta(
    node_meta: Any,
    clusters: list[list[tuple[int, str, np.ndarray]]],
    zones: list,
    cluster_layer_node: dict[tuple[int, str], int],
    cluster_layer_rot_node: dict[tuple[int, str], int],
    constraints: Constraints,
) -> None:
    """Populate geometry for virtual cluster reference nodes and stash the MPCs.

    Element-agnostic (operates purely on geometry/topology). Gives each junction's
    cluster **main** and **rot** reference nodes a real ``(y,z)`` at the junction
    centroid, the cluster reference tangent, an ``is_top`` flag (bottom layer = the
    inboard ``x=0`` face, top layer = the outboard ``x=L`` face), and a periodic
    ``pair_group`` id so :func:`homogenisation._build_node_pairs_by_group` pairs the
    inboard/outboard twins. The cluster MPCs are attached to ``node_meta.constraints``
    so the periodic homogeniser can compose them with periodic pairing.

    Main and rot nodes get distinct pair-group ids (one bottom + one top each).
    Mutates ``node_meta`` in place.
    """
    from ....pre_processing.mesh.shell_mesh_map import panel_end_tangent

    yz = node_meta.yz
    is_top = node_meta.is_top
    tangent = node_meta.tangent_yz
    pair_group = node_meta.pair_group
    base_group = (int(pair_group.max()) + 1) if pair_group.size else 0

    for cid, cluster in enumerate(clusters):
        if not cluster:
            continue
        coords = np.array([np.asarray(coord, dtype=float) for _, _, coord in cluster])
        centroid = coords.mean(axis=0)
        # Cluster reference tangent: dominant eigenvector of sum_j t_j t_j^T (matches
        # build_cluster_basis_constraints so the rotated basis is consistent).
        t_mats = [
            np.outer(t, t)
            for pi, which, _ in cluster
            for t in (panel_end_tangent(np.asarray(panels[pi].nodes, dtype=float), which),)
        ]
        M = np.sum(np.stack(t_mats, axis=0), axis=0)
        evals, evecs = np.linalg.eigh(M)
        t_ref = np.asarray(evecs[:, int(np.argmax(evals))], dtype=float)
        if float(np.linalg.norm(t_ref)) < 1e-12:
            t_ref = np.array([1.0, 0.0])
        if float(t_ref[0]) < 0.0:
            t_ref = -t_ref
        t_ref = t_ref / max(float(np.linalg.norm(t_ref)), 1e-12)

        for role, gmap in enumerate((cluster_layer_node, cluster_layer_rot_node)):
            grp = base_group + 2 * cid + role
            for layer, top in (("bottom", False), ("top", True)):
                gn = gmap.get((cid, layer))
                if gn is None:
                    continue
                yz[gn] = centroid
                tangent[gn] = t_ref
                is_top[gn] = top
                pair_group[gn] = grp

    node_meta.constraints = dict(constraints)


def build_clamp_supports(
    panel_maps: list[PanelGlobalMap],
    *,
    ndof: int,
    bc_mode: str = DEFAULT_BC_MODE,
    clamp_dofs: tuple[int, ...] = (W, BETA_S, BETA_X),
    add_rbm_pin: bool = True,
) -> set[int]:
    """Build the fixed-DOF set for shared/independent node numbering.

    Clamps ``clamp_dofs`` at panel support endpoints (both bottom+top layers
    when ``contour_support``), plus a minimal rigid-body pin (u_x at node 0
    and 1, u_s at node 0) to remove the in-plane translation/rotation null
    space.
    """
    mode = normalize_bc_mode(bc_mode)
    fixed: set[int] = set()
    if add_rbm_pin:
        fixed.add(gdof(0, U_X, ndof))
        fixed.add(gdof(0, U_S, ndof))
        fixed.add(gdof(1, U_X, ndof))
    for pm in panel_maps:
        if len(pm.s_nodes) == 0:
            continue
        n = len(pm.s_nodes)
        support_nodes = (0, n - 1, n, 2 * n - 1) if mode == "contour_support" else (0, n - 1)
        for ln in support_nodes:
            gn = pm.global_nodes[ln]
            for dof_type in clamp_dofs:
                fixed.add(gdof(gn, dof_type, ndof))
    return fixed


__all__ = [
    "U_X", "U_S", "W", "BETA_S", "BETA_X", "BETA_ZET",
    "Constraints",
    "DEFAULT_BC_MODE",
    "gdof",
    "canonicalize_bc_mode",
    "normalize_bc_mode",
    "is_contour_support",
    "assign_global_nodes_shared",
    "assign_global_nodes_independent",
    "assign_global_nodes_cluster_basis",
    "build_cluster_basis_constraints",
    "build_cluster_basis_supports",
    "attach_cluster_node_meta",
    "build_clamp_supports",
]
