"""Global node assignment and cluster topology helpers.

These helpers are parameterised by ``ndof`` and operate purely on
:class:`PanelGlobalMap` / cluster topology. They do not construct MPCs or
supports; constraint ownership lives in ``processing.static.constraints``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.analysis_dof_layout import (
    BETA_S,
    U_S,
    W,
    gdof,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.topology.shell_mesh_map import (
    PanelGlobalMap,
)

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
    stay unique and are tied to the cluster reference DOFs by MPCs. Mutates
    ``panel_maps`` global ids.

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


def _kind_from_panel_label(label: object) -> str:
    text = str(label or "")
    return text.split(":", 1)[0] if ":" in text else (text or "panel")


def cluster_is_kind_crossing(
    cluster: list[tuple[int, str, np.ndarray]],
    panel_maps: list[PanelGlobalMap],
) -> bool:
    """True when a junction cluster joins ≥2 distinct component kinds."""
    kinds: set[str] = set()
    for pi, _, _ in cluster:
        if 0 <= int(pi) < len(panel_maps):
            kinds.add(_kind_from_panel_label(getattr(panel_maps[int(pi)], "panel_label", "")))
        else:
            kinds.add("panel")
    return len(kinds) >= 2


def assign_global_nodes_hybrid_tb_clusters(
    panel_maps: list[PanelGlobalMap],
    clusters: list[list[tuple[int, str, np.ndarray]]],
    use_tb_flags: dict[int, bool],
) -> dict[str, object]:
    """Shared nodes except on clusters flagged for transformed-basis kinematics.

    Study hybrid numbering: flagged clusters get virtual cluster refs + unique
    panel endpoints (TB MPCs attached later); others share endpoint nodes like
    production ``shared``. Mutates ``panel_maps`` global ids.
    """
    cluster_layer_node: dict[tuple[int, str], int] = {}
    cluster_layer_rot_node: dict[tuple[int, str], int] = {}
    endpoint_cluster_node: dict[tuple[int, str, str], int] = {}
    endpoint_cluster_rot_node: dict[tuple[int, str, str], int] = {}
    shared_endpoint: dict[tuple[int, str, str], int] = {}
    next_node = 0
    for cid, c in enumerate(clusters):
        if not bool(use_tb_flags.get(cid, False)):
            g_bottom = next_node
            g_top = next_node + 1
            next_node += 2
            for pi, which, _ in c:
                shared_endpoint[(pi, which, "bottom")] = g_bottom
                shared_endpoint[(pi, which, "top")] = g_top
            continue
        for layer in ("bottom", "top"):
            cluster_layer_node[(cid, layer)] = next_node
            next_node += 1
            cluster_layer_rot_node[(cid, layer)] = next_node
            next_node += 1
        for pi, which, _ in c:
            for layer in ("bottom", "top"):
                endpoint_cluster_node[(pi, which, layer)] = cluster_layer_node[(cid, layer)]
                endpoint_cluster_rot_node[(pi, which, layer)] = cluster_layer_rot_node[
                    (cid, layer)
                ]
    for pm in panel_maps:
        if len(pm.s_nodes) == 0:
            continue
        n = len(pm.s_nodes)
        for local, which, layer in (
            (0, "start", "bottom"),
            (n - 1, "end", "bottom"),
            (n, "start", "top"),
            (2 * n - 1, "end", "top"),
        ):
            key = (pm.panel_index, which, layer)
            if key in shared_endpoint:
                pm.global_nodes[local] = shared_endpoint[key]
            else:
                pm.global_nodes[local] = next_node
                next_node += 1
        for li in range(1, n - 1):
            pm.global_nodes[li] = next_node
            next_node += 1
        for li in range(n + 1, 2 * n - 1):
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


def assign_global_nodes_hybrid_collinear(
    panel_maps: list[PanelGlobalMap],
    clusters: list[list[tuple[int, str, np.ndarray]]],
    cluster_is_collinear_flags: dict[int, bool],
) -> dict[str, object]:
    """Shared nodes on collinear clusters; cluster-basis on non-collinear.

    Study-only ``hybrid_collinear`` numbering: collinear junctions keep production
    shared-node kinematics; cusps / T-joints get transformed-basis virtual refs +
    MPCs. Mutates ``panel_maps`` global ids.
    """
    use_tb = {
        cid: not bool(cluster_is_collinear_flags.get(cid, False))
        for cid in range(len(clusters))
    }
    return assign_global_nodes_hybrid_tb_clusters(panel_maps, clusters, use_tb)


def assign_global_nodes_hybrid_kind_crossing(
    panel_maps: list[PanelGlobalMap],
    clusters: list[list[tuple[int, str, np.ndarray]]],
    cluster_is_collinear_flags: dict[int, bool],
) -> dict[str, object]:
    """Shared everywhere except kind-crossing ∩ non-collinear clusters (TB).

    Study-only ``hybrid_kind_crossing``: narrower than ``hybrid_collinear`` —
    same-kind non-collinear cusps stay shared; only multi-kind non-collinear
    joints get transformed-basis refs. Cap ties remain ``skin_offset``.
    """
    use_tb = {
        cid: (
            not bool(cluster_is_collinear_flags.get(cid, False))
            and cluster_is_kind_crossing(c, panel_maps)
        )
        for cid, c in enumerate(clusters)
    }
    return assign_global_nodes_hybrid_tb_clusters(panel_maps, clusters, use_tb)


def build_shared_cluster_frame_rotations(
    panel_maps: list[PanelGlobalMap],
    zones: list,
    clusters: list[list[tuple[int, str, np.ndarray]]],
    cluster_is_collinear_flags: dict[int, bool],
) -> dict[str, object]:
    """Panel→cluster 2-D frame rotations for kind-crossing ∩ non-collinear cusps.

    Study-only ``shared_cluster_frame``: numbering stays production ``shared``;
    these rotations are applied as Ke congruence transforms so endpoint DOFs are
    interpreted in the cluster ``t_ref`` frame without allocating TB virtual refs.
    """
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.topology.shell_mesh_map import (
        normal_2d,
        panel_end_tangent,
    )

    endpoint_frame_R: dict[tuple[int, str], tuple[float, float, float, float]] = {}
    cids: list[int] = []
    for cid, cluster in enumerate(clusters):
        if bool(cluster_is_collinear_flags.get(cid, False)):
            continue
        if not cluster_is_kind_crossing(cluster, panel_maps):
            continue
        if len(cluster) < 1:
            continue
        t_mats = [
            np.outer(
                panel_end_tangent(np.asarray(zones[pi].nodes, dtype=float), which),
                panel_end_tangent(np.asarray(zones[pi].nodes, dtype=float), which),
            )
            for pi, which, _ in cluster
        ]
        M = np.sum(np.stack(t_mats, axis=0), axis=0)
        evals, evecs = np.linalg.eigh(M)
        t_ref = np.asarray(evecs[:, int(np.argmax(evals))], dtype=float)
        if float(np.linalg.norm(t_ref)) < 1e-12:
            t_ref = np.array([1.0, 0.0])
        if float(t_ref[0]) < 0.0:
            t_ref = -t_ref
        t_ref = t_ref / max(float(np.linalg.norm(t_ref)), 1e-12)
        n_ref = normal_2d(t_ref)
        cids.append(int(cid))
        for pi, which, _ in cluster:
            t_p = panel_end_tangent(np.asarray(zones[pi].nodes, dtype=float), which)
            n_p = normal_2d(t_p)
            ts, tn = float(np.dot(t_p, t_ref)), float(np.dot(t_p, n_ref))
            ns, nn = float(np.dot(n_p, t_ref)), float(np.dot(n_p, n_ref))
            endpoint_frame_R[(int(pi), str(which))] = (ts, tn, ns, nn)
    return {"endpoint_frame_R": endpoint_frame_R, "cids": cids}


def apply_shared_cluster_frame_to_cache(
    cache: Any,
    panel_maps: list[PanelGlobalMap],
    endpoint_frame_R: dict[tuple[int, str], tuple[float, float, float, float]],
) -> int:
    """Congruence-transform Ke/Fe at flagged endpoints (panel → cluster frame).

    For each element corner that sits on a flagged ``(panel, start|end)``, the
    local ``(U_S, W)`` and ``BETA_S`` blocks are mapped by the 2-D rotation so
    shared global DOFs are interpreted in the cluster ``t_ref`` frame.
    Returns the number of element matrices touched.
    """
    if not endpoint_frame_R:
        return 0
    n_dof = int(cache.n_dof_node)
    touched = 0
    for stiff, force in zip(cache.stiffness_objects, cache.force_objects):
        pm = panel_maps[int(stiff.panel_index)]
        n = len(pm.s_nodes)
        if n == 0:
            continue
        i0, i1, i2, i3 = pm.elements[int(stiff.local_element_index)]
        corner_locals = (int(i0), int(i1), int(i2), int(i3))
        T = np.eye(4 * n_dof, dtype=float)
        any_r = False
        for corner_i, li in enumerate(corner_locals):
            if li == 0 or li == n:
                which = "start"
            elif li == n - 1 or li == 2 * n - 1:
                which = "end"
            else:
                continue
            R = endpoint_frame_R.get((int(pm.panel_index), which))
            if R is None:
                continue
            ts, tn, ns, nn = R
            base = corner_i * n_dof
            T[base + U_S, base + U_S] = ts
            T[base + U_S, base + W] = tn
            T[base + W, base + U_S] = ns
            T[base + W, base + W] = nn
            T[base + BETA_S, base + BETA_S] = ts
            any_r = True
        if not any_r:
            continue
        ke = np.asarray(stiff.K_e, dtype=float)
        fe = np.asarray(force.F_e, dtype=float)
        stiff.K_e = T.T @ ke @ T
        force.F_e = T.T @ fe
        touched += 1
    return touched


def attach_cluster_node_meta(
    node_meta: Any,
    clusters: list[list[tuple[int, str, np.ndarray]]],
    zones: list,
    cluster_layer_node: dict[tuple[int, str], int],
    cluster_layer_rot_node: dict[tuple[int, str], int],
    constraints: dict[int, list[tuple[int, float]]] | None = None,
) -> None:
    """Populate geometry for virtual cluster reference nodes.

    Gives each junction's cluster **main** and **rot** reference nodes a real
    ``(y,z)`` at the junction centroid, the cluster reference tangent, an
    ``is_top`` flag, and a periodic ``pair_group`` id.

    During migration, ``constraints`` may be attached to
    ``node_meta.constraints`` as a deprecated compatibility view.
    Mutates ``node_meta`` in place.
    """
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.topology.shell_mesh_map import (
        panel_end_tangent,
    )

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
        t_mats = [
            np.outer(t, t)
            for pi, which, _ in cluster
            for t in (panel_end_tangent(np.asarray(zones[pi].nodes, dtype=float), which),)
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

    if constraints is not None:
        node_meta.constraints = dict(constraints)


__all__ = [
    "DEFAULT_BC_MODE",
    "apply_shared_cluster_frame_to_cache",
    "assign_global_nodes_cluster_basis",
    "assign_global_nodes_hybrid_collinear",
    "assign_global_nodes_hybrid_kind_crossing",
    "assign_global_nodes_hybrid_tb_clusters",
    "assign_global_nodes_independent",
    "assign_global_nodes_shared",
    "attach_cluster_node_meta",
    "build_shared_cluster_frame_rotations",
    "canonicalize_bc_mode",
    "cluster_is_kind_crossing",
    "gdof",
    "is_contour_support",
    "normalize_bc_mode",
]
