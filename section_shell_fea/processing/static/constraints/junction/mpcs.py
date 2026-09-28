"""Junction cluster-basis MPC equations."""

from __future__ import annotations

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.analysis_dof_layout import (
    BETA_S,
    BETA_X,
    BETA_ZET,
    U_S,
    U_X,
    W,
    gdof,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.topology.shell_mesh_map import (
    PanelGlobalMap,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.types import (
    Constraints,
)


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

    Ports the MITC4 cluster-basis constraints to any ``ndof``: u_x / beta_x /
    beta_z (and ``identity_extra_dofs``) are tied identity to the cluster **main**
    node; the in-plane (u_s, w) and tangential rotation beta_s are mapped by the
    2-D rotation between each panel tangent and the cluster reference tangent.
    """
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.topology.shell_mesh_map import (
        normal_2d,
        panel_end_tangent,
    )

    constraints: Constraints = {}
    for cluster in clusters:
        if len(cluster) < 1:
            continue
        t_mats: list[np.ndarray] = []
        for pi, which, _ in cluster:
            t = panel_end_tangent(np.asarray(zones[pi].nodes, dtype=float), which)
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
            t_p = panel_end_tangent(np.asarray(zones[pi].nodes, dtype=float), end)
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

    for key, gn_rot in cluster_layer_rot_node.items():
        gn_main = cluster_layer_node[key]
        for d in (U_X, U_S, W, BETA_X, BETA_ZET, *identity_extra_dofs):
            constraints[gdof(gn_rot, d, ndof)] = [(gdof(gn_main, d, ndof), 1.0)]
    return constraints


__all__ = ["build_cluster_basis_constraints"]
