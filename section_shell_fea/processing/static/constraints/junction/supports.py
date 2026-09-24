"""Junction contour / cluster / hybrid essential-support selection."""

from __future__ import annotations

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.analysis_dof_layout import (
    BETA_S,
    BETA_X,
    U_S,
    U_X,
    W,
    gdof,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.global_numbering import (
    DEFAULT_BC_MODE,
    normalize_bc_mode,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.shell_mesh_map import (
    PanelGlobalMap,
)


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


def build_hybrid_collinear_supports(
    panel_maps: list[PanelGlobalMap],
    endpoint_cluster_index: dict[tuple[int, str], int],
    cluster_layer_node: dict[tuple[int, str], int],
    cluster_layer_rot_node: dict[tuple[int, str], int],
    *,
    ndof: int,
    bc_mode: str = DEFAULT_BC_MODE,
    clamp_dofs: tuple[int, ...] = (W, BETA_S, BETA_X),
) -> set[int]:
    """Supports for ``hybrid_collinear``: clamp shared endpoints + TB cluster refs."""
    mode = normalize_bc_mode(bc_mode)
    fixed: set[int] = set()
    fixed.add(gdof(0, U_X, ndof))
    fixed.add(gdof(0, U_S, ndof))
    fixed.add(gdof(1, U_X, ndof))
    done_tb: set[tuple[int, str]] = set()
    for pm in panel_maps:
        if len(pm.s_nodes) == 0:
            continue
        n = len(pm.s_nodes)
        support_local = (0, n - 1, n, 2 * n - 1) if mode == "contour_support" else (0, n - 1)
        for ln in support_local:
            which = "start" if ln in (0, n) else "end"
            layer = "bottom" if ln in (0, n - 1) else "top"
            cid = endpoint_cluster_index.get((pm.panel_index, which), -1)
            key = (cid, layer)
            if cid >= 0 and key in cluster_layer_node:
                if key in done_tb:
                    continue
                done_tb.add(key)
                gn_m = cluster_layer_node[key]
                gn_r = cluster_layer_rot_node[key]
                for d in clamp_dofs:
                    fixed.add(gdof(gn_m, d, ndof))
                fixed.add(gdof(gn_r, BETA_S, ndof))
            else:
                gn = int(pm.global_nodes[ln])
                for d in clamp_dofs:
                    fixed.add(gdof(gn, d, ndof))
    return fixed


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
    "build_clamp_supports",
    "build_cluster_basis_supports",
    "build_hybrid_collinear_supports",
]
