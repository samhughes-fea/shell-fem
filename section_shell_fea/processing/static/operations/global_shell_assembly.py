"""Global shell assembly helpers (assembly-only; no full-solve recovery spine).

Production paths use :func:`assemble_global_shell` /
:func:`assemble_section_from_zones`. Contour supports live in
:mod:`connected_assembly_core`.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp

from ....pre_processing.containers.fem_config import MITC4_DEFAULTS
from ....pre_processing.containers.types import ShellTheory
from ....pre_processing.constitutive.panel_abd import panel_abd_tuple_from_lam  # noqa: F401 — re-exported for tests
from ....pre_processing.topology.shell_mesh_map import (
    GlobalNodeMeta,
    PanelGlobalMap as _PanelGlobalMap,
    normal_2d as _normal_2d,
    panel_end_tangent as _panel_end_tangent,
)
from .connected_assembly_core import DEFAULT_BC_MODE
from .section_assembly import assemble_section_from_zones

_NDOF_NODE = MITC4_DEFAULTS.n_dof_node
_U_X = 0
_U_S = 1
_W = 2
_BETA_S = 3
_BETA_X = 4
_BETA_ZET = 5   # drilling rotation about shell normal (MITC4+)



def _dof(node_idx: int, local: int) -> int:
    return node_idx * _NDOF_NODE + local


def _element_global_dofs(
    global_node_ids: Sequence[int],
    *,
    n_dof_node: int = _NDOF_NODE,
) -> np.ndarray:
    """Global DOF indices for one strip element (4 nodes ├ù n_dof_node)."""
    g_dofs = np.empty(4 * n_dof_node, dtype=np.int64)
    for i, gn in enumerate(global_node_ids):
        base = i * n_dof_node
        g_dofs[base : base + n_dof_node] = gn * n_dof_node + np.arange(n_dof_node, dtype=np.int64)
    return g_dofs


def _coo_triplets_from_ke(
    ke: np.ndarray,
    global_node_ids: Sequence[int],
    *,
    n_dof_node: int = _NDOF_NODE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised COO triplets for one element stiffness block."""
    g_dofs = _element_global_dofs(global_node_ids, n_dof_node=n_dof_node)
    n = g_dofs.size
    rows = np.repeat(g_dofs, n)
    cols = np.tile(g_dofs, n)
    vals = np.asarray(ke, dtype=float).ravel()
    return rows, cols, vals


# Type alias for the unified multi-master constraint format.
# Each slave DOF maps to a list of (master_dof, coefficient) pairs so that
# q_slave = sum_k coeff_k * q_master_k.
_Constraints = dict[int, list[tuple[int, float]]]


def _build_basis_transform_constraints(
    panel_maps: list[_PanelGlobalMap],
    zones: list[Any],
    endpoint_clusters: list[list[tuple[int, str, np.ndarray]]],
) -> _Constraints:
    """
    Build physically correct 2-D kinematic compatibility MPCs.

    For each junction cluster the master panel (first in cluster) defines a local
    (u_s, w) basis.  All slave panels' endpoint DOFs are expressed in terms of the
    master's DOFs via the 2-D rotation:

      u_s_s = (┼Ø_s┬À┼Ø_m) u_s_m + (┼Ø_s┬Àn╠é_m) w_m
      w_s   = (n╠é_s┬À┼Ø_m) u_s_m + (n╠é_s┬Àn╠é_m) w_m

    where n╠é = CCW_90(┼Ø) is the panel's out-of-plane direction in the (Y,Z) plane.

    Spanwise (u_x) and drilling (╬▓_x) DOFs are shared 1:1 since X is common to all
    panels.  The tangential rotation ╬▓_s is projected by (┼Ø_s┬À┼Ø_m).
    """
    constraints: _Constraints = {}
    for cluster in endpoint_clusters:
        if len(cluster) < 2:
            continue
        pi_m, end_m, _ = cluster[0]
        pm_m = panel_maps[pi_m]
        n_m = len(pm_m.s_nodes)
        row_m = 0 if end_m == "start" else n_m - 1
        nodes_m = np.asarray(getattr(zones[pi_m], "nodes"), dtype=float)
        t_m = _panel_end_tangent(nodes_m, end_m)
        n_m_hat = _normal_2d(t_m)

        for pi_s, end_s, _ in cluster[1:]:
            pm_s = panel_maps[pi_s]
            n_s = len(pm_s.s_nodes)
            row_s = 0 if end_s == "start" else n_s - 1
            nodes_s = np.asarray(getattr(zones[pi_s], "nodes"), dtype=float)
            t_s = _panel_end_tangent(nodes_s, end_s)
            n_s_hat = _normal_2d(t_s)

            # 2├ù2 rotation matrix entries: R = [[ts_tm, ts_nm],[ns_tm, ns_nm]]
            # q_slave = R * q_master   for (u_s, w) block.
            ts_tm = float(np.dot(t_s, t_m))
            ts_nm = float(np.dot(t_s, n_m_hat))
            ns_tm = float(np.dot(n_s_hat, t_m))
            ns_nm = float(np.dot(n_s_hat, n_m_hat))

            for off_m, off_s in ((row_m, row_s), (row_m + n_m, row_s + n_s)):
                gn_m = pm_m.global_nodes[off_m]
                gn_s = pm_s.global_nodes[off_s]

                # Spanwise: u_x is common to all panels (no rotation of X-axis).
                constraints[_dof(gn_s, _U_X)] = [(_dof(gn_m, _U_X), 1.0)]

                # 2-D in-plane displacement: u_s_s = ts_tm*u_s_m + ts_nm*w_m
                us_terms = [(_dof(gn_m, _U_S), ts_tm)]
                if abs(ts_nm) > 1e-12:
                    us_terms.append((_dof(gn_m, _W), ts_nm))
                constraints[_dof(gn_s, _U_S)] = us_terms

                # 2-D out-of-plane: w_s = ns_tm*u_s_m + ns_nm*w_m
                w_terms = [(_dof(gn_m, _U_S), ns_tm)]
                if abs(ns_nm) > 1e-12:
                    w_terms.append((_dof(gn_m, _W), ns_nm))
                constraints[_dof(gn_s, _W)] = w_terms

                # Rotation about span: ╬▓_x is common.
                constraints[_dof(gn_s, _BETA_X)] = [(_dof(gn_m, _BETA_X), 1.0)]

                # Tangential rotation: project ╬▓_s by the tangent dot product.
                # ╬▓_s represents rotation about the panel tangent; its (Y,Z) component
                # along the slave tangent is (┼Ø_s┬À┼Ø_m)*╬▓_s_m.
                constraints[_dof(gn_s, _BETA_S)] = [(_dof(gn_m, _BETA_S), ts_tm)]

                # Drilling rotation: tied as identity (legacy path, shared mode preferred for MITC4+)
                constraints[_dof(gn_s, _BETA_ZET)] = [(_dof(gn_m, _BETA_ZET), 1.0)]

    return constraints


def _build_cluster_basis_constraints(
    panel_maps: list["_PanelGlobalMap"],
    zones: list[Any],
    endpoint_clusters: list[list[tuple[int, str, np.ndarray]]],
    endpoint_cluster_node: dict[tuple[int, str, str], int],
    endpoint_cluster_rot_node: dict[tuple[int, str, str], int],
) -> _Constraints:
    """
    Build transformed-basis constraints against cluster reference DOFs.

    This avoids panel-to-panel master/slave chains and makes BC propagation robust:
    all panel endpoint DOFs are slaves, cluster reference DOFs are masters.
    """
    constraints: _Constraints = {}
    for cluster in endpoint_clusters:
        if len(cluster) < 1:
            continue
        # Reference tangent from principal direction of endpoint tangents.
        t_mats: list[np.ndarray] = []
        for pi, which, _ in cluster:
            nodes = np.asarray(getattr(zones[pi], "nodes"), dtype=float)
            t = _panel_end_tangent(nodes, which)
            t_mats.append(np.outer(t, t))
        if t_mats:
            M = np.sum(np.stack(t_mats, axis=0), axis=0)
            evals, evecs = np.linalg.eigh(M)
            t_ref = np.asarray(evecs[:, int(np.argmax(evals))], dtype=float)
            if float(np.linalg.norm(t_ref)) < 1e-12:
                t_ref = np.array([1.0, 0.0], dtype=float)
        else:
            t_ref = np.array([1.0, 0.0], dtype=float)
        if float(t_ref[0]) < 0.0:
            t_ref = -t_ref
        t_ref = t_ref / max(float(np.linalg.norm(t_ref)), 1e-12)
        n_ref = _normal_2d(t_ref)

        for pi, end, _ in cluster:
            pm = panel_maps[pi]
            n = len(pm.s_nodes)
            if n == 0:
                continue
            row = 0 if end == "start" else n - 1
            nodes = np.asarray(getattr(zones[pi], "nodes"), dtype=float)
            t_p = _panel_end_tangent(nodes, end)
            n_p = _normal_2d(t_p)
            ts = float(np.dot(t_p, t_ref))
            tn = float(np.dot(t_p, n_ref))
            ns = float(np.dot(n_p, t_ref))
            nn = float(np.dot(n_p, n_ref))

            for off, layer in ((row, "bottom"), (row + n, "top")):
                gn_p = pm.global_nodes[off]
                gn_c_main = endpoint_cluster_node[(pi, end, layer)]
                gn_c_rot = endpoint_cluster_rot_node[(pi, end, layer)]
                constraints[_dof(gn_p, _U_X)] = [(_dof(gn_c_main, _U_X), 1.0)]
                us_terms = [(_dof(gn_c_main, _U_S), ts)]
                if abs(tn) > 1e-12:
                    us_terms.append((_dof(gn_c_main, _W), tn))
                constraints[_dof(gn_p, _U_S)] = us_terms
                w_terms = [(_dof(gn_c_main, _U_S), ns)]
                if abs(nn) > 1e-12:
                    w_terms.append((_dof(gn_c_main, _W), nn))
                constraints[_dof(gn_p, _W)] = w_terms
                constraints[_dof(gn_p, _BETA_X)] = [(_dof(gn_c_main, _BETA_X), 1.0)]
                beta_terms = [(_dof(gn_c_main, _BETA_S), ts)]
                if abs(tn) > 1e-12:
                    beta_terms.append((_dof(gn_c_rot, _BETA_S), tn))
                constraints[_dof(gn_p, _BETA_S)] = beta_terms
                constraints[_dof(gn_p, _BETA_ZET)] = [(_dof(gn_c_main, _BETA_ZET), 1.0)]
    return constraints


def assemble_global_shell(
    zones: list[Any],
    Nx_panels: list[np.ndarray],
    Nxy_panels: list[np.ndarray],
    *,
    section_strip_mesh: Any,
    zones_outboard: list[Any] | None = None,
    L_x: float = 1.0,
    endpoint_tol: float = 1e-6,
    bc_mode: str = DEFAULT_BC_MODE,
    interface_constraint_mode: str = "shared",
) -> tuple[sp.csr_matrix, np.ndarray, GlobalNodeMeta, list[_PanelGlobalMap], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Assemble global shell stiffness (and load vector) without MPC solve/recovery.

    Thin wrapper over :func:`assemble_section_from_zones`. ``bc_mode`` is accepted
    for API/cache-key compatibility but does not affect the assembled K/f.
    ``section_strip_mesh`` is required (BL-005).
    """
    del bc_mode  # assembly-only; contour supports applied on traction / CCX paths
    return assemble_section_from_zones(
        zones,
        Nx_panels,
        Nxy_panels,
        section_strip_mesh=section_strip_mesh,
        zones_outboard=zones_outboard,
        L_x=L_x,
        endpoint_tol=endpoint_tol,
        interface_constraint_mode=interface_constraint_mode,
        shell_theory=ShellTheory.FSDT,
    ).as_legacy_8tuple()



