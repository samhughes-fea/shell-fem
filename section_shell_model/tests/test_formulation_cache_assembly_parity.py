"""Parity: formulation-cache assembly matches legacy inline scatter."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.mesh.api.shell_panel_topology import build_shell_panel_topology
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.containers.types import ShellTheory
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.element_library.formulation_factory import build_formulation_cache
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.element_library.mitc4.element import mitc4_stiffness
from blade_structure_utilities.fem_utilities.section_shell_model.processing.static.operations.global_shell_assembly import (
    _NDOF_NODE,
    _PanelGlobalMap,
    _coo_triplets_from_ke,
    _dof,
    panel_abd_tuple_from_lam,
)
from blade_structure_utilities.mesh_utilities.section_mesh.discretise import (
    discretise_strip,
)
from blade_structure_utilities.fem_utilities.section_shell_model.processing.static.operations.section_assembly import assemble_section_system


def _legacy_assemble(panel_maps, panels, nx, nxy, *, L_x=1.0, n_gdof: int):
    row_chunks: list[np.ndarray] = []
    col_chunks: list[np.ndarray] = []
    val_chunks: list[np.ndarray] = []
    f_global = np.zeros(n_gdof)
    for pi, pm in enumerate(panel_maps):
        if len(pm.s_nodes) == 0:
            continue
        p = panels[pi]
        nodes_yz = np.asarray(getattr(p, "nodes"), dtype=float)
        ABD, thickness, _, _, G_eff = panel_abd_tuple_from_lam(p.lam)
        s_nodes = pm.s_nodes
        Nx_nodes = np.interp(s_nodes, np.asarray(p.s, dtype=float), np.asarray(nx[pi], dtype=float))
        Nxy_nodes = np.interp(s_nodes, np.asarray(p.s, dtype=float), np.asarray(nxy[pi], dtype=float))
        for e in pm.elements:
            i0, i1, i2, i3 = e
            L_s = float(abs(s_nodes[i1] - s_nodes[i0]))
            if L_s < 1e-30:
                continue
            Ke = mitc4_stiffness(L_s, L_x, ABD, thickness, G_eff=G_eff)
            gids = [pm.global_nodes[ln] for ln in (i0, i1, i2, i3)]
            r, c, v = _coo_triplets_from_ke(Ke, gids)
            row_chunks.append(r)
            col_chunks.append(c)
            val_chunks.append(v)
            Nx_avg = 0.5 * (Nx_nodes[i0] + Nx_nodes[i1])
            Nxy_avg = 0.5 * (Nxy_nodes[i0] + Nxy_nodes[i1])
            f_node_x = Nx_avg * L_s / 2.0
            f_node_s = Nxy_avg * L_s / 2.0
            for ln in (i2, i3):
                gn = pm.global_nodes[ln]
                f_global[_dof(gn, 0)] += f_node_x
                f_global[_dof(gn, 1)] += f_node_s
        if len(nodes_yz) >= 3:
            y_p, z_p = nodes_yz[:, 0], nodes_yz[:, 1]
            s_panel_full = np.asarray(p.s, dtype=float)
            dy_p, dz_p = np.diff(y_p), np.diff(z_p)
            ds_p = np.maximum(np.hypot(dy_p, dz_p), 1e-30)
            theta_p = np.arctan2(dz_p, dy_p)
            kappa_p = np.zeros(len(nodes_yz))
            for ii in range(1, len(nodes_yz) - 1):
                dth = theta_p[ii] - theta_p[ii - 1]
                dth = (dth + np.pi) % (2 * np.pi) - np.pi
                arc = 0.5 * (ds_p[ii - 1] + ds_p[ii])
                kappa_p[ii] = dth / arc if arc > 1e-30 else 0.0
            if len(kappa_p) > 1:
                kappa_p[0] = kappa_p[1]
                kappa_p[-1] = kappa_p[-2]
            kappa_nodes = np.interp(s_nodes, s_panel_full, kappa_p)
            for e in pm.elements:
                i0, i1, i2, i3 = e
                L_s = float(abs(s_nodes[i1] - s_nodes[i0]))
                if L_s < 1e-30:
                    continue
                Nx_avg = 0.5 * (Nx_nodes[i0] + Nx_nodes[i1])
                kappa_avg = 0.5 * (kappa_nodes[i0] + kappa_nodes[i1])
                q_n = Nx_avg * kappa_avg
                f_node_w = q_n * L_s * L_x / 4.0
                for ln in (i0, i1, i2, i3):
                    gn = pm.global_nodes[ln]
                    f_global[_dof(gn, 2)] += f_node_w
    if row_chunks:
        rows = np.concatenate(row_chunks)
        cols = np.concatenate(col_chunks)
        vals = np.concatenate(val_chunks)
    else:
        rows = cols = np.array([], dtype=np.int64)
        vals = np.array([], dtype=float)
    K = sp.coo_matrix((vals, (rows, cols)), shape=(n_gdof, n_gdof)).tocsr()
    return K, f_global


def test_formulation_cache_matches_legacy_inline_assembly():
    panel_topology = build_shell_panel_topology("2412", [0.35], chord_m=1.0)
    panels = panel_topology.panels
    nx = [np.linspace(0, 1000, len(np.asarray(p.s, dtype=float))) for p in panels]
    nxy = [np.zeros_like(n) for n in nx]

    panel_maps = []
    next_g = 0
    for pi, p in enumerate(panels):
        s_panel = np.asarray(p.s, dtype=float)
        if len(s_panel) < 2:
            panel_maps.append(_PanelGlobalMap(np.array([]), [], [], f"p{pi}", pi))
            continue
        s_nodes, elems = discretise_strip(s_panel, 4, panel_label="", nodes_yz=np.asarray(p.nodes))
        gn = [-1] * (2 * len(s_nodes))
        for i in range(2 * len(s_nodes)):
            gn[i] = next_g
            next_g += 1
        panel_maps.append(_PanelGlobalMap(s_nodes, elems, gn, f"p{pi}", pi))
    n_gdof = next_g * _NDOF_NODE

    K_legacy, f_legacy = _legacy_assemble(panel_maps, panels, nx, nxy, n_gdof=n_gdof)
    cache = build_formulation_cache(
        panel_maps, panels, nx, nxy, shell_theory=ShellTheory.FSDT, L_x=1.0
    )
    K_new, f_new, _ = assemble_section_system(cache, n_gdof)

    diff_k = float((K_legacy - K_new).max())
    diff_f = float(np.max(np.abs(f_legacy - f_new)))
    assert diff_k < 1e-10, f"K mismatch max={diff_k}"
    assert diff_f < 1e-10, f"f mismatch max={diff_f}"
