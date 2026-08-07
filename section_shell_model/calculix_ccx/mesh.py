"""Global strip mesh assembly for CalculiX ccx export."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.mesh.shell_mesh_map import (
    NElements,
    PanelGlobalMap,
    build_panel_mesh_maps,
    cluster_points,
    effective_n_elements_spec,
    interp_yz_nodes,
)


def _assign_shared_nodes(
    panel_maps: list[PanelGlobalMap],
    clusters: list[list[tuple[int, str, np.ndarray]]],
) -> int:
    """Assign global node ids with shared junction nodes at endpoint clusters.

    Each cluster gets one bottom + one top shared node; interior nodes are unique.
    (Relocated from the removed native S4R assembly; used by the ccx section mesh.)
    """
    next_global_node = 0
    endpoint_global_node: dict[tuple[int, str, str], int] = {}
    for c in clusters:
        g_bottom = next_global_node
        g_top = next_global_node + 1
        next_global_node += 2
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
            pm.global_nodes[li] = next_global_node
            next_global_node += 1
        for li in range(n + 1, 2 * n - 1):
            pm.global_nodes[li] = next_global_node
            next_global_node += 1
    return next_global_node


def _assign_unique_nodes(panel_maps: list[PanelGlobalMap]) -> int:
    """Assign every panel node a unique global id (no shared junctions)."""
    next_global_node = 0
    for pm in panel_maps:
        if len(pm.s_nodes) == 0:
            continue
        for li in range(2 * len(pm.s_nodes)):
            pm.global_nodes[li] = next_global_node
            next_global_node += 1
    return next_global_node


@dataclass
class CcxElement:
    eid: int
    node_ids: list[int]
    panel_index: int
    local_element_index: int


@dataclass
class CcxSectionMesh:
    """Global mesh for CalculiX deck export."""

    nodes: dict[int, tuple[float, float, float]]
    elements: list[CcxElement]
    elsets: dict[str, list[int]]
    nsets: dict[str, list[int]]
    panel_maps: list[PanelGlobalMap]
    clusters: list[list[tuple[int, str, np.ndarray]]]
    n_global_nodes: int
    l_x: float = 1.0
    global_to_ccx_node: dict[int, int] = field(default_factory=dict)
    element_to_panel: dict[int, tuple[int, int]] = field(default_factory=dict)


def _sanitize_elset_name(label: str) -> str:
    s = str(label).replace(":", "_").replace(" ", "_").replace(".", "_")
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in s)[:80] or "PANEL"


def build_ccx_section_mesh(
    panels: list[Any],
    n_elements_per_panel: NElements | None = None,
    *,
    target_element_length_m: float | None = None,
    min_elements_per_panel: int = 2,
    endpoint_tol: float = 1e-6,
    interface_constraint_mode: str = "shared",
    l_x: float = 1.0,
) -> CcxSectionMesh:
    """Build global node/element maps for ccx export (v1: shared junction nodes)."""
    n_elem_spec = effective_n_elements_spec(
        panels,
        n_elements_per_panel,
        target_element_length_m,
        min_n=int(min_elements_per_panel),
    )
    panel_maps, endpoints = build_panel_mesh_maps(panels, n_elem_spec)
    clusters = cluster_points(endpoints, endpoint_tol)

    if interface_constraint_mode == "shared":
        n_global_nodes = _assign_shared_nodes(panel_maps, clusters)
    else:
        n_global_nodes = _assign_unique_nodes(panel_maps)

    nodes_xyz: dict[int, tuple[float, float, float]] = {}
    for pi, pm in enumerate(panel_maps):
        if len(pm.s_nodes) == 0:
            continue
        p = panels[pi]
        yz_nodes = interp_yz_nodes(pm.s_nodes, p)
        n = len(pm.s_nodes)
        for li in range(2 * n):
            gn = pm.global_nodes[li]
            if gn < 0 or gn in nodes_xyz:
                continue
            si = li if li < n else li - n
            x_val = 0.0 if li < n else float(l_x)
            y_val = float(yz_nodes[si, 0])
            z_val = float(yz_nodes[si, 1])
            nodes_xyz[gn] = (x_val, y_val, z_val)

    global_to_ccx = {gn: i + 1 for i, gn in enumerate(sorted(nodes_xyz.keys()))}
    ccx_nodes = {global_to_ccx[gn]: nodes_xyz[gn] for gn in nodes_xyz}

    elements: list[CcxElement] = []
    elsets: dict[str, list[int]] = {}
    element_to_panel: dict[int, tuple[int, int]] = {}
    eid = 1

    for pi, pm in enumerate(panel_maps):
        if len(pm.s_nodes) == 0:
            continue
        elset_name = _sanitize_elset_name(pm.panel_label)
        elsets.setdefault(elset_name, [])
        for ei, elem in enumerate(pm.elements):
            nids = [global_to_ccx[pm.global_nodes[ln]] for ln in elem]
            elements.append(
                CcxElement(
                    eid=eid,
                    node_ids=nids,
                    panel_index=pi,
                    local_element_index=ei,
                )
            )
            elsets[elset_name].append(eid)
            element_to_panel[eid] = (pi, ei)
            eid += 1

    nsets: dict[str, list[int]] = {}
    if ccx_nodes:
        anchor_ids = sorted(ccx_nodes.keys())[:2]
        nsets["ANCHOR"] = anchor_ids

    return CcxSectionMesh(
        nodes=ccx_nodes,
        elements=elements,
        elsets=elsets,
        nsets=nsets,
        panel_maps=panel_maps,
        clusters=clusters,
        n_global_nodes=n_global_nodes,
        l_x=float(l_x),
        global_to_ccx_node=global_to_ccx,
        element_to_panel=element_to_panel,
    )


__all__ = ["CcxElement", "CcxSectionMesh", "build_ccx_section_mesh"]
