"""Build connected panel strip meshes and assign global node numbers.

Shared by the thin formulation-cache assembly path (periodic K6) and the
assembly helpers so node numbering stays in one place
(:mod:`connected_assembly_core` helpers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..pre_processing.mesh.shell_mesh_map import (
    NElements,
    PanelGlobalMap,
    cluster_is_collinear,
    cluster_points,
    effective_n_elements_spec,
    panel_endpoints,
    resolve_n_elements,
)
from ..pre_processing.mesh.strip_discretise import discretise_strip
from .connected_assembly_core import (
    assign_global_nodes_cluster_basis,
    assign_global_nodes_independent,
    assign_global_nodes_shared,
)


@dataclass
class ConnectedPanelTopology:
    """Mesh maps + junction topology after global node assignment."""

    panel_maps: list[PanelGlobalMap]
    next_global_node: int
    clusters: list[list[tuple[int, str, np.ndarray]]]
    endpoint_cluster_index: dict[tuple[int, str], int]
    cluster_is_collinear_flags: dict[int, bool]
    effective_mode: str
    interface_constraint_mode: str
    cluster_layer_node: dict[tuple[int, str], int]
    cluster_layer_rot_node: dict[tuple[int, str], int]
    endpoint_cluster_node: dict[tuple[int, str, str], int]
    endpoint_cluster_rot_node: dict[tuple[int, str, str], int]


def build_connected_panel_maps(
    panels: list[Any],
    *,
    n_elements_per_panel: NElements | None = None,
    target_element_length_m: float | None = None,
    endpoint_tol: float = 1e-6,
    min_elements_per_panel: int = 2,
    interface_constraint_mode: str = "shared",
) -> ConnectedPanelTopology:
    """Discretise panels, cluster endpoints, and assign global node IDs."""
    panel_maps: list[PanelGlobalMap] = []
    endpoints: list[tuple[int, str, np.ndarray]] = []
    n_elem_spec = effective_n_elements_spec(
        panels,
        n_elements_per_panel,
        target_element_length_m,
        min_n=int(min_elements_per_panel),
    )

    for pi, p in enumerate(panels):
        s_panel = np.asarray(p.s, dtype=float)
        if len(s_panel) < 2:
            panel_maps.append(
                PanelGlobalMap(np.array([]), [], [], str(getattr(p, "label", f"panel_{pi}")), pi)
            )
            continue
        nodes_yz = np.asarray(getattr(p, "nodes"), dtype=float)
        s_nodes, elems = discretise_strip(
            s_panel,
            resolve_n_elements(n_elem_spec, pi),
            panel_label=str(getattr(p, "label", "") or ""),
            nodes_yz=nodes_yz,
        )
        n_local_nodes = 2 * len(s_nodes)
        global_nodes = [-1] * n_local_nodes
        panel_maps.append(
            PanelGlobalMap(s_nodes, elems, global_nodes, str(getattr(p, "label", f"panel_{pi}")), pi)
        )
        ep = panel_endpoints(nodes_yz)
        endpoints.append((pi, "start", ep["start"]))
        endpoints.append((pi, "end", ep["end"]))

    clusters = cluster_points(endpoints, endpoint_tol)
    endpoint_cluster_index: dict[tuple[int, str], int] = {}
    for cluster_id, c in enumerate(clusters):
        for pi, which, _ in c:
            endpoint_cluster_index[(pi, which)] = cluster_id

    cluster_is_collinear_flags: dict[int, bool] = {
        cid: cluster_is_collinear(c, panels, tol_deg=5.0) for cid, c in enumerate(clusters)
    }

    effective_mode = interface_constraint_mode
    cluster_layer_node: dict[tuple[int, str], int] = {}
    cluster_layer_rot_node: dict[tuple[int, str], int] = {}
    endpoint_cluster_node: dict[tuple[int, str, str], int] = {}
    endpoint_cluster_rot_node: dict[tuple[int, str, str], int] = {}

    if interface_constraint_mode == "shared":
        next_global_node = assign_global_nodes_shared(panel_maps, clusters)
    elif interface_constraint_mode in ("transformed_basis", "shared_rotated") or effective_mode == "shared_rotated":
        basis = assign_global_nodes_cluster_basis(panel_maps, clusters)
        next_global_node = int(basis["next_node"])
        cluster_layer_node = basis["cluster_layer_node"]  # type: ignore[assignment]
        cluster_layer_rot_node = basis["cluster_layer_rot_node"]  # type: ignore[assignment]
        endpoint_cluster_node = basis["endpoint_cluster_node"]  # type: ignore[assignment]
        endpoint_cluster_rot_node = basis["endpoint_cluster_rot_node"]  # type: ignore[assignment]
    else:
        next_global_node = assign_global_nodes_independent(panel_maps)

    return ConnectedPanelTopology(
        panel_maps=panel_maps,
        next_global_node=next_global_node,
        clusters=clusters,
        endpoint_cluster_index=endpoint_cluster_index,
        cluster_is_collinear_flags=cluster_is_collinear_flags,
        effective_mode=effective_mode,
        interface_constraint_mode=interface_constraint_mode,
        cluster_layer_node=cluster_layer_node,
        cluster_layer_rot_node=cluster_layer_rot_node,
        endpoint_cluster_node=endpoint_cluster_node,
        endpoint_cluster_rot_node=endpoint_cluster_rot_node,
    )


__all__ = ["ConnectedPanelTopology", "build_connected_panel_maps"]
