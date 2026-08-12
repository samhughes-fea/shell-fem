"""Build connected panel strip meshes and assign global node numbers.

Mesh-prep for the thin formulation-cache assembly path (periodic K6) and
assembly helpers. Global node assignment helpers live in
:mod:`~blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.connected_assembly_core`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.shell_mesh_map import (
    NElements,
    PanelGlobalMap,
    cluster_is_collinear,
    cluster_points,
    panel_maps_from_section_strip_mesh,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.connected_assembly_core import (
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
    zones: list[Any],
    *,
    n_elements_per_panel: NElements | None = None,
    target_element_length_m: float | None = None,
    endpoint_tol: float = 1e-6,
    min_elements_per_panel: int = 2,
    interface_constraint_mode: str = "shared",
    section_strip_mesh: Any | None = None,
) -> ConnectedPanelTopology:
    """Consume a prebuilt strip mesh, cluster endpoints, assign global IDs.

    ``section_strip_mesh`` is required (BL-005). Density kwargs are rejected —
    build the mesh with ``mesh_utilities.section_mesh.build_section_strip_mesh``.
    """
    del min_elements_per_panel
    if section_strip_mesh is None:
        raise ValueError(
            "section_strip_mesh is required; build it with "
            "mesh_utilities.section_mesh.build_section_strip_mesh "
            "(FE remesh via discretise_strip is removed — BL-005)."
        )
    if n_elements_per_panel is not None or target_element_length_m is not None:
        raise ValueError(
            "Density kwargs are not accepted on build_connected_panel_maps; "
            "pass a prebuilt section_strip_mesh= instead."
        )

    panel_maps, endpoints = panel_maps_from_section_strip_mesh(section_strip_mesh, zones)
    clusters = cluster_points(endpoints, endpoint_tol)
    endpoint_cluster_index: dict[tuple[int, str], int] = {}
    for cluster_id, c in enumerate(clusters):
        for pi, which, _ in c:
            endpoint_cluster_index[(pi, which)] = cluster_id

    cluster_is_collinear_flags: dict[int, bool] = {
        cid: cluster_is_collinear(c, zones, tol_deg=5.0) for cid, c in enumerate(clusters)
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
