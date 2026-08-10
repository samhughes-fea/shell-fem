"""Build connected panel strip meshes and assign global node numbers.

Mesh-prep for the thin formulation-cache assembly path (periodic K6) and
assembly helpers. Global node assignment helpers live in
:mod:`~blade_structure_utilities.fem_utilities.section_shell_model.processing.static.operations.connected_assembly_core`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.mesh.shell_mesh_map import (
    NElements,
    PanelGlobalMap,
    cluster_is_collinear,
    cluster_points,
    effective_n_elements_spec,
    panel_endpoints,
    resolve_n_elements,
)
from blade_structure_utilities.mesh_utilities.section_mesh.discretise import (
    cap_required_stations_from_panels,
    discretise_strip,
)
from blade_structure_utilities.fem_utilities.section_shell_model.processing.static.operations.connected_assembly_core import (
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
    section_strip_mesh: Any | None = None,
) -> ConnectedPanelTopology:
    """Discretise panels (or consume prebuilt strip mesh), cluster, assign global IDs.

    When ``section_strip_mesh`` is provided, skip ``discretise_strip`` and use
    each ``StripPanelMesh`` ``s_nodes`` / ``elements`` (mesh-worker ownership).

    .. deprecated::
        The ``section_strip_mesh is None`` remesh branch (``discretise_strip``)
        is the FE dual-mesher path (BL-005). Linear-static production callers
        must pass a prebuilt ``SectionStripMesh``; remesh remains only until
        CalculiX / remaining consumers are migrated.
    """
    panel_maps: list[PanelGlobalMap] = []
    endpoints: list[tuple[int, str, np.ndarray]] = []
    strip_meshes = None
    if section_strip_mesh is not None:
        strip_meshes = list(getattr(section_strip_mesh, "panel_meshes", []) or [])
        if len(strip_meshes) != len(panels):
            raise ValueError(
                f"section_strip_mesh.panel_meshes length {len(strip_meshes)} "
                f"!= panels length {len(panels)}"
            )
        n_elem_spec = None
    else:
        warnings.warn(
            "connected_panel_topology remesh path is deprecated (BL-005); "
            "pass section_strip_mesh= from build_section_strip_mesh.",
            DeprecationWarning,
            stacklevel=2,
        )
        n_elem_spec = effective_n_elements_spec(
            panels,
            n_elements_per_panel,
            target_element_length_m,
            min_n=int(min_elements_per_panel),
        )
    required_per_panel = cap_required_stations_from_panels(panels)

    for pi, p in enumerate(panels):
        label = str(getattr(p, "label", f"panel_{pi}"))
        nodes_yz = np.asarray(getattr(p, "nodes"), dtype=float)
        if strip_meshes is not None:
            pm = strip_meshes[pi]
            pm_label = str(getattr(pm, "panel_label", ""))
            if pm_label and pm_label != label:
                raise ValueError(
                    f"panel[{pi}] label {label!r} != strip mesh label {pm_label!r}"
                )
            s_nodes = np.asarray(pm.s_nodes, dtype=float)
            elems = [list(e) for e in (pm.elements or [])]
            if s_nodes.size < 2:
                panel_maps.append(PanelGlobalMap(np.array([]), [], [], label, pi))
                continue
        else:
            s_panel = np.asarray(p.s, dtype=float)
            if len(s_panel) < 2:
                panel_maps.append(PanelGlobalMap(np.array([]), [], [], label, pi))
                continue
            s_nodes, elems = discretise_strip(
                s_panel,
                resolve_n_elements(n_elem_spec, pi),
                panel_label=label,
                nodes_yz=nodes_yz,
                required_s=required_per_panel[pi],
            )
        n_local_nodes = 2 * len(s_nodes)
        global_nodes = [-1] * n_local_nodes
        panel_maps.append(
            PanelGlobalMap(s_nodes, elems, global_nodes, label, pi)
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
