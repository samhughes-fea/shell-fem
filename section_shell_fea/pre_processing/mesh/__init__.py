"""FE mesh adaptation over BSU ``mesh_utilities.section_mesh``.

Internal home for FE maps, global numbering, junction clustering, and
``ShellPanelTopology``. Strip-mesh generation and typed handoff remain owned by
the public ``mesh_utilities.section_mesh`` barrel. Callers supply
``SectionStripMesh`` / ``ShellMeshInputs``; this package must not execute
``SectionMeshRunner``.
"""

from __future__ import annotations

from .api.shell_panel_topology import (
    ShellPanelTopology,
    TaperedShellPanelTopology,
    build_shell_panel_topology,
    build_tapered_shell_panel_topology,
    shell_panel_topology_from_shell_inputs,
)
from .connected_panel_topology import ConnectedPanelTopology, build_connected_panel_maps
from .global_numbering import (
    DEFAULT_BC_MODE,
    apply_shared_cluster_frame_to_cache,
    assign_global_nodes_cluster_basis,
    assign_global_nodes_hybrid_collinear,
    assign_global_nodes_hybrid_kind_crossing,
    assign_global_nodes_hybrid_tb_clusters,
    assign_global_nodes_independent,
    assign_global_nodes_shared,
    attach_cluster_node_meta,
    build_shared_cluster_frame_rotations,
    canonicalize_bc_mode,
    cluster_is_kind_crossing,
    is_contour_support,
    normalize_bc_mode,
)
from .shell_mesh_map import (
    GlobalNodeMeta,
    NElements,
    PanelGlobalMap,
    build_panel_mesh_maps,
    cluster_is_collinear,
    cluster_points,
    clusters_from_section_strip_mesh,
    collect_global_node_meta,
    effective_n_elements_spec,
    interp_yz_nodes,
    normal_2d,
    panel_end_tangent,
    panel_endpoints,
    panel_maps_from_section_strip_mesh,
    resolve_n_elements,
    scatter_element_ke,
    strip_identity_at_node,
    strip_membership_from_node_meta,
)

__all__ = [
    "ShellPanelTopology",
    "TaperedShellPanelTopology",
    "build_shell_panel_topology",
    "build_tapered_shell_panel_topology",
    "shell_panel_topology_from_shell_inputs",
    "ConnectedPanelTopology",
    "build_connected_panel_maps",
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
    "is_contour_support",
    "normalize_bc_mode",
    "GlobalNodeMeta",
    "NElements",
    "PanelGlobalMap",
    "build_panel_mesh_maps",
    "cluster_is_collinear",
    "cluster_points",
    "clusters_from_section_strip_mesh",
    "collect_global_node_meta",
    "effective_n_elements_spec",
    "interp_yz_nodes",
    "normal_2d",
    "panel_end_tangent",
    "panel_endpoints",
    "panel_maps_from_section_strip_mesh",
    "resolve_n_elements",
    "scatter_element_ke",
    "strip_identity_at_node",
    "strip_membership_from_node_meta",
]
