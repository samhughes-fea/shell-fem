"""Compat shim — import from ``pre_processing.mesh.panel_node_assembly``."""

from __future__ import annotations

import warnings

warnings.warn(
    "Import shell_mesh_map from "
    "blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.panel_node_assembly.",
    DeprecationWarning,
    stacklevel=2,
)

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.panel_node_assembly import (  # noqa: E402,F401
    PanelElementCountSpec,
    PanelNodeAssemblyMap,
    SectionNodeAssemblyMetadata,
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

# Deprecated type aliases (2027-03-31)
PanelGlobalMap = PanelNodeAssemblyMap
GlobalNodeMeta = SectionNodeAssemblyMetadata

__all__ = [
    "GlobalNodeMeta",
    "PanelElementCountSpec",
    "PanelGlobalMap",
    "PanelNodeAssemblyMap",
    "SectionNodeAssemblyMetadata",
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
