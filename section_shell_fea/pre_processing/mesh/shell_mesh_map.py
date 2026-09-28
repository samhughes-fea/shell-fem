"""Compat shim — import from ``pre_processing.topology.shell_mesh_map``."""

from __future__ import annotations

import warnings

warnings.warn(
    "Import shell_mesh_map from "
    "blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.topology.shell_mesh_map.",
    DeprecationWarning,
    stacklevel=2,
)

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.topology.shell_mesh_map import (  # noqa: E402,F401
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
    "GlobalNodeMeta",
    "NElements",
    "PanelGlobalMap",
    "build_panel_mesh_maps",
    "cluster_is_collinear",
    "cluster_points",
    "collect_global_node_meta",
    "effective_n_elements_spec",
    "interp_yz_nodes",
    "normal_2d",
    "panel_end_tangent",
    "panel_endpoints",
    "clusters_from_section_strip_mesh",
    "panel_maps_from_section_strip_mesh",
    "resolve_n_elements",
    "scatter_element_ke",
    "strip_identity_at_node",
    "strip_membership_from_node_meta",
]
