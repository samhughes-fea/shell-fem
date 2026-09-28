"""Compatibility shim — numbering and DOF constants live in pre-processing.

Canonical homes:
- DOF layout: ``pre_processing.containers.analysis_dof_layout``
- numbering: ``pre_processing.topology.global_numbering``
- MPCs / supports: ``processing.static.constraints``
"""

from __future__ import annotations

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.analysis_dof_layout import (
    BETA_S,
    BETA_X,
    BETA_ZET,
    U_S,
    U_X,
    W,
    gdof,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.topology.global_numbering import (
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
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.junction.mpcs import (
    build_cluster_basis_constraints,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.junction.supports import (
    build_clamp_supports,
    build_cluster_basis_supports,
    build_hybrid_collinear_supports,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.types import (
    Constraints,
)

__all__ = [
    "U_X", "U_S", "W", "BETA_S", "BETA_X", "BETA_ZET",
    "Constraints",
    "DEFAULT_BC_MODE",
    "gdof",
    "canonicalize_bc_mode",
    "normalize_bc_mode",
    "is_contour_support",
    "assign_global_nodes_shared",
    "assign_global_nodes_independent",
    "assign_global_nodes_cluster_basis",
    "assign_global_nodes_hybrid_collinear",
    "assign_global_nodes_hybrid_kind_crossing",
    "assign_global_nodes_hybrid_tb_clusters",
    "build_shared_cluster_frame_rotations",
    "apply_shared_cluster_frame_to_cache",
    "cluster_is_kind_crossing",
    "build_cluster_basis_constraints",
    "build_cluster_basis_supports",
    "build_hybrid_collinear_supports",
    "attach_cluster_node_meta",
    "build_clamp_supports",
]
