"""Compat shim — import from ``pre_processing.topology.global_numbering``."""

from __future__ import annotations

import warnings

warnings.warn(
    "Import global_numbering from "
    "blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.topology.global_numbering.",
    DeprecationWarning,
    stacklevel=2,
)

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.topology.global_numbering import (  # noqa: E402,F401
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
    gdof,
    is_contour_support,
    normalize_bc_mode,
)

__all__ = [
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
    "gdof",
    "is_contour_support",
    "normalize_bc_mode",
]
