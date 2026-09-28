"""Build connected panel strip meshes and assign global node numbers.

Mesh-prep for the thin formulation-cache assembly path. Global node assignment
helpers live in :mod:`pre_processing.mesh.global_numbering`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.panel_node_assembly import (
    PanelElementCountSpec,
    PanelNodeAssemblyMap,
    cluster_is_collinear,
    clusters_from_section_strip_mesh,
    panel_maps_from_section_strip_mesh,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.global_numbering import (
    assign_global_nodes_cluster_basis,
    assign_global_nodes_hybrid_collinear,
    assign_global_nodes_hybrid_kind_crossing,
    assign_global_nodes_independent,
    assign_global_nodes_shared,
    build_shared_cluster_frame_rotations,
)


@dataclass
class SectionShellAssemblyMap:
    """Mesh maps + junction assembly after global node assignment."""

    panel_maps: list[PanelNodeAssemblyMap]
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
    # Study-only shared_cluster_frame: (pi, which) -> (ts, tn, ns, nn)
    endpoint_frame_R: dict[tuple[int, str], tuple[float, float, float, float]] | None = None
    shared_cluster_frame_cids: tuple[int, ...] = ()


def _validate_producer_clusters(
    clusters: list[list[tuple[int, str, np.ndarray]]],
    endpoints: list[tuple[int, str, np.ndarray]],
) -> None:
    """Ensure strip-mesh clusters cover each panel endpoint exactly once (FEA-JN-001)."""
    endpoint_keys = {(int(pi), str(which)) for pi, which, _ in endpoints}
    seen: set[tuple[int, str]] = set()
    for cid, cluster in enumerate(clusters):
        if not cluster:
            raise ValueError(
                f"section_strip_mesh cluster {cid} is empty (FEA-JN-001)"
            )
        for pi, which, _pt in cluster:
            key = (int(pi), str(which))
            if key in seen:
                raise ValueError(
                    f"endpoint {key} appears in multiple strip-mesh clusters "
                    f"(FEA-JN-001)"
                )
            seen.add(key)
    missing = endpoint_keys - seen
    extra = seen - endpoint_keys
    if missing or extra:
        raise ValueError(
            "section_strip_mesh.clusters do not match panel endpoints "
            f"(missing={sorted(missing)!r}, extra={sorted(extra)!r}; FEA-JN-001); "
            "refusing FE re-clustering by proximity"
        )


def build_section_shell_assembly_map(
    zones: list[Any],
    *,
    n_elements_per_panel: PanelElementCountSpec | None = None,
    target_element_length_m: float | None = None,
    endpoint_tol: float = 1e-6,
    min_elements_per_panel: int = 2,
    interface_constraint_mode: str = "shared",
    section_strip_mesh: Any | None = None,
) -> SectionShellAssemblyMap:
    """Consume a prebuilt strip mesh, use producer clusters, assign global IDs.

    ``section_strip_mesh`` is required (BL-005). Density kwargs are rejected —
    build the mesh with ``mesh_utilities.section_mesh.build_section_strip_mesh``.

    Junction clusters come from ``section_strip_mesh.clusters`` (same as CalculiX
    ``clusters_from_section_strip_mesh``). FE does **not** re-run
    ``cluster_points`` by proximity (FEA-JN-001).
    """
    del min_elements_per_panel, endpoint_tol
    if section_strip_mesh is None:
        raise ValueError(
            "section_strip_mesh is required; build it with "
            "mesh_utilities.section_mesh.build_section_strip_mesh "
            "(FE remesh via discretise_strip is removed — BL-005)."
        )
    if n_elements_per_panel is not None or target_element_length_m is not None:
        raise ValueError(
            "Density kwargs are not accepted on build_section_shell_assembly_map; "
            "pass a prebuilt section_strip_mesh= instead."
        )

    panel_maps, endpoints = panel_maps_from_section_strip_mesh(section_strip_mesh, zones)
    clusters = clusters_from_section_strip_mesh(section_strip_mesh)
    _validate_producer_clusters(clusters, endpoints)
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
    endpoint_frame_R: dict[tuple[int, str], tuple[float, float, float, float]] | None = None
    shared_cluster_frame_cids: tuple[int, ...] = ()

    if interface_constraint_mode == "shared":
        next_global_node = assign_global_nodes_shared(panel_maps, clusters)
    elif interface_constraint_mode == "shared_cluster_frame":
        # Shared node IDs (production topology) + panel→cluster frame rotations
        # for kind-crossing ∩ non-collinear cusps only (applied at Ke scatter).
        next_global_node = assign_global_nodes_shared(panel_maps, clusters)
        frame = build_shared_cluster_frame_rotations(
            panel_maps, zones, clusters, cluster_is_collinear_flags
        )
        endpoint_frame_R = frame["endpoint_frame_R"]  # type: ignore[assignment]
        shared_cluster_frame_cids = tuple(frame["cids"])  # type: ignore[assignment]
    elif interface_constraint_mode == "hybrid_collinear":
        basis = assign_global_nodes_hybrid_collinear(
            panel_maps, clusters, cluster_is_collinear_flags
        )
        next_global_node = int(basis["next_node"])
        cluster_layer_node = basis["cluster_layer_node"]  # type: ignore[assignment]
        cluster_layer_rot_node = basis["cluster_layer_rot_node"]  # type: ignore[assignment]
        endpoint_cluster_node = basis["endpoint_cluster_node"]  # type: ignore[assignment]
        endpoint_cluster_rot_node = basis["endpoint_cluster_rot_node"]  # type: ignore[assignment]
    elif interface_constraint_mode == "hybrid_kind_crossing":
        basis = assign_global_nodes_hybrid_kind_crossing(
            panel_maps, clusters, cluster_is_collinear_flags
        )
        next_global_node = int(basis["next_node"])
        cluster_layer_node = basis["cluster_layer_node"]  # type: ignore[assignment]
        cluster_layer_rot_node = basis["cluster_layer_rot_node"]  # type: ignore[assignment]
        endpoint_cluster_node = basis["endpoint_cluster_node"]  # type: ignore[assignment]
        endpoint_cluster_rot_node = basis["endpoint_cluster_rot_node"]  # type: ignore[assignment]
    elif interface_constraint_mode in ("transformed_basis", "shared_rotated") or effective_mode == "shared_rotated":
        basis = assign_global_nodes_cluster_basis(panel_maps, clusters)
        next_global_node = int(basis["next_node"])
        cluster_layer_node = basis["cluster_layer_node"]  # type: ignore[assignment]
        cluster_layer_rot_node = basis["cluster_layer_rot_node"]  # type: ignore[assignment]
        endpoint_cluster_node = basis["endpoint_cluster_node"]  # type: ignore[assignment]
        endpoint_cluster_rot_node = basis["endpoint_cluster_rot_node"]  # type: ignore[assignment]
    else:
        next_global_node = assign_global_nodes_independent(panel_maps)

    return SectionShellAssemblyMap(
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
        endpoint_frame_R=endpoint_frame_R,
        shared_cluster_frame_cids=shared_cluster_frame_cids,
    )


__all__ = ["SectionShellAssemblyMap", "build_section_shell_assembly_map"]
