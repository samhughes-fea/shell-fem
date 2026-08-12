"""Multi-row lofted root-segment shell mesh for CalculiX diagnostic solves.

Builds a structured S4R quad grid by stacking contour-discretised station slices
along the span (global ``x``). Junction nodes are shared within each span layer
(``interface_constraint_mode="shared"``). Production periodic unit-cell meshes
remain in :mod:`.mesh`; this builder is diagnostic-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.shell_mesh_map import (
    PanelGlobalMap,
    clusters_from_section_strip_mesh,
    panel_maps_from_section_strip_mesh,
    interp_yz_nodes,
)
from .mesh import _sanitize_elset_name


@dataclass
class CcxRootElement:
    """One S4R element in a multi-row root segment."""

    eid: int
    node_ids: list[int]
    panel_index: int
    contour_element_index: int
    span_row: int  # 0 = root-adjacent element row


@dataclass
class CcxRootSegmentMesh:
    """Structured multi-row section mesh for a clamped root-segment diagnostic."""

    nodes: dict[int, tuple[float, float, float]]
    elements: list[CcxRootElement]
    elsets: dict[str, list[int]]
    nsets: dict[str, list[int]]
    panel_maps: list[PanelGlobalMap]
    clusters: list[list[tuple[int, str, np.ndarray]]]
    span_coords: np.ndarray  # node-layer x positions, shape (n_layers,)
    n_span_rows: int
    segment_length_m: float
    element_to_key: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    # eid -> (panel_index, contour_element_index, span_row)
    ref_node_id: int = 0
    rot_node_id: int = 0
    root_diameter_m: float = 0.0


def estimate_root_diameter_m(zones: list[Any]) -> float:
    """Characteristic cross-section diameter from panel YZ extents [m]."""
    ys: list[float] = []
    zs: list[float] = []
    for p in zones:
        nodes = np.asarray(getattr(p, "nodes"), dtype=float)
        if nodes.ndim != 2 or nodes.shape[0] == 0:
            continue
        ys.extend(nodes[:, 0].tolist())
        zs.extend(nodes[:, 1].tolist())
    if not ys:
        return 1.0
    return float(max(max(ys) - min(ys), max(zs) - min(zs), 1e-3))


def _layer_node_assignment(
    panel_maps: list[PanelGlobalMap],
    clusters: list[list[tuple[int, str, np.ndarray]]],
    *,
    shared: bool,
    next_id: int,
) -> tuple[dict[tuple[int, int], int], int]:
    """Assign unique ids for one span layer: (panel_index, local_contour_i) -> nid."""
    assignment: dict[tuple[int, int], int] = {}
    if shared:
        endpoint: dict[tuple[int, str], int] = {}
        for c in clusters:
            nid = next_id
            next_id += 1
            for pi, which, _ in c:
                endpoint[(pi, which)] = nid
        for pm in panel_maps:
            n = len(pm.s_nodes)
            if n == 0:
                continue
            assignment[(pm.panel_index, 0)] = endpoint[(pm.panel_index, "start")]
            assignment[(pm.panel_index, n - 1)] = endpoint[(pm.panel_index, "end")]
            for li in range(1, n - 1):
                assignment[(pm.panel_index, li)] = next_id
                next_id += 1
    else:
        for pm in panel_maps:
            n = len(pm.s_nodes)
            for li in range(n):
                assignment[(pm.panel_index, li)] = next_id
                next_id += 1
    return assignment, next_id


def _interp_yz_normalized(s_nodes_ref: np.ndarray, panel: Any) -> np.ndarray:
    """Interpolate panel YZ onto ``s_nodes_ref`` by normalised contour parameter.

    Absolute arc length can change between loft guide stations (taper / blend);
    using xi = (s - s0) / (s1 - s0) keeps the same number of contour nodes while
    mapping end-to-end.
    """
    s_ref = np.asarray(s_nodes_ref, dtype=float)
    s_p = np.asarray(panel.s, dtype=float)
    nodes_yz = np.asarray(getattr(panel, "nodes"), dtype=float)
    if s_ref.size == 0 or s_p.size < 2:
        return interp_yz_nodes(s_ref, panel)
    s0_r, s1_r = float(s_ref[0]), float(s_ref[-1])
    den_r = s1_r - s0_r
    xi = np.zeros_like(s_ref) if abs(den_r) <= 1e-30 else (s_ref - s0_r) / den_r
    s0_p, s1_p = float(s_p[0]), float(s_p[-1])
    s_query = s0_p + xi * (s1_p - s0_p)
    y = np.interp(s_query, s_p, nodes_yz[:, 0])
    z = np.interp(s_query, s_p, nodes_yz[:, 1])
    return np.column_stack([y, z])


def _yz_for_panel_layer(
    zones: list[Any],
    pm: PanelGlobalMap,
    *,
    station_panels: list[list[Any]] | None,
    layer_index: int,
    n_layers: int,
) -> np.ndarray:
    """Contour YZ at one span layer (prismatic or linearly lofted)."""
    if station_panels is None:
        return interp_yz_nodes(pm.s_nodes, zones[pm.panel_index])
    if len(station_panels) == n_layers:
        return _interp_yz_normalized(pm.s_nodes, station_panels[layer_index][pm.panel_index])
    # Fewer guide stations than node layers: interpolate along normalised span.
    n_guide = len(station_panels)
    if n_guide < 2:
        return _interp_yz_normalized(pm.s_nodes, station_panels[0][pm.panel_index])
    xi = float(layer_index) / float(max(n_layers - 1, 1))
    g = xi * float(n_guide - 1)
    i0 = int(np.floor(g))
    i1 = min(i0 + 1, n_guide - 1)
    w = g - float(i0)
    yz0 = _interp_yz_normalized(pm.s_nodes, station_panels[i0][pm.panel_index])
    yz1 = _interp_yz_normalized(pm.s_nodes, station_panels[i1][pm.panel_index])
    return (1.0 - w) * yz0 + w * yz1


def build_ccx_root_segment_mesh(
    zones: list[Any],
    *,
    section_strip_mesh: Any,
    segment_length_m: float,
    n_span_rows: int = 8,
    interface_constraint_mode: str = "shared",
    station_panels: list[list[Any]] | None = None,
    station_span_coords: np.ndarray | None = None,
) -> CcxRootSegmentMesh:
    """Build a multi-row lofted (or prismatic) root-segment S4R mesh.

    Contour discretisation is frozen from ``section_strip_mesh`` (BL-005); spanwise
    layers loft YZ only — they do not remesh the contour.

    Parameters
    ----------
    zones
        Contour topology / laminates at the root (layer 0).
    section_strip_mesh
        Caller-owned strip mesh (same contour for every span layer).
    segment_length_m
        Spanwise extent from the clamped root face (``x=0``) to the loaded cut.
    n_span_rows
        Number of element rows along the span (>= 1).
    station_panels
        Optional guide sections for lofting. Length may be ``n_span_rows + 1``
        (exact node layers) or any ``>= 2`` sequence that is linearly interpolated
        onto the span layers. When ``None``, the mesh is prismatic.
    station_span_coords
        Optional absolute ``x`` coordinates for the node layers. Defaults to a
        uniform linspace over ``[0, segment_length_m]``.
    """
    if section_strip_mesh is None:
        raise ValueError(
            "section_strip_mesh is required; build it with "
            "mesh_utilities.section_mesh.build_section_strip_mesh "
            "(or build_section_strip_mesh_from_panels)."
        )
    n_rows = max(int(n_span_rows), 1)
    n_layers = n_rows + 1
    L = float(segment_length_m)
    if L <= 0.0:
        raise ValueError(f"segment_length_m must be positive; got {segment_length_m!r}")
    if station_panels is not None and len(station_panels) < 1:
        raise ValueError("station_panels must be non-empty when provided")
    if station_panels is not None:
        for si, sp in enumerate(station_panels):
            if len(sp) != len(zones):
                raise ValueError(
                    f"station_panels[{si}] has {len(sp)} panels; expected {len(zones)}"
                )

    panel_maps, _endpoints = panel_maps_from_section_strip_mesh(section_strip_mesh, zones)
    clusters = clusters_from_section_strip_mesh(section_strip_mesh)
    shared = interface_constraint_mode == "shared"

    if station_span_coords is not None:
        span = np.asarray(station_span_coords, dtype=float).ravel()
        if span.size != n_layers:
            raise ValueError(
                f"station_span_coords length {span.size} != n_layers {n_layers}"
            )
    else:
        span = np.linspace(0.0, L, n_layers)

    # layer_assignments[ri][(pi, li)] = temporary id; remapped to ccx 1-based later
    next_id = 0
    layer_assignments: list[dict[tuple[int, int], int]] = []
    for _ri in range(n_layers):
        layer_map, next_id = _layer_node_assignment(
            panel_maps, clusters, shared=shared, next_id=next_id
        )
        layer_assignments.append(layer_map)

    nodes_xyz: dict[int, tuple[float, float, float]] = {}
    for ri in range(n_layers):
        x_val = float(span[ri])
        for pm in panel_maps:
            if len(pm.s_nodes) == 0:
                continue
            yz = _yz_for_panel_layer(
                zones,
                pm,
                station_panels=station_panels,
                layer_index=ri,
                n_layers=n_layers,
            )
            for li in range(len(pm.s_nodes)):
                nid = layer_assignments[ri][(pm.panel_index, li)]
                if nid in nodes_xyz:
                    continue
                nodes_xyz[nid] = (x_val, float(yz[li, 0]), float(yz[li, 1]))

    # Compact to 1-based CalculiX ids
    old_to_ccx = {oid: i + 1 for i, oid in enumerate(sorted(nodes_xyz.keys()))}
    ccx_nodes = {old_to_ccx[oid]: nodes_xyz[oid] for oid in nodes_xyz}
    for ri in range(n_layers):
        layer_assignments[ri] = {
            k: old_to_ccx[v] for k, v in layer_assignments[ri].items()
        }

    elements: list[CcxRootElement] = []
    elsets: dict[str, list[int]] = {}
    element_to_key: dict[int, tuple[int, int, int]] = {}
    eid = 1
    for pi, pm in enumerate(panel_maps):
        n = len(pm.s_nodes)
        if n < 2:
            continue
        elset_name = _sanitize_elset_name(pm.panel_label)
        elsets.setdefault(elset_name, [])
        # Contour connectivity: same layout as strip_quad_connectivity per row
        for ei in range(n - 1):
            for row in range(n_rows):
                n00 = layer_assignments[row][(pi, ei)]
                n10 = layer_assignments[row][(pi, ei + 1)]
                n11 = layer_assignments[row + 1][(pi, ei + 1)]
                n01 = layer_assignments[row + 1][(pi, ei)]
                node_ids = [n00, n10, n11, n01]
                elements.append(
                    CcxRootElement(
                        eid=eid,
                        node_ids=node_ids,
                        panel_index=pi,
                        contour_element_index=ei,
                        span_row=row,
                    )
                )
                elsets[elset_name].append(eid)
                element_to_key[eid] = (pi, ei, row)
                eid += 1

    root_ids = sorted(
        {
            layer_assignments[0][(pm.panel_index, li)]
            for pm in panel_maps
            for li in range(len(pm.s_nodes))
            if len(pm.s_nodes) > 0
        }
    )
    outboard_ids = sorted(
        {
            layer_assignments[n_layers - 1][(pm.panel_index, li)]
            for pm in panel_maps
            for li in range(len(pm.s_nodes))
            if len(pm.s_nodes) > 0
        }
    )
    nsets = {
        "ROOT_RING": root_ids,
        "OUTBOARD_RING": outboard_ids,
    }

    # Reference node at outboard elastic centroid (geometry for coupling)
    if outboard_ids:
        y_c = float(np.mean([ccx_nodes[n][1] for n in outboard_ids]))
        z_c = float(np.mean([ccx_nodes[n][2] for n in outboard_ids]))
        x_c = float(span[-1])
    else:
        y_c = z_c = 0.0
        x_c = float(span[-1])
    ref_node_id = (max(ccx_nodes.keys()) + 1) if ccx_nodes else 1
    rot_node_id = ref_node_id + 1
    ccx_nodes[ref_node_id] = (x_c, y_c, z_c)
    # Distinct nearby location for the rigid-body rotation node (CalculiX req.).
    ccx_nodes[rot_node_id] = (x_c, y_c + 1.0e-3, z_c)
    nsets["REFNODE"] = [ref_node_id]
    nsets["ROTNODE"] = [rot_node_id]

    # Stash contour node ids on panel_maps.global_nodes as layer-0 + layer-last for diagnostics
    for pm in panel_maps:
        n = len(pm.s_nodes)
        if n == 0:
            continue
        pm.global_nodes = [-1] * (2 * n)
        for li in range(n):
            pm.global_nodes[li] = layer_assignments[0][(pm.panel_index, li)]
            pm.global_nodes[li + n] = layer_assignments[n_layers - 1][(pm.panel_index, li)]

    return CcxRootSegmentMesh(
        nodes=ccx_nodes,
        elements=elements,
        elsets=elsets,
        nsets=nsets,
        panel_maps=panel_maps,
        clusters=clusters,
        span_coords=span,
        n_span_rows=n_rows,
        segment_length_m=float(span[-1] - span[0]),
        element_to_key=element_to_key,
        ref_node_id=ref_node_id,
        rot_node_id=rot_node_id,
        root_diameter_m=estimate_root_diameter_m(zones),
    )


__all__ = [
    "CcxRootElement",
    "CcxRootSegmentMesh",
    "build_ccx_root_segment_mesh",
    "estimate_root_diameter_m",
]
