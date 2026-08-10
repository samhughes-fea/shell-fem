"""Kernel-agnostic shell strip mesh mapping and global node numbering."""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from blade_structure_utilities.mesh_utilities.section_mesh.discretise import (
    cap_required_stations_from_panels,
    discretise_strip,
)
from blade_structure_utilities.mesh_utilities.section_mesh.mesh_topology import (
    NElements,
    cluster_points,
    panel_endpoints,
)


def resolve_n_elements(spec: NElements, pi: int, default: int = 10) -> int:
    if spec is None:
        return int(default)
    if isinstance(spec, numbers.Integral):
        return int(spec)
    if isinstance(spec, Mapping):
        return int(spec.get(pi, default))
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        return int(spec[pi]) if pi < len(spec) else default
    raise TypeError(f"n_elements_per_panel must be int, Mapping, or Sequence; got {type(spec)}")


def effective_n_elements_spec(
    panels: list[Any],
    n_elements_per_panel: NElements,
    target_element_length_m: float | None,
    *,
    default_n: int = 10,
    min_n: int = 1,
) -> NElements:
    """Resolve per-panel element counts.

    Production default is **length-based**: callers should pass
    ``target_element_length_m`` from ``TARGET_ELEMENT_LENGTH_FRAC * chord_m``
    with ``n_elements_per_panel=None``. An explicit integral count (other than
    the legacy sentinel ``default_n``) overrides the length policy.
    """
    if (
        target_element_length_m is None
        or not np.isfinite(target_element_length_m)
        or float(target_element_length_m) <= 0.0
    ):
        return default_n if n_elements_per_panel is None else n_elements_per_panel
    if n_elements_per_panel is None:
        pass
    elif isinstance(n_elements_per_panel, numbers.Integral) and int(n_elements_per_panel) != default_n:
        return n_elements_per_panel
    elif isinstance(n_elements_per_panel, Mapping):
        return n_elements_per_panel
    elif isinstance(n_elements_per_panel, Sequence) and not isinstance(
        n_elements_per_panel, (str, bytes)
    ):
        return n_elements_per_panel

    tgt = float(target_element_length_m)
    floor = max(1, int(min_n))
    mapping: dict[int, int] = {}
    for pi, p in enumerate(panels):
        s_p = np.asarray(getattr(p, "s", None), dtype=float)
        if s_p.size < 2:
            mapping[pi] = floor
        else:
            arc = float(s_p[-1] - s_p[0])
            mapping[pi] = max(floor, int(round(arc / tgt)))
    return mapping


def panel_end_tangent(nodes_yz: np.ndarray, which: str) -> np.ndarray:
    if len(nodes_yz) < 2:
        return np.array([1.0, 0.0], dtype=float)
    if which == "start":
        t = np.asarray(nodes_yz[1] - nodes_yz[0], dtype=float)
    else:
        t = np.asarray(nodes_yz[-1] - nodes_yz[-2], dtype=float)
    n = float(np.linalg.norm(t))
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    return t / n


def normal_2d(t: np.ndarray) -> np.ndarray:
    return np.array([-float(t[1]), float(t[0])], dtype=float)


def cluster_is_collinear(
    cluster: list[tuple[int, str, Any]],
    panels: list[Any],
    tol_deg: float = 1.0,
) -> bool:
    tangents: list[np.ndarray] = []
    for pi, _which, _ in cluster:
        nodes = np.asarray(getattr(panels[pi], "nodes"), dtype=float)
        tangents.append(panel_end_tangent(nodes, _which))
    if len(tangents) < 2:
        return True
    tol_rad = tol_deg * np.pi / 180.0
    t0 = tangents[0]
    for t in tangents[1:]:
        cos_a = float(np.clip(np.dot(t0, t), -1.0, 1.0))
        angle = float(np.arccos(abs(cos_a)))
        if angle > tol_rad:
            return False
    return True


@dataclass
class PanelGlobalMap:
    s_nodes: np.ndarray
    elements: list[list[int]]
    global_nodes: list[int]
    panel_label: str
    panel_index: int


@dataclass
class GlobalNodeMeta:
    n_nodes: int
    yz: np.ndarray
    is_top: np.ndarray
    tangent_yz: np.ndarray
    arc_length_s: np.ndarray
    pair_group: np.ndarray
    # Cluster-basis (transformed_basis) MPCs, when the assembly used virtual cluster
    # reference nodes. Carried here so the periodic homogeniser can compose the
    # junction-compatibility constraints with periodic pairing. None for shared mode.
    constraints: dict[int, list[tuple[int, float]]] | None = None


def build_panel_mesh_maps(
    panels: list[Any],
    n_elem_spec: NElements,
) -> tuple[list[PanelGlobalMap], list[tuple[int, str, np.ndarray]]]:
    panel_maps: list[PanelGlobalMap] = []
    endpoints: list[tuple[int, str, np.ndarray]] = []
    required_per_panel = cap_required_stations_from_panels(panels)
    for pi, p in enumerate(panels):
        s_panel = np.asarray(p.s, dtype=float)
        if len(s_panel) < 2:
            panel_maps.append(PanelGlobalMap(np.array([]), [], [], str(getattr(p, "label", f"panel_{pi}")), pi))
            continue
        nodes_yz = np.asarray(getattr(p, "nodes"), dtype=float)
        s_nodes, elems = discretise_strip(
            s_panel,
            resolve_n_elements(n_elem_spec, pi),
            panel_label=str(getattr(p, "label", "") or ""),
            nodes_yz=nodes_yz,
            required_s=required_per_panel[pi],
        )
        n_local_nodes = 2 * len(s_nodes)
        panel_maps.append(
            PanelGlobalMap(s_nodes, elems, [-1] * n_local_nodes, str(getattr(p, "label", f"panel_{pi}")), pi)
        )
        ep = panel_endpoints(nodes_yz)
        endpoints.append((pi, "start", ep["start"]))
        endpoints.append((pi, "end", ep["end"]))
    return panel_maps, endpoints


def interp_yz_nodes(s_nodes: np.ndarray, panel: Any) -> np.ndarray:
    nodes_yz = np.asarray(getattr(panel, "nodes"), dtype=float)
    s_panel = np.asarray(panel.s, dtype=float)
    y = np.interp(s_nodes, s_panel, nodes_yz[:, 0])
    z = np.interp(s_nodes, s_panel, nodes_yz[:, 1])
    return np.column_stack([y, z])


def collect_global_node_meta(
    panel_maps: list[PanelGlobalMap],
    panels: list[Any],
    panels_outboard: list[Any] | None,
    n_total_nodes: int,
    *,
    l_x: float = 1.0,
) -> GlobalNodeMeta:
    yz = np.full((n_total_nodes, 2), np.nan)
    is_top = np.zeros(n_total_nodes, dtype=bool)
    tangent_yz = np.full((n_total_nodes, 2), np.nan)
    arc_s = np.full(n_total_nodes, np.nan)
    pair_group = np.full(n_total_nodes, -1, dtype=np.int32)
    next_pair_group = 0

    for pi, pm in enumerate(panel_maps):
        if len(pm.s_nodes) == 0:
            continue
        p_in = panels[pi]
        p_out = panels_outboard[pi] if panels_outboard is not None else p_in
        nodes_yz_in = np.asarray(getattr(p_in, "nodes"), dtype=float)
        s_panel_in = np.asarray(p_in.s, dtype=float)
        nodes_yz_out = np.asarray(getattr(p_out, "nodes"), dtype=float)
        s_panel_out = np.asarray(p_out.s, dtype=float)
        s_nodes = pm.s_nodes
        n = len(s_nodes)

        def _interp_geom(
            nodes_yz_p: np.ndarray,
            s_panel: np.ndarray,
            s_nodes_eval: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            y_m = np.interp(s_nodes_eval, s_panel, nodes_yz_p[:, 0])
            z_m = np.interp(s_nodes_eval, s_panel, nodes_yz_p[:, 1])
            if len(nodes_yz_p) >= 2:
                ty_p = np.gradient(nodes_yz_p[:, 0], s_panel)
                tz_p = np.gradient(nodes_yz_p[:, 1], s_panel)
                ty_m = np.interp(s_nodes_eval, s_panel, ty_p)
                tz_m = np.interp(s_nodes_eval, s_panel, tz_p)
            else:
                ty_m = np.zeros(len(s_nodes_eval))
                tz_m = np.ones(len(s_nodes_eval))
            nrm = np.maximum(np.hypot(ty_m, tz_m), 1e-30)
            return y_m, z_m, ty_m / nrm, tz_m / nrm

        y_in, z_in, ty_in, tz_in = _interp_geom(nodes_yz_in, s_panel_in, s_nodes)
        if panels_outboard is None:
            y_out, z_out, ty_out, tz_out = y_in, z_in, ty_in, tz_in
            s_nodes_out = s_nodes
        else:
            s0_in = float(s_panel_in[0])
            s1_in = float(s_panel_in[-1])
            xi = np.zeros_like(s_nodes) if abs(s1_in - s0_in) <= 1e-30 else (s_nodes - s0_in) / (s1_in - s0_in)
            s0_out = float(s_panel_out[0])
            s1_out = float(s_panel_out[-1])
            s_nodes_out = s0_out + xi * (s1_out - s0_out)
            y_out, z_out, ty_out, tz_out = _interp_geom(nodes_yz_out, s_panel_out, s_nodes_out)

        for li in range(2 * n):
            si = li if li < n else li - n
            gn = pm.global_nodes[li]
            if gn < 0:
                continue
            if np.isnan(yz[gn, 0]):
                if li < n:
                    yz[gn] = [y_in[si], z_in[si]]
                    tangent_yz[gn] = [ty_in[si], tz_in[si]]
                    arc_s[gn] = s_nodes[si]
                else:
                    yz[gn] = [y_out[si], z_out[si]]
                    tangent_yz[gn] = [ty_out[si], tz_out[si]]
                    arc_s[gn] = s_nodes_out[si]
            if pair_group[gn] < 0:
                pair_group[gn] = next_pair_group + si
            is_top[gn] = li >= n
        next_pair_group += n

    return GlobalNodeMeta(
        n_nodes=n_total_nodes,
        yz=yz,
        is_top=is_top,
        tangent_yz=tangent_yz,
        arc_length_s=arc_s,
        pair_group=pair_group,
    )


def scatter_element_ke(
    ke: np.ndarray,
    global_node_ids: list[int],
    n_dof_node: int,
    row_chunks: list[np.ndarray],
    col_chunks: list[np.ndarray],
    val_chunks: list[np.ndarray],
) -> None:
    g_dofs = np.empty(4 * n_dof_node, dtype=np.int64)
    for i, gn in enumerate(global_node_ids):
        base = i * n_dof_node
        g_dofs[base : base + n_dof_node] = gn * n_dof_node + np.arange(n_dof_node, dtype=np.int64)
    n = g_dofs.size
    row_chunks.append(np.repeat(g_dofs, n))
    col_chunks.append(np.tile(g_dofs, n))
    val_chunks.append(np.asarray(ke, dtype=float).ravel())


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
    "resolve_n_elements",
    "scatter_element_ke",
]
