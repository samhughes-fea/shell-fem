"""
section_strip_mesh
==================
Build an explicit FE-agnostic strip mesh from per-subcomponent midlines
produced by :func:`build_shell_mesh_inputs`.

The global coupled solvers discretise each :class:`Panel` into ``n_elements``
quad elements as one spanwise strip (``L_x = 1`` m). This module:

* Interpolates the midline polyline onto the same ``s_nodes`` the solvers use.
* Surfaces endpoint-clustering topology for inspection before loads are applied.

Returned :class:`SectionStripMesh` can be fed to any strip kernel (MITC4, S4R, ccx).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_utilities.mesh.line_1d import (
    LineMeshSpec,
    PolylineLine,
    arc_lengths_from_objects,
    build_line_mesh_1d,
    effective_n_segments_spec,
    strip_quad_connectivity,
)

from .mesh_topology import (
    NElements,
    cluster_points as _cluster_points,
    panel_endpoints as _panel_endpoints,
)
from .shell_inputs_from_section import ShellMeshInputs
from .topology import build_section_v2

# Chord-relative strip-mesh sizing defaults.
TARGET_ELEMENT_LENGTH_FRAC = 0.012  # ~1.2% chord (example-derived)
ENDPOINT_TOL_FRAC = 0.006  # ~0.6% chord (1e-2 m at chord 1.656 m)
MIN_ELEMENTS_PER_PANEL = 2
# Junction clustering must bridge the fixed cap/skin mid-surface wall offset
# (~0.5*(t_skin+t_cap)), which does NOT scale with chord. A purely chord-relative
# endpoint tolerance falls below this gap at small-chord (outboard) stations and
# orphans the spar-cap endpoints. This safety factor floors the chord-relative
# tolerance at the physical wall offset so caps stay connected at every station.
ENDPOINT_TOL_WALL_OFFSET_SAFETY = 1.5


def target_element_length_m_from_chord(
    chord_m: float,
    *,
    frac: float = TARGET_ELEMENT_LENGTH_FRAC,
) -> float:
    """Physical target segment length [m] from chord and fractional sizing."""
    return float(frac) * float(chord_m)


def _resolve_strip_mesh_sizing(
    chord_m: float,
    *,
    target_element_length_m: float | None,
    target_element_length_frac: float | None,
    endpoint_tol: float | None,
    endpoint_tol_frac: float | None,
) -> tuple[float | None, float]:
    if target_element_length_m is not None and target_element_length_frac is not None:
        raise ValueError(
            "Pass either target_element_length_m or target_element_length_frac, not both."
        )
    if endpoint_tol is not None and endpoint_tol_frac is not None:
        raise ValueError("Pass either endpoint_tol or endpoint_tol_frac, not both.")

    if target_element_length_frac is not None:
        tgt_len = target_element_length_m_from_chord(chord_m, frac=target_element_length_frac)
    else:
        tgt_len = target_element_length_m

    if endpoint_tol_frac is not None:
        tol = float(endpoint_tol_frac) * float(chord_m)
    elif endpoint_tol is not None:
        tol = float(endpoint_tol)
    else:
        tol = 1e-6

    return tgt_len, tol


@dataclass(frozen=True)
class StripPanelMesh:
    """Explicit strip mesh for a single :class:`Panel`."""

    panel_index: int
    panel_label: str
    kind: str
    n_elements: int
    s_nodes: NDArray[np.float64]
    yz_nodes: NDArray[np.float64]
    elements: list[list[int]]
    thickness_m: float
    arc_length_m: float
    midline_kind_closed: bool


@dataclass(frozen=True)
class StripJunctionCluster:
    """One endpoint-cluster (junction in section topology)."""

    cluster_id: int
    point_yz: tuple[float, float]
    members: list[tuple[int, str]]
    panel_labels: list[str]


@dataclass(frozen=True)
class SectionStripMesh:
    """Full per-section strip mesh from per-subcomponent midlines."""

    panels: list[Any]
    n_elements_per_panel: list[int]
    panel_meshes: list[StripPanelMesh]
    clusters: list[StripJunctionCluster]
    n_total_nodes: int
    n_total_elements: int
    layout_key: str | None
    twist_rad: float
    chord_m: float

    def summary(self) -> dict[str, Any]:
        per_kind: dict[str, dict[str, Any]] = {}
        for pm in self.panel_meshes:
            d = per_kind.setdefault(
                pm.kind,
                {
                    "n_panels": 0,
                    "n_elements": 0,
                    "n_nodes": 0,
                    "arc_length_m": 0.0,
                    "thickness_m_avg": 0.0,
                },
            )
            d["n_panels"] += 1
            d["n_elements"] += pm.n_elements
            d["n_nodes"] += int(pm.yz_nodes.shape[0]) * 2
            d["arc_length_m"] += pm.arc_length_m
            d["thickness_m_avg"] += pm.thickness_m
        for kind, d in per_kind.items():
            n = max(int(d["n_panels"]), 1)
            d["thickness_m_avg"] = float(d["thickness_m_avg"]) / n
        return {
            "layout_key": self.layout_key,
            "twist_rad": float(self.twist_rad),
            "chord_m": float(self.chord_m),
            "n_panels": int(len(self.panels)),
            "n_total_nodes": int(self.n_total_nodes),
            "n_total_elements": int(self.n_total_elements),
            "n_clusters": int(len(self.clusters)),
            "junctions_by_size": {
                k: sum(1 for c in self.clusters if len(c.members) == k)
                for k in sorted({len(c.members) for c in self.clusters})
            },
            "per_kind": per_kind,
        }


def _kind_from_label(label: str) -> str:
    return label.split(":", 1)[0] if ":" in label else "panel"


def build_section_strip_mesh(
    shell_inputs: ShellMeshInputs,
    *,
    n_elements_per_panel: NElements | None = None,
    target_element_length_m: float | None = None,
    target_element_length_frac: float | None = None,
    min_elements_per_panel: int = MIN_ELEMENTS_PER_PANEL,
    endpoint_tol: float | None = None,
    endpoint_tol_frac: float | None = None,
    skin_lam: Any | None = None,
    web_lam: Any | None = None,
    cap_lam: Any | None = None,
) -> SectionStripMesh:
    """Build the strip mesh for one cross-section (FE-agnostic topology).

    When ``target_element_length_frac`` or ``endpoint_tol_frac`` are set, they
    are resolved against ``shell_inputs.chord_m``. With ``n_elements_per_panel``
    unset, skin and web panels use uniform arc-length spacing at the resolved
    target segment length (minimum ``min_elements_per_panel`` elements per panel).
    """
    chord_m = float(shell_inputs.chord_m)
    target_element_length_m, endpoint_tol = _resolve_strip_mesh_sizing(
        chord_m,
        target_element_length_m=target_element_length_m,
        target_element_length_frac=target_element_length_frac,
        endpoint_tol=endpoint_tol,
        endpoint_tol_frac=endpoint_tol_frac,
    )
    panels, _booms, _webs_geom, _n_cells = build_section_v2(
        shell_inputs,
        skin_lam=skin_lam,
        web_lam=web_lam,
        cap_lam=cap_lam,
    )
    arc_lengths = arc_lengths_from_objects(panels)
    n_seg_spec = effective_n_segments_spec(
        arc_lengths,
        n_elements_per_panel,
        target_element_length_m,
        default_n=10,
        min_n=int(min_elements_per_panel),
    )

    label_to_kind = {f"{m.kind}:{m.label}": m.kind for m in shell_inputs.midlines}
    label_to_closed = {f"{m.kind}:{m.label}": bool(m.closed) for m in shell_inputs.midlines}
    label_to_thickness = {
        f"{m.kind}:{m.label}": float(m.thickness_m) for m in shell_inputs.midlines
    }

    # Floor the (chord-relative) endpoint tolerance at the fixed cap/skin wall
    # offset so cap endpoints keep clustering with the web/skin junction at
    # small-chord stations (the gap ~0.5*(t_skin+t_cap) does not scale with chord).
    if endpoint_tol_frac is not None:
        t_cap = max(
            (float(m.thickness_m) for m in shell_inputs.midlines if m.kind == "cap"),
            default=0.0,
        )
        t_skin = max(
            (float(m.thickness_m) for m in shell_inputs.midlines if m.kind == "skin"),
            default=0.0,
        )
        wall_offset = 0.5 * (t_cap + t_skin)
        if wall_offset > 0.0:
            endpoint_tol = max(endpoint_tol, ENDPOINT_TOL_WALL_OFFSET_SAFETY * wall_offset)

    line_spec = LineMeshSpec(
        n_segments=10,
        target_segment_length_m=target_element_length_m,
    )
    polyline_lines: list[PolylineLine] = []
    for p in panels:
        nodes_yz = np.asarray(p.nodes, dtype=float)
        lab = str(p.label)
        polyline_lines.append(
            PolylineLine(
                id=lab,
                kind=label_to_kind.get(lab, _kind_from_label(lab)),
                points=nodes_yz,
                closed=label_to_closed.get(lab, False),
                thickness_m=label_to_thickness.get(lab, float(p.lam.t)),
            )
        )

    line_mesh = build_line_mesh_1d(
        polyline_lines,
        line_spec,
        n_segments_per_line=n_seg_spec,
        target_segment_length_m=None,
    )

    panel_meshes: list[StripPanelMesh] = []
    n_elements_resolved: list[int] = []
    n_total_nodes = 0
    n_total_elements = 0
    endpoints: list[tuple[int, str, NDArray[np.float64]]] = []

    for pi, p in enumerate(panels):
        s_panel = np.asarray(p.s, dtype=float)
        nodes_yz = np.asarray(p.nodes, dtype=float)
        lab = str(p.label)
        if s_panel.size < 2:
            n_elements_resolved.append(0)
            panel_meshes.append(
                StripPanelMesh(
                    panel_index=pi,
                    panel_label=lab,
                    kind=_kind_from_label(lab),
                    n_elements=0,
                    s_nodes=np.empty(0),
                    yz_nodes=np.empty((0, 2)),
                    elements=[],
                    thickness_m=label_to_thickness.get(lab, float(p.lam.t)),
                    arc_length_m=0.0,
                    midline_kind_closed=label_to_closed.get(lab, False),
                )
            )
            continue
        field = line_mesh.fields[lab]
        s_nodes = field.s_nodes
        yz_nodes = field.nodes
        elems = strip_quad_connectivity(int(s_nodes.shape[0]))
        n_elem_actual = int(len(elems))
        n_elements_resolved.append(n_elem_actual)

        n_local_nodes = 2 * int(s_nodes.shape[0])
        n_total_nodes += n_local_nodes
        n_total_elements += len(elems)

        ep = _panel_endpoints(nodes_yz)
        endpoints.append((pi, "start", ep["start"]))
        endpoints.append((pi, "end", ep["end"]))

        panel_meshes.append(
            StripPanelMesh(
                panel_index=pi,
                panel_label=str(p.label),
                kind=label_to_kind.get(p.label, _kind_from_label(p.label)),
                n_elements=n_elem_actual,
                s_nodes=s_nodes,
                yz_nodes=yz_nodes,
                elements=[list(map(int, e)) for e in elems],
                thickness_m=label_to_thickness.get(p.label, float(p.lam.t)),
                arc_length_m=float(s_panel[-1] - s_panel[0]),
                midline_kind_closed=label_to_closed.get(p.label, False),
            )
        )

    raw_clusters = _cluster_points(endpoints, endpoint_tol)
    clusters: list[StripJunctionCluster] = []
    for cid, c in enumerate(raw_clusters):
        members = [(int(pi), str(end)) for pi, end, _pt in c]
        labels = [str(panels[pi].label) for pi, _e in members]
        pt = c[0][2]
        clusters.append(
            StripJunctionCluster(
                cluster_id=int(cid),
                point_yz=(float(pt[0]), float(pt[1])),
                members=members,
                panel_labels=labels,
            )
        )

    return SectionStripMesh(
        panels=panels,
        n_elements_per_panel=n_elements_resolved,
        panel_meshes=panel_meshes,
        clusters=clusters,
        n_total_nodes=int(n_total_nodes),
        n_total_elements=int(n_total_elements),
        layout_key=shell_inputs.layout_key,
        twist_rad=float(shell_inputs.twist_rad),
        chord_m=float(shell_inputs.chord_m),
    )


__all__ = [
    "ENDPOINT_TOL_FRAC",
    "MIN_ELEMENTS_PER_PANEL",
    "TARGET_ELEMENT_LENGTH_FRAC",
    "StripPanelMesh",
    "StripJunctionCluster",
    "SectionStripMesh",
    "build_section_strip_mesh",
    "target_element_length_m_from_chord",
]
