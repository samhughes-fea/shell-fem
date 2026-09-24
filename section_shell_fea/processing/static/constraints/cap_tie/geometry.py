"""Cap-to-skin projection, discovery, and typed link report.

Backend-neutral geometry primitives. CCX may import projection/discovery here;
node/surface DTOs, ``*TIE`` emission, and indexing stay in ``calculix_ccx``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.shell_mesh_map import (
    GlobalNodeMeta,
    PanelGlobalMap,
)

CAP_TIE_MODES = ("none", "skin_offset")


def kind_of(label: str) -> str:
    """Panel kind from its label prefix (``skin:``/``cap:``/``web:``)."""
    return label.split(":", 1)[0] if ":" in label else "panel"


def project_onto_polyline(
    q: NDArray[np.float64], poly: NDArray[np.float64]
) -> tuple[float, int, float]:
    """Closest point on ``poly`` to ``q`` as ``(distance, segment_index, param)``.

    ``param`` is the clamped position within the segment, so the projected point
    is ``poly[i] + param * (poly[i + 1] - poly[i])``.
    """
    best_d, best_i, best_t = np.inf, -1, 0.0
    for j in range(poly.shape[0] - 1):
        a = poly[j]
        ab = poly[j + 1] - a
        l2 = float(np.dot(ab, ab))
        if l2 <= 0.0:
            continue
        t = float(np.clip(np.dot(q - a, ab) / l2, 0.0, 1.0))
        d = float(np.linalg.norm(q - (a + t * ab)))
        if d < best_d:
            best_d, best_i, best_t = d, j, t
    return best_d, best_i, best_t


# Compatibility names for CCX / legacy importers.
_kind_of = kind_of
_project_onto_polyline = project_onto_polyline


@dataclass(frozen=True)
class CapTieLink:
    """One rigid link: cap global node ``slave_node`` to a point on the skin."""

    cap_label: str
    slave_node: int
    skin_label: str
    master_nodes: tuple[int, int]
    weights: tuple[float, float]
    offset_yz: tuple[float, float]
    """``r = cap_node - skin_point``; its norm is the link length."""
    layer: str

    @property
    def length_m(self) -> float:
        return float(np.hypot(*self.offset_yz))


@dataclass
class CapTieReport:
    """What the tie builder did, for logging and convergence-sweep provenance."""

    mode: str
    links: list[CapTieLink] = field(default_factory=list)
    caps_seen: int = 0
    caps_tied: int = 0
    caps_skipped_already_shared: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def n_links(self) -> int:
        return len(self.links)

    def offset_stats_mm(self) -> tuple[float, float, float]:
        if not self.links:
            return (0.0, 0.0, 0.0)
        d = np.array([lk.length_m for lk in self.links]) * 1e3
        return (float(d.min()), float(np.median(d)), float(d.max()))

    def summary(self) -> str:
        lo, med, hi = self.offset_stats_mm()
        return (
            f"cap ties [{self.mode}]: {self.caps_tied}/{self.caps_seen} caps, "
            f"{self.n_links} links, offset {lo:.3f}/{med:.3f}/{hi:.3f} mm "
            f"(min/med/max)"
        )


def build_cap_skin_ties(
    panel_maps: list[PanelGlobalMap],
    node_meta: GlobalNodeMeta,
    *,
    cap_kind: str = "cap",
    skin_kind: str = "skin",
) -> CapTieReport:
    """Locate the skin point beneath every cap node, per layer.

    Geometry is read from ``node_meta`` (the same source the assembler uses for
    element geometry), so the link length reflects the mesh actually solved,
    including any inboard/outboard taper difference between the two layers.
    """
    report = CapTieReport(mode="skin_offset")

    skin_maps = [pm for pm in panel_maps if kind_of(pm.panel_label) == skin_kind and len(pm.s_nodes)]
    cap_maps = [pm for pm in panel_maps if kind_of(pm.panel_label) == cap_kind and len(pm.s_nodes)]
    if not cap_maps:
        report.notes.append("no cap panels in this section; nothing to tie")
        return report
    if not skin_maps:
        report.notes.append("no skin panels found; caps left untied")
        return report

    owner_count: dict[int, int] = {}
    for pm in panel_maps:
        for gn in set(pm.global_nodes):
            if gn >= 0:
                owner_count[gn] = owner_count.get(gn, 0) + 1

    for cap in cap_maps:
        report.caps_seen += 1
        n_cap = len(cap.s_nodes)
        tied_any = False
        shared_here = 0
        for layer, base in (("bottom", 0), ("top", n_cap)):
            skin_polys: list[tuple[str, list[int], NDArray[np.float64]]] = []
            for skin in skin_maps:
                n_s = len(skin.s_nodes)
                gids = [skin.global_nodes[base_s] for base_s in range(
                    0 if layer == "bottom" else n_s,
                    n_s if layer == "bottom" else 2 * n_s,
                )]
                if any(g < 0 for g in gids) or len(gids) < 2:
                    continue
                pts = node_meta.yz[np.asarray(gids, dtype=int)]
                if not np.all(np.isfinite(pts)):
                    continue
                skin_polys.append((skin.panel_label, gids, pts))
            if not skin_polys:
                continue

            for li in range(n_cap):
                gn_cap = cap.global_nodes[base + li]
                if gn_cap < 0:
                    continue
                if owner_count.get(gn_cap, 0) > 1:
                    shared_here += 1
                    continue
                q = node_meta.yz[gn_cap]
                if not np.all(np.isfinite(q)):
                    continue
                best = None
                for lab, gids, pts in skin_polys:
                    d, seg, t = project_onto_polyline(q, pts)
                    if seg >= 0 and (best is None or d < best[0]):
                        best = (d, lab, gids, pts, seg, t)
                if best is None:
                    continue
                _d, lab, gids, pts, seg, t = best
                p = pts[seg] + t * (pts[seg + 1] - pts[seg])
                report.links.append(
                    CapTieLink(
                        cap_label=cap.panel_label,
                        slave_node=int(gn_cap),
                        skin_label=lab,
                        master_nodes=(int(gids[seg]), int(gids[seg + 1])),
                        weights=(float(1.0 - t), float(t)),
                        offset_yz=(float(q[0] - p[0]), float(q[1] - p[1])),
                        layer=layer,
                    )
                )
                tied_any = True
        if tied_any:
            report.caps_tied += 1
        if shared_here:
            report.caps_skipped_already_shared.append(f"{cap.panel_label} ({shared_here} node(s))")

    if report.caps_skipped_already_shared:
        report.notes.append(
            "some cap nodes already share a global id and were left to the "
            "junction path: " + "; ".join(report.caps_skipped_already_shared)
        )
    return report


__all__ = [
    "CAP_TIE_MODES",
    "CapTieLink",
    "CapTieReport",
    "build_cap_skin_ties",
    "kind_of",
    "project_onto_polyline",
    "_kind_of",
    "_project_onto_polyline",
]
