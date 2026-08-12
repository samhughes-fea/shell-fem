"""Cap–skin ``*TIE`` surface geometry for CalculiX section decks.

Mirrors MITC4 ``cap_tie_mode="skin_offset"`` (BL-003): caps do not share junction
nodes with the skin, so a bonded interface must be emitted explicitly. CalculiX
analogue is ``*TIE`` with ``POSITION TOLERANCE`` covering the measured midline
offset (≈ ``(t_skin + t_cap) / 2``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.shell_mesh_map import (
    PanelGlobalMap,
    interp_yz_nodes,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.cap_tie_constraints import (
    CAP_TIE_MODES,
    _kind_of,
    _project_onto_polyline,
)

from .mesh import CcxSectionMesh, _sanitize_elset_name

_TOL_GAP_SLACK = 1.25
_TOL_ELEM_FRAC = 0.05
_MIN_TOL_M = 1e-6


@dataclass(frozen=True)
class CcxCapSkinTieSurface:
    """One cap→skin bonded interface for CalculiX ``*TIE`` export."""

    name: str
    cap_label: str
    skin_labels: tuple[str, ...]
    slave_ccx_nodes: tuple[int, ...]
    master_ccx_elements: tuple[int, ...]
    position_tolerance: float
    max_gap_m: float
    n_projections: int


def _ccx_id(mesh: CcxSectionMesh, gn: int) -> int | None:
    return mesh.global_to_ccx_node.get(int(gn))


def _owner_count(panel_maps: list[PanelGlobalMap]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for pm in panel_maps:
        for gn in set(int(g) for g in pm.global_nodes if int(g) >= 0):
            counts[gn] = counts.get(gn, 0) + 1
    return counts


def _skin_polyline(
    skin: PanelGlobalMap,
    zones: list,
    *,
    layer: str,
) -> tuple[list[int], NDArray[np.float64]] | None:
    n = len(skin.s_nodes)
    if n < 2:
        return None
    yz = interp_yz_nodes(skin.s_nodes, zones[skin.panel_index])
    if layer == "bottom":
        gids = [int(skin.global_nodes[i]) for i in range(n)]
    else:
        gids = [int(skin.global_nodes[n + i]) for i in range(n)]
    if any(g < 0 for g in gids):
        return None
    pts = np.asarray(yz, dtype=float)
    if pts.shape[0] != n or not np.all(np.isfinite(pts)):
        return None
    return gids, pts


def _median_element_length(mesh: CcxSectionMesh, zones: list, skin_maps: list[PanelGlobalMap]) -> float:
    lengths: list[float] = []
    for skin in skin_maps:
        s = np.asarray(skin.s_nodes, dtype=float)
        if s.size < 2:
            continue
        lengths.extend(float(abs(s[i + 1] - s[i])) for i in range(len(s) - 1))
    if not lengths:
        return 1e-3
    return float(np.median(lengths))


def build_cap_skin_tie_surfaces(
    mesh: CcxSectionMesh,
    zones: list,
    *,
    cap_kind: str = "cap",
    skin_kind: str = "skin",
) -> list[CcxCapSkinTieSurface]:
    """Project each unique cap node onto the skin and collect ``*TIE`` surfaces.

    Slave side: all CCX node ids on the cap (both strip layers) that are not
    already shared-junction nodes. Master side: skin S4R elements whose contour
    segments were hit by at least one projection.
    """
    panel_maps = mesh.panel_maps
    skin_maps = [
        pm for pm in panel_maps if _kind_of(pm.panel_label) == skin_kind and len(pm.s_nodes)
    ]
    cap_maps = [
        pm for pm in panel_maps if _kind_of(pm.panel_label) == cap_kind and len(pm.s_nodes)
    ]
    if not cap_maps or not skin_maps:
        return []

    owners = _owner_count(panel_maps)
    elem_len = _median_element_length(mesh, zones, skin_maps)
    # Map (skin panel_index, local bottom node i) -> element eids using that segment.
    skin_seg_to_eids: dict[tuple[int, int], list[int]] = {}
    for e in mesh.elements:
        pm = panel_maps[e.panel_index]
        if _kind_of(pm.panel_label) != skin_kind:
            continue
        # Strip quads: local nodes [i0, i1, i1+n, i0+n]; segment index = i0.
        i0 = int(pm.elements[e.local_element_index][0])
        skin_seg_to_eids.setdefault((e.panel_index, i0), []).append(int(e.eid))

    surfaces: list[CcxCapSkinTieSurface] = []
    for cap in cap_maps:
        n_cap = len(cap.s_nodes)
        slave_nodes: set[int] = set()
        master_eids: set[int] = set()
        skin_labels: set[str] = set()
        gaps: list[float] = []
        n_proj = 0

        for layer, base in (("bottom", 0), ("top", n_cap)):
            skin_polys: list[tuple[str, int, list[int], NDArray[np.float64]]] = []
            for skin in skin_maps:
                poly = _skin_polyline(skin, zones, layer=layer)
                if poly is None:
                    continue
                gids, pts = poly
                skin_polys.append((skin.panel_label, skin.panel_index, gids, pts))
            if not skin_polys:
                continue

            for li in range(n_cap):
                gn = int(cap.global_nodes[base + li])
                if gn < 0 or owners.get(gn, 0) > 1:
                    continue
                ccx_nid = _ccx_id(mesh, gn)
                if ccx_nid is None:
                    continue
                q = interp_yz_nodes(cap.s_nodes, zones[cap.panel_index])[li]
                if not np.all(np.isfinite(q)):
                    continue
                best = None
                for lab, spi, gids, pts in skin_polys:
                    d, seg, t = _project_onto_polyline(q, pts)
                    if seg >= 0 and (best is None or d < best[0]):
                        best = (d, lab, spi, seg, t)
                if best is None:
                    continue
                d, lab, spi, seg, t_param = best
                slave_nodes.add(int(ccx_nid))
                skin_labels.add(lab)
                gaps.append(float(d))
                n_proj += 1
                for eid in skin_seg_to_eids.get((spi, int(seg)), []):
                    master_eids.add(int(eid))
                # Adjacent segment when projection lands near a knot.
                if float(t_param) < 0.05 and seg > 0:
                    for eid in skin_seg_to_eids.get((spi, int(seg) - 1), []):
                        master_eids.add(int(eid))
                if float(t_param) > 0.95:
                    for eid in skin_seg_to_eids.get((spi, int(seg) + 1), []):
                        master_eids.add(int(eid))

        if not slave_nodes or not master_eids:
            continue

        max_gap = float(max(gaps)) if gaps else 0.0
        tol = max(max_gap * _TOL_GAP_SLACK, _TOL_ELEM_FRAC * elem_len, _MIN_TOL_M)
        tag = _sanitize_elset_name(cap.panel_label)
        surfaces.append(
            CcxCapSkinTieSurface(
                name=tag,
                cap_label=cap.panel_label,
                skin_labels=tuple(sorted(skin_labels)),
                slave_ccx_nodes=tuple(sorted(slave_nodes)),
                master_ccx_elements=tuple(sorted(master_eids)),
                position_tolerance=float(tol),
                max_gap_m=max_gap,
                n_projections=n_proj,
            )
        )
    return surfaces


def resolve_cap_tie_mode(cap_tie_mode: str, *, has_caps: bool) -> str:
    """Normalize ``cap_tie_mode``; default ``skin_offset`` when caps are present."""
    mode = str(cap_tie_mode or ("skin_offset" if has_caps else "none")).strip().lower()
    if mode not in CAP_TIE_MODES:
        raise ValueError(f"cap_tie_mode must be one of {CAP_TIE_MODES}, got {cap_tie_mode!r}")
    return mode


__all__ = [
    "CcxCapSkinTieSurface",
    "build_cap_skin_tie_surfaces",
    "resolve_cap_tie_mode",
]
