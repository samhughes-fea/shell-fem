"""Cap–skin ``*TIE`` surface geometry for CalculiX section decks.

Mirrors MITC4-FSDT ``cap_tie_mode="skin_offset"`` (BL-003 / FEA-CT-001): caps do
not share junction nodes with the skin, so a bonded interface must be emitted
explicitly. Membership follows SG-017 ``StripBond`` (named host), not nearest
skin projection. CalculiX analogue is ``*TIE`` with ``POSITION TOLERANCE``
covering the measured midline offset (≈ ``(t_skin + t_cap) / 2``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.panel_node_assembly import (
    PanelNodeAssemblyMap,
    interp_yz_nodes,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.cap_tie.geometry import (
    CAP_TIE_MODES,
    _bond_locate_tol_m,
    _locate_on_bonds,
    _resolve_skin_panel,
    bonds_by_cap_panel_label,
    kind_of as _kind_of,
    project_onto_polyline as _project_onto_polyline,
)

from .mesh import CcxSectionMesh, _sanitize_elset_name

_TOL_GAP_SLACK = 1.25
_TOL_ELEM_FRAC = 0.05
_MIN_TOL_M = 1e-6


@dataclass(frozen=True)
class CcxCapEquationLink:
    """Per-slave weighted masters for ``*EQUATION`` export (FEA-CCX-002).

    Mirrors MITC4 ``CapTieLink``: bond-named host polyline foot → two master
    CCX nodes with interpolating weights. No nearest-master rediscovery.
    """

    slave_ccx_node: int
    master_ccx_nodes: tuple[int, int]
    weights: tuple[float, float]


@dataclass(frozen=True)
class CcxCapSkinTieSurface:
    """One cap→skin bonded interface for CalculiX ``*TIE`` / ``*EQUATION`` export."""

    name: str
    cap_label: str
    skin_labels: tuple[str, ...]
    slave_ccx_nodes: tuple[int, ...]
    master_ccx_elements: tuple[int, ...]
    position_tolerance: float
    max_gap_m: float
    n_projections: int
    equation_links: tuple[CcxCapEquationLink, ...] = ()


def _ccx_id(mesh: CcxSectionMesh, gn: int) -> int | None:
    return mesh.global_to_ccx_node.get(int(gn))


def _owner_count(panel_maps: list[PanelNodeAssemblyMap]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for pm in panel_maps:
        for gn in set(int(g) for g in pm.global_nodes if int(g) >= 0):
            counts[gn] = counts.get(gn, 0) + 1
    return counts


def _skin_polyline(
    skin: PanelNodeAssemblyMap,
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


def _median_element_length(mesh: CcxSectionMesh, zones: list, skin_maps: list[PanelNodeAssemblyMap]) -> float:
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
    shell_midlines: Sequence[Any] | None = None,
    cap_kind: str = "cap",
    skin_kind: str = "skin",
) -> list[CcxCapSkinTieSurface]:
    """Build ``*TIE`` surfaces from SG-017 bonds (FEA-CT-001).

    Slave side: exclusive cap CCX nodes. Master side: skin S4R elements on the
    bond-named host whose contour segments hold the bond foot.
    """
    panel_maps = mesh.panel_maps
    skin_maps = [
        pm for pm in panel_maps if _kind_of(pm.panel_label) == skin_kind and len(pm.s_nodes)
    ]
    cap_maps = [
        pm for pm in panel_maps if _kind_of(pm.panel_label) == cap_kind and len(pm.s_nodes)
    ]
    if not cap_maps:
        return []
    if not skin_maps:
        raise ValueError(
            "cap_tie_mode='skin_offset' but no skin panels found; cannot build *TIE surfaces"
        )
    if shell_midlines is None:
        raise ValueError(
            "cap_tie_mode='skin_offset' requires shell_midlines with StripBond "
            "entries (SG-017); none were passed (FEA-CT-001)"
        )

    bonds_by_cap = bonds_by_cap_panel_label(shell_midlines)
    skin_by_label = {pm.panel_label: pm for pm in skin_maps}
    owners = _owner_count(panel_maps)
    elem_len = _median_element_length(mesh, zones, skin_maps)
    skin_seg_to_eids: dict[tuple[int, int], list[int]] = {}
    for e in mesh.elements:
        pm = panel_maps[e.panel_index]
        if _kind_of(pm.panel_label) != skin_kind:
            continue
        i0 = int(pm.elements[e.local_element_index][0])
        skin_seg_to_eids.setdefault((e.panel_index, i0), []).append(int(e.eid))

    surfaces: list[CcxCapSkinTieSurface] = []
    for cap in cap_maps:
        bonds = bonds_by_cap.get(cap.panel_label, ())
        if not bonds:
            raise ValueError(
                f"cap {cap.panel_label!r} has no StripBond entries under skin_offset; "
                "refusing nearest-skin projection (FEA-CT-001 / SG-017)"
            )

        n_cap = len(cap.s_nodes)
        slave_nodes: set[int] = set()
        master_eids: set[int] = set()
        skin_labels: set[str] = set()
        equation_links: list[CcxCapEquationLink] = []
        gaps: list[float] = []
        n_proj = 0
        cap_yz = interp_yz_nodes(cap.s_nodes, zones[cap.panel_index])

        for layer, base in (("bottom", 0), ("top", n_cap)):
            locate_tol = _bond_locate_tol_m(np.asarray(cap_yz, dtype=float))
            for li in range(n_cap):
                gn = int(cap.global_nodes[base + li])
                if gn < 0 or owners.get(gn, 0) > 1:
                    continue
                ccx_nid = _ccx_id(mesh, gn)
                if ccx_nid is None:
                    continue
                q = np.asarray(cap_yz[li], dtype=float)
                if not np.all(np.isfinite(q)):
                    continue
                bond, foot, _d_bond = _locate_on_bonds(
                    q, bonds, max_distance_m=locate_tol, cap_label=cap.panel_label
                )
                skin = _resolve_skin_panel(
                    str(bond.host_label), skin_by_label, cap_label=cap.panel_label
                )
                poly = _skin_polyline(skin, zones, layer=layer)
                if poly is None:
                    raise ValueError(
                        f"cap {cap.panel_label!r}: host skin {skin.panel_label!r} "
                        f"layer {layer!r} has no usable mesh polyline"
                    )
                gids, pts = poly
                d_mesh, seg, t_param = _project_onto_polyline(foot, pts)
                if seg < 0:
                    raise ValueError(
                        f"cap {cap.panel_label!r}: bond foot could not be located on "
                        f"host skin mesh {skin.panel_label!r}"
                    )
                m0_gn = int(gids[int(seg)])
                m1_gn = int(gids[int(seg) + 1])
                m0 = _ccx_id(mesh, m0_gn)
                m1 = _ccx_id(mesh, m1_gn)
                if m0 is None or m1 is None:
                    raise ValueError(
                        f"cap {cap.panel_label!r}: host skin masters "
                        f"{m0_gn}/{m1_gn} have no CCX node ids"
                    )
                slave_nodes.add(int(ccx_nid))
                skin_labels.add(skin.panel_label)
                gaps.append(float(np.linalg.norm(q - foot)))
                n_proj += 1
                equation_links.append(
                    CcxCapEquationLink(
                        slave_ccx_node=int(ccx_nid),
                        master_ccx_nodes=(int(m0), int(m1)),
                        weights=(float(1.0 - t_param), float(t_param)),
                    )
                )
                spi = int(skin.panel_index)
                for eid in skin_seg_to_eids.get((spi, int(seg)), []):
                    master_eids.add(int(eid))
                if float(t_param) < 0.05 and seg > 0:
                    for eid in skin_seg_to_eids.get((spi, int(seg) - 1), []):
                        master_eids.add(int(eid))
                if float(t_param) > 0.95:
                    for eid in skin_seg_to_eids.get((spi, int(seg) + 1), []):
                        master_eids.add(int(eid))
                _ = d_mesh  # diagnostic only

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
                equation_links=tuple(equation_links),
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
    "CcxCapEquationLink",
    "CcxCapSkinTieSurface",
    "build_cap_skin_tie_surfaces",
    "resolve_cap_tie_mode",
]
