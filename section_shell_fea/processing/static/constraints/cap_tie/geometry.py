"""Cap-to-skin bond membership and typed link report (FEA-CT-001).

Backend-neutral geometry. Membership comes from SG-017 ``StripBond`` on
``ShellMidlineStrip.bonds`` — never from nearest-skin rediscovery.
``project_onto_polyline`` remains for locating a known point on a named polyline
and for diagnostic gap checks only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.shell_mesh_map import (
    GlobalNodeMeta,
    PanelGlobalMap,
)
from blade_structure_utilities.geometry_utilities.thin_wall_section import (
    COINCIDENCE_TOL_M,
)

CAP_TIE_MODES = ("none", "skin_offset")

# Densified mid-edge nodes may sit off the coarser bond polyline by a fraction
# of the local mesh edge; beyond that the node is not on the producer bond.
_BOND_MESH_EDGE_FRAC = 0.25


def kind_of(label: str) -> str:
    """Panel kind from its label prefix (``skin:``/``cap:``/``web:``)."""
    return label.split(":", 1)[0] if ":" in label else "panel"


def project_onto_polyline(
    q: NDArray[np.float64], poly: NDArray[np.float64]
) -> tuple[float, int, float]:
    """Closest point on ``poly`` to ``q`` as ``(distance, segment_index, param)``.

    ``param`` is the clamped position within the segment, so the projected point
    is ``poly[i] + param * (poly[i + 1] - poly[i])``. Used to locate a point on a
    *named* polyline (bond or host mesh), never to choose among skins.
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


def panel_label_for_strip(*, kind: str, label: str) -> str:
    """FE panel label for a contract strip (``kind:label``)."""
    return f"{kind}:{label}"


def bonds_by_cap_panel_label(
    midlines: Sequence[Any] | None,
) -> dict[str, tuple[Any, ...]]:
    """Map ``cap:<strip.label>`` → that strip's ``bonds`` tuple."""
    out: dict[str, tuple[Any, ...]] = {}
    for m in midlines or ():
        if str(getattr(m, "kind", "")) != "cap":
            continue
        key = panel_label_for_strip(kind="cap", label=str(m.label))
        out[key] = tuple(getattr(m, "bonds", ()) or ())
    return out


def _median_segment_length(pts: NDArray[np.float64]) -> float:
    if pts.shape[0] < 2:
        return 1e-3
    lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    lengths = lengths[np.isfinite(lengths) & (lengths > 0.0)]
    if lengths.size == 0:
        return 1e-3
    return float(np.median(lengths))


def _bond_locate_tol_m(cap_mesh_pts: NDArray[np.float64]) -> float:
    return max(
        float(COINCIDENCE_TOL_M),
        _BOND_MESH_EDGE_FRAC * _median_segment_length(cap_mesh_pts),
    )


def _locate_on_bonds(
    q: NDArray[np.float64],
    bonds: Sequence[Any],
    *,
    max_distance_m: float,
    cap_label: str,
) -> tuple[Any, NDArray[np.float64], float]:
    """Return ``(bond, host_foot, distance_to_bond_points)`` for cap node ``q``."""
    best: tuple[float, Any, NDArray[np.float64]] | None = None
    for bond in bonds:
        pts = np.asarray(bond.points, dtype=float)
        host = np.asarray(bond.host_points, dtype=float)
        if pts.shape[0] < 2 or host.shape != pts.shape:
            raise ValueError(
                f"cap {cap_label!r}: StripBond to {getattr(bond, 'host_label', '?')!r} "
                f"has malformed points/host_points shapes {pts.shape}/{host.shape}"
            )
        d, seg, t = project_onto_polyline(q, pts)
        if seg < 0:
            continue
        if best is None or d < best[0]:
            foot = host[seg] + t * (host[seg + 1] - host[seg])
            best = (d, bond, np.asarray(foot, dtype=float))
    if best is None:
        raise ValueError(
            f"cap {cap_label!r}: exclusive node could not be located on any "
            "StripBond.points polyline"
        )
    d, bond, foot = best
    if d > max_distance_m:
        raise ValueError(
            f"cap {cap_label!r}: exclusive node is {d:.3e} m from StripBond.points "
            f"(limit {max_distance_m:.3e} m); refusing nearest-skin fallback (FEA-CT-001)"
        )
    return bond, foot, d


def _resolve_skin_panel(
    host_label: str,
    skin_by_label: Mapping[str, PanelGlobalMap],
    *,
    cap_label: str,
) -> PanelGlobalMap:
    key = panel_label_for_strip(kind="skin", label=str(host_label))
    if key in skin_by_label:
        return skin_by_label[key]
    if host_label in skin_by_label:
        return skin_by_label[host_label]
    raise ValueError(
        f"cap {cap_label!r}: StripBond.host_label={host_label!r} has no matching "
        f"skin panel (tried {key!r})"
    )


def _cap_tie_geometry_identity(
    panel_maps: list[PanelGlobalMap],
    node_meta: GlobalNodeMeta,
    *,
    shell_midlines: Sequence[Any] | None,
    cap_kind: str,
    skin_kind: str,
) -> tuple[Any, ...]:
    """Immutable identity for caching ``build_cap_skin_ties`` products."""
    panels: list[tuple[Any, ...]] = []
    for pm in panel_maps:
        g_arr = np.asarray(pm.global_nodes, dtype=int).ravel()
        s_arr = np.asarray(pm.s_nodes, dtype=float).ravel()
        panels.append(
            (
                str(pm.panel_label),
                tuple(int(g) for g in g_arr.tolist()),
                tuple(float(s) for s in s_arr.tolist()),
            )
        )
    yz = np.asarray(node_meta.yz, dtype=np.float64)
    yz_fp = (
        tuple(yz.shape),
        hash(yz.tobytes()) if yz.size else 0,
    )
    bonds: list[tuple[Any, ...]] = []
    for m in shell_midlines or ():
        kind = str(getattr(m, "kind", ""))
        label = str(getattr(m, "label", ""))
        bond_rows: list[tuple[Any, ...]] = []
        for bond in tuple(getattr(m, "bonds", ()) or ()):
            pts = np.asarray(getattr(bond, "points", ()), dtype=np.float64)
            host = np.asarray(getattr(bond, "host_points", ()), dtype=np.float64)
            bond_rows.append(
                (
                    str(getattr(bond, "host_label", "")),
                    hash(pts.tobytes()) if pts.size else 0,
                    hash(host.tobytes()) if host.size else 0,
                    tuple(pts.shape),
                    tuple(host.shape),
                )
            )
        bonds.append((kind, label, tuple(bond_rows)))
    return (str(cap_kind), str(skin_kind), tuple(panels), yz_fp, tuple(bonds))


@dataclass
class CapTieGeometryCache:
    """Exact-key cache of immutable cap-tie link geometry (WP4 D4.3)."""

    key: tuple[Any, ...] | None = None
    report: CapTieReport | None = None
    hits: int = 0
    misses: int = 0

    def clear(self) -> None:
        self.key = None
        self.report = None
        self.hits = 0
        self.misses = 0


_CAP_TIE_GEOMETRY_CACHE = CapTieGeometryCache()


def clear_cap_tie_geometry_cache() -> None:
    """Drop the process-local cap-tie geometry cache (tests / station reset)."""
    _CAP_TIE_GEOMETRY_CACHE.clear()


def build_cap_skin_ties(
    panel_maps: list[PanelGlobalMap],
    node_meta: GlobalNodeMeta,
    *,
    shell_midlines: Sequence[Any] | None = None,
    cap_kind: str = "cap",
    skin_kind: str = "skin",
    geometry_cache: CapTieGeometryCache | None = None,
) -> CapTieReport:
    """Build rigid-link geometry from SG-017 bonds (FEA-CT-001).

    Every exclusive cap node is placed on that cap's ``StripBond.points``, the
    aligned ``host_points`` foot is read, and masters/weights come only from the
    named host skin panel. ``shell_midlines`` is required whenever any cap panel
    is present.

    Immutable link geometry is cached under a complete content identity
    (panel numbering, node coordinates, StripBond polylines, kinds).
    """
    cache = geometry_cache if geometry_cache is not None else _CAP_TIE_GEOMETRY_CACHE
    identity = _cap_tie_geometry_identity(
        panel_maps,
        node_meta,
        shell_midlines=shell_midlines,
        cap_kind=cap_kind,
        skin_kind=skin_kind,
    )
    if cache.key == identity and cache.report is not None:
        cache.hits += 1
        # Return a shallow copy so callers can mutate notes without corrupting
        # the cached product; links remain the immutable CapTieLink tuples.
        cached = cache.report
        return CapTieReport(
            mode=cached.mode,
            links=list(cached.links),
            caps_seen=cached.caps_seen,
            caps_tied=cached.caps_tied,
            caps_skipped_already_shared=list(cached.caps_skipped_already_shared),
            notes=list(cached.notes),
        )
    report = CapTieReport(mode="skin_offset")

    skin_maps = [pm for pm in panel_maps if kind_of(pm.panel_label) == skin_kind and len(pm.s_nodes)]
    cap_maps = [pm for pm in panel_maps if kind_of(pm.panel_label) == cap_kind and len(pm.s_nodes)]
    if not cap_maps:
        report.notes.append("no cap panels in this section; nothing to tie")
        return report
    if not skin_maps:
        raise ValueError(
            "cap_tie_mode='skin_offset' but no skin panels found; cannot build ties"
        )

    bonds_by_cap = bonds_by_cap_panel_label(shell_midlines)
    if not bonds_by_cap and shell_midlines is None:
        raise ValueError(
            "cap_tie_mode='skin_offset' requires shell_midlines with StripBond "
            "entries (SG-017); none were passed (FEA-CT-001)"
        )

    skin_by_label = {pm.panel_label: pm for pm in skin_maps}

    owner_count: dict[int, int] = {}
    for pm in panel_maps:
        for gn in set(pm.global_nodes):
            if gn >= 0:
                owner_count[gn] = owner_count.get(gn, 0) + 1

    diag_gaps: list[float] = []

    for cap in cap_maps:
        report.caps_seen += 1
        bonds = bonds_by_cap.get(cap.panel_label)
        if bonds is None:
            # Midlines present but this panel label missing, or empty midlines map.
            if shell_midlines is not None and not any(
                str(getattr(m, "kind", "")) == "cap"
                and panel_label_for_strip(kind="cap", label=str(m.label)) == cap.panel_label
                for m in shell_midlines
            ):
                raise ValueError(
                    f"cap panel {cap.panel_label!r} has no matching ShellMidlineStrip "
                    "in shell_midlines (FEA-CT-001)"
                )
            bonds = ()
        if not bonds:
            raise ValueError(
                f"cap {cap.panel_label!r} has no StripBond entries under skin_offset; "
                "refusing nearest-skin projection (FEA-CT-001 / SG-017)"
            )

        n_cap = len(cap.s_nodes)
        tied_any = False
        shared_here = 0
        for layer, base in (("bottom", 0), ("top", n_cap)):
            # Cap mesh polyline for this layer (identification bound).
            cap_gids = [cap.global_nodes[base + li] for li in range(n_cap)]
            if any(g < 0 for g in cap_gids):
                continue
            cap_pts = node_meta.yz[np.asarray(cap_gids, dtype=int)]
            if not np.all(np.isfinite(cap_pts)):
                continue
            locate_tol = _bond_locate_tol_m(cap_pts)

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

                bond, foot, _d_bond = _locate_on_bonds(
                    q, bonds, max_distance_m=locate_tol, cap_label=cap.panel_label
                )
                skin = _resolve_skin_panel(
                    str(bond.host_label), skin_by_label, cap_label=cap.panel_label
                )
                n_s = len(skin.s_nodes)
                gids = [
                    skin.global_nodes[base_s]
                    for base_s in range(
                        0 if layer == "bottom" else n_s,
                        n_s if layer == "bottom" else 2 * n_s,
                    )
                ]
                if any(g < 0 for g in gids) or len(gids) < 2:
                    raise ValueError(
                        f"cap {cap.panel_label!r}: host skin {skin.panel_label!r} "
                        f"layer {layer!r} has incomplete global nodes"
                    )
                pts = node_meta.yz[np.asarray(gids, dtype=int)]
                if not np.all(np.isfinite(pts)):
                    raise ValueError(
                        f"cap {cap.panel_label!r}: host skin {skin.panel_label!r} "
                        f"layer {layer!r} has non-finite mesh coordinates"
                    )
                d_mesh, seg, t = project_onto_polyline(foot, pts)
                if seg < 0:
                    raise ValueError(
                        f"cap {cap.panel_label!r}: bond foot could not be located on "
                        f"host skin mesh {skin.panel_label!r}"
                    )
                # Diagnostic: distance from bond foot to mesh polyline (should be ~0).
                diag_gaps.append(float(d_mesh))
                # Masters interpolate to ``p``; offset must use the same point so
                # rigid-body motion satisfies the MPC identically.
                p = pts[seg] + t * (pts[seg + 1] - pts[seg])
                report.links.append(
                    CapTieLink(
                        cap_label=cap.panel_label,
                        slave_node=int(gn_cap),
                        skin_label=skin.panel_label,
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
            report.caps_skipped_already_shared.append(
                f"{cap.panel_label} ({shared_here} node(s))"
            )

    if report.caps_skipped_already_shared:
        report.notes.append(
            "some cap nodes already share a global id and were left to the "
            "junction path: " + "; ".join(report.caps_skipped_already_shared)
        )
    if diag_gaps:
        report.notes.append(
            "bond-foot→named-host mesh gap "
            f"max={max(diag_gaps):.3e} m med={float(np.median(diag_gaps)):.3e} m "
            "(diagnostic only; membership from StripBond)"
        )
    cache.key = identity
    cache.report = CapTieReport(
        mode=report.mode,
        links=list(report.links),
        caps_seen=report.caps_seen,
        caps_tied=report.caps_tied,
        caps_skipped_already_shared=list(report.caps_skipped_already_shared),
        notes=list(report.notes),
    )
    cache.misses += 1
    return report


__all__ = [
    "CAP_TIE_MODES",
    "CapTieGeometryCache",
    "CapTieLink",
    "CapTieReport",
    "bonds_by_cap_panel_label",
    "build_cap_skin_ties",
    "clear_cap_tie_geometry_cache",
    "kind_of",
    "panel_label_for_strip",
    "project_onto_polyline",
    "_kind_of",
    "_project_onto_polyline",
]
