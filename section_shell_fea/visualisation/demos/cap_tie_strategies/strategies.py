"""Candidate strategies for tying spar-cap strips into the shell FE (BL-003).

Spar caps are currently disconnected from the assembled system: junction
clustering keys on ``panel_endpoints`` (first/last node only), a cap's endpoints
sit mid-skin-segment, and its web crossings are interior to both strips — so no
cluster forms. Confirmed at assembly level: 0 cap global node ids are shared
with any other panel (5D-F 230 cap nodes, 2D-CN 120, both zero).

Each strategy is a pure function of geometry: it proposes *ties* without
touching the assembler, so alternatives can be compared before committing to an
assembly change. A tie is a ``CapTie`` — one cap node coupled to a target point
on another panel, plus the offset between them.

The offset matters. A cap sits on the inner skin surface, so a cap-to-skin tie
is a rigid link of length ``(t_skin + t_cap) / 2``, not a shared node. Any
strategy that pretends otherwise silently moves the cap onto the skin midline
and loses the very lever arm that makes a cap worth having.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CapTie:
    """One proposed coupling: cap node ``i`` of ``cap_label`` to ``target_label``."""

    cap_label: str
    cap_node_index: int
    cap_yz: tuple[float, float]
    target_label: str
    target_yz: tuple[float, float]
    offset_m: float
    """Distance between the coupled points. 0 => shared node; >0 => rigid link."""


@dataclass
class TieProposal:
    """What a strategy proposes for one section, plus why it may be incomplete."""

    strategy: str
    ties: list[CapTie] = field(default_factory=list)
    caps_seen: int = 0
    caps_with_no_tie: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    topology_change: bool = False
    """True when the strategy removes the cap panel rather than constraining it."""


def _panels_of_kind(mesh, kind: str):
    return [pm for pm in mesh.panel_meshes if pm.kind == kind and pm.yz_nodes.size]


def _nearest_on_polyline(
    q: NDArray[np.float64], poly: NDArray[np.float64]
) -> tuple[float, NDArray[np.float64]]:
    """Closest point on a polyline to ``q``; returns ``(distance, point)``."""
    best_d, best_p = np.inf, poly[0]
    for j in range(poly.shape[0] - 1):
        a = poly[j]
        ab = poly[j + 1] - a
        l2 = float(np.dot(ab, ab))
        if l2 <= 0.0:
            continue
        t = float(np.clip(np.dot(q - a, ab) / l2, 0.0, 1.0))
        p = a + t * ab
        d = float(np.linalg.norm(q - p))
        if d < best_d:
            best_d, best_p = d, p
    return best_d, best_p


# --------------------------------------------------------------------------
# S1 — tie cap nodes that coincide with a web polyline
# --------------------------------------------------------------------------
def tie_cap_web_crossings(mesh, thicknesses, *, tol_m: float = 1e-6) -> TieProposal:
    """Shared node wherever a cap crosses a web.

    Minimal change: reuses the existing shared-node cluster semantics, just
    extended from strip endpoints to interior coincidences. Cheap and exact
    where it applies — but it applies only where caps actually cross webs, which
    is layout-dependent, so it cannot be the whole answer on its own.
    """
    out = TieProposal(strategy="S1_cap_web_crossings")
    webs = [(pm.panel_label, np.asarray(pm.yz_nodes, float)) for pm in _panels_of_kind(mesh, "web")]
    for cap in _panels_of_kind(mesh, "cap"):
        out.caps_seen += 1
        nodes = np.asarray(cap.yz_nodes, float)
        found = 0
        for i, q in enumerate(nodes):
            best = None
            for wlab, w in webs:
                d, p = _nearest_on_polyline(q, w)
                if d <= tol_m and (best is None or d < best[0]):
                    best = (d, wlab, p)
            if best is not None:
                out.ties.append(
                    CapTie(cap.panel_label, i, tuple(q), best[1], tuple(best[2]), float(best[0]))
                )
                found += 1
        if found == 0:
            out.caps_with_no_tie.append(cap.panel_label)
    if out.caps_with_no_tie:
        out.notes.append(
            f"{len(out.caps_with_no_tie)} cap(s) cross no web -> left completely free"
        )
    return out


# --------------------------------------------------------------------------
# S2 — offset rigid link from every cap node to the skin
# --------------------------------------------------------------------------
def tie_cap_skin_offset(mesh, thicknesses, *, tol_frac: float | None = None) -> TieProposal:
    """Couple every cap node to the skin midline through a rigid offset link.

    Physically the truest of the constraint-based options: a cap is bonded to
    the skin over its whole footprint, not only where a web happens to pass. It
    is also the only one that is layout-independent — every cap has a skin under
    it by construction, whereas web crossings are not guaranteed.

    The link length is **measured from the mesh**, not computed from
    thicknesses. An earlier version gated ties on the separation matching
    ``(t_skin + t_cap) / 2`` and scored 10.7 % coverage failures; that gate was
    the bug, not the geometry.

    Two separate things were wrong with the prediction. The cap mid-surface used
    to be offset *vertically* from the outer surface while the skin midline is a
    true normal offset — since fixed, so the measured normal gap now matches the
    formula to within 1e-3. What remains is that the section builder **clamps and
    rescales** requested thicknesses to fit the local cavity, so a prediction
    built from the caller's requested ``skin_thickness`` is wrong by whatever the clamp
    did, regardless of how good the geometry is. Measuring sidesteps both.

    Pass ``tol_frac`` to re-enable the analytic gate (rejecting nodes whose
    separation differs from the stacked prediction by more than that fraction);
    it is kept only so the original scoring can be reproduced.
    """
    out = TieProposal(strategy="S2_cap_skin_offset")
    skins = [(pm.panel_label, np.asarray(pm.yz_nodes, float)) for pm in _panels_of_kind(mesh, "skin")]
    skin_thickness = float(thicknesses.get("skin", 0.0))
    for cap in _panels_of_kind(mesh, "cap"):
        out.caps_seen += 1
        nodes = np.asarray(cap.yz_nodes, float)
        t_cap = float(cap.thickness_m)
        expected = 0.5 * (skin_thickness + t_cap)
        found = 0
        for i, q in enumerate(nodes):
            best = None
            for slab, s in skins:
                d, p = _nearest_on_polyline(q, s)
                if best is None or d < best[0]:
                    best = (d, slab, p)
            if best is None:
                continue
            d, slab, p = best
            if tol_frac is not None and expected > 0 and abs(d - expected) > tol_frac * expected:
                continue
            out.ties.append(CapTie(cap.panel_label, i, tuple(q), slab, tuple(p), float(d)))
            found += 1
        if found == 0:
            out.caps_with_no_tie.append(cap.panel_label)
    out.notes.append("link length is read off the mesh, not derived from thicknesses")
    return out


# --------------------------------------------------------------------------
# S3 — tie only the cap endpoints to the skin
# --------------------------------------------------------------------------
def tie_cap_endpoints_only(mesh, thicknesses) -> TieProposal:
    """Cheapest constraint: pin each cap's two ends to the skin, nothing between.

    Included as the low-water mark. It restores a load path and so should stop
    the divergence, but leaves a long unsupported span whose shear is still
    governed by the strip's own stiffness — expect it to under-constrain.
    """
    out = TieProposal(strategy="S3_cap_endpoints_only")
    skins = [(pm.panel_label, np.asarray(pm.yz_nodes, float)) for pm in _panels_of_kind(mesh, "skin")]
    for cap in _panels_of_kind(mesh, "cap"):
        out.caps_seen += 1
        nodes = np.asarray(cap.yz_nodes, float)
        found = 0
        for i in (0, nodes.shape[0] - 1):
            q = nodes[i]
            best = None
            for slab, s in skins:
                d, p = _nearest_on_polyline(q, s)
                if best is None or d < best[0]:
                    best = (d, slab, p)
            if best is not None:
                out.ties.append(CapTie(cap.panel_label, i, tuple(q), best[1], tuple(best[2]), float(best[0])))
                found += 1
        if found == 0:
            out.caps_with_no_tie.append(cap.panel_label)
    out.notes.append("two ties per cap regardless of span; expect under-constraint")
    return out


# --------------------------------------------------------------------------
# S4 — no cap panel: fold the cap into the skin laminate
# --------------------------------------------------------------------------
def merge_cap_into_skin(mesh, thicknesses) -> TieProposal:
    """Drop the cap strip; thicken the skin laminate over the cap's span.

    No constraints to get wrong, and it is what the *load* channel already
    assumes — ``apply_stacked_cap_shear`` sets ``t_shear = t_skin + t_cap`` on
    the overlapping skin wall and hands the cap only ``q x Gt_share``. So this
    makes the FE agree with the load model instead of contradicting it, and it
    is trivially robust to layout and thickness.

    Cost: the cap stops being a separately reported component, so per-component
    FI for the spar has to be recovered from ply ranges within the stacked skin
    laminate rather than from a panel. That is a reporting change, not a
    modelling compromise, but it is not free.
    """
    out = TieProposal(strategy="S4_merge_cap_into_skin", topology_change=True)
    caps = _panels_of_kind(mesh, "cap")
    out.caps_seen = len(caps)
    out.notes.append("no ties required; cap becomes stacked skin laminate")
    out.notes.append("per-component spar FI must come from ply ranges, not a panel")
    return out


STRATEGIES: dict[str, Callable] = {
    "S1_cap_web_crossings": tie_cap_web_crossings,
    "S2_cap_skin_offset": tie_cap_skin_offset,
    "S3_cap_endpoints_only": tie_cap_endpoints_only,
    "S4_merge_cap_into_skin": merge_cap_into_skin,
}

__all__ = ["CapTie", "TieProposal", "STRATEGIES", *STRATEGIES.keys()]
