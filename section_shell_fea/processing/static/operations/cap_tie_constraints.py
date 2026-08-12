"""Offset rigid-link MPCs tying spar-cap strips to the skin (BL-003, strategy S2).

Spar caps are structurally disconnected in the assembled shell: junction
clustering keys on panel *endpoints*, a cap's endpoints sit mid-skin-segment,
and its web crossings are interior to both strips, so no cluster forms and no
global node is ever shared. Cap in-plane shear therefore diverges under mesh
refinement while skin and web converge.

This module closes that gap without changing topology: every cap node is tied
to the point directly beneath it on the skin midline by a **rigid link**, and
the link is emitted as ordinary master/slave MPCs on the existing constraint
path (``ConstrainGlobalShellSystem`` -> ``ReduceConstrainedShellSystem``). No
new solver machinery is involved.

Why a link and not a shared node
--------------------------------
A cap sits inboard of the skin, so the two midlines are separated. Collapsing
them onto one node would move the cap onto the skin midline and destroy the
lever arm that makes a cap worth having.

The link length is taken **from the meshed geometry**, not from
``(t_skin + t_cap)/2``. The two now agree: since the cap midline was moved onto
the true normal-offset locus, the measured normal gap is ``(t_skin + t_cap)/2``
to within 1e-3 across the catalogue. But that holds only for the thicknesses the
section builder *actually used* — requested values are clamped and rescaled to
fit the local cavity, so a link length derived from the caller's requested
thicknesses is wrong by whatever the clamp did. Measuring sidesteps that, and
degenerates gracefully where the separation is genuinely zero (C-family cap
endpoints sit on the skin polyline by construction).

Kinematics
----------
Nodal DOFs are ``[u_x, u_s, w, beta_s, beta_x, beta_z]`` in each panel's
curvilinear strip frame **C** (≡ **E** on each facet): ``u_s`` along the panel
tangent ``t``, ``w`` along its normal
``n``, ``beta_s`` about ``t``, ``beta_x`` about the beam axis ``e_x``,
``beta_z`` about ``n``. Frames differ between the cap and the skin, so the link
carries a frame rotation as well as an offset.

For a slave cap node ``C`` and a master point ``P`` on the skin, separated by
the in-plane vector ``r = C - P``, rigid-body motion gives ``u_C = u_P +
theta_P x r`` and ``theta_C = theta_P``. With ``cross2(a, b) = a_y b_z -
a_z b_y`` and ``P`` interpolated between skin nodes ``k`` with weights ``N_k``::

    u_x^C  = sum_k N_k [ u_x^k + beta_s^k cross2(t_k, r) + beta_z^k cross2(n_k, r) ]
    u_s^C  = sum_k N_k [ u_s^k (t_k.t_c) + w^k (n_k.t_c) + beta_x^k cross2(r, t_c) ]
    w^C    = sum_k N_k [ u_s^k (t_k.n_c) + w^k (n_k.n_c) + beta_x^k cross2(r, n_c) ]
    beta_s^C = sum_k N_k [ beta_s^k (t_k.t_c) + beta_z^k (n_k.t_c) ]
    beta_x^C = sum_k N_k [ beta_x^k ]
    beta_z^C = sum_k N_k [ beta_s^k (t_k.n_c) + beta_z^k (n_k.n_c) ]

``r = 0`` reduces this to the pure frame rotation, which is exactly the
shared-node compatibility condition, so coincident cap/skin nodes need no
special case.

Master DOFs must be *independent* for the reduction to be exact:
:class:`ReduceConstrainedShellSystem` builds ``T`` by looking each master up in
the independent set and silently drops any master that is itself a slave. This
module raises rather than let a constraint be quietly weakened, which rules out
combining it with ``interface_constraint_mode='transformed_basis'`` where skin
endpoint DOFs are slaves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ....pre_processing.mesh.shell_mesh_map import GlobalNodeMeta, PanelGlobalMap
from .connected_assembly_core import (
    BETA_S,
    BETA_X,
    BETA_ZET,
    U_S,
    U_X,
    W,
    Constraints,
    gdof,
)

CAP_TIE_MODES = ("none", "skin_offset")


def _kind_of(label: str) -> str:
    """Panel kind from its label prefix (``skin:``/``cap:``/``web:``)."""
    return label.split(":", 1)[0] if ":" in label else "panel"


def _cross2(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """2-D cross product ``a_y b_z - a_z b_y`` (the x-component of ``a x b``)."""
    return float(a[0] * b[1] - a[1] * b[0])


def _project_onto_polyline(
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

    skin_maps = [pm for pm in panel_maps if _kind_of(pm.panel_label) == skin_kind and len(pm.s_nodes)]
    cap_maps = [pm for pm in panel_maps if _kind_of(pm.panel_label) == cap_kind and len(pm.s_nodes)]
    if not cap_maps:
        report.notes.append("no cap panels in this section; nothing to tie")
        return report
    if not skin_maps:
        report.notes.append("no skin panels found; caps left untied")
        return report

    # A cap node already sharing a global id with another panel is coupled by the
    # ordinary junction path and must not be given a second, conflicting rule.
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
                    d, seg, t = _project_onto_polyline(q, pts)
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


def cap_tie_constraints(
    report: CapTieReport,
    node_meta: GlobalNodeMeta,
    *,
    ndof: int,
    existing: Constraints | None = None,
) -> Constraints:
    """Turn located ties into rigid-link MPCs on the six shared DOFs.

    Raises when a link would collide with ``existing`` constraints — either the
    cap node is already a slave, or a skin master is itself a slave (which the
    T-matrix reduction would silently drop, weakening the link without warning).
    """
    existing = existing or {}
    existing_slaves = set(existing.keys())
    constraints: Constraints = {}
    has_drilling = int(ndof) >= 6

    tangents = node_meta.tangent_yz
    for lk in report.links:
        t_c = np.asarray(tangents[lk.slave_node], dtype=float)
        if not np.all(np.isfinite(t_c)):
            raise ValueError(
                f"cap node {lk.slave_node} ({lk.cap_label}) has no tangent in "
                "node_meta; cannot orient a rigid link"
            )
        n_c = np.array([-t_c[1], t_c[0]], dtype=float)
        r = np.asarray(lk.offset_yz, dtype=float)

        slave_dofs = {
            U_X: gdof(lk.slave_node, U_X, ndof),
            U_S: gdof(lk.slave_node, U_S, ndof),
            W: gdof(lk.slave_node, W, ndof),
            BETA_S: gdof(lk.slave_node, BETA_S, ndof),
            BETA_X: gdof(lk.slave_node, BETA_X, ndof),
        }
        if has_drilling:
            slave_dofs[BETA_ZET] = gdof(lk.slave_node, BETA_ZET, ndof)
        clash = [d for d in slave_dofs.values() if d in existing_slaves or d in constraints]
        if clash:
            raise ValueError(
                f"cap node {lk.slave_node} ({lk.cap_label}) is already a "
                f"constraint slave on DOFs {sorted(clash)}; refusing to add a "
                "second, conflicting rule"
            )

        terms: dict[int, list[tuple[int, float]]] = {d: [] for d in slave_dofs}
        for gm, n_k in zip(lk.master_nodes, lk.weights):
            if n_k == 0.0:
                continue
            t_k = np.asarray(tangents[gm], dtype=float)
            if not np.all(np.isfinite(t_k)):
                raise ValueError(
                    f"skin master node {gm} ({lk.skin_label}) has no tangent in "
                    "node_meta; cannot orient a rigid link"
                )
            n_kv = np.array([-t_k[1], t_k[0]], dtype=float)
            tt = float(np.dot(t_k, t_c))
            tn = float(np.dot(t_k, n_c))
            nt = float(np.dot(n_kv, t_c))
            nn = float(np.dot(n_kv, n_c))
            cr_t = _cross2(t_k, r)
            cr_n = _cross2(n_kv, r)
            cr_rt = _cross2(r, t_c)
            cr_rn = _cross2(r, n_c)

            m_ux = gdof(gm, U_X, ndof)
            m_us = gdof(gm, U_S, ndof)
            m_w = gdof(gm, W, ndof)
            m_bs = gdof(gm, BETA_S, ndof)
            m_bx = gdof(gm, BETA_X, ndof)
            m_bz = gdof(gm, BETA_ZET, ndof) if has_drilling else None

            masters = [m_ux, m_us, m_w, m_bs, m_bx] + ([m_bz] if has_drilling else [])
            bad = [d for d in masters if d in existing_slaves]
            if bad:
                raise ValueError(
                    f"skin master node {gm} ({lk.skin_label}) has DOFs {sorted(bad)} "
                    "that are themselves constraint slaves; the reduction would drop "
                    "them and silently weaken the cap tie. Use "
                    "interface_constraint_mode='shared' with cap ties."
                )

            terms[U_X] += [(m_ux, n_k), (m_bs, n_k * cr_t)]
            terms[U_S] += [(m_us, n_k * tt), (m_w, n_k * nt), (m_bx, n_k * cr_rt)]
            terms[W] += [(m_us, n_k * tn), (m_w, n_k * nn), (m_bx, n_k * cr_rn)]
            terms[BETA_S] += [(m_bs, n_k * tt)]
            terms[BETA_X] += [(m_bx, n_k)]
            if has_drilling:
                terms[U_X] += [(m_bz, n_k * cr_n)]
                terms[BETA_S] += [(m_bz, n_k * nt)]
                terms[BETA_ZET] += [(m_bs, n_k * tn), (m_bz, n_k * nn)]

        for local, sdof in slave_dofs.items():
            kept = [(m, c) for m, c in terms[local] if abs(c) > 1e-14]
            if kept:
                constraints[sdof] = kept

    return constraints


def build_cap_tie_constraints(
    panel_maps: list[PanelGlobalMap],
    node_meta: GlobalNodeMeta,
    *,
    ndof: int,
    mode: str = "none",
    existing: Constraints | None = None,
) -> tuple[Constraints, CapTieReport]:
    """Locate and emit cap ties in one call. ``mode='none'`` is a no-op."""
    mode = str(mode or "none")
    if mode not in CAP_TIE_MODES:
        raise ValueError(f"cap_tie_mode must be one of {CAP_TIE_MODES}; got {mode!r}")
    if mode == "none":
        return {}, CapTieReport(mode="none", notes=["cap ties disabled"])
    report = build_cap_skin_ties(panel_maps, node_meta)
    return cap_tie_constraints(report, node_meta, ndof=ndof, existing=existing), report


__all__ = [
    "CAP_TIE_MODES",
    "CapTieLink",
    "CapTieReport",
    "build_cap_skin_ties",
    "build_cap_tie_constraints",
    "cap_tie_constraints",
]
