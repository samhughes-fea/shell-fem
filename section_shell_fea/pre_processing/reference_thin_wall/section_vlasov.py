"""
E-weighted thin-wall sectorial Vlasov warping for a blade cross-section.

Physics
-------
For a closed thin-walled section, the warping function ω̂(s) must be
single-valued (continuous) around each cell.  The open-outline sectorial
coordinate ω_open (computed by ``omega_vertices_open_chain``) has a jump
discontinuity at the cut point equal to twice the cell area.  The
**Batho (1/t) correction** removes this jump:

  Single cell:
    c = ∮ ω_open (ds/t) / ∮ (ds/t)
    ω̂ = ω_open − c

  Multi-cell (n cells, solved simultaneously):
    Flexibility matrix:  δ_ii = ∮_i ds/t,   δ_ij = −∫_{shared web ij} ds/t
    RHS:                 R_i  = ∮_i ω_open ds/t
    System:              [δ] c = R  →  c_i per cell
    ω̂(s) = ω_open(s) − c_{cell(s)}

This replaces the t-weighted mean used in ``normalized_warping``, which is
only correct for sections with uniform thickness.

E-weighted warping constant
---------------------------
  I_ω_E = Σ_panels ∫ E_axial(s) ω̂²(s) t(s) ds   (correct units: Pa·m⁶)

Warping stresses (given bimoment B and its gradient dB/dx)
----------------------------------------------------------
  σ_ω(s) = B · ω̂(s) / I_ω_E                  [Pa]
  q_ω(s) = −(dB/dx / I_ω_E) ∫₀ˢ E ω̂ t ds'    [N/m]
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.geometry_utilities.coordinate_frames import (
    FRAME_B,
    FrameTag,
    require_frame_tag,
)
from blade_structure_utilities.geometry_utilities.section_thin_wall_model import (
    COINCIDENCE_TOL_M,
    graph_from_midlines,
    sectorial_warping,
    wall_labels_from_loops,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.reference_thin_wall.sectorial_warping import (
    open_outline_from_airfoil,
    omega_vertices_open_chain,
    shear_center_outer_skin_loop,
)


def _panel_kind(label: str) -> str:
    """Panel kind from label prefix (``skin:``/``cap:``/``web:``).

    Neutral local copy of the cap-tie ``kind_of`` helper so this prep home does
    not import ``processing`` (dependency direction).
    """
    return label.split(":", 1)[0] if ":" in label else "panel"


@dataclass
class SectionVlasovResult:
    """Sectorial warping quantities for one cross-section."""

    y_sc: float              # shear centre y-coordinate [m]
    z_sc: float              # shear centre z-coordinate [m]
    omega_hat_v: NDArray     # ω̂ at outline vertices [m²] (n_verts,)
    I_omega_E: float         # E-weighted warping constant [Pa·m⁶ / E_ref]
    n_cells: int             # number of closed cells detected
    web_panel_indices: list[int]   # panel indices identified as spar webs
    # Per-panel arrays (one entry per panel, values at element mid-stations)
    panel_s_mids: list[NDArray]    # arc-length at element mids [m]
    sigma_omega: list[NDArray]     # warping normal stress [Pa] (zero for webs)
    q_omega: list[NDArray]         # secondary warping shear flow [N/m] (zero for webs)
    # Shear-centre pierce coords are frame B (ARCH-MUST-016 / FEA-R-002).
    pierce_frame: FrameTag = FRAME_B

    def __post_init__(self) -> None:
        self.pierce_frame = require_frame_tag(self.pierce_frame)
        if self.pierce_frame != FRAME_B:
            raise ValueError(
                f"SectionVlasovResult.pierce_frame must be {FRAME_B!r}; "
                f"got {self.pierce_frame!r} (FEA-R-002)"
            )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_section_vlasov(
    airfoil: NDArray,
    zones: list,
    B: float,
    dB_dx: float,
    webs_geom: list | None = None,
    t_web: float | None = None,
    *,
    shell_midlines: Sequence[Any] | None = None,
    cell_wall_loops: Sequence[Any] | None = None,
) -> SectionVlasovResult:
    """
    Compute E-weighted closed-cell sectorial warping for a blade cross-section.

    Parameters
    ----------
    airfoil   : (N, 2) array [y, z] — legacy outline when midlines are absent
                and there are no webs.
    zones     : MaterialZone objects (``material_utilities.section_bind.MaterialZone``)
    B         : bimoment [N·m²]
    dB_dx     : rate of bimoment along span [N·m]
    webs_geom : list of (upper_pt, lower_pt) tuples for spar webs; if provided
                without ``shell_midlines`` / ``cell_wall_loops``, raises FEA-L-004.
    t_web     : representative web thickness [m] for flexibility integrals;
                defaults to the skin t_mean when omitted
    shell_midlines, cell_wall_loops
        Producer identity (SG-014). Required for multi-cell Batho; outline and
        web feet come from skin/web midlines (FEA-L-004), not airfoil nearest.
    """
    # --- weighted-mean panel properties -----------------------------------
    total_len, E_sum, t_sum = 0.0, 0.0, 0.0
    for p in zones:
        if len(p.s) < 2:
            continue
        L_p = float(p.s[-1])
        total_len += L_p
        E_sum += float(p.lam.E) * L_p
        t_sum += float(p.lam.t) * L_p
    E_mean = E_sum / max(total_len, 1e-30)
    t_mean = t_sum / max(total_len, 1e-30)
    if t_web is None:
        t_web = t_mean

    webs = list(webs_geom or [])
    midlines = list(shell_midlines or [])
    loops = list(cell_wall_loops or [])

    if webs and not (midlines and loops):
        raise ValueError(
            "compute_section_vlasov: multi-cell Batho requires shell_midlines and "
            "cell_wall_loops (producer identity); refusing airfoil nearest-web "
            "rediscovery (FEA-L-004)"
        )

    # 1. Open outline & shear centre
    if midlines and loops:
        verts = _open_outline_from_skin_midlines(midlines, loops)
        from blade_structure_utilities.geometry_utilities.section_properties.reference_axes.shear_centre import (
            shear_centre_from_midlines,
        )

        sc = shear_centre_from_midlines(midlines, cell_wall_loops=loops)
        y_sc, z_sc = float(sc.xs), float(sc.ys)
        web_feet = _web_feet_from_midlines(midlines)
        if webs and not web_feet:
            raise ValueError(
                "compute_section_vlasov: webs_geom was supplied but shell_midlines "
                "contain no web strips; refusing webs_geom coordinate fallback "
                "(FEA-L-004)"
            )
    else:
        verts = open_outline_from_airfoil(airfoil)
        y_sc, z_sc, _, _ = shear_center_outer_skin_loop(verts, t_mean, E_mean)
        web_feet = []

    # 2–4. Sectorial ω̂ + Batho + I_ω (TW-002: section_thin_wall_model owns the theory)
    if midlines and loops:
        G_mean = E_mean / (2.0 * 1.3)
        axial: dict[str, float] = {}
        shear: dict[str, float] = {}
        for strip in midlines:
            kind = str(getattr(strip, "kind", "") or "").lower()
            if kind not in ("skin", "web", "cap"):
                continue
            label = str(getattr(strip, "label", ""))
            t = float(getattr(strip, "thickness_m", 0.0) or 0.0)
            if not np.isfinite(t) or t <= 0.0:
                raise ValueError(
                    f"Vlasov wall graph: strip {label!r} needs "
                    f"positive thickness_m (TW-002)"
                )
            axial[label] = float(E_mean) * t
            shear[label] = 0.0 if kind == "cap" else float(G_mean) * t
        graph = graph_from_midlines(
            midlines,
            cell_wall_loops=loops,
            axial_stiffness_by_label=axial,
            shear_stiffness_by_label=shear,
        )
        warp = sectorial_warping(graph, (y_sc, z_sc))
        omega_hat_v = _map_omega_nodes_to_outline(verts, graph.node_points, warp.omega_node)
        n_cells = int(warp.n_cells)
        I_omega_E = float(warp.I_omega)
    else:
        omega_open = omega_vertices_open_chain(verts, y_sc, z_sc)
        omega_hat_v, n_cells = _batho_correction(
            verts, omega_open, t_mean, web_feet, t_web
        )
        I_omega_E = _integrate_panels_E_omega2(zones, verts, omega_hat_v)

    if I_omega_E < 1e-60:
        raise ValueError(
            "compute_section_vlasov: E-weighted I_ω is ~0; refusing silent "
            "I_omega_E=1.0 clamp that would invent finite σ_ω/q_ω (FEA-L-005)"
        )

    # 5. Per-panel stresses and shear flows
    s_outline = _outline_arc_lengths(verts)
    panel_s_mids, sigma_panels, q_panels = [], [], []
    web_panel_indices: list[int] = []
    cumulative_integral = 0.0

    for pi, p in enumerate(zones):
        n_elem = max(len(p.s) - 1, 0)
        if len(p.s) < 2:
            panel_s_mids.append(np.array([]))
            sigma_panels.append(np.array([]))
            q_panels.append(np.array([]))
            continue

        is_web = _panel_kind(str(getattr(p, "label", ""))) == "web"
        is_cap = _panel_kind(str(getattr(p, "label", ""))) == "cap"
        if is_web or is_cap:
            # Webs: Batho uses midline feet; cap axial load sheds via StripBond
            # (TW-001) — neither is placed on the open skin outline (FEA-L-004).
            if is_web:
                web_panel_indices.append(pi)
            s_mids = 0.5 * (p.s[:-1] + p.s[1:])
            panel_s_mids.append(s_mids)
            sigma_panels.append(np.zeros(n_elem))
            q_panels.append(np.zeros(n_elem))
            continue

        E_p, t_p = float(p.lam.E), float(p.lam.t)
        s_global = _panel_global_s(p, verts)
        omega_nodes = np.interp(s_global, s_outline, omega_hat_v)

        sigma_w = B * omega_nodes / I_omega_E

        integral_panel = np.zeros(len(p.s))
        for i in range(1, len(p.s)):
            ds = p.s[i] - p.s[i - 1]
            om_mid = 0.5 * (omega_nodes[i] + omega_nodes[i - 1])
            integral_panel[i] = integral_panel[i - 1] + E_p * om_mid * t_p * ds
        q_w = -(dB_dx / I_omega_E) * (cumulative_integral + integral_panel)
        cumulative_integral += float(integral_panel[-1])

        s_mids = 0.5 * (p.s[:-1] + p.s[1:])
        panel_s_mids.append(s_mids)
        sigma_panels.append(0.5 * (sigma_w[:-1] + sigma_w[1:]))
        q_panels.append(0.5 * (q_w[:-1] + q_w[1:]))

    return SectionVlasovResult(
        y_sc=y_sc,
        z_sc=z_sc,
        omega_hat_v=omega_hat_v,
        I_omega_E=I_omega_E,
        n_cells=n_cells,
        web_panel_indices=web_panel_indices,
        panel_s_mids=panel_s_mids,
        sigma_omega=sigma_panels,
        q_omega=q_panels,
    )


# ---------------------------------------------------------------------------
# TW-002: wall-graph sectorial warping (kernel) + legacy outline Batho
# ---------------------------------------------------------------------------

def _map_omega_nodes_to_outline(
    verts: NDArray,
    node_points: NDArray,
    omega_node: NDArray,
) -> NDArray:
    """Map graph-node ω̂ onto the open outline by 1 nm coincidence (TW-002)."""
    out = np.zeros(len(verts), dtype=float)
    for i, v in enumerate(verts):
        d = np.hypot(node_points[:, 0] - v[0], node_points[:, 1] - v[1])
        j = int(np.argmin(d))
        if float(d[j]) > COINCIDENCE_TOL_M:
            raise ValueError(
                f"Vlasov outline vertex {i} at {np.asarray(v).tolist()} is "
                f"{float(d[j]):.3e} m from the wall graph (tol={COINCIDENCE_TOL_M}; TW-002)"
            )
        out[i] = float(omega_node[j])
    return out


# ---------------------------------------------------------------------------
# Batho multi-cell closed-section warping correction (legacy outline path)
# ---------------------------------------------------------------------------

def _outline_arc_lengths(verts: NDArray) -> NDArray:
    n = len(verts)
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = s[i - 1] + float(np.hypot(verts[i, 0] - verts[i - 1, 0],
                                           verts[i, 1] - verts[i - 1, 1]))
    return s


def _batho_correction(
    verts: NDArray,
    omega_open: NDArray,
    t_skin: float,
    webs_geom: list,
    t_web: float,
) -> tuple[NDArray, int]:
    """
    Apply Batho (1/t)-weighted closed-cell warping correction.

    For a single-cell section (no webs), the unique correction is:
        c = ∮ ω_open (ds/t) / ∮ (ds/t)
    For multi-cell: solve the n_c × n_c flexibility system.

    Returns (omega_hat, n_cells).
    """
    s_v = _outline_arc_lengths(verts)
    n_v = len(verts)

    # Locate web split points on the outline (upper and lower intersections)
    web_splits = _web_split_indices(verts, s_v, webs_geom)  # list of (i_upper, i_lower)
    n_cells = len(web_splits) + 1

    if n_cells == 1:
        # Single-cell: analytic formula
        num, den = 0.0, 0.0
        for i in range(n_v - 1):
            ds = s_v[i + 1] - s_v[i]
            om_mid = 0.5 * (omega_open[i] + omega_open[i + 1])
            num += om_mid * ds / t_skin
            den += ds / t_skin
        # Closing edge (TE → LE): not in open outline; its contribution is small
        # (the open outline already accounts for the dominant skin area)
        c = num / max(den, 1e-60)
        return omega_open - c, 1

    # Multi-cell: identify the s-ranges belonging to each cell
    # Outline runs: LE → (split_upper_0) → … → TE → (split_lower_{n-1}) → LE (reversed lower)
    # We use the outline as-is and split it at the web upper and lower vertices.
    # Cell i: outline segment from web_{i-1}_lower to web_i_upper (upper skin portion)
    #         plus outline segment from web_i_lower to web_{i-1}_upper (lower skin portion)
    # This is approximate for a 2D thin-wall section.

    cell_segments = _cell_outline_segments(n_v, web_splits)

    # Flexibility matrix and RHS
    delta = np.zeros((n_cells, n_cells))
    R = np.zeros(n_cells)

    for ci, segs in enumerate(cell_segments):
        for (i0, i1) in segs:
            for i in range(min(i0, i1), max(i0, i1)):
                ds = abs(s_v[i + 1] - s_v[i])
                om_mid = 0.5 * (omega_open[i] + omega_open[i + 1])
                delta[ci, ci] += ds / t_skin
                R[ci] += om_mid * ds / t_skin

    # Web contributions: each web is shared between cell ci and ci+1
    for wi, (i_up, i_lo) in enumerate(web_splits):
        ci = wi  # web wi is between cell wi and wi+1
        cj = wi + 1
        # Web length (approximate: straight line from upper to lower web point)
        y_u, z_u = verts[i_up]
        y_l, z_l = verts[i_lo]
        L_web = float(np.hypot(y_u - y_l, z_u - z_l))
        flex_web = L_web / t_web
        delta[ci, ci] += flex_web
        delta[cj, cj] += flex_web
        delta[ci, cj] -= flex_web
        delta[cj, ci] -= flex_web
        # Web warping contribution: use mid-value of omega_open at web endpoints
        om_web_mid = 0.5 * (omega_open[i_up] + omega_open[i_lo])
        R[ci] += om_web_mid * flex_web
        R[cj] += om_web_mid * flex_web

    # Solve for correction constants — singular systems must not silently
    # fall back to the open-section ω̂ (FEA-L-005).
    try:
        c_vec = np.linalg.solve(delta, R)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"Batho multi-cell system is singular (n_cells={n_cells}); "
            "refusing open-section ω̂ fallback (FEA-L-005)"
        ) from exc

    # Apply correction: each vertex belongs to a cell; assign correction by s-position
    vertex_cell = _assign_vertex_to_cell(n_v, web_splits)
    omega_hat = omega_open - c_vec[vertex_cell]

    return omega_hat, n_cells


def _vertex_index_exact(
    pt: NDArray[np.floating],
    verts: NDArray[np.floating],
    *,
    role: str,
) -> int:
    """Nearest outline vertex only if within ``COINCIDENCE_TOL_M`` (FEA-L-004)."""
    p = np.asarray(pt, dtype=np.float64).reshape(2)
    d = np.hypot(verts[:, 0] - p[0], verts[:, 1] - p[1])
    i = int(np.argmin(d))
    if float(d[i]) > COINCIDENCE_TOL_M:
        raise ValueError(
            f"Vlasov {role}: point {p.tolist()} is {float(d[i]):.3e} m from the "
            f"nearest outline vertex (tol={COINCIDENCE_TOL_M}; FEA-L-004)"
        )
    return i


def _is_skin_strip(strip: Any) -> bool:
    from blade_structure_utilities.geometry_utilities.section_properties.reference_axes.wall_network import (
        _strip_kind,
        _strip_label,
    )

    kind = str(_strip_kind(strip)).lower()
    lab = str(_strip_label(strip))
    if kind == "web" or kind in ("cap", "spar", "spar_cap"):
        return False
    if _panel_kind(lab) == "web" or _panel_kind(lab) == "cap":
        return False
    return kind == "skin" or _panel_kind(lab) == "skin" or kind not in ("web", "cap")


def _chain_skin_polylines(
    strips: Sequence[Any],
    *,
    role: str,
) -> NDArray[np.float64]:
    """Concatenate skin strips into one polyline via tip coincidence (FEA-L-004)."""
    from blade_structure_utilities.geometry_utilities.section_properties.reference_axes.wall_network import (
        _strip_label,
        _strip_points,
    )

    if not strips:
        raise ValueError(f"open outline {role}: no skin strips (FEA-L-004)")
    remaining: list[tuple[str, NDArray[np.float64]]] = [
        (str(_strip_label(s)), np.asarray(_strip_points(s), dtype=np.float64).copy())
        for s in strips
    ]
    for lab, pts in remaining:
        if pts.shape[0] < 2:
            raise ValueError(f"skin strip {lab!r} needs ≥2 points (FEA-L-004)")

    all_pts = np.vstack([pts for _lab, pts in remaining])
    chord_col = int(np.argmax(np.ptp(all_pts, axis=0)))

    # Prefer starting at the LE-ward tip (min along the chord axis).
    def _le_key(item: tuple[str, NDArray[np.float64]]) -> float:
        pts = item[1]
        return min(float(pts[0, chord_col]), float(pts[-1, chord_col]))

    start_lab, start_pts = min(remaining, key=_le_key)
    remaining = [item for item in remaining if item[0] != start_lab]
    # Orient so the LE tip is at the start of the chain.
    if float(start_pts[-1, chord_col]) < float(start_pts[0, chord_col]):
        start_pts = start_pts[::-1]
    parts: list[NDArray[np.float64]] = [start_pts]

    while remaining:
        tip = parts[-1][-1]
        best_i = -1
        best_d = float("inf")
        best_rev = False
        for i, (_lab, pts) in enumerate(remaining):
            d0 = float(np.hypot(*(pts[0] - tip)))
            d1 = float(np.hypot(*(pts[-1] - tip)))
            if d0 < best_d:
                best_d, best_i, best_rev = d0, i, False
            if d1 < best_d:
                best_d, best_i, best_rev = d1, i, True
        if best_i < 0 or best_d > COINCIDENCE_TOL_M:
            left = [lab for lab, _ in remaining]
            raise ValueError(
                f"open outline {role}: cannot chain remaining skin strips {left} "
                f"onto tip {tip.tolist()} (best gap {best_d:.3e} m; FEA-L-004)"
            )
        _lab, pts = remaining.pop(best_i)
        if best_rev:
            pts = pts[::-1]
        parts.append(pts[1:])
    return np.vstack(parts)


def _open_outline_from_skin_midlines(
    midlines: Sequence[Any],
    cell_wall_loops: Sequence[Any],
) -> NDArray[np.float64]:
    """LE-cut open skin outline from producer midlines (FEA-L-004).

    Same topology as ``open_outline_from_airfoil``: suction LE→TE, then
    reversed pressure TE→LE without repeating the TE vertex. Skin membership
    is taken from ``cell_wall_loops.wall_labels`` (SG-014); within each surface
    the strips are chained by tip coincidence only.
    """
    def _strip_label(m: Any) -> str:
        return str(getattr(m, "label", "") or "")

    by_lab = {_strip_label(m): m for m in midlines}
    skin_labels: set[str] = set()
    for cell in wall_labels_from_loops(cell_wall_loops):
        for lab in cell:
            strip = by_lab.get(lab)
            if strip is not None and _is_skin_strip(strip):
                skin_labels.add(lab)
    if not skin_labels:
        raise ValueError(
            "open outline from midlines needs ≥1 skin wall "
            f"(got none; FEA-L-004)"
        )

    suction: list[Any] = []
    pressure: list[Any] = []
    other: list[Any] = []
    for lab in sorted(skin_labels):
        strip = by_lab[lab]
        side = str(getattr(strip, "skin_side", None) or "").lower()
        if side in ("suction_surface", "upper", "suction"):
            suction.append(strip)
        elif side in ("pressure_surface", "lower", "pressure"):
            pressure.append(strip)
        else:
            other.append(strip)

    if suction and pressure:
        upper = _chain_skin_polylines(suction, role="suction")
        lower_le_to_te = _chain_skin_polylines(pressure, role="pressure")
        lower_te_to_le = lower_le_to_te[::-1]
        # Match open_outline_from_airfoil: upper ∪ lower_rev[1:] (drop TE dup).
        tip = upper[-1]
        d0 = float(np.hypot(*(lower_te_to_le[0] - tip)))
        if d0 > COINCIDENCE_TOL_M:
            raise ValueError(
                f"suction TE and pressure TE disagree by {d0:.3e} m (FEA-L-004)"
            )
        return np.vstack([upper, lower_te_to_le[1:]])

    # Single closed skin family (no side tags): one tip-chained loop, cut at LE.
    all_skin = [by_lab[lab] for lab in sorted(skin_labels)]
    if other and not (suction or pressure):
        all_skin = other
    closed = _chain_skin_polylines(all_skin, role="outer-skin")
    if (
        closed.shape[0] >= 2
        and float(np.hypot(*(closed[0] - closed[-1]))) <= COINCIDENCE_TOL_M
    ):
        closed = closed[:-1]
    if closed.shape[0] < 3:
        raise ValueError(
            f"open outline from midlines: closed skin has too few vertices "
            f"({closed.shape[0]}; FEA-L-004)"
        )
    # LE = minimum along the longer in-plane axis (Y_B in FRAME_B; x in toys).
    spans = np.ptp(closed, axis=0)
    chord_col = int(np.argmax(spans))
    i_le = int(np.argmin(closed[:, chord_col]))
    opened = np.vstack([closed[i_le:], closed[:i_le]])
    # Match airfoil open outline: start and end both at the LE vertex.
    return np.vstack([opened, opened[:1]])



def _web_feet_from_midlines(
    midlines: Sequence[Any],
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    from blade_structure_utilities.geometry_utilities.section_properties.reference_axes.wall_network import (
        _strip_kind,
        _strip_label,
        _strip_points,
    )

    feet: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    for m in midlines:
        lab = str(_strip_label(m))
        kind = str(_strip_kind(m)).lower()
        if kind != "web" and _panel_kind(lab) != "web":
            continue
        pts = np.asarray(_strip_points(m), dtype=np.float64)
        if pts.shape[0] < 2:
            raise ValueError(f"web midline {lab!r} needs ≥2 points (FEA-L-004)")
        feet.append((pts[0].copy(), pts[-1].copy()))
    return feet


def _web_split_indices(
    verts: NDArray, s_v: NDArray, webs_geom: list
) -> list[tuple[int, int]]:
    """Map each web foot to an outline vertex index (Batho cell splits).

    Feet must coincide with outline vertices within ``COINCIDENCE_TOL_M``
    (FEA-L-004); no silent nearest rediscovery.
    """
    splits = []
    for wi, (up, lo) in enumerate(webs_geom):
        up = np.asarray(up, dtype=float)
        lo = np.asarray(lo, dtype=float)
        i_up = _vertex_index_exact(up, verts, role=f"web[{wi}] upper")
        i_lo = _vertex_index_exact(lo, verts, role=f"web[{wi}] lower")
        splits.append((i_up, i_lo))
    splits.sort(key=lambda p: s_v[p[0]])
    return splits


def _cell_outline_segments(
    n_v: int, web_splits: list[tuple[int, int]]
) -> list[list[tuple[int, int]]]:
    """
    Build the list of (i_start, i_end) index pairs for each cell's outline portion.

    The open outline runs from index 0 (LE upper) to n_v-1 (LE lower).
    Webs split the outline at (i_up, i_lo) pairs.
    Cell 0 is from LE to the first web upper point (upper skin part)
    plus from the first web lower point to LE (lower skin part).
    """
    n_cells = len(web_splits) + 1
    # Collect all split indices on the outline (sorted ascending)
    split_up = sorted(p[0] for p in web_splits)
    split_lo = sorted((p[1] for p in web_splits), reverse=True)  # lower half goes backward

    # Upper boundary split points: 0, up_0, up_1, …, up_{n-1}, n_v//2
    # Lower boundary split points (reversed): n_v//2, lo_{n-1}, …, lo_0, n_v-1
    n_half = n_v // 2
    up_pts = [0] + split_up + [n_half]
    lo_pts = [n_half] + split_lo + [n_v - 1]

    cell_segs: list[list[tuple[int, int]]] = []
    for ci in range(n_cells):
        segs = []
        # Upper skin segment of cell ci
        i0_up, i1_up = up_pts[ci], up_pts[ci + 1]
        if i1_up > i0_up:
            segs.append((i0_up, i1_up))
        # Lower skin segment of cell ci (reversed: lo_pts goes from mid to LE)
        i0_lo, i1_lo = lo_pts[ci], lo_pts[ci + 1]
        if i1_lo > i0_lo:
            segs.append((i0_lo, i1_lo))
        cell_segs.append(segs)
    return cell_segs


def _assign_vertex_to_cell(n_v: int, web_splits: list[tuple[int, int]]) -> NDArray:
    """
    Return an int array of shape (n_v,) giving cell index for each outline vertex.
    """
    cell_idx = np.zeros(n_v, dtype=int)
    if not web_splits:
        return cell_idx

    split_up = sorted(p[0] for p in web_splits)
    n_half = n_v // 2
    up_pts = [0] + split_up + [n_half]
    split_lo = sorted((p[1] for p in web_splits), reverse=True)
    lo_pts = [n_half] + split_lo + [n_v - 1]

    n_cells = len(web_splits) + 1
    for ci in range(n_cells):
        for i in range(up_pts[ci], up_pts[ci + 1] + 1):
            if 0 <= i < n_v:
                cell_idx[i] = ci
        for i in range(lo_pts[ci], lo_pts[ci + 1] + 1):
            if 0 <= i < n_v:
                cell_idx[i] = ci
    return cell_idx


# ---------------------------------------------------------------------------
# Warping constant and panel mapping helpers
# ---------------------------------------------------------------------------

def _panel_global_s(panel, verts: NDArray) -> NDArray:
    """Map panel local arc-length to global open-outline arc-length (FEA-L-004)."""
    s_outline = _outline_arc_lengths(verts)
    p0 = np.asarray(panel.nodes[0], dtype=float)
    i_start = _vertex_index_exact(
        p0, verts, role=f"panel {getattr(panel, 'label', '<unknown>')!r} start"
    )
    return float(s_outline[i_start]) + panel.s


def _integrate_panels_E_omega2(zones: list, verts: NDArray, omega_hat_v: NDArray) -> float:
    """I_ω_E = Σ_skin ∫ E_p ω̂² t_p ds (trapezoid).

    Caps are excluded (bond into skin — TW-001). Webs use ω̂ at the two outline
    feet with linear interpolation along the web arc (FEA-L-004).
    """
    s_outline = _outline_arc_lengths(verts)
    acc = 0.0
    for p in zones:
        if len(p.s) < 2:
            continue
        lab = str(getattr(p, "label", ""))
        if _panel_kind(lab) == "cap":
            continue
        E_p, t_p = float(p.lam.E), float(p.lam.t)
        if _panel_kind(lab) == "web":
            i0 = _vertex_index_exact(
                np.asarray(p.nodes[0], dtype=float), verts, role=f"web {lab!r} foot0"
            )
            i1 = _vertex_index_exact(
                np.asarray(p.nodes[-1], dtype=float), verts, role=f"web {lab!r} foot1"
            )
            omega_nodes = np.linspace(
                float(omega_hat_v[i0]), float(omega_hat_v[i1]), len(p.s)
            )
        else:
            s_global = _panel_global_s(p, verts)
            omega_nodes = np.interp(s_global, s_outline, omega_hat_v)
        acc += float(np.trapezoid(E_p * omega_nodes ** 2 * t_p, p.s))
    return acc
