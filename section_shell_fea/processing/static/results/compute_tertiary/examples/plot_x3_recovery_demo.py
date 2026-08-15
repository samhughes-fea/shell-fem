"""
Demo plots for tertiary through-thickness (laminate axis-3) stations and CLPT stresses.

Not a pytest. Run from repo root:

  python blade_structure_utilities/fem_utilities/section_shell_fea/processing/static/results/compute_tertiary/examples/plot_x3_recovery_demo.py

Writes PNGs under examples/outputs/ for embedding in compute_tertiary/docs/README.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[7]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.material_utilities.laminate import (
    FACE_INNER,
    FACE_MID,
    FACE_OUTER,
    LaminateDefinition,
    laminate_midstrains_curvatures,
    ply_face_stresses_laminate,
)
from blade_structure_utilities.material_utilities.orthotropic import OrthotropicPly

_OUT = _HERE / "outputs"


def _ply(
    *,
    name: str,
    t: float,
    E1: float,
    E2: float,
    G12: float = 4e9,
    nu12: float = 0.28,
) -> OrthotropicPly:
    return OrthotropicPly(
        name=name,
        E1=E1,
        E2=E2,
        G12=G12,
        nu12=nu12,
        rho=1800.0,
        t_ply=t,
        Xt=800e6,
        Xc=600e6,
        Yt=40e6,
        Yc=130e6,
        S12=70e6,
        Zt=40e6,
        S13=70e6,
        S23=70e6,
    )


def build_isotropic_monolayer() -> LaminateDefinition:
    """Single isotropic-like ply (E1=E2) — constant Q_bar through thickness."""
    E = 70e9
    return LaminateDefinition(
        plies=[(_ply(name="iso", t=2.0e-3, E1=E, E2=E, G12=E / (2.0 * (1.0 + 0.30)), nu12=0.30), 0.0)]
    )


def build_composite_stack() -> LaminateDefinition:
    """Uneven multi-ply orthotropic stack (0/90/+45/-45) with unequal ply thickness."""
    return LaminateDefinition(
        plies=[
            (_ply(name="ud0", t=0.25e-3, E1=40e9, E2=10e9), 0.0),
            (_ply(name="ud90", t=0.50e-3, E1=40e9, E2=10e9), 90.0),
            (_ply(name="ud45", t=0.35e-3, E1=40e9, E2=10e9), 45.0),
            (_ply(name="udm45", t=0.40e-3, E1=40e9, E2=10e9), -45.0),
        ]
    )


def _ply_index_at_x3(x3: float, x3_interfaces: NDArray[np.float64]) -> int:
    """Active ply index for station x3 (interfaces inclusive on upper face of each ply)."""
    for k in range(len(x3_interfaces) - 1):
        lo, hi = float(x3_interfaces[k]), float(x3_interfaces[k + 1])
        if lo <= x3 <= hi or (k == len(x3_interfaces) - 2 and np.isclose(x3, hi)):
            return k
    if x3 < float(x3_interfaces[0]):
        return 0
    return len(x3_interfaces) - 2


def sigma123_along_x3(
    lam: LaminateDefinition,
    eps0: NDArray[np.float64],
    kappa: NDArray[np.float64],
    x3: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Plane-stress ``[σ11, σ22, τ12]`` in **E** (laminate) at each ``x3`` [Pa]."""
    bundle = lam.stiffness_bundle()
    Q_bar = np.asarray(bundle.Q_bar, dtype=np.float64)
    ifaces = lam.ply_stack_x3_stations().x3_interfaces
    eps0_v = np.asarray(eps0, dtype=np.float64).ravel()
    kappa_v = np.asarray(kappa, dtype=np.float64).ravel()
    out = np.zeros((x3.size, 3), dtype=np.float64)
    for i, xi in enumerate(np.asarray(x3, dtype=np.float64).ravel()):
        k = _ply_index_at_x3(float(xi), ifaces)
        eps = eps0_v + float(xi) * kappa_v
        out[i] = Q_bar[k] @ eps
    return out


def _pure_bending_m11(lam: LaminateDefinition, M11: float = 100.0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    abd = lam.stiffness_bundle().ABD
    N = np.zeros(3, dtype=np.float64)
    M = np.array([M11, 0.0, 0.0], dtype=np.float64)
    return laminate_midstrains_curvatures(abd, N, M)


def plot_stations_isotropic(lam: LaminateDefinition, outpath: Path) -> None:
    st = lam.ply_stack_x3_stations()
    faces = st.x3_ply_faces[0]  # one plate: outer, mid, inner
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.axhline(0.0, color="0.85", linewidth=0.8)
    ax.axvline(0.0, color="0.85", linewidth=0.8)
    ax.plot([0, 0], [-0.5 * st.h, 0.5 * st.h], color="0.3", linewidth=2.0, label="laminate axis 3")
    colors = ["#1E90FF", "#C0392B", "#2E8B57"]
    markers = ["^", "s", "v"]
    names = ["FACE_OUTER (+h/2)", "FACE_MID", "FACE_INNER (−h/2)"]
    for x3, c, m, name in zip(faces, colors, markers, names):
        ax.scatter([0.0], [x3], s=90, zorder=5, color=c, marker=m, label=name)
        ax.annotate(name.split()[0], (0.05, x3), fontsize=8, va="center", color=c)
    ax.set_ylabel(r"$x_3$ [m] (laminate axis 3, through-thickness)")
    ax.set_xlabel("(schematic)")
    ax.set_xlim(-0.5, 1.2)
    ax.set_title("Isotropic monolayer — production face product")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_stations_composite(lam: LaminateDefinition, outpath: Path) -> None:
    st = lam.ply_stack_x3_stations()
    faces = st.x3_ply_faces  # (n_ply, 3)
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    ax.axhline(0.0, color="0.85", linewidth=0.8)
    ax.plot([0, 0], [-0.5 * st.h, 0.5 * st.h], color="0.3", linewidth=2.0)
    for y in st.x3_interfaces:
        ax.axhline(y, color="0.75", linestyle=":", linewidth=0.9)
    # Offset each plate's face triple slightly in x for readability
    for k in range(faces.shape[0]):
        xs = np.full(3, 0.08 * (k + 1))
        ax.scatter(xs[0], faces[k, FACE_OUTER], s=55, color="#1E90FF", marker="^", zorder=5)
        ax.scatter(xs[1], faces[k, FACE_MID], s=55, color="#C0392B", marker="s", zorder=5)
        ax.scatter(xs[2], faces[k, FACE_INNER], s=55, color="#2E8B57", marker="v", zorder=5)
        ax.annotate(f"ply {k}", (xs[0] + 0.02, faces[k, FACE_MID]), fontsize=7, va="center")
    # Legend proxies
    ax.scatter([], [], color="#1E90FF", marker="^", s=55, label="FACE_OUTER")
    ax.scatter([], [], color="#C0392B", marker="s", s=55, label="FACE_MID")
    ax.scatter([], [], color="#2E8B57", marker="v", s=55, label="FACE_INNER")
    ax.set_ylabel(r"$x_3$ [m] (laminate axis 3)")
    ax.set_xlabel("plate index offset (schematic)")
    ax.set_xlim(-0.2, 0.08 * (faces.shape[0] + 2))
    ax.set_title(r"Composite stack — per-plate outer / mid / inner ($x3\_ply\_faces$)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_sigma11_bending(
    lam: LaminateDefinition,
    outpath: Path,
    *,
    title: str,
) -> None:
    """Dense σ11(x3) with production face markers from ply_face_stresses_laminate."""
    eps0, kappa = _pure_bending_m11(lam, M11=100.0)
    st = lam.ply_stack_x3_stations()
    x3_line = np.linspace(-0.5 * st.h, 0.5 * st.h, 400)
    sig = sigma123_along_x3(lam, eps0, kappa, x3_line)
    faces = ply_face_stresses_laminate(lam, eps0, kappa)  # (n_ply, 3, 3)
    x3_faces = st.x3_ply_faces

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.plot(sig[:, 0] / 1e6, x3_line, color="#1E90FF", linewidth=1.6, label=r"$\sigma_{11}(x_3)$ dense")
    ax.scatter(
        faces[:, FACE_OUTER, 0] / 1e6,
        x3_faces[:, FACE_OUTER],
        s=55,
        color="#1E90FF",
        marker="^",
        zorder=5,
        edgecolors="k",
        linewidths=0.4,
        label="outer face",
    )
    ax.scatter(
        faces[:, FACE_MID, 0] / 1e6,
        x3_faces[:, FACE_MID],
        s=55,
        color="#C0392B",
        marker="s",
        zorder=5,
        edgecolors="k",
        linewidths=0.4,
        label="mid face",
    )
    ax.scatter(
        faces[:, FACE_INNER, 0] / 1e6,
        x3_faces[:, FACE_INNER],
        s=55,
        color="#2E8B57",
        marker="v",
        zorder=5,
        edgecolors="k",
        linewidths=0.4,
        label="inner face",
    )
    ax.axhline(0.0, color="0.85", linewidth=0.7)
    ax.axvline(0.0, color="0.85", linewidth=0.7)
    ax.set_xlabel(r"$\sigma_{11}$ [MPa] (laminate 123)")
    ax.set_ylabel(r"$x_3$ [m]")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_free_surfaces_vs_mids(lam: LaminateDefinition, outpath: Path) -> None:
    """Highlight laminate ±h/2 (extreme-fibre stations) vs plate mids under pure M11."""
    eps0, kappa = _pure_bending_m11(lam, M11=100.0)
    st = lam.ply_stack_x3_stations()
    faces = ply_face_stresses_laminate(lam, eps0, kappa)
    x3_faces = st.x3_ply_faces
    x3_line = np.linspace(-0.5 * st.h, 0.5 * st.h, 400)
    sig = sigma123_along_x3(lam, eps0, kappa, x3_line)

    # Free surfaces: top-plate outer, bottom-plate inner
    x3_outer = float(x3_faces[-1, FACE_OUTER])
    x3_inner = float(x3_faces[0, FACE_INNER])
    sig_outer = float(faces[-1, FACE_OUTER, 0])
    sig_inner = float(faces[0, FACE_INNER, 0])

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.plot(sig[:, 0] / 1e6, x3_line, color="0.55", linewidth=1.4, label=r"$\sigma_{11}(x_3)$")
    ax.scatter(
        faces[:, FACE_MID, 0] / 1e6,
        x3_faces[:, FACE_MID],
        s=50,
        color="#C0392B",
        marker="s",
        zorder=4,
        label="mid faces (Hashin / VM input)",
    )
    ax.scatter(
        [sig_outer / 1e6, sig_inner / 1e6],
        [x3_outer, x3_inner],
        s=110,
        color="#2E8B57",
        marker="D",
        zorder=6,
        edgecolors="k",
        linewidths=0.6,
        label=r"free surfaces $\pm h/2$ (extreme-fibre)",
    )
    ax.annotate(r"$+h/2$", (sig_outer / 1e6, x3_outer), textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax.annotate(r"$-h/2$", (sig_inner / 1e6, x3_inner), textcoords="offset points", xytext=(8, -10), fontsize=8)
    ax.axhline(0.0, color="0.85", linewidth=0.7)
    ax.axvline(0.0, color="0.85", linewidth=0.7)
    ax.set_xlabel(r"$\sigma_{11}$ [MPa] (laminate 123)")
    ax.set_ylabel(r"$x_3$ [m]")
    ax.set_title(r"Composite — mid faces vs free surfaces under pure $M_{11}$")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# Representative structural systems for face-semantics demos (NACA 2412, untwisted).
FACE_SEMANTICS_LAYOUTS: tuple[str, ...] = (
    "1C-CN",  # single shear web + sized caps
    "2C-CN",  # two-web C family
    "2D-CN",  # continuous spar box (canonical doc figure)
    "2D-F",   # flapwise webs
    "3D-CN",  # three-web box
)


def plot_panel_face_semantics(
    outpath: Path,
    *,
    layout_key: str = "2D-CN",
    naca: str = "2412",
) -> None:
    """Real MultiCellSection midlines with circle markers for FACE_OUTER / FACE_INNER.

    Uses section_geometry + mesh-input winding normalize so markers match production
    +x3: skin fluid/void, spar-cap skin/void, web fore (LE) / aft (TE).

    Face markers on upper+lower skin (aft of spar box), all caps, and all webs.
    Inline ``P{i}`` labels match strip-mesh ``panel_index`` (labellines if installed).
    """
    from matplotlib.lines import Line2D

    from blade_structure_utilities.fem_utilities.section_shell_fea.visualisation.section_axes import (
        finish_section_axes,
        section_hv,
        section_point_hv,
    )
    from blade_structure_utilities.geometry_utilities.section_geometry.api.factory import (
        build_section_view,
        resolve_system_type,
    )
    from blade_structure_utilities.geometry_utilities.section_geometry.sdf.airfoil import (
        AirfoilSDF,
    )
    from blade_structure_utilities.mesh_utilities.section_mesh.handoff.panel_x3_orientation import (
        mid_segment_frame,
        normal_2d,
    )
    from blade_structure_utilities.mesh_utilities.section_mesh.handoff._section_to_shell_inputs import (
        shell_mesh_inputs_from_section,
    )

    try:
        from labellines import labelLines
    except ImportError:
        labelLines = None

    key = str(layout_key).strip()
    af = AirfoilSDF.from_naca(str(naca).zfill(4), n_points=140, chord=1.0)
    layout = resolve_system_type(key)
    section = build_section_view(af, layout, twist_angle_rad=0.0)
    shell = shell_mesh_inputs_from_section(section, twist_rad=0.0, layout_key=key)
    strips = shell.midlines
    le = np.asarray(shell.leading_edge_b, dtype=float)
    te = np.asarray(shell.trailing_edge_b, dtype=float)

    # RGB kind palette (blue skin / red cap / green web)
    kind_style = {
        "skin": dict(color="#1F77B4", lw=1.6, zorder=2),
        "cap": dict(color="#D62728", lw=2.0, zorder=3),
        "web": dict(color="#2CA02C", lw=2.0, zorder=3),
    }
    face_colors = {
        "skin": ("#1F77B4", "#9BCAE0"),
        "cap": ("#D62728", "#F5B7B1"),
        "web": ("#2CA02C", "#A9DFBF"),
    }

    def _arc_length(xy: NDArray[np.float64]) -> float:
        d = np.diff(xy, axis=0)
        return float(np.sum(np.hypot(d[:, 0], d[:, 1])))

    def _point_at_frac(xy: NDArray[np.float64], frac: float) -> NDArray[np.float64]:
        if xy.shape[0] < 2:
            return np.asarray(xy[0], dtype=float)
        seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(s[-1])
        if total <= 0.0:
            return np.asarray(xy[0], dtype=float)
        target = float(np.clip(frac, 0.0, 1.0)) * total
        j = int(np.clip(np.searchsorted(s, target, side="right") - 1, 0, len(seg) - 1))
        den = seg[j] if seg[j] > 0.0 else 1.0
        t = (target - s[j]) / den
        return (1.0 - t) * xy[j] + t * xy[j + 1]

    chord = max(float(te[0] - le[0]), 1e-9)
    # Skip LE/TE junction stubs — they crowd the outline without aiding face reading
    min_label_arc = 0.06 * chord

    def _panel_description(s) -> str:
        """Short human label for the panel-id legend."""
        raw = str(s.label)
        xy = np.asarray(s.midline_b, dtype=float)
        mid = np.mean(xy, axis=0)
        if s.kind == "skin":
            side = "upper" if float(mid[1]) >= 0.0 else "lower"
            seg = raw.split("seg")[-1] if "seg" in raw else "?"
            return f"skin {side} seg{seg}"
        if s.kind == "cap":
            surf = s.surface or ("upper" if "UPPER" in raw.upper() else "lower")
            return f"spar cap {surf}"
        if s.kind == "web":
            return raw
        return f"{s.kind}: {raw}"

    web_strips_all = [s for s in strips if s.kind == "web"]
    web_y_means = [
        float(np.mean(np.asarray(s.midline_b, dtype=float)[:, 0])) for s in web_strips_all
    ]
    y_fore = min(web_y_means) if web_y_means else 0.0
    y_aft = max(web_y_means) if web_y_means else 0.0

    def _web_role(s) -> str:
        y_m = float(np.mean(np.asarray(s.midline_b, dtype=float)[:, 0]))
        if len(web_y_means) <= 1:
            return str(s.label)
        if abs(y_m - y_fore) <= 1e-9:
            return f"{s.label} (fore)"
        if abs(y_m - y_aft) <= 1e-9:
            return f"{s.label} (aft)"
        return f"{s.label} (mid)"

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    member_lines: list = []
    panel_id_rows: list[tuple[int, str, str, bool]] = []  # i, kind, description, labeled
    for i, s in enumerate(strips):
        xy = np.asarray(s.midline_b, dtype=float)
        st = kind_style.get(s.kind, dict(color="0.4", lw=1.2, zorder=1))
        # Same index as strip-mesh panel_index; hide stub labels from labellines
        keep = s.kind in ("cap", "web") or _arc_length(xy) >= min_label_arc
        lab = f"P{i}" if keep else f"_P{i}"
        h, v = section_hv(xy)
        (ln,) = ax.plot(h, v, label=lab, **st)
        member_lines.append(ln)
        desc = _web_role(s) if s.kind == "web" else _panel_description(s)
        panel_id_rows.append((i, s.kind, desc, keep))

    def _skin_station_away_from_caps(
        skin_strips: list,
        cap_strips: list,
        *,
        prefer_upper: bool,
        margin_frac: float = 0.18,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """Pick a skin mid-segment well clear of the spar box on upper or lower surface."""
        if not skin_strips:
            return None
        if cap_strips:
            cap_xy = np.vstack([np.asarray(c.midline_b, dtype=float) for c in cap_strips])
            x_lo, x_hi = float(cap_xy[:, 0].min()), float(cap_xy[:, 0].max())
        else:
            x_lo, x_hi = 0.25, 0.55
        chord = max(float(te[0] - le[0]), 1e-9)
        x_aft = x_hi + margin_frac * chord
        x_fore = x_lo - margin_frac * chord

        best: tuple[float, NDArray[np.float64], NDArray[np.float64]] | None = None
        for skin in skin_strips:
            pts = np.asarray(skin.midline_b, dtype=float)
            if pts.shape[0] < 2:
                continue
            for j in range(pts.shape[0] - 1):
                m = 0.5 * (pts[j] + pts[j + 1])
                m_x, m_z = float(m[0]), float(m[1])
                in_box = x_lo - 0.03 * chord <= m_x <= x_hi + 0.03 * chord
                if in_box:
                    continue
                on_upper = m_z >= 0.0
                if on_upper != prefer_upper:
                    continue
                score = abs(m_x - x_aft)
                if m_x < x_lo:
                    score = abs(m_x - x_fore) + 0.12 * chord
                n_hat = normal_2d(pts[j + 1] - pts[j])
                if best is None or score < best[0]:
                    best = (score, m, n_hat)
        if best is None:
            return None
        return best[1], best[2]

    eps = 0.012  # normal offset [m] for face markers
    cap_strips = [s for s in strips if s.kind == "cap"]
    skin_strips = [s for s in strips if s.kind == "skin"]
    web_strips = [s for s in strips if s.kind == "web"]

    stations: list[tuple[NDArray[np.float64], NDArray[np.float64], str, str]] = []
    for prefer_upper in (False, True):
        frame = _skin_station_away_from_caps(
            skin_strips, cap_strips, prefer_upper=prefer_upper,
        )
        if frame is not None:
            c_o, c_i = face_colors["skin"]
            stations.append((frame[0], frame[1], c_o, c_i))

    for strip in sorted(cap_strips, key=lambda s: (s.surface != "upper", s.surface or "")):
        m, n_hat = mid_segment_frame(strip.midline_b)
        c_o, c_i = face_colors["cap"]
        stations.append((m, n_hat, c_o, c_i))

    for strip in sorted(web_strips, key=lambda s: float(np.mean(s.midline_b[:, 0]))):
        m, n_hat = mid_segment_frame(strip.midline_b)
        c_o, c_i = face_colors["web"]
        stations.append((m, n_hat, c_o, c_i))

    for m, n_hat, c_outer, c_inner in stations:
        p_outer = m + eps * n_hat
        p_inner = m - eps * n_hat
        h_o, v_o = section_point_hv(float(p_outer[0]), float(p_outer[1]))
        h_i, v_i = section_point_hv(float(p_inner[0]), float(p_inner[1]))
        ax.scatter(
            [h_o], [v_o], s=55, c=c_outer, edgecolors="k", linewidths=0.5,
            zorder=6, marker="o",
        )
        ax.scatter(
            [h_i], [v_i], s=55, c=c_inner, edgecolors="k", linewidths=0.5,
            zorder=6, marker="o",
        )

    h_le, v_le = section_point_hv(float(le[0]), float(le[1]))
    h_te, v_te = section_point_hv(float(te[0]), float(te[1]))
    ax.scatter([h_le], [v_le], s=40, c="k", marker="^", zorder=5)
    ax.scatter([h_te], [v_te], s=40, c="k", marker="v", zorder=5)
    ax.annotate("LE", (h_le, v_le), textcoords="offset points", xytext=(-12, 4), fontsize=8)
    ax.annotate("TE", (h_te, v_te), textcoords="offset points", xytext=(6, 4), fontsize=8)

    # Inline panel ids — stagger along each panel away from markers / other labels
    from matplotlib import patheffects as pe

    labeled = [
        ln for ln in member_lines
        if ln.get_label() and not str(ln.get_label()).startswith("_")
    ]
    # Face stations (on-line + offset markers) + LE/TE — labels must clear these
    marker_pts: list[NDArray[np.float64]] = [le, te]
    for m, n_hat, _co, _ci in stations:
        marker_pts.append(np.asarray(m, dtype=float))
        marker_pts.append(m + eps * n_hat)
        marker_pts.append(m - eps * n_hat)
    marker_clear = 0.045 * chord
    label_clear = 0.035 * chord

    def _min_dist(p: NDArray[np.float64], pts: list[NDArray[np.float64]]) -> float:
        if not pts:
            return 1e9
        d = np.linalg.norm(np.asarray(pts, dtype=float) - np.asarray(p, dtype=float), axis=1)
        return float(np.min(d))

    def _is_near_vertical(xy: NDArray[np.float64]) -> bool:
        dx = abs(float(xy[-1, 0] - xy[0, 0]))
        dy = abs(float(xy[-1, 1] - xy[0, 1]))
        return dy > 2.0 * max(dx, 1e-15)

    line_items: list[tuple[object, NDArray[np.float64], float]] = []
    for ln in labeled:
        lab = str(ln.get_label())
        if not (lab.startswith("P") and lab[1:].isdigit()):
            continue
        xy = np.asarray(strips[int(lab[1:])].midline_b, dtype=float)
        if xy.shape[0] == 0:
            continue
        line_items.append((ln, xy, _arc_length(xy)))
    # Longer panels first so short stubs fill remaining gaps
    line_items.sort(key=lambda t: -t[2])

    placed: list[NDArray[np.float64]] = []
    ok_lines: list = []
    xvals: list[float] = []
    annotate_jobs: list[tuple[object, NDArray[np.float64]]] = []
    staggered: list[tuple[object, NDArray[np.float64]]] = []
    fracs = np.linspace(0.12, 0.88, 15)

    def _project_s(xy: NDArray[np.float64], p: NDArray[np.float64]) -> float:
        """Arc-length station of the closest polyline vertex/segment to ``p``."""
        seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        best_s = 0.0
        best_d = np.inf
        for j in range(len(seg)):
            a, b = xy[j], xy[j + 1]
            ab = b - a
            L2 = float(np.dot(ab, ab))
            t = 0.0 if L2 <= 0.0 else float(np.clip(np.dot(p - a, ab) / L2, 0.0, 1.0))
            q = a + t * ab
            d = float(np.linalg.norm(p - q))
            if d < best_d:
                best_d = d
                best_s = float(s[j] + t * seg[j])
        return best_s

    def _web_label_point(xy: NDArray[np.float64], *, below_marker: bool) -> NDArray[np.float64]:
        """Equidistant (arc-length) between mid-face markers and one spar-cap end.

        ``below_marker=True`` → lower cap side (−Z_B of markers); else upper cap (+Z_B).
        """
        i_cap = int(np.argmin(xy[:, 1])) if below_marker else int(np.argmax(xy[:, 1]))
        p_cap = xy[i_cap]
        p_mark, _n = mid_segment_frame(xy)
        s_cap = _project_s(xy, p_cap)
        s_mark = _project_s(xy, p_mark)
        s_mid = 0.5 * (s_cap + s_mark)
        total = float(_arc_length(xy))
        frac = s_mid / total if total > 0.0 else (0.35 if below_marker else 0.65)
        pt = _point_at_frac(xy, float(np.clip(frac, 0.0, 1.0)))
        # Slight aft offset so the glyph clears the fore/aft face markers
        n_hat = normal_2d(xy[-1] - xy[0])
        return pt - 0.012 * n_hat

    def _line_is_web(ln, xy: NDArray[np.float64]) -> bool:
        lab = str(ln.get_label())
        if lab.startswith("P") and lab[1:].isdigit():
            return strips[int(lab[1:])].kind == "web"
        return _is_near_vertical(xy)

    for ln, xy, _arc in line_items:
        if _line_is_web(ln, xy):
            # Foremost web: below markers; all other webs: above markers (+Z_B)
            y_m = float(np.mean(xy[:, 0]))
            below = len(web_y_means) >= 2 and abs(y_m - y_fore) <= 1e-9
            best_pt = _web_label_point(xy, below_marker=below)
            placed.append(best_pt)
            staggered.append((ln, best_pt))
            annotate_jobs.append((ln, best_pt))
            continue

        best_pt = None
        best_score = -1.0
        for frac in fracs:
            pt = _point_at_frac(xy, float(frac))
            d_mark = _min_dist(pt, marker_pts)
            d_lab = _min_dist(pt, placed)
            # Hard preference: clear markers, then stagger from other labels
            if d_mark < marker_clear:
                score = 0.01 * d_mark
            elif d_lab < label_clear:
                score = 0.5 * d_lab + 0.05 * d_mark
            else:
                score = d_lab + 0.25 * d_mark
            if score > best_score:
                best_score = score
                best_pt = pt
        if best_pt is None:
            continue
        placed.append(best_pt)
        staggered.append((ln, best_pt))
        h_xy, _v_xy = section_hv(xy)
        lo, hi = float(np.min(h_xy)), float(np.max(h_xy))
        span = max(hi - lo, 1e-15)
        h_pt, _v_pt = section_point_hv(float(best_pt[0]), float(best_pt[1]))
        ok_lines.append(ln)
        xvals.append(float(np.clip(h_pt, lo + 1e-9 * span, hi - 1e-9 * span)))

    used_labellines = False
    if labelLines is not None and ok_lines:
        try:
            labelLines(
                ok_lines,
                align=False,
                fontsize=7,
                xvals=xvals,
                zorder=7,
                outline_color="white",
                outline_width=1.5,
            )
            used_labellines = True
        except Exception:
            used_labellines = False

    label_pe = [pe.withStroke(linewidth=1.5, foreground="white")]
    draw_annotate = annotate_jobs if used_labellines else staggered
    for ln, pt in draw_annotate:
        h_pt, v_pt = section_point_hv(float(pt[0]), float(pt[1]))
        ax.annotate(
            str(ln.get_label()),
            (h_pt, v_pt),
            fontsize=7,
            ha="center",
            va="center",
            path_effects=label_pe,
            zorder=7,
        )

    ax.set_title(
        f"Panel face semantics — NACA {str(naca).zfill(4)} / {key} midlines\n"
        r"(markers at $\mathbf{m}\pm\varepsilon\hat{n}$; inline $P_i$ = panel_index)",
        fontsize=10,
    )
    ax.grid(True, alpha=0.25)

    # Matplotlib packs ncol>1 column-first → col0 Midline | col1 Outer | col2 Inner
    def _mk(color: str, lab: str) -> Line2D:
        return Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=color,
            markeredgecolor="k", markersize=7, linestyle="None", label=lab,
        )

    has_cap = any(s.kind == "cap" for s in strips)
    has_web = any(s.kind == "web" for s in strips)
    handles = [
        Line2D([0], [0], color=kind_style["skin"]["color"], lw=1.6, label="skin midline"),
    ]
    if has_cap:
        handles.append(
            Line2D([0], [0], color=kind_style["cap"]["color"], lw=2.0, label="cap midline")
        )
    if has_web:
        handles.append(
            Line2D([0], [0], color=kind_style["web"]["color"], lw=2.0, label="web midline")
        )
    handles.extend([
        _mk(face_colors["skin"][0], "Skin FACE_OUTER — fluid"),
        _mk(face_colors["skin"][1], "Skin FACE_INNER — void"),
    ])
    if has_cap:
        handles.extend([
            _mk(face_colors["cap"][0], "Cap FACE_OUTER — skin side"),
            _mk(face_colors["cap"][1], "Cap FACE_INNER — void"),
        ])
    if has_web:
        handles.extend([
            _mk(face_colors["web"][0], "Web FACE_OUTER — fore (LE)"),
            _mk(face_colors["web"][1], "Web FACE_INNER — aft (TE)"),
        ])

    # Tight padded limits in section-plot coords (h=Y_B, v=X_B).
    pts = np.vstack([np.asarray(s.midline_b, dtype=float) for s in strips])
    h_all, v_all = section_hv(pts)
    xmin, xmax = float(h_all.min()), float(h_all.max())
    ymin, ymax = float(v_all.min()), float(v_all.max())
    pad = 0.08 * max(xmax - xmin, ymax - ymin, 1e-9)
    x0, x1 = xmin - pad, xmax + pad
    y0, y1 = ymin - pad, ymax + pad
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_box_aspect((y1 - y0) / (x1 - x0))
    finish_section_axes(ax, square=False, equal_aspect=True)

    # Face-semantics legend, then panel-id key underneath
    fig.canvas.draw()
    ax_tight = ax.get_tightbbox(fig.canvas.get_renderer())
    ax_fig = ax_tight.transformed(fig.transFigure.inverted())
    x_mid = 0.5 * (ax_fig.x0 + ax_fig.x1)
    leg_faces = fig.legend(
        handles=handles,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(x_mid, ax_fig.y0 - 0.005),
        ncol=3,
        framealpha=0.95,
        borderaxespad=0.0,
        columnspacing=1.2,
        handletextpad=0.4,
        labelspacing=0.25,
    )
    fig.canvas.draw()
    faces_bb = leg_faces.get_window_extent(fig.canvas.get_renderer())
    faces_fig = faces_bb.transformed(fig.transFigure.inverted())

    panel_handles = [
        Line2D(
            [0], [0],
            color=kind_style.get(kind, {}).get("color", "0.4"),
            lw=1.5,
            label=(f"P{i}: {desc}" if labeled else f"P{i}: {desc} (stub)"),
        )
        for i, kind, desc, labeled in panel_id_rows
    ]
    fig.legend(
        handles=panel_handles,
        fontsize=6.5,
        loc="upper center",
        bbox_to_anchor=(x_mid, faces_fig.y0 - 0.012),
        ncol=4,
        framealpha=0.95,
        borderaxespad=0.0,
        columnspacing=0.9,
        handletextpad=0.35,
        labelspacing=0.2,
        title=r"$P_i$ = strip-mesh panel_index",
        title_fontsize=7,
    )
    fig.savefig(outpath, dpi=150, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    iso = build_isotropic_monolayer()
    comp = build_composite_stack()

    plot_stations_isotropic(iso, _OUT / "x3_stations_isotropic.png")
    plot_stations_composite(comp, _OUT / "x3_stations_composite.png")
    plot_sigma11_bending(
        iso,
        _OUT / "sigma11_x3_isotropic_bending.png",
        title=r"Isotropic — $\sigma_{11}(x_3)$ under pure $M_{11}$ (outer / mid / inner)",
    )
    plot_sigma11_bending(
        comp,
        _OUT / "sigma11_x3_composite_bending.png",
        title=r"Composite — $\sigma_{11}(x_3)$ under pure $M_{11}$ (per-plate faces)",
    )
    plot_free_surfaces_vs_mids(comp, _OUT / "free_surfaces_vs_mids_composite.png")
    import shutil

    for layout_key in FACE_SEMANTICS_LAYOUTS:
        out = _OUT / f"panel_face_semantics_{layout_key}.png"
        plot_panel_face_semantics(out, layout_key=layout_key)
    # Canonical doc alias
    shutil.copyfile(
        _OUT / "panel_face_semantics_2D-CN.png",
        _OUT / "panel_face_semantics.png",
    )
    print(f"Wrote plots under {_OUT}")


if __name__ == "__main__":
    main()
