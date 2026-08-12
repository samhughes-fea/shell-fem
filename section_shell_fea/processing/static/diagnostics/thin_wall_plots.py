"""Thin-wall distribution ribbons for section_shell_fea diagnostics.

Adapted from ``examples/area-diagrams-strip/ribbon_plots.py`` into the package
so demos no longer depend on the vendor thin-wall plot helpers.

Geometry is canonical **B** ``(X_B, Y_B)``; figures follow the section plot
contract via ``visualisation.section_axes``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.loading.thin_wall.sectorial_warping import (
    open_outline_from_airfoil,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.visualisation.section_axes import (
    finish_section_axes,
    section_hv,
)


def outward_normals(nodes: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unit outward normals perpendicular to the polyline tangent (CCW 90°).

    Vectorised form of the ``+x3`` sense owned by
    ``coordinate_frames.section_normal_2d`` — same convention, evaluated at every
    polyline vertex at once (pinned in ``tests/test_section_frame_canon.py``).
    """
    p = np.asarray(nodes, dtype=float)
    n = len(p)
    if n < 2:
        return np.array([[0.0, 1.0]], dtype=float)
    tvec = np.zeros_like(p)
    tvec[0] = p[1] - p[0]
    tvec[-1] = p[-1] - p[-2]
    if n > 2:
        tvec[1:-1] = p[2:] - p[:-2]
    ln = np.linalg.norm(tvec, axis=1, keepdims=True)
    ln[ln < 1e-10] = 1e-10
    tvec /= ln
    return np.column_stack([-tvec[:, 1], tvec[:, 0]])


def auto_scale(
    values_lists: list[NDArray[np.float64]],
    span_fraction: float = 0.07,
) -> float:
    """Scale so max |value| ribbon occupies ``span_fraction`` of unit chord."""
    all_vals = np.concatenate(
        [np.asarray(v, dtype=float).ravel() for v in values_lists if len(v)] or [np.array([0.0])]
    )
    vmax = float(np.max(np.abs(all_vals))) if all_vals.size else 0.0
    return span_fraction / max(vmax, 1e-30)


def _insert_zero_crossings(
    nodes: NDArray[np.float64],
    values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    aug_n = [nodes[0:1]]
    aug_v = [values[0:1]]
    for i in range(len(nodes) - 1):
        v0, v1 = float(values[i]), float(values[i + 1])
        if v0 * v1 < 0.0:
            t = v0 / (v0 - v1)
            crossing = nodes[i] + t * (nodes[i + 1] - nodes[i])
            aug_n.append(crossing[None])
            aug_v.append(np.array([0.0]))
        aug_n.append(nodes[i + 1 : i + 2])
        aug_v.append(values[i + 1 : i + 2])
    return np.vstack(aug_n), np.concatenate(aug_v)


def fill_distribution(
    ax: Any,
    nodes: NDArray[np.float64],
    values: NDArray[np.float64],
    scale: float,
    pos_color: str,
    neg_color: str,
    *,
    alpha: float = 0.45,
    zorder: int = 4,
    envelope_line_color: str | None = None,
) -> None:
    """Filled magnitude-area ribbon offset outward from a panel midline."""
    nodes = np.asarray(nodes, dtype=float)
    values = np.asarray(values, dtype=float).ravel()
    n = min(len(nodes), len(values))
    if n < 2:
        return
    nodes, values = _insert_zero_crossings(nodes[:n], values[:n])
    normals = outward_normals(nodes)
    mag = np.abs(values)
    offset = nodes + scale * mag[:, None] * normals

    # Section plot contract: horizontal = Y_B (column 1), vertical = X_B (column 0).
    for i in range(len(nodes) - 1):
        v_mid = 0.5 * (values[i] + values[i + 1])
        color = pos_color if v_mid >= 0.0 else neg_color
        ax.fill(
            [nodes[i, 1], offset[i, 1], offset[i + 1, 1], nodes[i + 1, 1]],
            [nodes[i, 0], offset[i, 0], offset[i + 1, 0], nodes[i + 1, 0]],
            color=color,
            alpha=alpha,
            linewidth=0,
            zorder=zorder,
        )

    eline = envelope_line_color if envelope_line_color is not None else pos_color
    e_h = np.where(mag > 0, offset[:, 1], np.nan)
    e_v = np.where(mag > 0, offset[:, 0], np.nan)
    ax.plot(e_h, e_v, color=eline, lw=0.9, ls="--", alpha=0.85, zorder=zorder + 1)


def plot_section_distributions(
    ax: Any,
    panel_nodes_list: Sequence[NDArray[np.float64]],
    q_values_list: Sequence[NDArray[np.float64]] | None,
    sigma_values_list: Sequence[NDArray[np.float64]] | None,
    *,
    airfoil_outline: NDArray[np.float64] | None = None,
    web_endpoints: list[tuple[Any, Any]] | None = None,
    spar_caps: list[dict[str, float]] | None = None,
    title: str = "Section stress distributions",
    q_span: float = 0.07,
    sigma_span: float = 0.055,
    title_fontsize: float = 10.0,
) -> None:
    """Combined q(s) / σ(s) ribbon diagram on one axes (array-based API).

    All geometry (``panel_nodes_list``, ``airfoil_outline``, ``web_endpoints``,
    ``spar_caps``) must be canonical **B** ``(X_B, Y_B)``; the axes follow the
    section plot contract (horiz :math:`Y_B`, vert :math:`X_B`, LE on RHS).
    """
    import matplotlib.pyplot as plt

    if airfoil_outline is not None:
        af = np.asarray(airfoil_outline, dtype=float)
        if not np.allclose(af[0], af[-1]):
            af = np.vstack([af, af[:1]])
        af_h, af_v = section_hv(af)
        ax.plot(
            af_h, af_v,
            color="#ffffff", lw=2.2, solid_capstyle="butt", alpha=0.9, zorder=18,
        )

    for nodes in panel_nodes_list:
        nd = np.asarray(nodes, dtype=float)
        if len(nd) >= 2:
            nd_h, nd_v = section_hv(nd)
            ax.plot(nd_h, nd_v, color="#f0f4f8", lw=1.1, alpha=0.7, zorder=8)

    if web_endpoints:
        for top, bot in web_endpoints:
            u = np.asarray(top, dtype=float).ravel()
            l = np.asarray(bot, dtype=float).ravel()
            ax.plot([u[1], l[1]], [u[0], l[0]], color="#eef6ff", lw=2.5, alpha=0.9, ls="--", zorder=9)

    if q_values_list is not None:
        s_q = auto_scale([np.asarray(q, dtype=float) for q in q_values_list], q_span)
        for nodes, q in zip(panel_nodes_list, q_values_list):
            fill_distribution(
                ax, np.asarray(nodes, dtype=float), np.asarray(q, dtype=float),
                s_q, "#2ecc71", "#1abc9c", envelope_line_color="#8dd4b8",
            )

    if sigma_values_list is not None:
        s_s = auto_scale([np.asarray(s, dtype=float) for s in sigma_values_list], sigma_span)
        for nodes, sigma in zip(panel_nodes_list, sigma_values_list):
            fill_distribution(
                ax, np.asarray(nodes, dtype=float), np.asarray(sigma, dtype=float),
                s_s, "#3498db", "#e74c3c", envelope_line_color="#c4c8e0",
            )

    if spar_caps and sigma_values_list is not None:
        # Marker is wide along the chord (display horizontal, Y_B) and thin
        # through the flapwise depth (display vertical, X_B).
        cap_chord, cap_flap = 0.030, 0.012
        for cap in spar_caps:
            x_b, y_b = float(cap["y"]), float(cap["z"])
            sigma_cap = float(cap.get("sigma", 0.0))
            rect = plt.Rectangle(
                (y_b - cap_chord / 2, x_b - cap_flap / 2), cap_chord, cap_flap,
                lw=1.2, edgecolor="#f39c12", facecolor="#f39c12", alpha=0.6, zorder=22,
            )
            ax.add_patch(rect)
            ax.text(
                y_b, x_b + cap_flap * 1.3, f"{sigma_cap / 1e6:.2f} MPa",
                ha="center", va="bottom", fontsize=6.5, color="#f39c12", alpha=0.9, zorder=23,
            )

    ax.set_facecolor("#0f1117")
    ax.set_title(title, color="#e0e0e0", fontsize=title_fontsize, pad=5)
    finish_section_axes(ax, label_color="#e0e0e0")
    ax.xaxis.label.set_fontsize(8)
    ax.yaxis.label.set_fontsize(8)
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a3a")
    ax.grid(True, color="#2a2a3a", lw=0.4, alpha=0.5)


def plot_section(
    ax: Any,
    zones: list[Any],
    booms: list[Any],
    webs_geom: list,
    airfoil: np.ndarray,
    spar_positions: list[float],
    q_tot: list,
    sig_p: list,
    sig_b: list,
    title: str,
    *,
    scale_q: float | None = None,
    scale_s: float | None = None,
    q_span: float = 0.07,
    sigma_span: float = 0.055,
    plot_shear: bool = True,
    plot_bending: bool = True,
    title_fontsize: float = 10.0,
) -> None:
    """
    Vendor-thin-wall ``plot_section``-compatible ribbon plot.

    ``airfoil`` and every panel/boom/web coordinate must be canonical **B**
    ``(X_B, Y_B)`` — one frame per axes. ``spar_positions`` stay chord fractions
    in ``[0, 1]`` and are mapped onto the outline's own :math:`Y_B` span for the
    cell labels, so no chord/pivot has to be threaded in.

    Used by :func:`save_thin_wall_stress_figures`.
    """
    import matplotlib.pyplot as plt

    outline = open_outline_from_airfoil(np.asarray(airfoil, dtype=float))
    af = np.vstack([outline, outline[:1]])
    af_h, af_v = section_hv(af)
    ax.plot(af_h, af_v, color="#7eb8da", lw=2.4, alpha=0.95, zorder=2)
    ax.plot(af_h, af_v, color="#ffffff", lw=2.2, alpha=0.98, zorder=18)

    if scale_q is None and plot_shear:
        scale_q = auto_scale([np.asarray(q, dtype=float) for q in q_tot], q_span)
    if scale_s is None and plot_bending:
        scale_s = auto_scale([np.asarray(s, dtype=float) for s in sig_p], sigma_span)

    for p, q_p, s_p in zip(zones, q_tot, sig_p):
        nodes = np.asarray(p.nodes, dtype=float)
        if plot_shear and scale_q is not None:
            fill_distribution(
                ax, nodes, np.asarray(q_p, dtype=float), scale_q,
                "#2ecc71", "#1abc9c", envelope_line_color="#8dd4b8",
            )
        if plot_bending and scale_s is not None:
            fill_distribution(
                ax, nodes, np.asarray(s_p, dtype=float), scale_s,
                "#3498db", "#e74c3c", envelope_line_color="#c4c8e0",
            )
        if len(nodes) >= 2:
            nd_h, nd_v = section_hv(nodes)
            ax.plot(nd_h, nd_v, color="#f0f4f8", lw=1.1, alpha=0.75, zorder=8)

    for (u, l) in webs_geom:
        ax.plot([u[1], l[1]], [u[0], l[0]], color="#eef6ff", lw=3.0, alpha=0.95, ls="--", zorder=9)

    if plot_bending:
        # Wide along the chord (display horizontal, Y_B), thin through the
        # flapwise depth (display vertical, X_B).
        cap_chord, cap_flap = 0.03, 0.012
        for b, sb in zip(booms, sig_b):
            x_b = float(getattr(b, "y", b[0] if hasattr(b, "__getitem__") else 0.0))
            y_b = float(getattr(b, "z", b[1] if hasattr(b, "__getitem__") else 0.0))
            rect = plt.Rectangle(
                (y_b - cap_chord / 2, x_b - cap_flap / 2),
                cap_chord, cap_flap,
                lw=1.2, edgecolor="#f39c12", facecolor="#f39c12", alpha=0.6, zorder=22,
            )
            ax.add_patch(rect)
            ax.text(
                y_b, x_b + cap_flap * 1.3, f"{float(sb) / 1e6:.2f} MPa",
                ha="center", va="bottom", fontsize=6.5, color="#f39c12", alpha=0.9, zorder=23,
            )

    # Cell labels: chord fractions mapped onto the outline's own Y_B span, so the
    # annotation stays in the same frame as the geometry it labels.
    all_frac = [0.0] + list(spar_positions) + [1.0]
    n_half = len(airfoil) // 2
    upper = np.asarray(airfoil[:n_half], dtype=float)
    seg = upper[np.argsort(upper[:, 1])]
    y_le, y_te = float(np.min(af[:, 1])), float(np.max(af[:, 1]))
    for i in range(len(all_frac) - 1):
        frac_mid = 0.5 * (all_frac[i] + all_frac[i + 1])
        y_mid = y_le + frac_mid * (y_te - y_le)
        x_u = float(np.interp(y_mid, seg[:, 1], seg[:, 0]))
        ax.text(
            y_mid, x_u * 0.2, f"C{i + 1}",
            ha="center", va="center", fontsize=8, color="white", alpha=0.4, zorder=12,
        )

    ax.set_title(title, color="#e0e0e0", fontsize=title_fontsize, pad=5)
    finish_section_axes(ax, label_color="#e0e0e0")
    ax.xaxis.label.set_fontsize(8)
    ax.yaxis.label.set_fontsize(8)
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    ax.set_facecolor("#0f1117")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a3a")
    ax.grid(True, color="#2a2a3a", lw=0.4, alpha=0.5)


def plot_clpt_laminate_stress_fi(
    zones: list[Any],
    q_tot: list,
    sig_p: list,
    *,
    panel_index: int = 0,
    station_index: int | None = None,
    strengths: dict[str, float] | None = None,
    outfile: Path | str | None = None,
    title: str | None = None,
) -> Path | None:
    """
    One skin station: thin-wall (σ_xx, q) → membrane CLPT → ply σ, ε, Hashin FI.

    Uses ``blade_structure_utilities.material_utilities`` CLT recovery (no vendor helper).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from blade_structure_utilities.material_utilities.laminate import membrane_resultants_from_shell_stress
    from blade_structure_utilities.material_utilities.laminate import laminate_definition_from_legacy
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_tertiary.local_clpt_shell import (
        default_skin_strengths_pa,
        solve_station_clpt_shell,
    )
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
        ShellPanelResultants,
    )

    p = panels[panel_index]
    plies_legacy = p.lam.build_plies()
    lam = laminate_definition_from_legacy(p.lam, plies=plies_legacy)
    if lam is None:
        raise TypeError("plot_clpt_laminate_stress_fi requires a convertible panel laminate")

    npt = len(np.asarray(p.s, dtype=float))
    if npt < 2:
        raise ValueError("Panel has insufficient stations for CLPT strip plot.")
    j = npt // 2 if station_index is None else int(station_index)
    j = max(0, min(npt - 1, j))

    sig_xx = float(np.asarray(sig_p[panel_index], dtype=float).ravel()[j])
    q_here = float(np.asarray(q_tot[panel_index], dtype=float).ravel()[j])
    t_wall = float(getattr(p.lam, "t", None) or lam.total_thickness())
    tau_xy = q_here / max(t_wall, 1e-30)
    N_vec = membrane_resultants_from_shell_stress(sig_xx, 0.0, tau_xy, t_wall)
    st = strengths or default_skin_strengths_pa()

    row = ShellPanelResultants(
        N11=float(N_vec[0]),
        N22=float(N_vec[1]),
        N12=float(N_vec[2]),
        M11=0.0,
        M22=0.0,
        M12=0.0,
        panel_index=panel_index,
        station_index=j,
        thickness_m=t_wall,
    )
    res = solve_station_clpt_shell(
        row,
        plies_legacy,
        Xt=st["Xt"],
        Xc=st["Xc"],
        Yt=st["Yt"],
        Yc=st["Yc"],
        S12=st["S12"],
        lam=lam,
    )

    nply = len(res.plies)
    layers = np.arange(nply)
    s11 = np.zeros(nply)
    s22 = np.zeros(nply)
    t12 = np.zeros(nply)
    e11 = np.zeros(nply)
    e22 = np.zeros(nply)
    g12 = np.zeros(nply)
    angles: list[float] = []
    # M-frame ply stress/strain come off the CLPT result directly — solve_station_
    # clpt_shell already applied T_sigma / T_epsilon from coordinate_frames.
    sig_mat = res.sig_ply_mat_mid or []
    eps_mat = res.eps_ply_mat_mid or []
    for k in range(min(nply, len(sig_mat), len(eps_mat))):
        th = float(getattr(res.plies[k], "theta_deg", getattr(res.plies[k], "angle", 0.0)))
        angles.append(th)
        sm = np.asarray(sig_mat[k], dtype=float).ravel()[:3]
        em = np.asarray(eps_mat[k], dtype=float).ravel()[:3]
        s11[k], s22[k], t12[k] = sm[0], sm[1], sm[2]
        e11[k], e22[k], g12[k] = em[0], em[1], em[2]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor="#0f1117")
    fig.patch.set_facecolor("#0f1117")
    ax0, ax1, ax2, ax3 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    bh = 0.22
    ax0.barh(layers - bh, s11 / 1e6, height=bh * 0.92, label="σ11", color="#5dade2")
    ax0.barh(layers, s22 / 1e6, height=bh * 0.92, label="σ22", color="#ec7063")
    ax0.barh(layers + bh, t12 / 1e6, height=bh * 0.92, label="τ12", color="#af7ac5")
    ax0.axvline(0.0, color="#888888", lw=0.6)
    ax0.set_yticks(layers)
    ax0.set_yticklabels([f"P{k + 1} ({angles[k]:.0f}°)" for k in range(nply)])
    ax0.set_xlabel("Stress [MPa]")
    ax0.set_title("Ply σ — material axes")
    ax0.legend(loc="lower right", fontsize=7)

    ax1.barh(layers - bh, e11 * 1e6, height=bh * 0.92, label="ε11 (µε)", color="#82e0aa")
    ax1.barh(layers, e22 * 1e6, height=bh * 0.92, label="ε22 (µε)", color="#f7dc6f")
    ax1.barh(layers + bh, g12 * 1e6, height=bh * 0.92, label="γ12 (µε)", color="#f5b7b1")
    ax1.axvline(0.0, color="#888888", lw=0.6)
    ax1.set_yticks(layers)
    ax1.set_yticklabels([f"P{k + 1}" for k in range(nply)])
    ax1.set_xlabel("Strain [microstrain]")
    ax1.set_title("Ply ε — material axes")
    ax1.legend(loc="lower right", fontsize=7)

    ax2.bar(layers, res.fi, width=0.65, color="#bb8fce", label="Hashin envelope FI")
    ax2.axhline(1.0, color="#e74c3c", ls="--", lw=1.0, label="FI = 1")
    ax2.set_xticks(layers)
    ax2.set_xticklabels([f"P{k + 1}" for k in range(nply)])
    ax2.set_ylabel("Failure index")
    ax2.set_title("Hashin envelope per ply")
    ax2.legend(loc="upper right", fontsize=8)

    ax3.axis("off")
    lbl = getattr(p, "label", None) or f"panel_{panel_index}"
    fi_max = float(np.max(res.fi)) if nply else 0.0
    summary = (
        "Thin-wall → membrane CLPT at one station\n"
        f"  panel={lbl!r}  s-index={j}/{npt - 1}\n"
        f"  t={t_wall * 1e3:.3f} mm\n"
        f"  σ_xx={sig_xx / 1e6:.4f} MPa  q={q_here:.4f} N/m\n"
        f"  τ_xy=q/t={tau_xy / 1e6:.4f} MPa\n"
        f"  max Hashin FI={fi_max:.3e}\n"
        f"  ε⁰=[{res.eps0[0]:.3e}, {res.eps0[1]:.3e}, {res.eps0[2]:.3e}]\n"
        f"  Xt/Xc/Yt/Yc/S12 [MPa]="
        f"{st['Xt']/1e6:.0f}/{st['Xc']/1e6:.0f}/"
        f"{st['Yt']/1e6:.0f}/{st['Yc']/1e6:.0f}/{st['S12']/1e6:.0f}"
    )
    ax3.text(
        0.02, 0.98, summary, transform=ax3.transAxes, va="top", ha="left",
        fontsize=8, family="monospace", color="#e0e0e0", linespacing=1.35,
    )

    fig.suptitle(
        title or "CLT: one skin station → ply σ, ε, Hashin FI",
        color="#e0e0e0",
        fontsize=10,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if outfile is None:
        plt.close(fig)
        return None
    out = Path(outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return out


__all__ = [
    "auto_scale",
    "fill_distribution",
    "outward_normals",
    "plot_clpt_laminate_stress_fi",
    "plot_section",
    "plot_section_distributions",
]
