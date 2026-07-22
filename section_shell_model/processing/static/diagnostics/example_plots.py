"""
PNG outputs for the section_shell_model example: mesh strips and stress recovery.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from blade_structure_utilities.section_shell_model.pre_processing.loading.thin_wall.sectorial_warping import (
    open_outline_from_airfoil,
)


def save_mitc4_section_mesh_figure(
    outfile: "Path | str",
    panels: list[Any],
    all_panel_results: list[list[Any]],
    airfoil: np.ndarray,
    spar_positions: list[float],
    *,
    dpi: int = 150,
    figsize: tuple[float, float] = (10.0, 3.8),
) -> "Path":
    """Plot the actual MITC4 element mesh derived from per-element result data.

    Uses ``len(all_panel_results[pi])`` as the element count for each panel,
    then places element-boundary nodes at uniform arc-length positions along
    the panel midline.  This is the correct visualisation of the discretisation
    used by the MITC4 solver, not the fine-sampled Panel polylines.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0f1117")
    ax.set_facecolor("#0f1117")

    outline = open_outline_from_airfoil(airfoil)
    af = np.vstack([outline, outline[:1]])
    ax.plot(af[:, 0], af[:, 1], color="#5dade2", lw=1.0, alpha=0.4, zorder=1)

    import matplotlib.cm as cm_base
    cmap = cm_base.get_cmap("tab20")

    n_elem_total = 0
    n_node_total = 0
    for pi, (p, elem_results) in enumerate(zip(panels, all_panel_results)):
        if not elem_results:
            continue
        n_elem = len(elem_results)
        nodes = np.asarray(p.nodes, dtype=float)
        s_panel = np.asarray(p.s, dtype=float)
        if nodes.shape[0] < 2 or s_panel.size < 2:
            continue
        # Sample n_elem+1 boundary nodes at uniform s-spacing (matches default mesh)
        s_bounds = np.linspace(float(s_panel.min()), float(s_panel.max()), n_elem + 1)
        y_bounds = np.interp(s_bounds, s_panel, nodes[:, 0])
        z_bounds = np.interp(s_bounds, s_panel, nodes[:, 1])

        col = cmap((pi % 20) / 19.0)
        ax.plot(y_bounds, z_bounds, color=col, lw=2.0, solid_capstyle="round", zorder=3)
        ax.scatter(y_bounds, z_bounds, s=22, color=col, zorder=4, edgecolors="#ffffff",
                   linewidths=0.5)

        # Label panel at midpoint
        ym = float(np.interp(0.5 * (s_panel.min() + s_panel.max()), s_panel, nodes[:, 0]))
        zm = float(np.interp(0.5 * (s_panel.min() + s_panel.max()), s_panel, nodes[:, 1]))
        label = str(getattr(p, "label", "") or f"P{pi}")
        ax.text(ym, zm, label, fontsize=6, color="#bbbbbb", ha="center", va="center",
                zorder=5, alpha=0.8)

        n_elem_total += n_elem
        n_node_total += n_elem + 1

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("y/c [m]", color="#e0e0e0", fontsize=9)
    ax.set_ylabel("z/c [m]", color="#e0e0e0", fontsize=9)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    ax.grid(True, color="#2a2a3a", lw=0.3, alpha=0.4)
    ax.set_title(
        f"MITC4 shell mesh: quad elements per panel (uniform arc-length spacing)\n"
        f"(elements={n_elem_total}, boundary nodes={n_node_total}, panels={len([r for r in all_panel_results if r])})",
        color="#e0e0e0", fontsize=9,
    )
    plt.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return Path(outfile)


def save_shell_mesh_figure(
    outfile: Path | str,
    panels: list[Any],
    webs_geom: list,
    airfoil: np.ndarray,
    spar_positions: list[float],
    *,
    dpi: int = 150,
    figsize: tuple[float, float] = (10.0, 7.0),
) -> Path:
    """
    Plot midline thin-wall mesh: each edge between consecutive panel nodes is one
    linear strip element; nodes indicate discretisation density.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0f1117")
    ax.set_facecolor("#0f1117")

    outline = open_outline_from_airfoil(airfoil)
    af = np.vstack([outline, outline[:1]])
    ax.plot(
        af[:, 0],
        af[:, 1],
        color="#5dade2",
        lw=1.2,
        alpha=0.85,
        zorder=1,
    )

    import matplotlib.cm as cm_base

    cmap = cm_base.get_cmap("tab20")
    n_seg_total = 0
    n_node_samples = 0
    for pi, p in enumerate(panels):
        nodes = np.asarray(p.nodes, dtype=float)
        if len(nodes) < 2:
            continue
        col = cmap((pi % 20) / 19.0)
        ax.plot(
            nodes[:, 0],
            nodes[:, 1],
            color=col,
            lw=2.2,
            solid_capstyle="round",
            zorder=3,
        )
        n_seg_total += len(nodes) - 1
        n_node_samples += len(nodes)
        ax.scatter(
            nodes[:, 0],
            nodes[:, 1],
            c=[col],
            s=14,
            zorder=5,
            edgecolors="#ffffff",
            linewidths=0.35,
        )
        lbl = getattr(p, "label", None) or f"P{pi}"
        mid = nodes[len(nodes) // 2]
        ax.text(
            mid[0],
            mid[1],
            f" {lbl}",
            fontsize=6,
            color=col,
            alpha=0.95,
            zorder=6,
            clip_on=True,
        )

    for (u, l) in webs_geom:
        ax.plot(
            [u[0], l[0]],
            [u[1], l[1]],
            color="#ecf0f1",
            lw=2.5,
            ls="--",
            alpha=0.9,
            zorder=4,
        )

    all_x = [0.0] + list(spar_positions) + [1.0]
    n_half = len(airfoil) // 2
    upper = airfoil[:n_half]
    order = np.argsort(upper[:, 0])
    seg = upper[order]
    for i in range(len(all_x) - 1):
        xm = 0.5 * (all_x[i] + all_x[i + 1])
        z_u = float(np.interp(xm, seg[:, 0], seg[:, 1]))
        ax.text(
            xm,
            z_u * 0.15,
            f"C{i+1}",
            ha="center",
            va="center",
            fontsize=9,
            color="#aaaaaa",
            alpha=0.7,
        )

    ax.set_aspect("equal")
    ax.set_title(
        "Thin-wall mesh: linear strip elements along panel midlines\n"
        f"(segments={n_seg_total}, node samples={n_node_samples}, panels={len(panels)})",
        color="#e0e0e0",
        fontsize=10,
    )
    ax.set_xlabel("y/c [m]", color="#e0e0e0", fontsize=9)
    ax.set_ylabel("z/c [m]", color="#e0e0e0", fontsize=9)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    ax.grid(True, color="#2a2a3a", lw=0.4, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a3a")

    plt.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return outfile


def save_thin_wall_stress_figures(
    outfile_prefix: Path | str,
    panels: list[Any],
    booms: list[Any],
    webs_geom: list,
    airfoil: np.ndarray,
    spar_positions: list[float],
    q_tot: list,
    sig_p: list,
    sig_b: list,
    *,
    dpi: int = 150,
) -> tuple[Path, Path]:
    """
    Shear-flow and axial-stress ribbon plots via package ``thin_wall_plots.plot_section``.
    Writes ``{prefix}_shear_flow.png`` and ``{prefix}_axial_stress.png``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from blade_structure_utilities.section_shell_model.processing.static.diagnostics.thin_wall_plots import (
        plot_section,
    )

    base = Path(outfile_prefix)
    base.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")
    fig1, ax1 = plt.subplots(figsize=(9.0, 6.5), facecolor="#0f1117")
    plot_section(
        ax1,
        panels,
        booms,
        webs_geom,
        airfoil,
        spar_positions,
        q_tot,
        sig_p,
        sig_b,
        "Shear flow q(s) — thin-wall recovery",
        plot_shear=True,
        plot_bending=False,
        title_fontsize=11.0,
    )
    fig1.patch.set_facecolor("#0f1117")
    p1 = Path(f"{base}_shear_flow.png")
    fig1.savefig(p1, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9.0, 6.5), facecolor="#0f1117")
    plot_section(
        ax2,
        panels,
        booms,
        webs_geom,
        airfoil,
        spar_positions,
        q_tot,
        sig_p,
        sig_b,
        "Axial stress σ_xx(s) — thin-wall recovery",
        plot_shear=False,
        plot_bending=True,
        title_fontsize=11.0,
    )
    fig2.patch.set_facecolor("#0f1117")
    p2 = Path(f"{base}_axial_stress.png")
    fig2.savefig(p2, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig2)

    return p1, p2


def save_clpt_ply_figure(
    outfile: Path | str,
    panels: list[Any],
    q_tot: list,
    sig_p: list,
    *,
    panel_index: int = 0,
    station_index: int | None = None,
    strengths: dict[str, float] | None = None,
) -> Path:
    """Ply σ, ε, Hashin envelope FI at one station (package CLT helper; savefig dpi=150)."""
    from blade_structure_utilities.section_shell_model.processing.static.diagnostics.thin_wall_plots import (
        plot_clpt_laminate_stress_fi,
    )

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plot_clpt_laminate_stress_fi(
        panels,
        q_tot,
        sig_p,
        panel_index=panel_index,
        station_index=station_index,
        strengths=strengths,
        outfile=outfile,
        title="section_shell_model: CLPT ply σ, ε, Hashin (reference skin station)",
    )
    return outfile


def save_panel_along_contour_figure(
    outfile: Path | str,
    panels: list[Any],
    q_tot: list,
    sig_p: list,
    panel_index: int,
    *,
    sigma_omega_mids: np.ndarray | None = None,
    dpi: int = 150,
) -> Path:
    """
    q(s), σ_xx(s) — and optionally σ_ω(s) — along contour coordinate s [m] for one panel.

    Parameters
    ----------
    sigma_omega_mids : optional (n_elem,) array of Vlasov warping stress [Pa] at element mids.
        When provided, a third subplot is added showing the warping stress contribution.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    p = panels[panel_index]
    s_along = np.asarray(p.s, dtype=float)
    q_along = np.asarray(q_tot[panel_index], dtype=float).ravel()
    sig_along = np.asarray(sig_p[panel_index], dtype=float).ravel()
    n = min(len(s_along), len(q_along), len(sig_along))
    s_along = s_along[:n]
    q_along = q_along[:n]
    sig_along = sig_along[:n]

    lbl = getattr(p, "label", None) or f"panel_{panel_index}"
    has_omega = sigma_omega_mids is not None and len(sigma_omega_mids) > 0

    n_rows = 3 if has_omega else 2
    figheight = 8.5 if has_omega else 6.0

    plt.style.use("dark_background")
    fig, axes = plt.subplots(n_rows, 1, figsize=(9.0, figheight), sharex=True,
                             facecolor="#0f1117")
    fig.patch.set_facecolor("#0f1117")
    axq, axs = axes[0], axes[1]
    for ax in axes:
        ax.set_facecolor("#0f1117")

    axq.plot(s_along, q_along, color="#2ecc71", lw=1.3, label="q")
    axq.set_ylabel("q [N/m]", color="#e0e0e0", fontsize=9)
    axq.set_title(f"Along-panel resultants — {lbl}", color="#e0e0e0", fontsize=10)
    axq.grid(True, color="#2a2a3a", lw=0.4, alpha=0.5)
    axq.tick_params(colors="#aaaaaa", labelsize=8)
    axq.legend(loc="best", fontsize=8)

    axs.plot(s_along, sig_along / 1e6, color="#3498db", lw=1.3, label="σ_xx")
    axs.set_ylabel("σ_xx [MPa]", color="#e0e0e0", fontsize=9)
    axs.grid(True, color="#2a2a3a", lw=0.4, alpha=0.5)
    axs.tick_params(colors="#aaaaaa", labelsize=8)
    axs.legend(loc="best", fontsize=8)

    if has_omega:
        axw = axes[2]
        # sigma_omega_mids is at element mids; build corresponding s-mids
        s_mids = 0.5 * (s_along[:-1] + s_along[1:])
        n_om = min(len(s_mids), len(sigma_omega_mids))
        axw.plot(s_mids[:n_om], np.asarray(sigma_omega_mids[:n_om]) / 1e6,
                 color="#e67e22", lw=1.3, label="σ_ω (Vlasov)")
        axw.axhline(0.0, color="#555", lw=0.6, ls="--")
        axw.set_xlabel("s along panel [m]", color="#e0e0e0", fontsize=9)
        axw.set_ylabel("σ_ω [MPa]", color="#e0e0e0", fontsize=9)
        axw.grid(True, color="#2a2a3a", lw=0.4, alpha=0.5)
        axw.tick_params(colors="#aaaaaa", labelsize=8)
        axw.legend(loc="best", fontsize=8)
    else:
        axs.set_xlabel("s along panel [m]", color="#e0e0e0", fontsize=9)

    plt.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return outfile


def save_mitc4_fi_figure(
    outfile: Path | str,
    all_panel_fi: list,
    panel_labels: list[str] | None = None,
    *,
    dpi: int = 150,
) -> Path:
    """
    Heatmap of Hashin-envelope failure indices across all panels and elements.

    Parameters
    ----------
    all_panel_fi  : list[np.ndarray]  — FI per element per panel (max over plies)
    panel_labels  : optional panel label strings
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    n_panels = len(all_panel_fi)
    if n_panels == 0:
        return outfile

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11.0, 4.5), facecolor="#0f1117")
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # Build a 2D grid: rows = panels, cols = elements (variable lengths → pad)
    max_n = max(len(fi) for fi in all_panel_fi if len(fi) > 0)
    grid = np.full((n_panels, max_n), np.nan)
    for pi, fi in enumerate(all_panel_fi):
        if len(fi) > 0:
            grid[pi, : len(fi)] = fi

    norm = mcolors.Normalize(vmin=0.0, vmax=max(1.0, float(np.nanmax(grid))))
    im = ax.imshow(grid, aspect="auto", norm=norm, cmap="plasma", origin="upper",
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Hashin FI (max over plies)", color="#e0e0e0", fontsize=9)
    cb.ax.tick_params(colors="#aaaaaa", labelsize=7)

    ax.set_xlabel("element index along panel contour", color="#e0e0e0", fontsize=9)
    ax.set_ylabel("panel index", color="#e0e0e0", fontsize=9)
    ax.set_title("MITC4 full-section Hashin failure index sweep", color="#e0e0e0", fontsize=10)
    ax.tick_params(colors="#aaaaaa", labelsize=7)

    if panel_labels:
        ax.set_yticks(range(n_panels))
        ax.set_yticklabels(panel_labels, fontsize=7, color="#aaaaaa")

    # Contour at FI = 1 (failure threshold)
    if np.nanmax(grid) >= 1.0:
        ax.contour(grid, levels=[1.0], colors=["#e74c3c"], linewidths=1.2)

    plt.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return outfile


def save_multi_panel_hashin_montage(
    outfile: Path | str,
    all_panel_fi: list,
    panel_labels: list[str] | None = None,
    *,
    dpi: int = 150,
) -> Path:
    """One mini heatmap subplot per panel (Hashin FI along contour)."""
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    n = len(all_panel_fi)
    if n == 0:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "no panel FI", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return outfile
    ncols = int(math.ceil(math.sqrt(float(n))))
    nrows = int(math.ceil(n / float(ncols)))
    plt.style.use("dark_background")
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.2 * nrows), facecolor="#0f1117")
    if nrows * ncols == 1:
        ax_flat = [axes]
    else:
        ax_flat = axes.ravel()
    vals: list[float] = []
    for fi in all_panel_fi:
        a = np.asarray(fi, dtype=np.float64).ravel()
        if a.size:
            vals.append(float(np.nanmax(a)))
    mx = max(1.0, max(vals) if vals else 0.0)
    norm = mcolors.Normalize(vmin=0.0, vmax=mx)
    for k in range(nrows * ncols):
        ax = ax_flat[k]
        ax.set_facecolor("#0f1117")
        if k >= n:
            ax.axis("off")
            continue
        fi = np.asarray(all_panel_fi[k], dtype=np.float64).reshape(1, -1)
        ax.imshow(fi, aspect="auto", norm=norm, cmap="plasma", interpolation="nearest")
        lab = (panel_labels[k] if panel_labels and k < len(panel_labels) else f"panel_{k}")[:28]
        ax.set_title(lab, fontsize=7, color="#e0e0e0")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Per-panel Hashin FI (montage)", color="#e0e0e0", fontsize=10)
    plt.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return outfile


def _scalar_field_on_section_geometry(
    outfile: Path | str,
    airfoil: np.ndarray,
    webs_geom: list,
    spar_positions: list[float],
    panels: list[Any],
    per_panel_values: list[np.ndarray],
    *,
    title: str,
    cbar_label: str,
    cmap: str = "turbo",
    vmin: float | None = None,
    vmax: float | None = None,
    diverging: bool = False,
    dpi: int = 150,
    figsize: tuple[float, float] = (10.0, 7.0),
) -> Path:
    """
    Shared helper: draw any per-element scalar field mapped onto panel midlines.

    Each panel's values array length equals its number of MITC4 elements; positions
    are interpolated from ``p.s`` / ``p.nodes`` using a uniform s-linspace.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    segments: list = []
    values: list[float] = []
    mid_y: list[float] = []
    mid_z: list[float] = []
    for p, vals in zip(panels, per_panel_values):
        vals = np.asarray(vals, dtype=float).ravel()
        n_e = int(vals.size)
        if n_e == 0:
            continue
        s_panel = np.asarray(p.s, dtype=float)
        nodes = np.asarray(p.nodes, dtype=float)
        if s_panel.size < 2 or nodes.shape[0] < 2:
            continue
        s_min, s_max = float(s_panel.min()), float(s_panel.max())
        if s_max <= s_min + 1e-30:
            continue
        s_nodes = np.linspace(s_min, s_max, n_e + 1, dtype=np.float64)
        for k in range(n_e):
            sa, sb = float(s_nodes[k]), float(s_nodes[k + 1])
            y0 = float(np.interp(sa, s_panel, nodes[:, 0]))
            z0 = float(np.interp(sa, s_panel, nodes[:, 1]))
            y1 = float(np.interp(sb, s_panel, nodes[:, 0]))
            z1 = float(np.interp(sb, s_panel, nodes[:, 1]))
            segments.append(((y0, z0), (y1, z1)))
            values.append(float(vals[k]))
            sm = 0.5 * (sa + sb)
            mid_y.append(float(np.interp(sm, s_panel, nodes[:, 0])))
            mid_z.append(float(np.interp(sm, s_panel, nodes[:, 1])))

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0f1117")
    ax.set_facecolor("#1a1d2e")

    if not values:
        ax.set_title(f"No data — {title}", color="#e0e0e0", fontsize=10)
        fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
        plt.close(fig)
        return outfile

    v_arr = np.asarray(values, dtype=np.float64)
    v_lo = float(vmin) if vmin is not None else float(np.nanmin(v_arr))
    v_hi = float(vmax) if vmax is not None else float(np.nanmax(v_arr))
    if abs(v_hi - v_lo) < 1e-30:
        v_hi = v_lo + 1.0

    if diverging and v_lo < 0.0 < v_hi:
        norm: mcolors.Normalize = mcolors.TwoSlopeNorm(vmin=v_lo, vcenter=0.0, vmax=v_hi)
    else:
        norm = mcolors.Normalize(vmin=v_lo, vmax=v_hi)

    outline = open_outline_from_airfoil(airfoil)
    af = np.vstack([outline, outline[:1]])
    ax.plot(af[:, 0], af[:, 1], color="#5dade2", lw=1.0, alpha=0.5, zorder=1)
    for (u, w) in webs_geom:
        ax.plot([u[0], w[0]], [u[1], w[1]], color="#ecf0f1", lw=1.2, ls="--", alpha=0.5, zorder=2)

    segs = np.asarray(segments, dtype=np.float64)
    lc = LineCollection(segs, array=v_arr, cmap=cmap, norm=norm,
                        linewidths=7.0, capstyle="round", joinstyle="round",
                        zorder=4, antialiased=True)
    lc.set_clip_on(False)
    ax.add_collection(lc)
    my = np.asarray(mid_y, dtype=np.float64)
    mz = np.asarray(mid_z, dtype=np.float64)
    ax.scatter(my, mz, c=v_arr, cmap=cmap, norm=norm, s=72, zorder=5,
               edgecolors="#f8f8f8", linewidths=0.85, clip_on=False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(v_arr)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(cbar_label, color="#e0e0e0", fontsize=9)
    cbar.ax.tick_params(colors="#aaaaaa", labelsize=7)

    all_x = [0.0] + list(spar_positions) + [1.0]
    n_half = len(airfoil) // 2
    upper = airfoil[:n_half]
    order = np.argsort(upper[:, 0])
    seg_u = upper[order]
    for i in range(len(all_x) - 1):
        xm = 0.5 * (all_x[i] + all_x[i + 1])
        z_uf = float(np.interp(xm, seg_u[:, 0], seg_u[:, 1]))
        ax.text(xm, z_uf * 0.12, f"C{i+1}", ha="center", va="center",
                fontsize=8, color="#666666", alpha=0.9)

    el_pts = segs.reshape(-1, 2)
    all_xy = np.vstack([af, el_pts])
    wx = float(np.ptp(all_xy[:, 0]))
    wy = float(np.ptp(all_xy[:, 1]))
    pad = max(0.02 * max(wx, wy, 0.01), 1e-6)
    ax.set_xlim(float(all_xy[:, 0].min()) - pad, float(all_xy[:, 0].max()) + pad)
    ax.set_ylim(float(all_xy[:, 1].min()) - pad, float(all_xy[:, 1].max()) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("y/c [m]", color="#e0e0e0", fontsize=9)
    ax.set_ylabel("z/c [m]", color="#e0e0e0", fontsize=9)
    ax.set_title(title, color="#e0e0e0", fontsize=10)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    ax.grid(True, color="#2a2a3a", lw=0.3, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a3a")

    plt.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117",
                pil_kwargs={"compress_level": 6})
    plt.close(fig)
    return outfile


def save_clpt_fi_on_section_geometry(
    outfile: Path | str,
    airfoil: np.ndarray,
    webs_geom: list,
    spar_positions: list[float],
    panels: list[Any],
    all_panel_fi: list[np.ndarray],
    *,
    dpi: int = 150,
    figsize: tuple[float, float] = (10.0, 7.0),
    vmax: float | None = None,
    cmap: str = "turbo",
) -> Path:
    """
    Map per-MITC4-element max ply **Hashin FI (CLPT)** onto panel midlines in the
    section plane (y, z) [m].
    """
    all_vals = np.concatenate([np.asarray(fi, dtype=float).ravel()
                               for fi in all_panel_fi if len(fi)] or [np.array([])])
    fmax = float(np.nanmax(all_vals)) if all_vals.size else 0.0
    if vmax is not None:
        v_hi = max(float(vmax), 1e-30)
    elif 0.0 < fmax < 1.0:
        v_hi = fmax
    else:
        v_hi = max(1.0, fmax) if fmax > 0.0 else 1.0
    return _scalar_field_on_section_geometry(
        outfile, airfoil, webs_geom, spar_positions, panels, all_panel_fi,
        title=(
            "CLPT: max ply Hashin FI on section geometry (per MITC4 element along contour)\n"
            f"(max FI = {fmax:.3e}; colour scale 0 … {v_hi:.3e})"
        ),
        cbar_label="max ply Hashin FI (CLPT)",
        cmap=cmap, vmin=0.0, vmax=v_hi, dpi=dpi, figsize=figsize,
    )


def save_mitc4_resultants_figure(
    outfile: Path | str,
    panel_resultants: list,
    panel_label: str = "",
    *,
    dpi: int = 150,
) -> Path:
    """
    Eight-panel figure showing all MITC4 shell resultants along panel contour,
    including FSDT transverse shear resultants Qx and Qy.

    Parameters
    ----------
    panel_resultants : list[ShellPanelResultants]  — per-element ShellPanelResultants
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    indices = [r.station_index for r in panel_resultants]
    fields = [
        ("Nx",  [r.Nx  for r in panel_resultants], "N/m",   "#3498db"),
        ("Ny",  [r.Ny  for r in panel_resultants], "N/m",   "#2ecc71"),
        ("Nxy", [r.Nxy for r in panel_resultants], "N/m",   "#e74c3c"),
        ("Mx",  [r.Mx  for r in panel_resultants], "N·m/m", "#9b59b6"),
        ("My",  [r.My  for r in panel_resultants], "N·m/m", "#f39c12"),
        ("Mxy", [r.Mxy for r in panel_resultants], "N·m/m", "#1abc9c"),
        ("Qx",  [r.Qx if r.Qx is not None else 0.0 for r in panel_resultants], "N/m", "#e67e22"),
        ("Qy",  [r.Qy if r.Qy is not None else 0.0 for r in panel_resultants], "N/m", "#c0392b"),
    ]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(4, 2, figsize=(11.0, 12.0), facecolor="#0f1117")
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(
        f"MITC4 shell resultants — {panel_label or 'panel'}",
        color="#e0e0e0",
        fontsize=11,
    )

    for ax, (name, vals, unit, col) in zip(axes.flat, fields):
        ax.set_facecolor("#0f1117")
        ax.plot(indices, vals, color=col, lw=1.5, marker="o", markersize=3)
        ax.axhline(0.0, color="#555", lw=0.6, ls="--")
        ax.set_title(name, color="#e0e0e0", fontsize=10)
        ax.set_ylabel(unit, color="#aaaaaa", fontsize=8)
        ax.set_xlabel("element index", color="#aaaaaa", fontsize=8)
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        ax.grid(True, color="#2a2a3a", lw=0.4, alpha=0.5)

    plt.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return outfile


def save_loads_provenance_png(
    outfile: Path | str,
    *,
    N: float,
    Vy: float,
    Vz: float,
    My: float,
    Mz: float,
    T: float,
    station_tag: str = "",
    rz_m: float | None = None,
    n_elements_per_panel: int | None = None,
    dpi: int = 150,
) -> Path:
    """Single-panel text summary of section resultants (for final-pass provenance)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 2.2), facecolor="#0f1117")
    ax.axis("off")
    lines = [
        f"station_tag={station_tag!r}",
        f"N={N:.6g}  Vy={Vy:.6g}  Vz={Vz:.6g}  My={My:.6g}  Mz={Mz:.6g}  T={T:.6g}",
    ]
    if rz_m is not None:
        lines.append(f"r_z_m={float(rz_m):.6g} m")
    if n_elements_per_panel is not None:
        lines.append(f"n_elements_per_panel={int(n_elements_per_panel)}")
    ax.text(
        0.02,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        color="#e0e0e0",
    )
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return outfile


def save_mitc4_v2_section_mesh_figure(
    outfile: Path | str,
    mesh: Any,
    *,
    dpi: int = 150,
    caption_lines: list[str] | None = None,
) -> Path:
    """Plot MITC4 v2 section mesh (panel quads in ``y,z``) from :class:`SectionStripMesh`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9.0, 7.0), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")
    pmeshes = getattr(mesh, "panel_meshes", None) or []
    for pm in pmeshes:
        yz = np.asarray(getattr(pm, "yz_nodes", []), dtype=np.float64)
        if yz.ndim != 2 or yz.shape[1] < 2:
            continue
        for el in getattr(pm, "elements", []) or []:
            if len(el) < 3:
                continue
            loop = list(el) + [el[0]]
            ii = [int(i) for i in loop if 0 <= int(i) < yz.shape[0]]
            if len(ii) < 3:
                continue
            ax.plot(yz[ii, 0], yz[ii, 1], color="#5dade2", lw=0.9, alpha=0.85)
        ax.scatter(yz[:, 0], yz[:, 1], s=6, c="#ecf0f1", alpha=0.35, zorder=3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("y [m]", color="#e0e0e0", fontsize=9)
    ax.set_ylabel("z [m]", color="#e0e0e0", fontsize=9)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    ax.grid(True, color="#2a2a3a", lw=0.3, alpha=0.4)
    ttl = "MITC4 v2 — section mesh (midline strip quads)"
    if caption_lines:
        ttl = ttl + "\n" + " | ".join(caption_lines[:3])
    ax.set_title(ttl, color="#e0e0e0", fontsize=10)
    if caption_lines and len(caption_lines) > 3:
        fig.text(0.02, 0.02, "\n".join(caption_lines[3:]), fontsize=7, color="#aaaaaa", va="bottom")
    plt.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return outfile


def save_mitc4_v2_dashboard_figure(
    outfile: Path | str,
    mesh: Any,
    *,
    N: float,
    Vy: float,
    Vz: float,
    My: float,
    Mz: float,
    T: float,
    station_tag: str,
    rz_m: float,
    n_elements_per_panel: int,
    dpi: int = 150,
) -> Path:
    """Two-panel v2 dashboard: mesh (left) + JSON-style summary (right)."""
    import json as _json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(12.0, 6.0), facecolor="#0f1117", gridspec_kw={"width_ratios": [1.15, 1.0]}
    )
    ax0.set_facecolor("#0f1117")
    ax1.set_facecolor("#0f1117")
    pmeshes = getattr(mesh, "panel_meshes", None) or []
    for pm in pmeshes:
        yz = np.asarray(getattr(pm, "yz_nodes", []), dtype=np.float64)
        if yz.ndim != 2 or yz.shape[1] < 2:
            continue
        for el in getattr(pm, "elements", []) or []:
            if len(el) < 3:
                continue
            loop = list(el) + [el[0]]
            ii = [int(i) for i in loop if 0 <= int(i) < yz.shape[0]]
            if len(ii) < 3:
                continue
            ax0.plot(yz[ii, 0], yz[ii, 1], color="#5dade2", lw=0.9, alpha=0.9)
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_title("Mesh", color="#e0e0e0", fontsize=10)
    ax0.set_xlabel("y [m]", color="#aaaaaa", fontsize=8)
    ax0.set_ylabel("z [m]", color="#aaaaaa", fontsize=8)
    summ: dict[str, Any] = {}
    if hasattr(mesh, "summary"):
        try:
            summ = mesh.summary()
        except Exception:
            summ = {}
    txt = _json.dumps(
        {
            "station_tag": station_tag,
            "rz_m": float(rz_m),
            "resultants": {"N": N, "Vy": Vy, "Vz": Vz, "My": My, "Mz": Mz, "T": T},
            "n_elements_per_panel": int(n_elements_per_panel),
            "mesh_summary": summ,
        },
        indent=2,
    )
    ax1.axis("off")
    ax1.text(
        0.02,
        0.98,
        txt,
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        family="monospace",
        color="#e0e0e0",
    )
    fig.suptitle(f"MITC4 v2 station dashboard — {station_tag}", color="#e0e0e0", fontsize=11)
    plt.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return outfile


# ---------------------------------------------------------------------------
# Reserve factor on section geometry
# ---------------------------------------------------------------------------

def save_reserve_factor_on_section_geometry(
    outfile: Path | str,
    airfoil: np.ndarray,
    webs_geom: list,
    spar_positions: list[float],
    panels: list[Any],
    all_panel_fi: list[np.ndarray],
    *,
    dpi: int = 150,
    figsize: tuple[float, float] = (10.0, 7.0),
    rf_cap: float = 10.0,
    cmap: str = "RdYlGn",
) -> Path:
    """
    Map strength reserve factor (RF = 1/FI, capped at ``rf_cap``) onto panel
    midlines. Green = high reserve, red = near/at failure.
    """
    rf_panels: list[np.ndarray] = []
    for fi in all_panel_fi:
        fi_arr = np.asarray(fi, dtype=float).ravel()
        rf = np.where(fi_arr > 1e-6, 1.0 / fi_arr, rf_cap)
        rf_panels.append(np.clip(rf, 0.0, rf_cap))
    all_rf = np.concatenate([r for r in rf_panels if r.size] or [np.array([])])
    rf_min = float(np.nanmin(all_rf)) if all_rf.size else 0.0
    rf_max = float(np.nanmax(all_rf)) if all_rf.size else rf_cap
    return _scalar_field_on_section_geometry(
        outfile, airfoil, webs_geom, spar_positions, panels, rf_panels,
        title=(
            f"Strength reserve factor RF = 1/FI (capped at {rf_cap:.0f}) on section geometry\n"
            f"(min RF = {rf_min:.2f}; max RF = {rf_max:.2f})"
        ),
        cbar_label=f"RF = 1/FI (cap {rf_cap:.0f})",
        cmap=cmap, vmin=0.0, vmax=min(rf_max, rf_cap), dpi=dpi, figsize=figsize,
    )


# ---------------------------------------------------------------------------
# Per-ply stress / strain / FI / RF breakdown (reference panel)
# ---------------------------------------------------------------------------

def save_ply_stress_breakdown_figure(
    outfile: Path | str,
    sig_lam_mid: list[np.ndarray],
    eps_lam_mid: list[np.ndarray],
    fi: np.ndarray,
    plies: list[Any],
    *,
    station_label: str = "",
    dpi: int = 150,
) -> Path:
    """
    Four-panel bar chart for the reference panel's CLPT ply-level results:
    σ₁₁/σ₂₂/σ₁₂, ε₁₁/ε₂₂/γ₁₂, Hashin FI, and reserve factor (1/FI).
    x-axis ticks are ply angles extracted from the ply stack.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    n_plies = len(sig_lam_mid)
    if n_plies == 0:
        return outfile

    angles: list[str] = []
    for i, ply in enumerate(plies[:n_plies]):
        a = getattr(ply, "angle", None) or getattr(ply, "theta_deg", None)
        angles.append(f"{float(a):.0f}°" if a is not None else f"#{i}")

    ply_idx = np.arange(n_plies)
    sig_arr = np.array([np.asarray(s, dtype=float).ravel()[:3]
                        for s in sig_lam_mid], dtype=float) / 1e6  # MPa
    eps_arr = np.array([np.asarray(e, dtype=float).ravel()[:3]
                        for e in eps_lam_mid], dtype=float)
    fi_arr = np.asarray(fi, dtype=float).ravel()[:n_plies]
    rf_arr = np.clip(np.where(fi_arr > 1e-6, 1.0 / fi_arr, 20.0), 0.0, 20.0)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(4, 1, figsize=(max(8.0, n_plies * 0.7 + 2.0), 13.0),
                             facecolor="#0f1117")
    fig.patch.set_facecolor("#0f1117")
    title = "CLPT ply-level stress / strain / FI / RF — reference panel"
    if station_label:
        title += f"  [{station_label}]"
    fig.suptitle(title, color="#e0e0e0", fontsize=11)

    w = 0.26
    sig_cfg = [("σ₁₁", "#3498db"), ("σ₂₂", "#2ecc71"), ("σ₁₂", "#e74c3c")]
    eps_cfg = [("ε₁₁", "#9b59b6"), ("ε₂₂", "#f39c12"), ("γ₁₂", "#1abc9c")]
    ax0, ax1, ax2, ax3 = axes

    for j, (lbl, col) in enumerate(sig_cfg):
        ax0.bar(ply_idx + j * w, sig_arr[:, j], w, label=lbl, color=col, alpha=0.85)
    ax0.axhline(0, color="#555", lw=0.6, ls="--")
    ax0.set_ylabel("stress [MPa]", color="#e0e0e0", fontsize=9)
    ax0.set_title("Ply mid-surface stresses (material frame)", color="#e0e0e0", fontsize=9)
    ax0.legend(fontsize=8, loc="best")
    ax0.set_xticks(ply_idx + w)
    ax0.set_xticklabels(angles, fontsize=7)

    for j, (lbl, col) in enumerate(eps_cfg):
        ax1.bar(ply_idx + j * w, eps_arr[:, j], w, label=lbl, color=col, alpha=0.85)
    ax1.axhline(0, color="#555", lw=0.6, ls="--")
    ax1.set_ylabel("strain [—]", color="#e0e0e0", fontsize=9)
    ax1.set_title("Ply mid-surface strains (material frame)", color="#e0e0e0", fontsize=9)
    ax1.legend(fontsize=8, loc="best")
    ax1.set_xticks(ply_idx + w)
    ax1.set_xticklabels(angles, fontsize=7)

    fi_cols = ["#e74c3c" if f >= 1.0 else "#f39c12" if f >= 0.7 else "#2ecc71" for f in fi_arr]
    ax2.bar(ply_idx, fi_arr, color=fi_cols, alpha=0.9, zorder=3)
    ax2.axhline(1.0, color="#e74c3c", lw=1.2, ls="--", label="FI = 1 (failure)", zorder=4)
    ax2.set_ylabel("Hashin FI [—]", color="#e0e0e0", fontsize=9)
    ax2.set_title("Hashin failure index per ply", color="#e0e0e0", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.set_xticks(ply_idx)
    ax2.set_xticklabels(angles, fontsize=7)

    rf_cols = ["#e74c3c" if f < 1.0 else "#f39c12" if f < 2.0 else "#2ecc71" for f in rf_arr]
    ax3.bar(ply_idx, rf_arr, color=rf_cols, alpha=0.9, zorder=3)
    ax3.axhline(1.0, color="#e74c3c", lw=1.2, ls="--", label="RF = 1 (failure threshold)", zorder=4)
    ax3.set_ylabel("RF = 1/FI [—]", color="#e0e0e0", fontsize=9)
    ax3.set_title("Strength reserve factor per ply (clipped at 20)", color="#e0e0e0", fontsize=9)
    ax3.set_xlabel("ply (orientation)", color="#e0e0e0", fontsize=9)
    ax3.legend(fontsize=8)
    ax3.set_xticks(ply_idx)
    ax3.set_xticklabels(angles, fontsize=7)

    for ax in axes:
        ax.set_facecolor("#0f1117")
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        ax.grid(True, color="#2a2a3a", lw=0.4, alpha=0.5, axis="y")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a3a")

    plt.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return outfile


# ---------------------------------------------------------------------------
# Vlasov warping: all-panels σ_ω and q_ω montage
# ---------------------------------------------------------------------------

def save_vlasov_all_panels_figure(
    outfile: Path | str,
    vlasov_result: Any,
    panel_labels: list[str] | None = None,
    *,
    dpi: int = 150,
) -> Path:
    """
    Montage of Vlasov warping stress σ_ω [MPa] and secondary shear flow q_ω [N/m]
    for every panel. Web panels (zero values) are drawn but marked.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    s_mids = getattr(vlasov_result, "panel_s_mids", None) or []
    sigma_panels = getattr(vlasov_result, "sigma_omega", None) or []
    q_panels = getattr(vlasov_result, "q_omega", None) or []
    web_idx = set(getattr(vlasov_result, "web_panel_indices", []))
    n = len(s_mids)
    if n == 0:
        return outfile

    labels = list(panel_labels or [f"panel_{i}" for i in range(n)])

    plt.style.use("dark_background")
    fig, axes = plt.subplots(n, 2, figsize=(11.0, max(3.0, 2.2 * n)),
                             sharex=False, facecolor="#0f1117")
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(
        f"Vlasov warping — σ_ω (left) and q_ω (right) per panel\n"
        f"I_ω_E = {float(getattr(vlasov_result, 'I_omega_E', 0.0)):.3e} Pa·m⁶",
        color="#e0e0e0", fontsize=11,
    )
    if n == 1:
        axes = np.array([axes])

    for pi in range(n):
        ax_sig, ax_q = axes[pi, 0], axes[pi, 1]
        s = np.asarray(s_mids[pi] if pi < len(s_mids) else [], dtype=float)
        sig = np.asarray(sigma_panels[pi] if pi < len(sigma_panels) else [], dtype=float)
        q = np.asarray(q_panels[pi] if pi < len(q_panels) else [], dtype=float)
        lbl = labels[pi] if pi < len(labels) else f"panel_{pi}"
        is_web = pi in web_idx

        for ax in (ax_sig, ax_q):
            ax.set_facecolor("#0f1117")
            ax.tick_params(colors="#aaaaaa", labelsize=7)
            ax.grid(True, color="#2a2a3a", lw=0.4, alpha=0.5)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a2a3a")

        if s.size > 0 and sig.size > 0:
            ax_sig.plot(s[:len(sig)], sig[:len(s)] / 1e6, color="#e67e22", lw=1.3)
        ax_sig.axhline(0, color="#555", lw=0.5, ls="--")
        ax_sig.set_ylabel("σ_ω [MPa]", color="#aaaaaa", fontsize=7)
        ax_sig.set_title(f"{lbl}{'  [web]' if is_web else ''}", color="#e0e0e0", fontsize=8)

        if s.size > 0 and q.size > 0:
            ax_q.plot(s[:len(q)], q[:len(s)], color="#3498db", lw=1.3)
        ax_q.axhline(0, color="#555", lw=0.5, ls="--")
        ax_q.set_ylabel("q_ω [N/m]", color="#aaaaaa", fontsize=7)

        if pi == n - 1:
            ax_sig.set_xlabel("s [m]", color="#aaaaaa", fontsize=7)
            ax_q.set_xlabel("s [m]", color="#aaaaaa", fontsize=7)

    plt.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return outfile


# ---------------------------------------------------------------------------
# Vlasov warping stress on section geometry
# ---------------------------------------------------------------------------

def save_warping_stress_on_section_geometry(
    outfile: Path | str,
    airfoil: np.ndarray,
    webs_geom: list,
    spar_positions: list[float],
    panels: list[Any],
    vlasov_result: Any,
    *,
    dpi: int = 150,
    figsize: tuple[float, float] = (10.0, 7.0),
) -> Path:
    """
    Map Vlasov warping stress σ_ω [MPa] per panel element onto section geometry.
    Uses a diverging colourmap centred at zero (red = compression, blue = tension).
    """
    sigma_panels = getattr(vlasov_result, "sigma_omega", None) or []
    if not sigma_panels:
        return Path(outfile)

    per_panel: list[np.ndarray] = []
    for pi, p in enumerate(panels):
        if pi < len(sigma_panels):
            per_panel.append(np.asarray(sigma_panels[pi], dtype=float) / 1e6)
        else:
            per_panel.append(np.array([]))

    all_vals = np.concatenate([v for v in per_panel if v.size] or [np.array([])])
    v_abs = max(float(np.nanmax(np.abs(all_vals))) if all_vals.size else 1.0, 1e-30)

    return _scalar_field_on_section_geometry(
        outfile, airfoil, webs_geom, spar_positions, panels, per_panel,
        title="Vlasov warping stress σ_ω on section geometry [MPa]",
        cbar_label="σ_ω [MPa]",
        cmap="RdBu_r", vmin=-v_abs, vmax=v_abs, diverging=True,
        dpi=dpi, figsize=figsize,
    )


# ---------------------------------------------------------------------------
# MITC4 all-panels resultants montage
# ---------------------------------------------------------------------------

def save_mitc4_all_panels_resultants_montage(
    outfile: Path | str,
    all_panel_results: list[list[Any]],
    panel_labels: list[str] | None = None,
    *,
    dpi: int = 150,
) -> Path:
    """
    Grid figure: n_panels rows × 6 columns, one subplot per (panel, resultant).
    Extends ``save_mitc4_resultants_figure`` (reference panel only) to all panels.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    non_empty = [(pi, res) for pi, res in enumerate(all_panel_results) if res]
    if not non_empty:
        return outfile

    n = len(non_empty)
    labels = panel_labels or [f"panel_{pi}" for pi, _ in non_empty]

    field_cfg = [
        ("Nx",  "#3498db", "N/m"),
        ("Ny",  "#2ecc71", "N/m"),
        ("Nxy", "#e74c3c", "N/m"),
        ("Mx",  "#9b59b6", "N·m/m"),
        ("My",  "#f39c12", "N·m/m"),
        ("Mxy", "#1abc9c", "N·m/m"),
        ("Qx",  "#e67e22", "N/m"),
        ("Qy",  "#c0392b", "N/m"),
    ]

    plt.style.use("dark_background")
    row_h = 1.8
    n_cols = len(field_cfg)
    fig, axes = plt.subplots(n, n_cols, figsize=(max(15.0, 2.0 * n_cols), max(3.0, row_h * n)),
                             facecolor="#0f1117", squeeze=False)
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("MITC4 shell resultants — all panels", color="#e0e0e0", fontsize=11)

    for row, (pi, res) in enumerate(non_empty):
        lbl = labels[pi] if pi < len(labels) else f"panel_{pi}"
        indices = [r.station_index for r in res]
        for col, (fname, col_color, unit) in enumerate(field_cfg):
            ax = axes[row, col]
            ax.set_facecolor("#0f1117")
            vals = [float(getattr(r, fname) if getattr(r, fname) is not None else 0.0) for r in res]
            ax.plot(indices, vals, color=col_color, lw=1.2, marker=".", markersize=2)
            ax.axhline(0, color="#555", lw=0.5, ls="--")
            ax.tick_params(colors="#aaaaaa", labelsize=6)
            ax.grid(True, color="#2a2a3a", lw=0.3, alpha=0.4)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a2a3a")
            if row == 0:
                ax.set_title(f"{fname} [{unit}]", color="#e0e0e0", fontsize=8)
            if col == 0:
                ax.set_ylabel(lbl[:20], color="#aaaaaa", fontsize=7, rotation=45,
                              ha="right", va="center", labelpad=2)
            if row == n - 1:
                ax.set_xlabel("elem", color="#aaaaaa", fontsize=6)

    plt.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return outfile


# ---------------------------------------------------------------------------
# Panel stiffness properties (ABD diagonals, thickness, G_eff)
# ---------------------------------------------------------------------------

def save_panel_stiffness_properties_figure(
    outfile: Path | str,
    panel_labels: list[str],
    abd_matrices: Any,
    thickness_m: Any,
    g_eff: Any,
    *,
    dpi: int = 150,
) -> Path:
    """
    Three-panel figure showing per-panel laminate stiffness properties:

    * Left:   bar chart of panel thickness [mm]
    * Centre: bar chart of effective in-plane shear modulus G_eff [GPa]
    * Right:  heat grid of ABD diagonal terms [A11, A22, A66, D11, D22, D66]
              (rows = panels, cols = diagonal stiffness terms)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    n = len(panel_labels)
    if n == 0:
        return outfile

    t_arr = np.asarray(thickness_m, dtype=float).ravel()[:n] * 1e3 if thickness_m is not None else np.zeros(n)
    g_arr = np.asarray(g_eff, dtype=float).ravel()[:n] / 1e9 if g_eff is not None else np.zeros(n)

    abd_diag: np.ndarray | None = None
    abd_labels = ["A₁₁", "A₂₂", "A₆₆", "D₁₁", "D₂₂", "D₆₆"]
    if abd_matrices is not None:
        abd = np.asarray(abd_matrices, dtype=float)
        if abd.ndim == 3 and abd.shape[1] == 6 and abd.shape[2] == 6:
            abd_diag = abd[:n, [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]  # (n, 6)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(14.0, max(3.5, 0.4 * n + 2.0)),
                             facecolor="#0f1117",
                             gridspec_kw={"width_ratios": [1, 1, 1.6]})
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Panel laminate stiffness properties", color="#e0e0e0", fontsize=11)

    idx = np.arange(n)
    short_labels = [lb[:18] for lb in panel_labels]
    ax0, ax1, ax2 = axes
    for ax in axes:
        ax.set_facecolor("#0f1117")
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a3a")

    ax0.barh(idx, t_arr, color="#3498db", alpha=0.85)
    ax0.set_yticks(idx)
    ax0.set_yticklabels(short_labels, fontsize=7)
    ax0.set_xlabel("thickness [mm]", color="#e0e0e0", fontsize=9)
    ax0.set_title("Panel thickness", color="#e0e0e0", fontsize=9)
    ax0.grid(True, color="#2a2a3a", lw=0.4, alpha=0.5, axis="x")
    ax0.invert_yaxis()

    ax1.barh(idx, g_arr, color="#2ecc71", alpha=0.85)
    ax1.set_yticks(idx)
    ax1.set_yticklabels(short_labels, fontsize=7)
    ax1.set_xlabel("G_eff [GPa]", color="#e0e0e0", fontsize=9)
    ax1.set_title("Effective shear modulus", color="#e0e0e0", fontsize=9)
    ax1.grid(True, color="#2a2a3a", lw=0.4, alpha=0.5, axis="x")
    ax1.invert_yaxis()

    if abd_diag is not None and abd_diag.size:
        nonzero = abd_diag[abd_diag != 0.0]
        vmin_log = max(float(np.nanmin(np.abs(nonzero))), 1e-6) if nonzero.size else 1e-6
        vmax_log = float(np.nanmax(np.abs(abd_diag))) if abd_diag.size else 1.0
        im = ax2.imshow(abd_diag, aspect="auto", cmap="plasma",
                        norm=mcolors.LogNorm(vmin=vmin_log, vmax=max(vmax_log, vmin_log * 10)))
        cb = fig.colorbar(im, ax=ax2, fraction=0.05, pad=0.02)
        cb.set_label("stiffness [log scale]", color="#e0e0e0", fontsize=8)
        cb.ax.tick_params(colors="#aaaaaa", labelsize=7)
        ax2.set_xticks(range(6))
        ax2.set_xticklabels(abd_labels, fontsize=8, color="#e0e0e0")
        ax2.set_yticks(range(n))
        ax2.set_yticklabels(short_labels, fontsize=7, color="#aaaaaa")
        ax2.set_title("ABD diagonal stiffness terms", color="#e0e0e0", fontsize=9)
    else:
        ax2.text(0.5, 0.5, "ABD data not available", ha="center", va="center",
                 transform=ax2.transAxes, color="#aaaaaa", fontsize=9)
        ax2.axis("off")

    plt.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return outfile


# ---------------------------------------------------------------------------
# Panel buckling: critical load factor bar chart + modeshape plots
# ---------------------------------------------------------------------------

def save_panel_buckling_figure(
    outfile: Path | str,
    panel_buckling_results: list,
    *,
    lambda_alert: float = 1.0,
    lambda_warn: float = 2.0,
    dpi: int = 150,
) -> Path:
    """
    Two-panel figure summarising per-panel linear buckling results.

    Left: horizontal bar chart of λ₁ (lowest critical load factor) per panel.
    Right: w-modeshape (mode 1) along free-DOF index for all panels overlaid.

    Colour coding:
      λ₁ < lambda_alert  → red
      lambda_alert ≤ λ₁ < lambda_warn → amber
      λ₁ ≥ lambda_warn   → green
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    if not panel_buckling_results:
        return outfile

    panel_labels = [getattr(br, "panel_label", f"panel_{i}") for i, br in enumerate(panel_buckling_results)]
    lambdas_1 = []
    for br in panel_buckling_results:
        lam = getattr(br, "lambdas", None)
        if lam is not None and len(lam) > 0:
            lv = float(lam[0])
            lambdas_1.append(lv if np.isfinite(lv) else float("inf"))
        else:
            lambdas_1.append(float("inf"))

    n = len(lambdas_1)
    colors = []
    for lv in lambdas_1:
        if not np.isfinite(lv) or lv >= float(lambda_warn):
            colors.append("#2ecc71")
        elif lv >= float(lambda_alert):
            colors.append("#f39c12")
        else:
            colors.append("#e74c3c")

    plt.style.use("dark_background")
    fig, (ax_bar, ax_mode) = plt.subplots(1, 2, figsize=(12.0, max(3.5, 0.45 * n + 2.0)),
                                           facecolor="#0f1117")
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Panel linear buckling — critical load factors", color="#e0e0e0", fontsize=11)

    y_pos = np.arange(n)
    plot_lam = [min(lv, 10.0) if np.isfinite(lv) else 10.0 for lv in lambdas_1]
    ax_bar.barh(y_pos, plot_lam, color=colors, alpha=0.85, height=0.6)
    ax_bar.axvline(float(lambda_alert), color="#e74c3c", lw=1.2, ls="--", label=f"λ={lambda_alert:.2g}")
    ax_bar.axvline(float(lambda_warn), color="#f39c12", lw=0.9, ls=":", label=f"λ={lambda_warn:.2g}")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([lb[:20] for lb in panel_labels], fontsize=8, color="#e0e0e0")
    ax_bar.set_xlabel("λ₁ = N_crit / N_applied  [capped at 10]", color="#e0e0e0", fontsize=9)
    ax_bar.set_title("Mode-1 critical load factor", color="#e0e0e0", fontsize=9)
    ax_bar.legend(fontsize=7, loc="lower right")
    ax_bar.grid(True, axis="x", color="#2a2a3a", lw=0.4, alpha=0.5)
    ax_bar.invert_yaxis()
    ax_bar.set_facecolor("#0f1117")
    ax_bar.tick_params(colors="#aaaaaa", labelsize=7)

    import matplotlib.cm as cm_base
    cmap_m = cm_base.get_cmap("tab20")
    ax_mode.set_facecolor("#0f1117")
    ax_mode.set_title("Mode-1 w-modeshape (free DOFs)", color="#e0e0e0", fontsize=9)
    ax_mode.set_xlabel("free DOF index", color="#e0e0e0", fontsize=8)
    ax_mode.set_ylabel("amplitude [normalised]", color="#e0e0e0", fontsize=8)
    ax_mode.tick_params(colors="#aaaaaa", labelsize=7)
    ax_mode.grid(True, color="#2a2a3a", lw=0.3, alpha=0.4)

    for pi, br in enumerate(panel_buckling_results):
        ms = getattr(br, "modeshapes", None)
        if ms is None or len(ms) == 0 or ms.shape[1] == 0:
            continue
        mode0 = np.asarray(ms[0], dtype=float)
        if not np.all(np.isfinite(mode0)):
            continue
        norm = float(np.max(np.abs(mode0))) or 1.0
        col = cmap_m((pi % 20) / 19.0)
        ax_mode.plot(mode0 / norm, lw=1.2, alpha=0.85,
                     label=panel_labels[pi][:16], color=col)

    if n <= 12:
        ax_mode.legend(fontsize=6, loc="best")

    plt.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return outfile
