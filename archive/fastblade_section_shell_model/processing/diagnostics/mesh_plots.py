"""Section centerline mesh figures (taxonomy 2x2: subcomponent / strip / panel / element)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np


def _junction_member_kinds(mesh: Any, c: Any) -> set[str]:
    return {str(mesh.panel_meshes[int(pi)].kind) for pi, _ in c.members}


def _junction_display_label(mesh: Any, c: Any) -> str:
    """Architectural junction class for legend/title (not raw endpoint count).

    Web feet are always labelled ``skin–web`` when both kinds are present, even if a
    spar-cap endpoint co-locates (D-family T-junction). That matches the topology
    count ``2 * n_webs`` skin–shear-web junctions.
    """
    kinds = _junction_member_kinds(mesh, c)
    if "skin" in kinds and "web" in kinds:
        return "skin–web"
    if kinds == {"skin", "cap"}:
        return "skin–cap"
    if len(kinds) == 1:
        return next(iter(kinds))
    return "+".join(sorted(kinds))


def _tab20_colour(i: int) -> tuple[float, float, float, float]:
    """Stable tab20 colour for constituent index ``i`` (cycles every 20)."""
    from matplotlib import colormaps

    cmap = colormaps["tab20"]
    return tuple(cmap((int(i) % 20) / 19.0))


def _element_colour(ei: int) -> tuple[float, float, float, float]:
    """Distinct colour per element index (golden-ratio hue walk; no legend)."""
    from matplotlib.colors import hsv_to_rgb

    # Adjacent contour elements get well-separated hues.
    h = (float(ei) * 0.6180339887498949) % 1.0
    rgb = hsv_to_rgb((h, 0.75, 0.95))
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)


def _short_strip_label(strip: Any) -> str:
    kind = str(getattr(strip, "kind", "") or "")
    label = str(getattr(strip, "label", "") or "")
    if kind and label:
        return f"{kind}:{label}"
    return label or kind or "?"


def _short_panel_label(pm: Any) -> str:
    return str(getattr(pm, "panel_label", None) or getattr(pm, "kind", f"P{pm.panel_index}"))


def _constituent_colour_map(
    shell_inputs: Any | None = None,
    mesh: Any | None = None,
) -> dict[str, tuple[float, float, float, float]]:
    """Map ``kind:label`` → tab20 colour; strip order first, then any extra panels."""
    colours: dict[str, tuple[float, float, float, float]] = {}
    i = 0
    if shell_inputs is not None:
        for strip in getattr(shell_inputs, "midlines", []) or []:
            key = _short_strip_label(strip)
            if key not in colours:
                colours[key] = _tab20_colour(i)
                i += 1
    if mesh is not None:
        for pm in getattr(mesh, "panel_meshes", []) or []:
            key = _short_panel_label(pm)
            if key not in colours:
                colours[key] = _tab20_colour(i)
                i += 1
    return colours

def _midpoint_text(ax: Any, pts: np.ndarray, text: str, *, color: Any) -> None:
    if pts.shape[0] < 1:
        return
    mid = pts[pts.shape[0] // 2]
    ax.text(
        float(mid[0]),
        float(mid[1]),
        f" {text}",
        fontsize=5.5,
        color=color,
        alpha=0.95,
        zorder=12,
        clip_on=True,
    )


def _add_skin_inward_rays_class_c_near_caps(
    ax_mesh: Any,
    shell_inputs: Any,
    *,
    chord_m: float,
) -> None:
    """Draw Class C ray directions used by cap interior resampling."""
    from blade_precompute.section_shell_model.pre_processing.mesh.shell_inputs_from_section import (
        _SKIN_RAY_CAP_BBOX_MARGIN_M,
        _concat_skin_ring_vertices,
        _normal_2d_ccw,
        _pick_inward_ray_direction,
        _point_to_open_polyline_nearest,
        _skin_ring_vertex_tangents,
    )

    strips = shell_inputs.midlines
    cap_polys: list[Any] = []
    cap_centroids: list[Any] = []
    for st in strips:
        if st.kind != "cap":
            continue
        pts = np.asarray(st.midline_b, dtype=float)
        if pts.shape[0] < 2:
            continue
        cap_polys.append(pts)
        cap_centroids.append(np.mean(pts, axis=0).reshape(2))
    if not cap_polys:
        return

    skin_ring = _concat_skin_ring_vertices(strips)
    if skin_ring.shape[0] < 2:
        return
    tans = _skin_ring_vertex_tangents(skin_ring)
    mrg = float(_SKIN_RAY_CAP_BBOX_MARGIN_M)
    arrow_m = max(0.008, 0.012 * float(chord_m))

    xs: list[float] = []
    ys: list[float] = []
    us: list[float] = []
    vs: list[float] = []
    for i in range(skin_ring.shape[0]):
        q = skin_ring[i]
        best_d = float("inf")
        best_cmid = cap_centroids[0]
        found = False
        for cpts, c_mid in zip(cap_polys, cap_centroids, strict=True):
            xmin, xmax = float(cpts[:, 0].min()), float(cpts[:, 0].max())
            ymin, ymax = float(cpts[:, 1].min()), float(cpts[:, 1].max())
            if (
                q[0] < xmin - mrg
                or q[0] > xmax + mrg
                or q[1] < ymin - mrg
                or q[1] > ymax + mrg
            ):
                continue
            d, _ = _point_to_open_polyline_nearest(q, cpts)
            if d < best_d:
                best_d = float(d)
                best_cmid = c_mid
                found = True
        if not found:
            continue
        n_ccw = _normal_2d_ccw(tans[i])
        v = _pick_inward_ray_direction(q, n_ccw, best_cmid)
        xs.append(float(q[0]))
        ys.append(float(q[1]))
        us.append(float(v[0] * arrow_m))
        vs.append(float(v[1] * arrow_m))

    if not xs:
        return
    ax_mesh.quiver(
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(us, dtype=float),
        np.asarray(vs, dtype=float),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0024,
        color="#66ffcc",
        alpha=0.9,
        zorder=16,
        label="Class C ray dir (perp tangent, in cap bbox)",
    )


def taxonomy_plot_caption(layout_key: str, layout: Any) -> str:
    y = str(layout.structural_family)
    y_desc = {
        "A": "no spar caps",
        "B": "single cap band (pitch / max-t)",
        "C": "discrete caps per web",
        "D": "continuous box (first web to last web)",
    }.get(y, y)
    z = (
        "flapwise (B-frame vertical webs)"
        if layout.web_orientation == "flapwise"
        else "chord-normal (S-frame vertical webs)"
    )
    parts = [
        f"X = {layout.n_webs} web(s)",
        f"Y = {y} ({y_desc})",
        f"Z = {z}",
    ]
    return " | ".join(parts)


def draw_midline_strips_on_ax(
    ax: Any,
    shell_inputs: Any,
    *,
    le_b: tuple[float, float],
    te_b: tuple[float, float],
    title: str = "Strips (ShellMidlineStrip)",
    colour_by_label: dict[str, tuple[float, float, float, float]] | None = None,
) -> None:
    """Pre-mesh midlines + geometry sample vertices; no FE element edges."""
    midlines = list(getattr(shell_inputs, "midlines", []) or [])
    if colour_by_label is None:
        colour_by_label = _constituent_colour_map(shell_inputs=shell_inputs)
    for i, strip in enumerate(midlines):
        pts = np.asarray(strip.midline_b, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
            continue
        lbl = _short_strip_label(strip)
        col = colour_by_label.get(lbl, _tab20_colour(i))
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=col,
            lw=2.0,
            alpha=0.95,
            zorder=4,
            solid_capstyle="round",
            label=lbl if i < 20 else None,
        )
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            color=col,
            s=10,
            alpha=0.9,
            zorder=8,
            edgecolors="white",
            linewidths=0.3,
        )
        _midpoint_text(ax, pts, lbl, color=col)

    ax.scatter(
        [le_b[0], te_b[0]],
        [le_b[1], te_b[1]],
        color="#ffffff",
        s=60,
        marker="x",
        zorder=30,
        label="LE / TE",
    )
    ax.set_title(
        f"{title}\n{len(midlines)} strips · geometry samples (pre-mesh)",
        color="#e0e0e0",
        fontsize=10,
    )
    ax.legend(
        loc="lower right",
        fontsize=5.5,
        facecolor="#1a1a2e",
        edgecolor="#2a2a3a",
        labelcolor="#e0e0e0",
        ncol=2,
    )


def draw_panel_patches_on_ax(
    ax: Any,
    mesh: Any,
    *,
    le_b: tuple[float, float],
    te_b: tuple[float, float],
    title: str = "Panels (constitutive patches)",
    colour_by_label: dict[str, tuple[float, float, float, float]] | None = None,
) -> None:
    """Discretised panel contour as one thick patch + nodes; no junctions / rays."""
    summ = mesh.summary()
    if colour_by_label is None:
        colour_by_label = _constituent_colour_map(mesh=mesh)
    for pm in mesh.panel_meshes:
        yz = np.asarray(pm.yz_nodes, dtype=float)
        if yz.size == 0 or yz.ndim != 2 or yz.shape[0] < 2:
            continue
        lbl = _short_panel_label(pm)
        col = colour_by_label.get(lbl, _tab20_colour(int(pm.panel_index)))
        ax.plot(
            yz[:, 0],
            yz[:, 1],
            color=col,
            lw=3.0,
            alpha=0.95,
            zorder=4,
            solid_capstyle="round",
            label=lbl if int(pm.panel_index) < 20 else None,
        )
        ax.scatter(
            yz[:, 0],
            yz[:, 1],
            color=col,
            s=12,
            alpha=0.95,
            zorder=8,
            edgecolors="white",
            linewidths=0.35,
        )
        _midpoint_text(ax, yz, lbl, color=col)

    ax.scatter(
        [le_b[0], te_b[0]],
        [le_b[1], te_b[1]],
        color="#ffffff",
        s=60,
        marker="x",
        zorder=30,
        label="LE / TE",
    )
    ax.set_title(
        f"{title}\n{summ['n_panels']} panels · {summ['n_total_nodes']} nodes "
        f"(continuous patch, no element edges)",
        color="#e0e0e0",
        fontsize=10,
    )
    ax.legend(
        loc="lower right",
        fontsize=5.5,
        facecolor="#1a1a2e",
        edgecolor="#2a2a3a",
        labelcolor="#e0e0e0",
        ncol=2,
    )


def draw_centerline_mesh_on_ax(
    ax_mesh: Any,
    mesh: Any,
    *,
    le_b: tuple[float, float],
    te_b: tuple[float, float],
    kind_colour: dict[str, str] | None = None,
    title: str,
    shell_inputs: Any | None = None,
    chord_m: float | None = None,
    colour_mode: Literal["kind", "panel", "element"] = "kind",
    colour_by_label: dict[str, tuple[float, float, float, float]] | None = None,
) -> None:
    """Plot strip mesh midlines, element edges, nodes, and junction clusters.

    ``colour_mode``:
      - ``kind`` — colour by skin/web/cap
      - ``panel`` — colour by constitutive panel label (tab20)
      - ``element`` — unique colour per FE contour element; no per-element legend
    """
    if kind_colour is None:
        kind_colour = {"skin": "#7eb8da", "web": "#e74c3c", "cap": "#f1c40f"}
    if colour_mode == "panel" and colour_by_label is None:
        colour_by_label = _constituent_colour_map(shell_inputs=shell_inputs, mesh=mesh)

    def _colour_for_panel(pm: Any) -> Any:
        if colour_mode == "panel":
            lbl = _short_panel_label(pm)
            if colour_by_label is not None and lbl in colour_by_label:
                return colour_by_label[lbl]
            return _tab20_colour(int(pm.panel_index))
        return kind_colour.get(pm.kind, "#bbbbbb")

    # Midline underlay: skip for element mode (edges carry the colour story).
    if colour_mode != "element":
        drawn_keys: set[str] = set()
        for pm in mesh.panel_meshes:
            if pm.yz_nodes.size == 0:
                continue
            col = _colour_for_panel(pm)
            if colour_mode == "panel":
                key = f"panel:{_short_panel_label(pm)}"
                legend = _short_panel_label(pm) if key not in drawn_keys else None
            else:
                key = f"kind:{pm.kind}"
                legend = f"{pm.kind} midline" if key not in drawn_keys else None
            if legend is not None:
                drawn_keys.add(key)
            ax_mesh.plot(
                pm.yz_nodes[:, 0],
                pm.yz_nodes[:, 1],
                color=col,
                lw=1.4,
                alpha=0.55,
                zorder=3,
                label=legend,
            )

    elem_drawn: set[str] = set()
    elem_global = 0
    element_legend_set = False
    for pm in mesh.panel_meshes:
        if pm.yz_nodes.size == 0:
            continue
        panel_col = _colour_for_panel(pm)
        for i in range(pm.yz_nodes.shape[0] - 1):
            x0, y0 = pm.yz_nodes[i]
            x1, y1 = pm.yz_nodes[i + 1]
            legend = None
            if colour_mode == "element":
                col = _element_colour(elem_global)
                if not element_legend_set:
                    legend = "element (unique colour)"
                    element_legend_set = True
                elem_global += 1
            elif colour_mode == "panel":
                col = panel_col
            else:
                col = panel_col
                tag = f"{pm.kind}_elem"
                if tag not in elem_drawn:
                    elem_drawn.add(tag)
                    legend = f"{pm.kind} element"
            ax_mesh.plot(
                [x0, x1],
                [y0, y1],
                color=col,
                lw=2.4,
                alpha=0.95,
                zorder=5,
                solid_capstyle="round",
                label=legend,
            )
        node_col = "#ecf0f1" if colour_mode == "element" else panel_col
        ax_mesh.scatter(
            pm.yz_nodes[:, 0],
            pm.yz_nodes[:, 1],
            color=node_col,
            s=10 if colour_mode == "element" else 14,
            alpha=0.85 if colour_mode == "element" else 0.95,
            zorder=8,
            edgecolors="white",
            linewidths=0.35,
        )

    junction_colour_by_label = {
        "skin–web": "#f39c12",
        "skin–cap": "#2ecc71",
        "skin": "#7eb8da",
        "web": "#e74c3c",
        "cap": "#f1c40f",
    }
    cluster_drawn: set[str] = set()
    n_drawn_junctions = 0
    for c in mesh.clusters:
        if len(c.members) < 2:
            continue
        jlabel = _junction_display_label(mesh, c)
        n_drawn_junctions += 1
        col = junction_colour_by_label.get(jlabel, "#9b59b6")
        legend = None
        if jlabel not in cluster_drawn:
            cluster_drawn.add(jlabel)
            legend = f"{jlabel} junction"
        ax_mesh.scatter(
            [c.point_yz[0]],
            [c.point_yz[1]],
            facecolors="none",
            edgecolors=col,
            s=180,
            lw=1.6,
            zorder=20,
            label=legend,
        )

    ax_mesh.scatter(
        [le_b[0], te_b[0]],
        [le_b[1], te_b[1]],
        color="#ffffff",
        s=60,
        marker="x",
        zorder=30,
        label="LE / TE",
    )

    if shell_inputs is not None and chord_m is not None and chord_m > 0.0:
        _add_skin_inward_rays_class_c_near_caps(ax_mesh, shell_inputs, chord_m=chord_m)

    summ = mesh.summary()
    junction_class_counts: Counter[str] = Counter()
    for cl in mesh.clusters:
        if len(cl.members) < 2:
            continue
        junction_class_counts[_junction_display_label(mesh, cl)] += 1
    colour_note = ""
    if colour_mode == "element":
        colour_note = " · per-element colour (no element labels)"
    class_txt = ", ".join(f"{k}×{v}" for k, v in sorted(junction_class_counts.items()))
    ax_mesh.set_title(
        f"{title}\n{summ['n_panels']} panels · {summ['n_total_elements']} elements · "
        f"{summ['n_total_nodes']} nodes · {n_drawn_junctions} junctions"
        + (f" ({class_txt})" if class_txt else "")
        + colour_note,
        color="#e0e0e0",
        fontsize=10,
    )
    ax_mesh.legend(
        loc="lower right",
        fontsize=7 if colour_mode == "element" else (5.5 if colour_mode == "panel" else 7),
        facecolor="#1a1a2e",
        edgecolor="#2a2a3a",
        labelcolor="#e0e0e0",
        ncol=2,
    )


def _draw_section_sdf_panel(ax_geom: Any, section: Any, airfoil: Any, twist_rad: float) -> None:
    from matplotlib.patches import Patch

    kind_colour = {"skin": "#7eb8da", "web": "#e74c3c", "cap": "#f1c40f"}
    kind_alpha = {"skin": 0.30, "web": 0.55, "cap": 0.55}

    airfoil_b = airfoil.rotate(twist_rad) if abs(twist_rad) > 1e-10 else airfoil
    verts_b = np.asarray(airfoil_b.vertices, dtype=float)
    pad = 0.05 * float(max(np.ptp(verts_b[:, 0]), np.ptp(verts_b[:, 1])))
    x_min = float(verts_b[:, 0].min()) - pad
    x_max = float(verts_b[:, 0].max()) + pad
    y_min = float(verts_b[:, 1].min()) - pad
    y_max = float(verts_b[:, 1].max()) + pad
    gx = np.linspace(x_min, x_max, 600)
    gy = np.linspace(y_min, y_max, 360)
    GX, GY = np.meshgrid(gx, gy)

    label_to_kind: dict[str, str] = {}
    for label in section.labels:
        s = str(label).lower()
        if "skin" in s:
            label_to_kind[label] = "skin"
        elif "web" in s:
            label_to_kind[label] = "web"
        elif "cap" in s:
            label_to_kind[label] = "cap"
    n_sub = len(list(getattr(section, "labels", []) or []))
    for label, kind in label_to_kind.items():
        try:
            phi = np.asarray(section[label](GX, GY), dtype=float)
        except Exception:
            continue
        col = kind_colour[kind]
        ax_geom.contourf(
            GX, GY, phi, levels=[-1e9, 0.0], colors=[col], alpha=kind_alpha[kind], zorder=1,
        )
        ax_geom.contour(
            GX, GY, phi, levels=[0.0], colors=[col], linewidths=0.8, alpha=0.9, zorder=2,
        )

    ax_geom.legend(
        handles=[
            Patch(facecolor=kind_colour["skin"], alpha=kind_alpha["skin"], label="skin SDF"),
            Patch(facecolor=kind_colour["cap"], alpha=kind_alpha["cap"], label="cap SDF"),
            Patch(facecolor=kind_colour["web"], alpha=kind_alpha["web"], label="web SDF"),
        ],
        loc="lower right",
        fontsize=8,
        facecolor="#1a1a2e",
        edgecolor="#2a2a3a",
        labelcolor="#e0e0e0",
        ncol=1,
    )
    ax_geom.set_title(
        f"Subcomponents (SDF)\n{n_sub} subcomponents · coloured by kind",
        color="#e0e0e0",
        fontsize=10,
    )


def save_section_centerline_mesh_figure(
    outfile: Path | str,
    *,
    mesh: Any,
    section: Any,
    shell_inputs: Any,
    layout: Any,
    layout_key: str,
    airfoil: Any,
    twist_rad: float,
    twist_deg: float,
    target_element_length_m: float,
    station_tag: str = "",
    dpi: int = 160,
) -> Path:
    """Save 2x2 taxonomy figure: subcomponent / strip / panel / element."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(18, 16), sharex=True, sharey=True)
    fig.patch.set_facecolor("#0f1117")
    ax_geom, ax_strips = axes[0, 0], axes[0, 1]
    ax_panels, ax_mesh = axes[1, 0], axes[1, 1]
    for _ax in (ax_geom, ax_strips, ax_panels, ax_mesh):
        _ax.set_facecolor("#0f1117")

    _draw_section_sdf_panel(ax_geom, section, airfoil, twist_rad)

    le_b = tuple(shell_inputs.leading_edge_b)
    te_b = tuple(shell_inputs.trailing_edge_b)
    ax_geom.scatter([le_b[0], te_b[0]], [le_b[1], te_b[1]], color="#ffffff", s=60, marker="x", zorder=30)

    colour_by_label = _constituent_colour_map(shell_inputs=shell_inputs, mesh=mesh)
    draw_midline_strips_on_ax(
        ax_strips,
        shell_inputs,
        le_b=le_b,
        te_b=te_b,
        title="Strips (ShellMidlineStrip)",
        colour_by_label=colour_by_label,
    )
    draw_panel_patches_on_ax(
        ax_panels,
        mesh,
        le_b=le_b,
        te_b=te_b,
        title="Panels (constitutive patches)",
        colour_by_label=colour_by_label,
    )

    mesh_title = f"Elements — h_target={target_element_length_m:.4g} m"
    draw_centerline_mesh_on_ax(
        ax_mesh,
        mesh,
        le_b=le_b,
        te_b=te_b,
        title=mesh_title,
        shell_inputs=shell_inputs,
        chord_m=float(shell_inputs.chord_m),
        colour_mode="element",
        colour_by_label=colour_by_label,
    )

    for _ax in (ax_geom, ax_strips, ax_panels, ax_mesh):
        _ax.set_xlabel("y_B [m]", color="#e0e0e0")
        _ax.set_ylabel("z_B [m]", color="#e0e0e0")
        _ax.tick_params(colors="#aaaaaa")
        _ax.set_aspect("equal")
        _ax.grid(True, color="#2a2a3a", lw=0.4, alpha=0.5)

    tax = taxonomy_plot_caption(layout_key, layout)
    summ = mesh.summary()
    n_sub = len(getattr(section, "labels", []) or [])
    n_strips = len(getattr(shell_inputs, "midlines", []) or [])
    tag_txt = f" · {station_tag}" if station_tag else ""
    fig.suptitle(
        f"{layout_key}{tag_txt} — taxonomy: Subcomponent → Strip → Panel → Element\n"
        f"{tax}\n"
        f"twist = {twist_deg:.2f}° · "
        f"n_subcomponents={n_sub} · n_strips={n_strips} · "
        f"n_panels={summ['n_panels']} · n_elements={summ['n_total_elements']}",
        color="#e0e0e0",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(outfile, dpi=dpi, facecolor="#0f1117", bbox_inches="tight")
    plt.close(fig)
    return outfile


__all__ = [
    "draw_centerline_mesh_on_ax",
    "draw_midline_strips_on_ax",
    "draw_panel_patches_on_ax",
    "save_section_centerline_mesh_figure",
    "taxonomy_plot_caption",
]
