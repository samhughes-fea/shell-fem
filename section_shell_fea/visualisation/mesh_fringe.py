"""Section-mesh fringe helpers for shell-FE / CG visualisations.

Nodal scatter / elemental face colouring in the canonical-B section plane
:math:`(X_B, Y_B)`. Primary deformed overlay + companion deformed-mesh PNGs live
here too.

All coordinates arriving here are canonical B and are remapped for display by
``visualisation.section_axes`` (horiz :math:`Y_B`, vert :math:`X_B`, LE on RHS).
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from numpy.typing import NDArray

from .section_axes import finish_section_axes, section_hv, section_rings_hv


def dump_plot_json(
    output_path: Path,
    payload: dict[str, Any],
    *,
    indent: int | None = None,
) -> Path:
    """Write companion JSON next to a PNG (same stem, ``.json``)."""
    out = Path(output_path)
    json_path = out.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=indent, default=str), encoding="utf-8")
    return json_path


class ElementalFringeGeometry:
    """Cached polygon/segment topology for repeated elemental fringe recolors."""

    __slots__ = ("polys", "segments", "drawn_elem_idx", "drawn_modes", "n_elems")

    def __init__(
        self,
        polys: list[NDArray[np.floating]],
        segments: list[NDArray[np.floating]],
        drawn_elem_idx: list[int],
        drawn_modes: list[str],
        n_elems: int,
    ):
        self.polys = polys
        self.segments = segments
        self.drawn_elem_idx = drawn_elem_idx
        self.drawn_modes = drawn_modes
        self.n_elems = int(n_elems)


def build_elemental_fringe_geometry(
    elements: Iterable[dict],
    *,
    coords_key: str = "node_coords_yz_m",
) -> ElementalFringeGeometry:
    """Extract drawable mid-surface polys/segments once per station mesh."""
    elems = list(elements)
    polys: list[NDArray[np.floating]] = []
    segments: list[NDArray[np.floating]] = []
    drawn_elem_idx: list[int] = []
    drawn_modes: list[str] = []
    for i, el in enumerate(elems):
        coords = el.get(coords_key) if isinstance(el, dict) else None
        if not coords:
            continue
        pts = []
        for c in coords:
            if c is None:
                continue
            arr = np.asarray(c, dtype=float).ravel()
            if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
                pts.append(arr[:2])
        if len(pts) < 2:
            continue
        uniq = _unique_contour_points(pts)
        area = _poly_area(np.asarray(pts, dtype=float)) if len(pts) >= 3 else 0.0
        if uniq.shape[0] >= 3 and area > 1e-16:
            polys.append(np.asarray(pts, dtype=float))
            drawn_modes.append("poly")
        elif uniq.shape[0] >= 2:
            segments.append(uniq[:2])
            drawn_modes.append("segment")
        else:
            continue
        drawn_elem_idx.append(i)
    return ElementalFringeGeometry(polys, segments, drawn_elem_idx, drawn_modes, len(elems))


def _poly_area(pts: NDArray[np.floating]) -> float:
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _unique_contour_points(
    pts: Sequence[Sequence[float]] | NDArray[np.floating],
    *,
    atol: float = 1e-12,
) -> NDArray[np.float64]:
    """Collapse coincident strip-quad corners to unique mid-surface (y, z) points."""
    arr = np.asarray(pts, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=float)
    unique: list[NDArray[np.floating]] = []
    for p in arr[:, :2]:
        if not np.all(np.isfinite(p)):
            continue
        if unique and np.allclose(p, unique[-1], atol=atol):
            continue
        if unique and np.allclose(p, unique[0], atol=atol):
            continue
        unique.append(p)
    if not unique:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(unique, dtype=float)


def _finite_scale(values: NDArray[np.floating], *, deform_scale: float | None) -> float:
    if deform_scale is not None and np.isfinite(deform_scale) and deform_scale > 0:
        return float(deform_scale)
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    peak = float(np.nanmax(np.abs(finite)))
    if peak <= 0.0:
        return 1.0
    # Scale so peak in-plane displacement is ~5% of section chord span.
    return 1.0


def auto_deform_scale(
    node_yz: NDArray[np.floating],
    us: NDArray[np.floating],
    w: NDArray[np.floating],
    *,
    fraction: float = 0.05,
) -> float:
    """Choose α so max in-plane |u| maps to ``fraction`` of section diagonal."""
    yz = np.asarray(node_yz, dtype=float)
    if yz.size == 0:
        return 1.0
    span = float(np.linalg.norm(np.nanmax(yz, axis=0) - np.nanmin(yz, axis=0)))
    if not np.isfinite(span) or span <= 0.0:
        span = 1.0
    u_in = np.sqrt(np.asarray(us, dtype=float) ** 2 + np.asarray(w, dtype=float) ** 2)
    peak = float(np.nanmax(u_in)) if u_in.size else 0.0
    if not np.isfinite(peak) or peak <= 0.0:
        return 1.0
    return float(fraction * span / peak)


def save_nodal_fringe(
    *,
    node_yz: NDArray[np.floating],
    values: NDArray[np.floating],
    output_path: Path,
    title: str,
    colorbar_label: str,
    connectivity: Sequence[Sequence[int]] | None = None,
    dpi: int = 160,
    figsize: tuple[float, float] = (8.0, 5.0),
    cmap: str = "coolwarm",
    sequential: bool = False,
    deform_us: NDArray[np.floating] | None = None,
    deform_w: NDArray[np.floating] | None = None,
    deform_scale: float | None = None,
    overlay_deformed: bool = False,
    plot_on_deformed: bool = False,
    write_json: bool = False,
) -> None:
    """Colour unique nodes by ``values``; optionally overlay/plot deformed mesh.

    Companion JSON is opt-in (``write_json``) — default keeps PNG depth without
    duplicating full nodal arrays beside every fringe.
    """
    yz = np.asarray(node_yz, dtype=float)
    val = np.asarray(values, dtype=float).ravel()
    if yz.ndim != 2 or yz.shape[0] != val.size:
        raise ValueError("node_yz and values must align (n_nodes,)")

    us = None if deform_us is None else np.asarray(deform_us, dtype=float).ravel()
    ww = None if deform_w is None else np.asarray(deform_w, dtype=float).ravel()
    alpha = 1.0
    if us is not None and ww is not None and us.size == val.size:
        alpha = auto_deform_scale(yz, us, ww) if deform_scale is None else float(deform_scale)

    if plot_on_deformed and us is not None and ww is not None:
        plot_yz = np.column_stack([yz[:, 0] + alpha * us, yz[:, 1] + alpha * ww])
    else:
        plot_yz = yz

    fig, ax = plt.subplots(figsize=figsize)
    cmap_name = "viridis" if sequential else cmap
    finite = val[np.isfinite(val)]
    if sequential:
        vmin, vmax = 0.0, float(np.nanmax(finite)) if finite.size else 1.0
    else:
        peak = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
        vmin, vmax = -peak, peak

    if connectivity:
        polys = []
        for elem in connectivity:
            ids = [int(i) for i in elem if 0 <= int(i) < len(plot_yz)]
            if len(ids) >= 3:
                polys.append(plot_yz[ids])
        if polys:
            pc = PolyCollection(
                section_rings_hv(polys),
                facecolors="none",
                edgecolors="0.7",
                linewidths=0.4,
                zorder=1,
            )
            ax.add_collection(pc)

    plot_h, plot_v = section_hv(plot_yz)
    sc = ax.scatter(
        plot_h,
        plot_v,
        c=val,
        cmap=cmap_name,
        vmin=vmin,
        vmax=vmax,
        s=18,
        zorder=3,
        edgecolors="k",
        linewidths=0.2,
    )
    if (
        overlay_deformed
        and not plot_on_deformed
        and us is not None
        and ww is not None
        and us.size == val.size
    ):
        def_yz = np.column_stack([yz[:, 0] + alpha * us, yz[:, 1] + alpha * ww])
        def_h, def_v = section_hv(def_yz)
        ax.scatter(
            def_h,
            def_v,
            s=8,
            facecolors="none",
            edgecolors="0.35",
            linewidths=0.5,
            zorder=2,
            label=f"deformed (α={alpha:.3g})",
        )
        if connectivity:
            dpolys = []
            for elem in connectivity:
                ids = [int(i) for i in elem if 0 <= int(i) < len(def_yz)]
                if len(ids) >= 3:
                    dpolys.append(def_yz[ids])
            if dpolys:
                ax.add_collection(
                    PolyCollection(
                        section_rings_hv(dpolys),
                        facecolors="none",
                        edgecolors="0.35",
                        linewidths=0.35,
                        linestyles="--",
                        zorder=2,
                    )
                )
        ax.legend(fontsize=7, loc="best")

    finish_section_axes(ax)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(colorbar_label)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    if write_json:
        conn_list = [list(map(int, e)) for e in (connectivity or [])]
        coincident = 0
        for elem in conn_list:
            ids = [i for i in elem if 0 <= i < len(yz)]
            if len(ids) >= 2:
                pts = yz[ids]
                for k in range(len(ids)):
                    if np.allclose(pts[k], pts[(k + 1) % len(ids)], atol=1e-14):
                        coincident += 1
                        break
        dump_plot_json(
            output_path,
            {
                "kind": "nodal_fringe",
                "title": title,
                "colorbar_label": colorbar_label,
                "n_nodes": int(yz.shape[0]),
                "node_yz_m": yz[:, :2].tolist(),
                "values": val.tolist(),
                "connectivity": conn_list,
                "deform_scale_alpha": float(alpha),
                "plot_on_deformed": bool(plot_on_deformed),
                "overlay_deformed": bool(overlay_deformed),
                "diagnostics": {
                    "n_conn_elems": len(conn_list),
                    "n_conn_with_coincident_adjacent": coincident,
                    "value_finite": int(np.sum(np.isfinite(val))),
                    "value_abs_max": float(np.nanmax(np.abs(finite))) if finite.size else None,
                },
            },
        )


class FringeCanvas:
    """Reusable Agg figure for high-volume elemental fringe emission."""

    def __init__(self, figsize: tuple[float, float] = (8.0, 5.0)):
        self.figsize = figsize
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self._colorbar = None

    def close(self) -> None:
        plt.close(self.fig)
        self.fig = None  # type: ignore[assignment]
        self.ax = None  # type: ignore[assignment]

    def render_elemental(
        self,
        *,
        keep_poly: list[NDArray[np.floating]],
        keep_seg: list[NDArray[np.floating]],
        face_vals: NDArray[np.floating],
        seg_vals: NDArray[np.floating],
        output_path: Path,
        title: str,
        colorbar_label: str,
        dpi: int,
        sequential: bool,
        cmap: str,
        clim_min: float | None = None,
        clim_max: float | None = None,
    ) -> None:
        ax = self.ax
        fig = self.fig
        ax.clear()
        if self._colorbar is not None:
            try:
                self._colorbar.remove()
            except Exception:  # noqa: BLE001
                pass
            self._colorbar = None
        all_vals = (
            np.concatenate([face_vals, seg_vals])
            if face_vals.size and seg_vals.size
            else (face_vals if face_vals.size else seg_vals)
        )
        cmap_name = "viridis" if sequential else cmap
        if clim_min is not None and clim_max is not None:
            vmin, vmax = float(clim_min), float(clim_max)
        elif sequential:
            vmin, vmax = 0.0, float(np.nanmax(all_vals)) if all_vals.size else 1.0
        else:
            peak = float(np.nanmax(np.abs(all_vals))) if all_vals.size else 1.0
            vmin, vmax = -peak, peak
        mappable = None
        if keep_poly:
            pc = PolyCollection(
                section_rings_hv(keep_poly),
                array=face_vals,
                cmap=cmap_name,
                edgecolors="k",
                linewidths=0.25,
            )
            pc.set_clim(vmin, vmax)
            ax.add_collection(pc)
            mappable = pc
        if keep_seg:
            lc = LineCollection(
                section_rings_hv(keep_seg),
                array=seg_vals,
                cmap=cmap_name,
                linewidths=2.5,
                capstyle="round",
            )
            lc.set_clim(vmin, vmax)
            ax.add_collection(lc)
            if mappable is None:
                mappable = lc
        ax.autoscale()
        finish_section_axes(ax)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25)
        if mappable is not None:
            self._colorbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
            self._colorbar.set_label(colorbar_label)
        fig.subplots_adjust(left=0.12, right=0.88, bottom=0.12, top=0.90)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)


def _resolve_elemental_values(
    *,
    geometry: ElementalFringeGeometry,
    elements: Iterable[dict] | None,
    value_key: str | None,
    values: Sequence[float] | None,
) -> tuple[list[NDArray[np.floating]], list[NDArray[np.floating]], NDArray[np.floating], NDArray[np.floating]] | None:
    polys = geometry.polys
    segments = geometry.segments
    drawn_elem_idx = geometry.drawn_elem_idx
    drawn_modes = geometry.drawn_modes
    if not polys and not segments:
        return None
    elems = list(elements) if elements is not None else None
    face_vals: list[float] = []
    seg_vals: list[float] = []
    keep_poly: list[NDArray[np.floating]] = []
    keep_seg: list[NDArray[np.floating]] = []
    poly_i = 0
    seg_i = 0
    for j, ei in enumerate(drawn_elem_idx):
        mode = drawn_modes[j]
        if values is not None:
            v = float(values[ei]) if ei < len(values) else np.nan
        elif value_key is not None and elems is not None:
            raw: Any = elems[ei] if ei < len(elems) else None
            for part in value_key.split("."):
                raw = (raw or {}).get(part) if isinstance(raw, dict) else None
            try:
                v = float(raw) if raw is not None else np.nan
            except (TypeError, ValueError):
                v = np.nan
        else:
            v = np.nan
        if mode == "poly":
            src = polys[poly_i]
            poly_i += 1
            if not np.isfinite(v):
                continue
            keep_poly.append(src)
            face_vals.append(v)
        else:
            src = segments[seg_i]
            seg_i += 1
            if not np.isfinite(v):
                continue
            keep_seg.append(src)
            seg_vals.append(v)
    if not keep_poly and not keep_seg:
        return None
    return (
        keep_poly,
        keep_seg,
        np.asarray(face_vals, dtype=float),
        np.asarray(seg_vals, dtype=float),
    )


def _render_elemental_fringe_job(job: dict[str, Any]) -> str:
    """Process-pool worker: render one elemental fringe from a compact job dict."""
    canvas = FringeCanvas(figsize=tuple(job.get("figsize") or (8.0, 5.0)))
    try:
        keep_poly = [np.asarray(p, dtype=float) for p in job["keep_poly"]]
        keep_seg = [np.asarray(s, dtype=float) for s in job["keep_seg"]]
        canvas.render_elemental(
            keep_poly=keep_poly,
            keep_seg=keep_seg,
            face_vals=np.asarray(job["face_vals"], dtype=float),
            seg_vals=np.asarray(job["seg_vals"], dtype=float),
            output_path=Path(job["output_path"]),
            title=str(job["title"]),
            colorbar_label=str(job["colorbar_label"]),
            dpi=int(job.get("dpi") or 160),
            sequential=bool(job.get("sequential")),
            cmap=str(job.get("cmap") or "coolwarm"),
            clim_min=job.get("clim_min"),
            clim_max=job.get("clim_max"),
        )
    finally:
        canvas.close()
    return str(job["output_path"])


def run_elemental_fringe_jobs(
    jobs: Sequence[dict[str, Any]],
    *,
    workers: int | None = None,
) -> None:
    """Render fringe jobs serially or via a process pool.

    ``workers``:
      - ``None`` or ``<= 0`` — auto-allocate via ``allocate_core_budget``
      - ``>= 1`` — explicit override (``1`` forces serial)
    """
    if not jobs:
        return
    if workers is None or int(workers) <= 0:
        from blade_structure_utilities.parallel_utilities.core_budget import (
            allocate_core_budget,
        )

        n_workers = allocate_core_budget(
            n_outer_tasks=max(1, len(jobs))
        ).outer_workers
    else:
        n_workers = max(1, int(workers))
    if n_workers == 1 or len(jobs) == 1:
        canvas = FringeCanvas(figsize=tuple(jobs[0].get("figsize") or (8.0, 5.0)))
        try:
            for job in jobs:
                canvas.render_elemental(
                    keep_poly=[np.asarray(p, dtype=float) for p in job["keep_poly"]],
                    keep_seg=[np.asarray(s, dtype=float) for s in job["keep_seg"]],
                    face_vals=np.asarray(job["face_vals"], dtype=float),
                    seg_vals=np.asarray(job["seg_vals"], dtype=float),
                    output_path=Path(job["output_path"]),
                    title=str(job["title"]),
                    colorbar_label=str(job["colorbar_label"]),
                    dpi=int(job.get("dpi") or 160),
                    sequential=bool(job.get("sequential")),
                    cmap=str(job.get("cmap") or "coolwarm"),
                    clim_min=job.get("clim_min"),
                    clim_max=job.get("clim_max"),
                )
        finally:
            canvas.close()
        return
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_render_elemental_fringe_job, job) for job in jobs]
        for fut in as_completed(futs):
            fut.result()


def build_elemental_fringe_job(
    *,
    geometry: ElementalFringeGeometry,
    elements: Iterable[dict] | None = None,
    value_key: str | None = None,
    values: Sequence[float] | None = None,
    output_path: Path,
    title: str,
    colorbar_label: str,
    dpi: int = 160,
    figsize: tuple[float, float] = (8.0, 5.0),
    cmap: str = "coolwarm",
    sequential: bool = False,
    clim_min: float | None = None,
    clim_max: float | None = None,
) -> dict[str, Any] | None:
    """Build a picklable fringe job dict, or None if nothing to draw."""
    resolved = _resolve_elemental_values(
        geometry=geometry,
        elements=elements,
        value_key=value_key,
        values=values,
    )
    if resolved is None:
        return None
    keep_poly, keep_seg, face_vals, seg_vals = resolved
    return {
        "keep_poly": [p.tolist() for p in keep_poly],
        "keep_seg": [s.tolist() for s in keep_seg],
        "face_vals": face_vals.tolist(),
        "seg_vals": seg_vals.tolist(),
        "output_path": str(output_path),
        "title": title,
        "colorbar_label": colorbar_label,
        "dpi": int(dpi),
        "figsize": tuple(figsize),
        "cmap": cmap,
        "sequential": bool(sequential),
        "clim_min": None if clim_min is None else float(clim_min),
        "clim_max": None if clim_max is None else float(clim_max),
    }


def save_elemental_fringe(
    *,
    elements: Iterable[dict] | None = None,
    value_key: str | None = None,
    values: Sequence[float] | None = None,
    output_path: Path,
    title: str,
    colorbar_label: str,
    dpi: int = 160,
    figsize: tuple[float, float] = (8.0, 5.0),
    cmap: str = "coolwarm",
    sequential: bool = False,
    clim_min: float | None = None,
    clim_max: float | None = None,
    coords_key: str = "node_coords_yz_m",
    geometry: ElementalFringeGeometry | None = None,
    write_json: bool = False,
    canvas: FringeCanvas | None = None,
) -> ElementalFringeGeometry | None:
    """Colour element polygons from ``node_coords_yz_m`` + per-element values.

    Pass a reused ``geometry`` (from :func:`build_elemental_fringe_geometry` or
    a prior call) to avoid rebuilding mid-surface polys for every field/ply.
    Companion JSON is opt-in (``write_json``) — ply fringes are PNG-primary.
    """
    if geometry is None:
        if elements is None:
            return None
        geometry = build_elemental_fringe_geometry(elements, coords_key=coords_key)
    resolved = _resolve_elemental_values(
        geometry=geometry,
        elements=elements,
        value_key=value_key,
        values=values,
    )
    if resolved is None:
        return geometry
    keep_poly, keep_seg, face_vals_a, seg_vals_a = resolved
    all_vals = (
        np.concatenate([face_vals_a, seg_vals_a])
        if face_vals_a.size and seg_vals_a.size
        else (face_vals_a if face_vals_a.size else seg_vals_a)
    )
    own_canvas = canvas is None
    if canvas is None:
        canvas = FringeCanvas(figsize=figsize)
    try:
        canvas.render_elemental(
            keep_poly=keep_poly,
            keep_seg=keep_seg,
            face_vals=face_vals_a,
            seg_vals=seg_vals_a,
            output_path=Path(output_path),
            title=title,
            colorbar_label=colorbar_label,
            dpi=dpi,
            sequential=sequential,
            cmap=cmap,
            clim_min=clim_min,
            clim_max=clim_max,
        )
    finally:
        if own_canvas:
            canvas.close()

    if write_json:
        areas = [_poly_area(p) for p in keep_poly]
        seg_lengths = [
            float(np.linalg.norm(s[1] - s[0])) for s in keep_seg if s.shape[0] >= 2
        ]
        dump_plot_json(
            Path(output_path),
            {
                "kind": "elemental_fringe",
                "title": title,
                "colorbar_label": colorbar_label,
                "value_key": value_key,
                "n_faces": len(keep_poly),
                "n_segments": len(keep_seg),
                "face_values": face_vals_a.tolist(),
                "segment_values": seg_vals_a.tolist(),
                "diagnostics": {
                    "n_elems_in": geometry.n_elems,
                    "n_faces_drawn": len(keep_poly),
                    "n_segments_drawn": len(keep_seg),
                    "area_min": float(np.min(areas)) if areas else None,
                    "area_max": float(np.max(areas)) if areas else None,
                    "area_median": float(np.median(areas)) if areas else None,
                    "seg_length_median": float(np.median(seg_lengths)) if seg_lengths else None,
                    "value_abs_max": float(np.nanmax(np.abs(all_vals))) if all_vals.size else None,
                },
            },
        )
    return geometry


def save_gaussian_fringe(
    *,
    elements: Iterable[dict],
    gp_values: NDArray[np.floating],
    output_path: Path,
    title: str,
    colorbar_label: str,
    dpi: int = 160,
    figsize: tuple[float, float] = (8.0, 5.0),
    cmap: str = "coolwarm",
    sequential: bool = False,
    coords_key: str = "node_coords_yz_m",
    geometry: ElementalFringeGeometry | None = None,
    write_json: bool = False,
) -> ElementalFringeGeometry | None:
    """Element-face fringe coloured by mean of per-element Gauss-point values.

    ``gp_values`` shape ``(n_elem,)`` or ``(n_elem, n_gp)`` — the latter is
    reduced by nanmean over the GP axis (product feature for secondary GP viz).
    """
    arr = np.asarray(gp_values, dtype=float)
    if arr.ndim == 2:
        values = np.nanmean(arr, axis=1)
    else:
        values = arr.ravel()
    return save_elemental_fringe(
        elements=elements,
        values=values,
        output_path=output_path,
        title=title,
        colorbar_label=colorbar_label,
        dpi=dpi,
        figsize=figsize,
        cmap=cmap,
        sequential=sequential,
        coords_key=coords_key,
        geometry=geometry,
        write_json=write_json,
    )


__all__ = [
    "ElementalFringeGeometry",
    "FringeCanvas",
    "auto_deform_scale",
    "build_elemental_fringe_geometry",
    "build_elemental_fringe_job",
    "dump_plot_json",
    "run_elemental_fringe_jobs",
    "save_nodal_fringe",
    "save_elemental_fringe",
    "save_gaussian_fringe",
]
