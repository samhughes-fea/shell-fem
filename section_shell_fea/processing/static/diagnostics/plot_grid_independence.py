"""Plot ASME/Roache GCI grid-independence study outputs (ladder + summary CSVs).

Plots every finite primary scalar and secondary / tertiary / quaternary kind
peak present in the ladder (not only |N12|). Also writes per-layout composites
pairing section mid-surface geometry with CI ladder curves.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .asme_gci import FS_TWO_LEVEL, P_DEFAULT, gci_fine, gci_triplet
from .cascade_mesh_convergence import (
    DEFAULT_CHORD,
    DEFAULT_THICKNESS,
    DEFAULT_TWIST_DEG,
    KIND_PEAK_METRICS,
    PEAK_METRIC_LABELS,
    QUATERNARY_KIND_METRICS,
    _build_shell,
)

KINDS = ("skin", "cap", "web")
KIND_COLORS = {"skin": "#2563eb", "cap": "#dc2626", "web": "#059669"}

# Post-ownership CI watch keys (raw |N12| / mid-σ fine GCI > 10%).
CI_WATCH_LAYOUTS = frozenset({"3D-CN", "4D-CN", "4D-F", "5D-CN", "5D-F"})


def structural_family_for_layout(layout: str) -> str:
    """Return structural family letter (A–D) for layout key directory grouping."""
    from blade_structure_utilities.geometry_utilities.section_geometry import (
        resolve_system_type,
    )

    try:
        fam = str(resolve_system_type(str(layout)).structural_family).strip().upper()
    except Exception:  # noqa: BLE001
        fam = ""
    if fam in {"A", "B", "C", "D"}:
        return fam
    # Fallback: first letter after optional leading digits (e.g. 5D-F → D, 0A → A).
    for ch in str(layout):
        if ch.upper() in {"A", "B", "C", "D"}:
            return ch.upper()
    return "other"

# Kind-peak metrics to plot when present in the ladder.
PEAK_METRICS: tuple[tuple[str, str], ...] = tuple(
    (m, PEAK_METRIC_LABELS.get(m, m)) for m in KIND_PEAK_METRICS
)
# Also plot endpoint-excluded (interior) ladders for the same peaks.
INTERIOR_PEAK_METRICS: tuple[tuple[str, str], ...] = tuple(
    (f"{m}_interior", f"interior {PEAK_METRIC_LABELS.get(m, m)}") for m in KIND_PEAK_METRICS
)
# Scalar (not kind-keyed) ladder columns — primary + quaternary station scalar.
SCALAR_METRICS: tuple[tuple[str, str], ...] = (
    ("governing_fi", "governing FI (quaternary)"),
    ("solve_residual_rel", "solve residual ‖r‖/‖f‖ (primary)"),
    ("residual_rel_independent", "independent residual (primary)"),
    ("residual_rel_unreduced", "unreduced residual (primary)"),
    ("d_inf", "‖d‖_∞ (primary)"),
)

def _f(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or "nan")
    except ValueError:
        return float("nan")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _savefig(fig, path: Path, *, dpi: int = 140) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _has_finite_ladder(ladder: Sequence[dict[str, str]], col: str) -> bool:
    return any(np.isfinite(_f(r, col)) and abs(_f(r, col)) > 0.0 for r in ladder)


def _layouts(ladder: Sequence[dict[str, str]]) -> list[str]:
    seen: list[str] = []
    for r in ladder:
        lab = str(r.get("layout", ""))
        if lab and lab not in seen:
            seen.append(lab)
    return seen


def _series_for_layout(
    ladder: Sequence[dict[str, str]],
    layout: str,
    col: str,
) -> tuple[list[float], list[float]]:
    rows = sorted(
        (r for r in ladder if str(r.get("layout")) == layout),
        key=lambda r: _f(r, "h_m"),
    )
    h = [_f(r, "h_m") for r in rows]
    q = [_f(r, col) for r in rows]
    return h, q


def _enrich_summary_from_ladder(
    summary: list[dict[str, str]],
    ladder: Sequence[dict[str, str]],
    metrics: Sequence[str],
) -> list[dict[str, str]]:
    """Fill missing GCI / last-two columns from the ladder (e.g. peak_abs_n11)."""
    out: list[dict[str, str]] = []
    for row in summary:
        rec = dict(row)
        layout = str(rec.get("layout", ""))
        for metric in metrics:
            for kind in ("skin", "web", "cap"):
                gci_key = f"{metric}_gci_fine_{kind}"
                lt_key = f"{metric}_last_two_{kind}"
                det_key = f"{metric}_gci_detail_{kind}"
                ok_key = f"{metric}_gci_ok_{kind}"
                need = not np.isfinite(_f(rec, gci_key)) or str(rec.get(gci_key, "")) == ""
                if not need and np.isfinite(_f(rec, lt_key)):
                    continue
                col = f"{metric}_{kind}"
                h, q = _series_for_layout(ladder, layout, col)
                pairs = [
                    (hh, qq)
                    for hh, qq in zip(h, q)
                    if np.isfinite(hh) and np.isfinite(qq) and qq > 0.0
                ]
                pairs = sorted(pairs, key=lambda t: t[0])[:3]
                if len(pairs) >= 2 and (not np.isfinite(_f(rec, lt_key))):
                    a, b = pairs[-2][1], pairs[-1][1]
                    rec[lt_key] = str(max(a, b) / min(a, b))
                if len(pairs) >= 2 and need:
                    g = gci_triplet([p[0] for p in pairs], [p[1] for p in pairs])
                    rec[gci_key] = str(g.gci_fine)
                    rec[ok_key] = str(bool(g.ok))
                    rec[det_key] = str(g.detail)
                    rec[f"{metric}_p_obs_{kind}"] = str(g.p_obs)
        out.append(rec)
    return out


def plot_gci_fine_bars(
    summary_rows: Sequence[dict[str, str]],
    outfile: Path,
    *,
    metric: str = "peak_abs_n12",
    title: str | None = None,
    limit: float = 0.10,
    ylabel: str | None = None,
) -> Path | None:
    """Horizontal bars of fine-grid GCI (%) by layout for skin/web."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layouts = [r["layout"] for r in summary_rows]
    if not layouts:
        return None
    # Skip if both skin and web are all-NaN
    skin_vals = [_f(r, f"{metric}_gci_fine_skin") for r in summary_rows]
    web_vals = [_f(r, f"{metric}_gci_fine_web") for r in summary_rows]
    if not (any(np.isfinite(skin_vals)) or any(np.isfinite(web_vals))):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, max(4.5, 0.28 * len(layouts))), sharey=True)
    y = np.arange(len(layouts))
    for ax, kind, vals in zip(axes, ("skin", "web"), (skin_vals, web_vals)):
        vals_a = np.asarray(vals, dtype=float) * 100.0
        colors = []
        for r, v in zip(summary_rows, vals_a):
            detail = str(r.get(f"{metric}_gci_detail_{kind}", "") or "")
            if not np.isfinite(v):
                colors.append("#9ca3af")
            elif detail == "gci_non_monotonic":
                colors.append("#f59e0b")
            elif v > 100.0 * limit:
                colors.append("#dc2626")
            else:
                colors.append(KIND_COLORS[kind])
        ax.barh(y, np.nan_to_num(vals_a, nan=0.0), color=colors, height=0.7)
        ax.axvline(100.0 * limit, color="#111827", ls="--", lw=1.0, label=f"{100 * limit:.0f}% gate")
        ax.set_xlabel(f"{ylabel or metric} fine GCI [%]")
        ax.set_title(kind)
        xmax = float(np.nanmax(vals_a)) if np.isfinite(vals_a).any() else 25.0
        ax.set_xlim(0, max(25.0, xmax * 1.15))
        ax.grid(True, axis="x", alpha=0.3)
    axes[0].set_yticks(y, layouts, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(title or f"Fine-grid GCI — {ylabel or metric}", fontsize=12)
    fig.tight_layout()
    return _savefig(fig, outfile)


def plot_last_two_drift_bars(
    summary_rows: Sequence[dict[str, str]],
    outfile: Path,
    *,
    metric: str = "peak_abs_n12",
    title: str | None = None,
    limit: float = 2.0,
    ylabel: str | None = None,
) -> Path | None:
    """Horizontal bars of last-two drift by layout for skin/web."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layouts = [r["layout"] for r in summary_rows]
    if not layouts:
        return None
    skin_vals = [_f(r, f"{metric}_last_two_skin") for r in summary_rows]
    web_vals = [_f(r, f"{metric}_last_two_web") for r in summary_rows]
    if not (any(np.isfinite(skin_vals)) or any(np.isfinite(web_vals))):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, max(4.5, 0.28 * len(layouts))), sharey=True)
    y = np.arange(len(layouts))
    for ax, kind, vals in zip(axes, ("skin", "web"), (skin_vals, web_vals)):
        vals_a = np.asarray(vals, dtype=float)
        colors = [
            "#9ca3af"
            if not np.isfinite(v)
            else ("#dc2626" if v > limit else KIND_COLORS[kind])
            for v in vals_a
        ]
        ax.barh(y, np.nan_to_num(vals_a, nan=0.0), color=colors, height=0.7)
        ax.axvline(limit, color="#111827", ls="--", lw=1.0, label=f"{limit:g}× gate")
        ax.set_xlabel(f"{ylabel or metric} last-two drift [×]")
        ax.set_title(kind)
        xmax = float(np.nanmax(vals_a)) if np.isfinite(vals_a).any() else 3.0
        ax.set_xlim(0, max(3.0, xmax * 1.15))
        ax.grid(True, axis="x", alpha=0.3)
    axes[0].set_yticks(y, layouts, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(title or f"Last-two drift — {ylabel or metric}", fontsize=12)
    fig.tight_layout()
    return _savefig(fig, outfile)


def plot_ladder_small_multiples(
    ladder_rows: Sequence[dict[str, str]],
    outfile: Path,
    *,
    metric_col_prefix: str,
    kind: str | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    ncols: int = 8,
) -> Path | None:
    """Small-multiples of metric vs ``h_m`` for every layout.

    Pass ``kind`` for kind-keyed columns (``{prefix}_{kind}``), or ``None`` for
    a scalar column named ``metric_col_prefix``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = (
        metric_col_prefix
        if kind is None
        else f"{metric_col_prefix}_{kind}"
    )
    if not _has_finite_ladder(ladder_rows, col) and not any(
        np.isfinite(_f(r, col)) for r in ladder_rows
    ):
        return None

    by_layout: dict[str, list[dict[str, str]]] = {}
    for r in ladder_rows:
        by_layout.setdefault(str(r["layout"]), []).append(r)
    layouts = sorted(by_layout)
    if not layouts:
        return None
    n = len(layouts)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.2 * ncols, 1.8 * nrows), sharex=False, sharey=False
    )
    axes_flat = np.atleast_1d(axes).ravel()
    color = KIND_COLORS.get(kind or "skin", "#333333")
    for i, layout in enumerate(layouts):
        ax = axes_flat[i]
        rows = sorted(by_layout[layout], key=lambda r: _f(r, "h_m"), reverse=True)
        h = np.asarray([_f(r, "h_m") for r in rows], dtype=float)
        q = np.asarray([_f(r, col) for r in rows], dtype=float)
        mask = np.isfinite(h) & np.isfinite(q)
        if mask.any():
            ax.plot(h[mask], q[mask], "o-", color=color, ms=3, lw=1.2)
        ax.set_title(layout, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
        if i // ncols == nrows - 1:
            ax.set_xlabel("h [m]", fontsize=7)
        if i % ncols == 0:
            ax.set_ylabel(ylabel or col, fontsize=7)
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle(title or f"Ladder {col} vs h", fontsize=11)
    fig.tight_layout()
    return _savefig(fig, outfile)


def plot_scalar_gci_bars(
    summary_rows: Sequence[dict[str, str]],
    outfile: Path,
    *,
    gci_key: str = "governing_fi_gci_fine",
    detail_key: str = "governing_fi_gci_detail",
    title: str = "Fine-grid GCI — governing FI",
    limit: float = 0.10,
) -> Path | None:
    """Single-column GCI bars for a scalar summary metric (e.g. governing_fi)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layouts = [r["layout"] for r in summary_rows]
    vals = np.asarray([_f(r, gci_key) for r in summary_rows], dtype=float) * 100.0
    if not layouts or not np.isfinite(vals).any():
        return None
    colors = []
    for r, v in zip(summary_rows, vals):
        detail = str(r.get(detail_key, "") or "")
        if not np.isfinite(v):
            colors.append("#9ca3af")
        elif detail == "gci_non_monotonic":
            colors.append("#f59e0b")
        elif v > 100.0 * limit:
            colors.append("#dc2626")
        else:
            colors.append("#2563eb")
    fig, ax = plt.subplots(figsize=(7, max(4.5, 0.28 * len(layouts))))
    y = np.arange(len(layouts))
    ax.barh(y, np.nan_to_num(vals, nan=0.0), color=colors, height=0.7)
    ax.axvline(100.0 * limit, color="#111827", ls="--", lw=1.0)
    ax.set_yticks(y, layouts, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("fine GCI [%]")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return _savefig(fig, outfile)


def plot_grid_independence_study(
    out_dir: str | Path,
    *,
    ladder_name: str = "gci_ladder.csv",
    summary_name: str = "gci_summary.csv",
    plots_subdir: str = "plots",
    backend_label: str = "",
    write_hashin: bool = True,
) -> list[Path]:
    """Read study CSVs under ``out_dir`` and write overview PNGs into ``plots/``."""
    out = Path(out_dir)
    ladder_path = out / ladder_name
    summary_path = out / summary_name
    if not ladder_path.is_file():
        alt = out / "grid_independence_ladder.csv"
        if alt.is_file():
            ladder_path = alt
    if not summary_path.is_file():
        alt = out / "grid_independence_summary.csv"
        if alt.is_file():
            summary_path = alt
    if not ladder_path.is_file() or not summary_path.is_file():
        return []

    ladder = _read_csv(ladder_path)
    summary = _enrich_summary_from_ladder(
        _read_csv(summary_path),
        ladder,
        metrics=[m for m, _ in PEAK_METRICS],
    )
    plots = out / plots_subdir
    written: list[Path] = []
    label = f" ({backend_label})" if backend_label else ""

    def _skip_quaternary(metric: str) -> bool:
        if write_hashin:
            return False
        base = metric.replace("_interior", "")
        return base in QUATERNARY_KIND_METRICS or base.startswith("peak_hashin")

    for metric, pretty in PEAK_METRICS:
        if _skip_quaternary(metric):
            continue
        # Skip metrics that are all-NaN / zero on this backend (e.g. CCX Hashin).
        if not any(_has_finite_ladder(ladder, f"{metric}_{k}") for k in KINDS):
            continue
        for p in (
            plot_gci_fine_bars(
                summary,
                plots / f"gci_fine_{metric}_by_layout.png",
                metric=metric,
                title=f"Fine-grid GCI — {pretty}{label}",
                ylabel=pretty,
            ),
            plot_last_two_drift_bars(
                summary,
                plots / f"last_two_drift_{metric}_by_layout.png",
                metric=metric,
                title=f"Last-two drift — {pretty}{label}",
                ylabel=pretty,
            ),
        ):
            if p is not None:
                written.append(p)
        for kind in KINDS:
            p = plot_ladder_small_multiples(
                ladder,
                plots / f"ladder_{metric}_{kind}.png",
                metric_col_prefix=metric,
                kind=kind,
                title=f"Mesh ladder — {pretty} ({kind}){label}",
                ylabel=pretty,
            )
            if p is not None:
                written.append(p)

    for metric, pretty in INTERIOR_PEAK_METRICS:
        if _skip_quaternary(metric):
            continue
        if not any(_has_finite_ladder(ladder, f"{metric}_{k}") for k in ("skin", "web")):
            continue
        for kind in ("skin", "web"):
            p = plot_ladder_small_multiples(
                ladder,
                plots / f"ladder_{metric}_{kind}.png",
                metric_col_prefix=metric,
                kind=kind,
                title=f"Mesh ladder — {pretty} ({kind}){label}",
                ylabel=pretty,
            )
            if p is not None:
                written.append(p)

    for col, pretty in SCALAR_METRICS:
        if not any(np.isfinite(_f(r, col)) for r in ladder):
            continue
        p = plot_ladder_small_multiples(
            ladder,
            plots / f"ladder_{col}.png",
            metric_col_prefix=col,
            kind=None,
            title=f"Mesh ladder — {pretty}{label}",
            ylabel=pretty,
        )
        if p is not None:
            written.append(p)

    p = plot_scalar_gci_bars(
        summary,
        plots / "gci_fine_governing_fi_by_layout.png",
        title=f"Fine-grid GCI — governing FI{label}",
    )
    if p is not None:
        written.append(p)

    try:
        written.extend(
            plot_study_layout_convergence(
                out,
                ladder_name=ladder_path.name,
                summary_name=summary_path.name,
                out_subdir=str(Path(plots_subdir) / "layouts"),
            )
        )
    except Exception:  # noqa: BLE001 — layout composites must not abort overview plots
        pass

    return written


def _load_recipe_mapping(recipe: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    if recipe is None:
        return {}
    if isinstance(recipe, Mapping):
        return dict(recipe)
    path = Path(recipe)
    if not path.is_file():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def shell_from_recipe(
    layout: str,
    recipe: str | Path | Mapping[str, Any] | None = None,
) -> Any:
    """Build ``ShellMeshInputs`` for ``layout`` from a station recipe mapping/JSON."""
    data = _load_recipe_mapping(recipe)
    thickness = dict(data.get("thickness") or DEFAULT_THICKNESS)
    chord = float(data.get("chord_m", data.get("chord", DEFAULT_CHORD)))
    twist_deg = float(data.get("twist_deg", DEFAULT_TWIST_DEG))
    naca_raw = data.get("naca_dat")
    naca: str | Path | None = None
    if naca_raw:
        naca_path = Path(str(naca_raw))
        if naca_path.is_file():
            naca = naca_path
    return _build_shell(
        layout=str(layout),
        chord=chord,
        twist_deg=twist_deg,
        thickness=thickness,
        naca_dat=naca,
    )


def _draw_shell_geometry(ax, shell: Any, *, title: str) -> None:
    """Light print-friendly midlines (skin / web / cap) in B-frame plot axes."""
    from blade_structure_utilities.geometry_utilities.coordinate_frames import (
        fit_square_section_axes,
        section_plot_axes,
        setup_section_axes,
    )

    styles = {
        "skin": dict(color="0.35", lw=1.4, alpha=0.95),
        "web": dict(color="#059669", lw=1.8, alpha=0.95),
        "cap": dict(color="#dc2626", lw=2.0, alpha=0.95),
        "other": dict(color="0.55", lw=1.0, alpha=0.75, ls="--"),
    }
    midlines = list(getattr(shell, "midlines", []) or [])
    for strip in midlines:
        pts = np.asarray(getattr(strip, "midline_b", None), dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        st = styles.get(str(getattr(strip, "kind", "other")), styles["other"])
        h, v = section_plot_axes(pts)
        ax.plot(h, v, zorder=2, **st)

    le = getattr(shell, "leading_edge_b", None)
    te = getattr(shell, "trailing_edge_b", None)
    if le is not None and te is not None:
        ends = np.asarray([le, te], dtype=float)
        if ends.shape == (2, 2):
            h, v = section_plot_axes(ends)
            ax.scatter(h, v, c="#111827", s=36, marker="x", zorder=5, label="LE / TE")

    ax.set_facecolor("#f7f7f7")
    ax.grid(True, which="major", color="0.85", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    setup_section_axes(ax)
    fit_square_section_axes(ax, pad_frac=0.08)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)


def _gci_pct_label(summary_row: Mapping[str, str] | None, metric: str, kind: str) -> str:
    if not summary_row:
        return "n/a"
    v = _f(dict(summary_row), f"{metric}_gci_fine_{kind}")
    if not np.isfinite(v):
        return "n/a"
    flag = " FAIL" if v > 0.10 else ""
    return f"{100.0 * v:.1f}%{flag}"


def _local_relative_gci(
    h: np.ndarray,
    q: np.ndarray,
    *,
    fallback_gci: float = float("nan"),
) -> np.ndarray:
    """Per-rung relative GCI (fine→coarse), for ±GCI·|Q| error bands.

    At each index ``i`` (ascending ``h``, fine first) uses the three-level
    Roache/ASME estimate on ``(i,i+1,i+2)`` when available, else a two-level
    ``Fs=3`` estimate on ``(i,i+1)``, else ``fallback_gci``.
    """
    n = int(h.size)
    rel = np.full(n, float("nan"), dtype=float)
    if n == 0:
        return rel
    order = np.argsort(h)  # fine → coarse
    h_s = np.asarray(h[order], dtype=float)
    q_s = np.asarray(q[order], dtype=float)
    g_s = np.full(n, float("nan"), dtype=float)
    for i in range(n):
        g_i = float("nan")
        if i + 2 < n:
            g = gci_triplet(
                [h_s[i], h_s[i + 1], h_s[i + 2]],
                [q_s[i], q_s[i + 1], q_s[i + 2]],
            )
            if np.isfinite(g.gci_fine) and g.gci_fine >= 0.0:
                g_i = float(g.gci_fine)
        if not np.isfinite(g_i) and i + 1 < n:
            try:
                g_i = float(
                    gci_fine(
                        h_s[i],
                        h_s[i + 1],
                        q_s[i],
                        q_s[i + 1],
                        p=P_DEFAULT,
                        fs=FS_TWO_LEVEL,
                    )
                )
            except ValueError:
                g_i = float("nan")
        if not np.isfinite(g_i) and np.isfinite(fallback_gci) and fallback_gci >= 0.0:
            g_i = float(fallback_gci)
        g_s[i] = g_i
    # Propagate last known estimate to the coarsest point if still NaN.
    for i in range(n - 2, -1, -1):
        if not np.isfinite(g_s[i + 1]) and np.isfinite(g_s[i]):
            g_s[i + 1] = g_s[i]
    rel[order] = g_s
    return rel


def plot_layout_convergence_with_geometry(
    layout: str,
    ladder_rows: Sequence[dict[str, str]],
    summary_row: Mapping[str, str] | None,
    shell: Any,
    outfile: Path,
    *,
    title_suffix: str = "",
    mark_ci_watch: bool | None = None,
) -> Path | None:
    """One layout: section geometry (left) + CI ladder curves (right)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(
        (r for r in ladder_rows if str(r.get("layout")) == str(layout)),
        key=lambda r: _f(r, "h_m"),
        reverse=True,
    )
    if not rows:
        return None

    watch = (
        bool(mark_ci_watch)
        if mark_ci_watch is not None
        else str(layout) in CI_WATCH_LAYOUTS
    )
    watch_tag = "  [CI watch]" if watch else ""
    suffix = f" {title_suffix}" if title_suffix else ""

    fig, (ax_g, ax_l) = plt.subplots(
        1, 2, figsize=(11.0, 4.6), gridspec_kw={"width_ratios": [1.05, 1.25]}
    )
    geom_title = f"{layout}{watch_tag}\nsection midlines"
    if shell is not None:
        _draw_shell_geometry(ax_g, shell, title=geom_title)
    else:
        ax_g.text(0.5, 0.5, "geometry unavailable", ha="center", va="center")
        ax_g.set_axis_off()
        ax_g.set_title(geom_title, fontsize=10)

    h = np.asarray([_f(r, "h_m") for r in rows], dtype=float)
    ax_r = ax_l.twinx()
    series_specs = (
        ("peak_abs_n12", "skin", KIND_COLORS["skin"], "-", 1.8, ax_l, "|N12|"),
        ("peak_abs_n12", "web", KIND_COLORS["web"], "-", 1.6, ax_l, "|N12|"),
        ("peak_mid_sig", "skin", KIND_COLORS["skin"], "--", 1.5, ax_r, "mid-σ"),
        ("peak_mid_sig", "web", KIND_COLORS["web"], "--", 1.3, ax_r, "mid-σ"),
    )
    any_n12 = False
    any_mid = False
    band_labeled = False
    for metric, kind, color, ls, lw, ax, _family in series_specs:
        col = f"{metric}_{kind}"
        q = np.asarray([_f(r, col) for r in rows], dtype=float)
        mask = np.isfinite(h) & np.isfinite(q) & (q > 0.0)
        if not mask.any():
            continue
        h_m = h[mask]
        q_m = q[mask]
        # Sort ascending h for a clean fill_between ribbon.
        order = np.argsort(h_m)
        h_m = h_m[order]
        q_m = q_m[order]
        fallback = (
            _f(dict(summary_row), f"{metric}_gci_fine_{kind}")
            if summary_row
            else float("nan")
        )
        rel = _local_relative_gci(h_m, q_m, fallback_gci=fallback)
        band = np.where(np.isfinite(rel), np.abs(q_m) * rel, 0.0)
        if np.any(band > 0.0):
            ax.fill_between(
                h_m,
                np.maximum(q_m - band, 0.0),
                q_m + band,
                color=color,
                alpha=0.18,
                linewidth=0,
                zorder=1,
                label="± local GCI·|Q|" if not band_labeled else None,
            )
            band_labeled = True
        gci_s = _gci_pct_label(summary_row, metric, kind)
        pretty = "|N12|" if metric == "peak_abs_n12" else "mid-σ"
        ax.plot(
            h_m,
            q_m,
            ls,
            color=color,
            lw=lw,
            ms=4,
            marker="o",
            zorder=3,
            label=f"{pretty} {kind}  GCI={gci_s}",
        )
        if metric == "peak_abs_n12":
            any_n12 = True
        else:
            any_mid = True

    ax_l.set_xlabel("h [m]", fontsize=9)
    ax_l.set_ylabel("|N12| [N/m]", fontsize=9, color=KIND_COLORS["skin"])
    ax_r.set_ylabel("mid-σ [Pa]", fontsize=9, color="#6b7280")
    ax_l.tick_params(labelsize=8)
    ax_r.tick_params(labelsize=8)
    ax_l.grid(True, alpha=0.3)
    if any_n12 or any_mid:
        lines_l, labels_l = ax_l.get_legend_handles_labels()
        lines_r, labels_r = ax_r.get_legend_handles_labels()
        ax_l.legend(lines_l + lines_r, labels_l + labels_r, fontsize=7, loc="best")
    else:
        ax_l.text(0.5, 0.5, "no CI ladder series", ha="center", va="center", transform=ax_l.transAxes)

    # Call out CI gate failure on title when either skin/web N12 or mid-σ GCI > 10%.
    fail_bits: list[str] = []
    for metric, kind, _lab in (
        ("peak_abs_n12", "skin", "N12 skin"),
        ("peak_abs_n12", "web", "N12 web"),
        ("peak_mid_sig", "skin", "mid-σ skin"),
        ("peak_mid_sig", "web", "mid-σ web"),
    ):
        if not summary_row:
            break
        v = _f(dict(summary_row), f"{metric}_gci_fine_{kind}")
        if np.isfinite(v) and v > 0.10:
            fail_bits.append(f"{kind}/{metric}={100*v:.1f}%")
    gate = ("  ·  CI GCI>10%: " + ", ".join(fail_bits)) if fail_bits else ""
    ax_l.set_title(f"CI ladder vs h{suffix}{gate}", fontsize=10)

    fig.suptitle(f"{layout}{watch_tag} — geometry + convergence{suffix}", fontsize=12, y=1.02)
    fig.tight_layout()
    return _savefig(fig, Path(outfile), dpi=150)


def plot_study_layout_convergence(
    study_dir: str | Path,
    *,
    keys: Sequence[str] | None = None,
    out_subdir: str = "plots/layouts",
    ladder_name: str = "gci_ladder.csv",
    summary_name: str = "gci_summary.csv",
    recipe_path: str | Path | None = None,
    title_suffix: str = "",
    mark_ci_watch: bool | None = None,
) -> list[Path]:
    """Write per-layout geometry+ladder PNGs for a finished GCI study directory."""
    out = Path(study_dir)
    ladder_path = out / ladder_name
    summary_path = out / summary_name
    if not ladder_path.is_file():
        alt = out / "grid_independence_ladder.csv"
        if alt.is_file():
            ladder_path = alt
    if not summary_path.is_file():
        alt = out / "grid_independence_summary.csv"
        if alt.is_file():
            summary_path = alt
    if not ladder_path.is_file() or not summary_path.is_file():
        return []

    ladder = _read_csv(ladder_path)
    summary_rows = _read_csv(summary_path)
    summary_by = {str(r.get("layout", "")): r for r in summary_rows}
    layouts = _layouts(ladder)
    if keys is not None:
        want = {str(k).strip() for k in keys if str(k).strip()}
        layouts = [k for k in layouts if k in want]

    recipe = recipe_path if recipe_path is not None else out / "station_recipe.json"
    recipe_map = _load_recipe_mapping(recipe)
    chord = float(recipe_map.get("chord_m", DEFAULT_CHORD)) if recipe_map else float(DEFAULT_CHORD)
    twist = float(recipe_map.get("twist_deg", DEFAULT_TWIST_DEG)) if recipe_map else float(DEFAULT_TWIST_DEG)
    geom_note = f"c={chord:.3g} m, twist={twist:.2g}°"

    dest = out / out_subdir
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for layout in layouts:
        try:
            shell = shell_from_recipe(layout, recipe_map or None)
        except Exception:  # noqa: BLE001
            shell = None
        family = structural_family_for_layout(layout)
        family_dir = dest / family
        family_dir.mkdir(parents=True, exist_ok=True)
        outfile = family_dir / f"{layout}_convergence.png"
        suffix = title_suffix or geom_note
        path = plot_layout_convergence_with_geometry(
            layout,
            ladder,
            summary_by.get(layout),
            shell,
            outfile,
            title_suffix=suffix,
            mark_ci_watch=mark_ci_watch,
        )
        if path is not None:
            written.append(path)
    return written


def plot_compare_mitc4_ccx(
    mitc4_dir: str | Path,
    ccx_dir: str | Path,
    outfile: Path,
    *,
    metric: str = "peak_abs_n12",
    pretty: str | None = None,
) -> Path | None:
    """Side-by-side fine GCI bars for MITC4 vs CCX (skin)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _load(d: Path) -> list[dict[str, str]]:
        for name in ("gci_summary.csv", "grid_independence_summary.csv"):
            p = d / name
            if p.is_file():
                return _read_csv(p)
        return []

    def _ladder(d: Path) -> list[dict[str, str]]:
        for name in ("gci_ladder.csv", "grid_independence_ladder.csv"):
            p = d / name
            if p.is_file():
                return _read_csv(p)
        return []

    a = _enrich_summary_from_ladder(_load(Path(mitc4_dir)), _ladder(Path(mitc4_dir)), [metric])
    b = _enrich_summary_from_ladder(_load(Path(ccx_dir)), _ladder(Path(ccx_dir)), [metric])
    a_m = {r["layout"]: r for r in a}
    b_m = {r["layout"]: r for r in b}
    layouts = sorted(set(a_m) & set(b_m))
    if not layouts:
        return None
    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.28 * len(layouts))))
    y = np.arange(len(layouts))
    va = np.asarray([_f(a_m[k], f"{metric}_gci_fine_skin") for k in layouts]) * 100.0
    vb = np.asarray([_f(b_m[k], f"{metric}_gci_fine_skin") for k in layouts]) * 100.0
    h = 0.38
    ax.barh(y - h / 2, np.nan_to_num(va, nan=0.0), height=h, color="#2563eb", label="MITC4")
    ax.barh(y + h / 2, np.nan_to_num(vb, nan=0.0), height=h, color="#7c3aed", label="CCX")
    ax.axvline(10.0, color="#111827", ls="--", lw=1.0)
    ax.set_yticks(y, layouts, fontsize=8)
    ax.invert_yaxis()
    label = pretty or metric
    ax.set_xlabel(f"{label} fine GCI skin [%]")
    ax.set_title(f"MITC4 vs CCX — fine-grid GCI (skin {label})")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return _savefig(fig, outfile)


__all__ = [
    "CI_WATCH_LAYOUTS",
    "plot_compare_mitc4_ccx",
    "plot_gci_fine_bars",
    "plot_grid_independence_study",
    "plot_ladder_small_multiples",
    "plot_last_two_drift_bars",
    "plot_layout_convergence_with_geometry",
    "plot_scalar_gci_bars",
    "plot_study_layout_convergence",
    "shell_from_recipe",
    "structural_family_for_layout",
]
