"""Plot ASME/Roache GCI grid-independence study outputs (ladder + summary CSVs).

Plots every finite primary scalar and secondary / tertiary / quaternary kind
peak present in the ladder (not only |N12|).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from .asme_gci import gci_triplet
from .cascade_mesh_convergence import (
    KIND_PEAK_METRICS,
    PEAK_METRIC_LABELS,
    QUATERNARY_KIND_METRICS,
)

KINDS = ("skin", "cap", "web")
KIND_COLORS = {"skin": "#2563eb", "cap": "#dc2626", "web": "#059669"}

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
    "plot_compare_mitc4_ccx",
    "plot_gci_fine_bars",
    "plot_grid_independence_study",
    "plot_ladder_small_multiples",
    "plot_last_two_drift_bars",
    "plot_scalar_gci_bars",
]
