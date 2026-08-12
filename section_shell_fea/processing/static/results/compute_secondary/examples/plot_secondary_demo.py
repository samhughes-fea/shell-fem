"""
Demo plots for secondary shell panel resultants and panel nesting.

Not a pytest. Run from repo root:

  python blade_structure_utilities/fem_utilities/section_shell_fea/processing/static/results/compute_secondary/examples/plot_secondary_demo.py

Writes PNGs under examples/outputs/ for embedding in compute_secondary/docs/README.md.
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

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
    ShellPanelResultants,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_secondary.panel_row_nesting import (
    nest_resultant_rows_by_panel,
)

_OUT = _HERE / "outputs"

_PANEL_COLORS = {0: "#1E90FF", 1: "#C0392B", 2: "#2E8B57"}
_PANEL_LABELS = {0: "skin", 1: "spar", 2: "web"}


def _row(
    *,
    panel_index: int,
    station_index: int,
    N11: float,
    M11: float,
    Q1: float | None = None,
    Q2: float | None = None,
    thickness_m: float = 0.008,
) -> ShellPanelResultants:
    return ShellPanelResultants(
        N11=N11,
        N22=0.1 * N11,
        N12=0.05 * abs(N11),
        M11=M11,
        M22=0.2 * M11,
        M12=0.0,
        Q1=Q1,
        Q2=Q2,
        thickness_m=thickness_m,
        sigma_11_pa=N11 / max(thickness_m, 1e-30),
        tau_12_pa=0.05 * abs(N11),
        panel_label=_PANEL_LABELS[panel_index],
        panel_index=panel_index,
        station_index=station_index,
    )


def build_synthetic_rows_scrambled() -> list[ShellPanelResultants]:
    """Flat list intentionally *not* panel-contiguous (mimics mixed stiffness order)."""
    # station patterns per panel
    skin = [
        _row(panel_index=0, station_index=i, N11=1.0e5 + 2.0e4 * i, M11=50.0 + 5.0 * i, Q1=800.0, Q2=200.0)
        for i in range(5)
    ]
    spar = [
        _row(panel_index=1, station_index=i, N11=2.5e5 - 1.0e4 * i, M11=120.0 - 8.0 * i, Q1=1500.0, Q2=0.0)
        for i in range(4)
    ]
    web = [
        _row(panel_index=2, station_index=i, N11=8.0e4 + 5.0e3 * i, M11=20.0 + 3.0 * i, Q1=None, Q2=None)
        for i in range(3)
    ]
    # Interleave panels so nesting effect is visible
    flat: list[ShellPanelResultants] = []
    for i in range(5):
        if i < len(skin):
            flat.append(skin[i])
        if i < len(spar):
            flat.append(spar[i])
        if i < len(web):
            flat.append(web[i])
    return flat


def plot_panel_nesting_order(rows: list[ShellPanelResultants], outpath: Path) -> None:
    nested = nest_resultant_rows_by_panel(rows, n_panels=3)
    flat_after = [r for panel in nested for r in panel]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    for ax, seq, title in (
        (axes[0], rows, "Before nest (flat order)"),
        (axes[1], flat_after, "After nest_resultant_rows_by_panel"),
    ):
        xs = np.arange(len(seq))
        colors = [_PANEL_COLORS[int(r.panel_index)] for r in seq]
        ax.scatter(xs, [int(r.panel_index) for r in seq], c=colors, s=55, zorder=3)
        for i, r in enumerate(seq):
            ax.annotate(
                f"s{int(r.station_index)}",
                (i, int(r.panel_index)),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=7,
            )
        ax.set_xlabel("flat index")
        ax.set_yticks([0, 1, 2], [_PANEL_LABELS[i] for i in range(3)])
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("panel_index")
    fig.suptitle("Secondary panel nesting (station_index sorted within panel)", fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_resultant_by_panel(
    rows: list[ShellPanelResultants],
    *,
    attr: str,
    ylabel: str,
    title: str,
    outpath: Path,
    scale: float = 1.0,
) -> None:
    nested = nest_resultant_rows_by_panel(rows, n_panels=3)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for pi, panel_rows in enumerate(nested):
        if not panel_rows:
            continue
        xs = [int(r.station_index) for r in panel_rows]
        ys = [float(getattr(r, attr)) / scale for r in panel_rows]
        ax.plot(
            xs,
            ys,
            "o-",
            color=_PANEL_COLORS[pi],
            label=_PANEL_LABELS[pi],
            linewidth=1.5,
            markersize=6,
        )
    ax.set_xlabel("station_index (within panel)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_qx_qy_pass_through(rows: list[ShellPanelResultants], outpath: Path) -> None:
    nested = nest_resultant_rows_by_panel(rows, n_panels=3)
    labels: list[str] = []
    qx_vals: list[float] = []
    qy_vals: list[float] = []
    colors: list[str] = []
    present: list[bool] = []
    for pi, panel_rows in enumerate(nested):
        for r in panel_rows:
            labels.append(f"{_PANEL_LABELS[pi]}:s{int(r.station_index)}")
            has = r.Q1 is not None and r.Q2 is not None
            present.append(has)
            qx_vals.append(float(r.Q1) if r.Q1 is not None else 0.0)
            qy_vals.append(float(r.Q2) if r.Q2 is not None else 0.0)
            colors.append(_PANEL_COLORS[pi])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    w = 0.38
    ax.bar(x - w / 2, qx_vals, width=w, color=colors, alpha=0.85, label="Q1 [N/m]")
    ax.bar(x + w / 2, qy_vals, width=w, color=colors, alpha=0.45, edgecolor="black", linewidth=0.4, label="Q2 [N/m]")
    for i, ok in enumerate(present):
        if not ok:
            ax.axvspan(i - 0.45, i + 0.45, color="0.85", alpha=0.5, zorder=0)
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Transverse shear resultant [N/m]")
    ax.set_title("Optional Q1/Q2 pass-through (hatched grey = absent / RESERVED)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    rows = build_synthetic_rows_scrambled()
    plot_panel_nesting_order(rows, _OUT / "panel_nesting_order.png")
    plot_resultant_by_panel(
        rows,
        attr="N11",
        ylabel=r"$N_x$ [N/m] (strip axis 1)",
        title=r"Secondary elemental $N_x$ by panel (after nesting)",
        outpath=_OUT / "resultants_Nx_by_panel.png",
    )
    plot_resultant_by_panel(
        rows,
        attr="M11",
        ylabel=r"$M_x$ [N·m/m] (strip axis 1)",
        title=r"Secondary elemental $M_x$ by panel (after nesting)",
        outpath=_OUT / "resultants_Mx_by_panel.png",
    )
    plot_qx_qy_pass_through(rows, _OUT / "qx_qy_pass_through.png")
    print(f"Wrote plots under {_OUT}")


if __name__ == "__main__":
    main()
