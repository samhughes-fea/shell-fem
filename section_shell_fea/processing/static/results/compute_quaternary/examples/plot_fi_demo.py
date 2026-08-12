"""
Demo plots for quaternary failure indices (Von Mises, Hashin, extreme-fibre).

Not a pytest. Run from repo root:

  python blade_structure_utilities/fem_utilities/section_shell_fea/processing/static/results/compute_quaternary/examples/plot_fi_demo.py

Writes PNGs under examples/outputs/ for embedding in compute_quaternary/docs/README.md.
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

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_quaternary.compute_extreme_fibre_fi import (
    extreme_fibre_fi,
)
from blade_structure_utilities.material_utilities.failure import (
    hashin_fi,
    hashin_fi_modes,
    von_mises_plane_stress_fi,
)

_OUT = _HERE / "outputs"

# Illustrative strengths [Pa]
_XT, _XC = 800e6, 600e6
_YT, _YC = 40e6, 130e6
_S12 = 70e6
_SIGMA_Y = 250e6  # yield — Von Mises only
_SIGMA_UTS = 400e6  # ultimate tension — extreme-fibre
_SIGMA_UCS = 300e6  # ultimate compression — extreme-fibre
_TAU_US = 150e6  # ultimate shear — extreme-fibre


def plot_von_mises_isotropic(outpath: Path) -> None:
    """Von Mises FI contour in σ11–σ22 (τ12=0); FI=1 ellipse; strength = σ_y."""
    s_max = 1.4 * _SIGMA_Y
    s11 = np.linspace(-s_max, s_max, 220)
    s22 = np.linspace(-s_max, s_max, 220)
    S11, S22 = np.meshgrid(s11, s22)
    tau = np.zeros_like(S11)
    sigma = np.stack([S11, S22, tau], axis=-1)
    fi = von_mises_plane_stress_fi(sigma, np.array(_SIGMA_Y, dtype=float))

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    cf = ax.contourf(S11 / 1e6, S22 / 1e6, fi, levels=np.linspace(0, 2.0, 21), cmap="viridis")
    ax.contour(S11 / 1e6, S22 / 1e6, fi, levels=[1.0], colors=["#C0392B"], linewidths=2.0)
    fig.colorbar(cf, ax=ax, label=r"Von Mises FI $= \sigma_{\mathrm{vm}} / \sigma_y$")
    ax.axhline(0, color="0.85", linewidth=0.7)
    ax.axvline(0, color="0.85", linewidth=0.7)
    ax.set_xlabel(r"$\sigma_{11}$ [MPa] (material 123)")
    ax.set_ylabel(r"$\sigma_{22}$ [MPa]")
    ax.set_title(
        rf"Isotropic track — Von Mises FI ($\sigma_y={_SIGMA_Y/1e6:.0f}$ MPa, mid face, $\tau_{{12}}=0$)"
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_hashin_envelope(outpath: Path) -> None:
    """Hashin envelope FI=1 contour in σ11–σ22 at τ12=0."""
    s11 = np.linspace(-1.2 * _XC, 1.2 * _XT, 240)
    s22 = np.linspace(-1.2 * _YC, 1.2 * _YT, 240)
    S11, S22 = np.meshgrid(s11, s22)
    tau = np.zeros_like(S11)
    sigma = np.stack([S11, S22, tau], axis=-1)
    fi = hashin_fi(sigma, _XT, _XC, _YT, _YC, _S12)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    cf = ax.contourf(S11 / 1e6, S22 / 1e6, fi, levels=np.linspace(0, 2.5, 26), cmap="magma")
    ax.contour(S11 / 1e6, S22 / 1e6, fi, levels=[1.0], colors=["#1E90FF"], linewidths=2.0)
    fig.colorbar(cf, ax=ax, label="Hashin envelope FI (max of 4 modes)")
    ax.axhline(0, color="0.85", linewidth=0.7)
    ax.axvline(0, color="0.85", linewidth=0.7)
    ax.set_xlabel(r"$\sigma_{11}$ [MPa] (ply material 123)")
    ax.set_ylabel(r"$\sigma_{22}$ [MPa]")
    ax.set_title(r"Composite track — Hashin envelope (mid face, $\tau_{12}=0$)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_hashin_mode_breakdown(outpath: Path) -> None:
    """Bar chart of four Hashin mode FIs for one illustrative stress state."""
    sigma = np.array([400e6, 25e6, 30e6], dtype=float)
    ft, fc, mt, mc = hashin_fi_modes(sigma, _XT, _XC, _YT, _YC, _S12)
    labels = ["FT (fibre T)", "FC (fibre C)", "MT (matrix T)", "MC (matrix C)"]
    values = [float(ft), float(fc), float(mt), float(mc)]
    colors = ["#1E90FF", "#8E44AD", "#2E8B57", "#C0392B"]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.2, label="FI = 1 (onset)")
    ax.set_ylabel("Hashin mode FI")
    ax.set_title(
        r"Composite track — mode breakdown (mid face)"
        "\n"
        rf"$\sigma={{[{sigma[0]/1e6:.0f},\,{sigma[1]/1e6:.0f},\,{sigma[2]/1e6:.0f}]}}$ MPa"
    )
    ax.set_ylim(0, max(1.2, 1.15 * max(values)))
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_extreme_fibre_envelope(outpath: Path) -> None:
    """Extreme-fibre FI contour: sign of σ11/σ22 selects σ_uts vs σ_ucs; τ=0."""
    s_max = 1.3 * max(_SIGMA_UTS, _SIGMA_UCS)
    s11 = np.linspace(-s_max, s_max, 240)
    s22 = np.linspace(-s_max, s_max, 240)
    S11, S22 = np.meshgrid(s11, s22)
    tau = np.zeros_like(S11)
    # One surface only for the contour (identical outer=inner here)
    sig = np.stack([S11, S22, tau], axis=-1)  # (ny, nx, 3)
    sig_ext = np.stack([sig, sig], axis=-2)  # (ny, nx, 2, 3)
    # Flatten batch for helper, then reshape
    flat = sig_ext.reshape(-1, 2, 3)
    fi_flat = extreme_fibre_fi(flat, _SIGMA_UTS, _SIGMA_UCS, _TAU_US)
    fi = fi_flat.reshape(S11.shape)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    cf = ax.contourf(S11 / 1e6, S22 / 1e6, fi, levels=np.linspace(0, 2.0, 21), cmap="cividis")
    ax.contour(S11 / 1e6, S22 / 1e6, fi, levels=[1.0], colors=["#C0392B"], linewidths=2.0)
    fig.colorbar(cf, ax=ax, label=r"Extreme-fibre FI (uts / ucs / $\tau_{us}$)")
    ax.axhline(0, color="0.85", linewidth=0.7)
    ax.axvline(0, color="0.85", linewidth=0.7)
    ax.set_xlabel(r"$\sigma_{11}$ [MPa] (free surface)")
    ax.set_ylabel(r"$\sigma_{22}$ [MPa]")
    ax.set_title(
        r"Extreme-fibre track — free surfaces $\pm h/2$"
        "\n"
        rf"$\sigma_{{\mathrm{{uts}}}}={_SIGMA_UTS/1e6:.0f}$, "
        rf"$\sigma_{{\mathrm{{ucs}}}}={_SIGMA_UCS/1e6:.0f}$ MPa "
        rf"($\tau_{{12}}=0$)"
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_vm_vs_extreme_fibre_bending(outpath: Path) -> None:
    """Under pure bending-like σ11(x3), mid-face VM vs free-surface extreme-fibre FI."""
    # Synthetic linear σ11 through thickness (isotropic pure bending sketch)
    h = 2.0e-3
    x3 = np.linspace(-0.5 * h, 0.5 * h, 5)
    # Peak |σ| at free surfaces = 200 MPa
    sigma_peak = 200e6
    sig11 = sigma_peak * (x3 / (0.5 * h))
    # Mid face index 2 of 5 stations
    mid_i = 2
    outer_i, inner_i = -1, 0

    # Von Mises on mid (uniaxial → σ_vm = |σ11|)
    vm_mid = abs(float(sig11[mid_i])) / _SIGMA_Y
    # Extreme-fibre on free surfaces (uniaxial → |σ|/uts or ucs)
    sig_ext = np.array(
        [
            [
                [sig11[outer_i], 0.0, 0.0],
                [sig11[inner_i], 0.0, 0.0],
            ]
        ],
        dtype=float,
    )
    ef = float(extreme_fibre_fi(sig_ext, _SIGMA_UTS, _SIGMA_UCS, _TAU_US)[0])

    labels = [
        r"Von Mises mid" "\n" rf"($\sigma_y={_SIGMA_Y/1e6:.0f}$ MPa)",
        r"Extreme-fibre $\pm h/2$" "\n" rf"(uts/ucs={_SIGMA_UTS/1e6:.0f}/{_SIGMA_UCS/1e6:.0f})",
    ]
    values = [vm_mid, ef]
    colors = ["#1E90FF", "#2E8B57"]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0, label="FI = 1")
    ax.set_ylabel("FI")
    ax.set_title(
        r"Complementarity under bending-like $\sigma_{11}$"
        "\n"
        rf"$|\sigma_{{\mathrm{{mid}}}}|={abs(sig11[mid_i])/1e6:.0f}$ MPa, "
        rf"$|\sigma_{{\pm h/2}}|={sigma_peak/1e6:.0f}$ MPa"
    )
    ax.set_ylim(0, max(1.2, 1.2 * max(values)))
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_governing_track_schematic(outpath: Path) -> None:
    """Bar schematic: Hashin / interlaminar / Von Mises / extreme-fibre demo scalars."""
    hashin_max = 0.82
    interlaminar = 0.45
    von_mises = 0.91
    extreme_fibre = 1.05
    labels = [
        "Hashin in-plane\n(mid face)",
        "Hashin\ninterlaminar",
        "Von Mises\n($\\sigma_y$, mid)",
        "Extreme-fibre\n(uts/ucs/$\\tau_{us}$)",
    ]
    values = [hashin_max, interlaminar, von_mises, extreme_fibre]
    colors = ["#C0392B", "#E67E22", "#1E90FF", "#2E8B57"]
    criteria = ["hashin", "hashin", "von_mises", "extreme_fibre"]
    gov_i = int(np.argmax(values))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    bars[gov_i].set_linewidth(2.0)
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.set_ylabel("FI")
    ax.set_title(
        f"Governing selection (demo) — criterion={criteria[gov_i]!r}, FI={values[gov_i]:.2f}"
    )
    ax.set_ylim(0, 1.3)
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    plot_von_mises_isotropic(_OUT / "von_mises_fi_isotropic.png")
    plot_hashin_envelope(_OUT / "hashin_envelope_composite.png")
    plot_hashin_mode_breakdown(_OUT / "hashin_mode_breakdown.png")
    plot_extreme_fibre_envelope(_OUT / "extreme_fibre_fi_envelope.png")
    plot_vm_vs_extreme_fibre_bending(_OUT / "vm_vs_extreme_fibre_bending.png")
    plot_governing_track_schematic(_OUT / "governing_track_schematic.png")
    print(f"Wrote plots under {_OUT}")


if __name__ == "__main__":
    main()
