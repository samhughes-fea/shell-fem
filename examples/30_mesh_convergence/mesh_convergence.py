"""Mesh convergence: the ASME grid convergence index of full-blade results.

The blade and loads are those of example 29 (a 10 m NACA blade from the kernel's shell
mesh under aero, gravity and rotation, and ULS = 1.35 aero + 1.1 gravity + 1.0 rotation).
It is meshed on a ladder of systematically refined kernel meshes: level k has 10k span
bays and 2k elements across every chordwise region (``shell_mesh(n_span=10k + 1,
n_chord=2k + 1)``), so every element shrinks by the same ratio from one level to the
next. ``blade_convergence_study`` solves ULS on each and reports, for the tip flap and
edge deflection, the tip twist, the root bending moments and the ULS buckling load factor,
the observed order of convergence, the Richardson extrapolated value and the GCI of the
finest mesh (Celik et al. 2008).

A tube ladder checks the procedure against beam theory first: its GCI band on the tip
twist holds m L^2 / (2 G J).
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "29_full_blade_loading"))

import numpy as np

from _shared.plotting import REFERENCE, SERIES, TEXT_SECONDARY, plt, save
from full_blade_loading import ULS, blade, load_cases
from shell_fem import LoadCase
from shell_fem.pre_processing.load_library.blade_loads import add_sectional_force, add_sectional_torque
from shell_fem.simulation_runner.convergence import blade_convergence_study
from shell_fem.verification.blade_benchmarks import E, L, NU, R, T, tube_blade

LEVELS = (2, 3, 4, 6, 8)
TRACKED = (("tip_flap", "tip flap deflection (mm)", 1e3), ("tip_twist", "tip twist (degrees)", np.degrees(1.0)),
           ("buckling", "ULS buckling load factor", 1.0))


def kernel_blade(k):
    return blade(n_span=10 * k + 1, n_chord=2 * k + 1)


def tube_study():
    m = 1000.0

    def loads(b):
        return [add_sectional_torque(add_sectional_force(LoadCase("c"), b, lambda z: (0.0, 1000.0, 0.0)), b,
                                     lambda z: m)]

    results = blade_convergence_study(lambda n: tube_blade("quad4", 2 * n, 2 * n), loads, "c", [4, 6, 9, 13],
                                      atol={"tip_edge": 1e-9, "root_My": 1e-3}).execute().get_results()
    GJ = 2.0 * np.pi * R ** 3 * T * E / (2.0 * (1.0 + NU))
    return results, m * L ** 2 / (2.0 * GJ)


def main():
    warnings.simplefilter("ignore")
    tube, exact = tube_study()
    g = tube.gci("tip_twist")
    print("tube, Quad4, tip twist under a uniform torque per span (beam theory "
          f"{exact:.6e} rad):")
    print(f"  fine {g.fine:.6e}  extrapolated {g.extrapolated:.6e}  p = {g.order:.2f}  GCI {100 * g.gci:.2f} %  "
          f"band [{g.band[0]:.6e}, {g.band[1]:.6e}]")

    study = blade_convergence_study(kernel_blade, load_cases, "ULS", LEVELS, combinations=[ULS], buckling_modes=1)
    results = study.execute().get_results()
    print("\nkernel blade, ULS:")
    print(results.table())
    print("(GCI over the three finest meshes; * oscillatory convergence; ? not reliable: no positive order fits,")
    print(" or the order of the finest triple differs from the next by more than 30 %)")
    n = len(results)
    for name in results.quantities:
        g, g2 = results.gci(name), results.gci(name, range(n - 4, n - 1))
        print(f"  {name:<10} p {g2.order:6.2f} -> {g.order:6.2f}   asymptotic ratio {g.asymptotic_ratio:8.4f}"
              f"   reliable: {results.reliable(name)}")

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.6))
    h = results.h
    for col, (name, label, scale) in enumerate(TRACKED):
        g = results.gci(name)
        ok = results.reliable(name)
        ax = axes[0, col]
        ax.plot(h, scale * results.values(name), "o-", color=SERIES[0], label="kernel mesh ladder")
        if ok:
            ax.axhline(scale * g.extrapolated, color=REFERENCE, ls="--", lw=1.0, label="Richardson extrapolation")
            lo, hi = sorted((scale * g.band[0], scale * g.band[1]))
            ax.errorbar([h[-1]], [scale * g.fine], yerr=[[scale * g.fine - lo], [hi - scale * g.fine]], fmt="none",
                        ecolor=SERIES[1], capsize=4, lw=1.6, label="GCI of the finest mesh")
        ax.set_xlim(0.0, 1.08 * h[0])
        short = label.split(" (")[0]
        short = short[0].upper() + short[1:]
        if ok:
            ax.set_title(f"{short}: GCI {100 * g.gci:.2g} %, p = {g.order:.2g}")
        else:
            ax.set_title(f"{short}: not yet in the asymptotic range")
        ax.set_xlabel("element size h (m)")
        ax.set_ylabel(label)
        ax = axes[1, col]
        if ok:
            error = np.abs(results.values(name) - g.extrapolated) / abs(g.extrapolated)
            keep = error > 0.0
            ax.loglog(h[keep], error[keep], "o-", color=SERIES[0])
            slope = h ** g.order * error[-1] / h[-1] ** g.order
            ax.loglog(h, slope, color=REFERENCE, ls=":", lw=1.0, label=f"slope p = {g.order:.2g}")
            ax.set_title(f"Distance from the extrapolated {label.split(' (')[0]}")
            ax.legend(fontsize=8)
        else:   # no limit to measure from: show the change from each mesh to the next
            v = results.values(name)
            ax.semilogx(h[1:], 100.0 * (v[1:] - v[:-1]) / abs(v[-1]), "o-", color=SERIES[0])
            ax.axhline(0.0, color=TEXT_SECONDARY, lw=0.8)
            ax.set_title(f"Change in {label.split(' (')[0]} from the next coarser mesh")
            ax.set_ylabel("change (% of the finest value)")
        ax.set_xlabel("element size h (m)")
        if ok:
            ax.set_ylabel("relative difference")
    axes[0, 0].legend(fontsize=8)
    fig.text(0.01, 0.005, "Level k: 10k span bays, 2k elements per chordwise region; "
             f"{results.levels[0].n_dof} to {results.levels[-1].n_dof} DOFs. "
             "Example 29 blade and loads, ULS combination.", fontsize=8, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, __file__, "mesh_convergence.png")


if __name__ == "__main__":
    main()
