"""Ply failure: the Tsai-Wu index of every ply of a full blade, mapped over its surface.

The blade and loads are those of example 29 (a 10 m NACA blade from the kernel's shell
mesh under aero, gravity and rotation, and ULS = 1.35 aero + 1.1 gravity + 1.0 rotation).
``BladeStaticRunner(..., failure_criterion=TsaiWuCriterion())`` evaluates the criterion on
every ply of every shell element at bottom, mid and top of the ply and at each recovery
point, keeps the largest index per element and ply, and envelopes the cases. The figure
maps the governing index and ply over both skin surfaces, cuts the most loaded section,
and plots the worst index of each region along the span.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "29_full_blade_loading"))

import numpy as np

from _shared.plotting import SEQUENTIAL, SERIES, TEXT_SECONDARY, plt, save
from full_blade_loading import ULS, blade, load_cases
from shell_fem import TsaiWuCriterion
from shell_fem.post_processing.plots import section_fringe, surface_fringe
from shell_fem.simulation_runner.blade import BladeStaticRunner

ROOT_ZONE = 0.5   # m: the clamp holds the root section rigid, which raises local stresses there
REGIONS = (("spar_cap_suction_surface", "suction cap"), ("spar_cap_pressure_surface", "pressure cap"),
           ("skin_suction_surface", "suction skin"), ("skin_pressure_surface", "pressure skin"), ("webs", "webs"))


def main():
    warnings.simplefilter("ignore")
    b = blade()
    results = BladeStaticRunner(b, load_cases(b), [ULS], failure_criterion=TsaiWuCriterion()).execute().get_results()

    print(f"{'case':<10}{'max index':>11}{'reserve':>9}{'element':>9}{'ply':>5}  mode                 at (x, y, z) m")
    for name in results:
        w = results[name].failure.worst()
        print(f"{name:<10}{w['index']:11.4f}{w['strength_ratio']:9.2f}{w['element_id']:9d}{w['ply']:5d}  "
              f"{w['mode']:<20} ({w['location'][0]:.3f}, {w['location'][1]:.3f}, {w['location'][2]:.2f})")
    env = results.failure_envelope()
    field = env.field
    away = [int(e) for e, c in zip(field.element_ids, field.centroid[:, 2]) if c > ROOT_ZONE]
    w = field.worst(away)
    print(f"\nworst beyond z = {ROOT_ZONE:g} m (clear of the clamp): index {w['index']:.4f}, reserve "
          f"{w['strength_ratio']:.2f}, ply {w['ply']}, at z = {w['location'][2]:.2f} m")
    print("\nworst index by region (envelope of all cases):")
    for key, label in REGIONS:
        w = field.worst(b.elements(key))
        rows = field.for_elements([w["element_id"]])
        print(f"  {label:<14}{w['index']:8.4f}  reserve {w['strength_ratio']:6.2f}  ply {w['ply']:2d}  {w['mode']:<18}"
              f"  z = {w['location'][2]:.2f} m  case {env.case[rows[0], w['ply']]}")

    ids, index = field.element_ids, field.max_index
    vmax = float(np.nanmax(index))
    fig = plt.figure(figsize=(13.0, 8.2))
    grid = fig.add_gridspec(3, 3, height_ratios=(1.0, 1.0, 1.1))
    for row, (surface, title) in enumerate((("skin_suction_surface", "suction side"),
                                            ("skin_pressure_surface", "pressure side"))):
        ax = fig.add_subplot(grid[row, :2])
        shown = b.elements(surface) + b.elements(surface.replace("skin", "spar_cap"))
        fringe = surface_fringe(ax, b.mesh, ids, index, show=shown, cmap=SEQUENTIAL, vmin=0.0, vmax=vmax)
        ax.set_title(f"Governing Tsai-Wu index over all cases, {title}")
        ax.set_xlabel("span z (m)")
        ax.set_ylabel("chord x (m)")
        ax.grid(False)
        fig.colorbar(fringe, ax=ax, shrink=0.9, label="max ply index")

    away = [int(e) for e, c in zip(ids, field.centroid[:, 2]) if c > ROOT_ZONE]
    worst = field.worst(away)
    ax = fig.add_subplot(grid[0:2, 2])
    z_cut = worst["location"][2]
    cut = section_fringe(ax, b.mesh, ids, index, z_cut, cmap=SEQUENTIAL, vmin=0.0, vmax=vmax)
    ax.plot(*worst["location"][:2], "x", color=SERIES[7], ms=10, mew=2)
    ax.set_title(f"Section at z = {z_cut:.2f} m (x: worst point beyond {ROOT_ZONE:g} m)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(False)
    fig.colorbar(cut, ax=ax, orientation="horizontal", pad=0.15, label="max ply index")

    ax = fig.add_subplot(grid[2, :2])
    edges = np.asarray(b.span_positions)
    mids = 0.5 * (edges[:-1] + edges[1:])
    for colour, (key, label) in zip(SERIES, REGIONS):
        rows = field.for_elements(b.elements(key))
        z = field.centroid[rows, 2]
        bay = np.clip(np.searchsorted(edges, z) - 1, 0, mids.size - 1)
        peak = np.full(mids.size, np.nan)
        for k in np.unique(bay):
            peak[k] = np.nanmax(index[rows][bay == k])
        ax.plot(mids, peak, color=colour, label=label)
    ax.set_title("Worst ply index in each span bay, by region")
    ax.set_xlabel("span z (m)")
    ax.set_ylabel("max Tsai-Wu index")
    ax.legend(fontsize=8, ncol=5)

    ax = fig.add_subplot(grid[2, 2])
    rows = field.for_elements(b.elements("skin") + b.elements("caps"))
    plies, counts = np.unique(field.governing_ply[rows], return_counts=True)
    ax.bar(plies, counts, color=SERIES[0])
    ax.set_title("Governing ply (skin and caps)")
    ax.set_xlabel("ply (0 = innermost)")
    ax.set_ylabel("elements")
    fig.text(0.01, 0.005, f"Example 29 blade, kernel mesh h = 0.1 m ({b.mesh.n_elements} elements). Tsai-Wu at bottom, "
             "mid and top of every ply; envelope of aero, gravity, rotation and ULS.", fontsize=8,
             color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, __file__, "ply_failure.png")


if __name__ == "__main__":
    main()
