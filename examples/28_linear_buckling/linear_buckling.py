"""Linear buckling of every shell element against classical closed forms.

Simply supported plates in compression (aluminium and a cross-ply laminate), in shear
and heated with the edges held; a cantilever column; a cylinder in axial compression.
The problems live in ``shell_fem.verification.buckling_benchmarks``.

Panels 1-5: computed / closed-form critical load against degrees of freedom.
Panel 6: the first shear buckling mode of the square plate.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from matplotlib.ticker import NullFormatter

from _shared.plotting import DIVERGING, REFERENCE, SERIES, plt, save
from shell_fem import (BoundaryConditions, DOF, FSDTModel, Laminate, LinearBucklingRunner, LoadCase,
                       RectangularPlateBuilder, SolveBucklingEigenproblem, create_default_library)
from shell_fem.verification.buckling_benchmarks import PROBLEMS, plate_compression
from shell_fem.verification.shell_benchmarks import isotropic_laminate

ELEMENTS = {"Tri3": "tri3", "Quad4": "quad4", "Quad8": "quad8", "Tri6": "tri6", "Curved Quad8": "quad8_curved"}
COLOURS = {"Tri3": SERIES[0], "Quad4": SERIES[1], "Quad8": SERIES[2], "Tri6": SERIES[3], "Curved Quad8": SERIES[4]}
MESHES = {"plate_compression": (4, 8, 16), "cross_ply": (4, 8, 16), "plate_shear": (4, 8, 16),
          "plate_thermal": (4, 8, 16), "column": (4, 8, 16), "cylinder_axial": (8, 12, 16)}
TITLES = {
    "plate_compression": "Aluminium plate, uniaxial compression",
    "cross_ply": "[0/90]s plate, a/b = 2, compression",
    "plate_shear": "Aluminium plate, shear",
    "plate_thermal": "Aluminium plate, heated, edges held",
    "column": "Cantilever column",
    "cylinder_axial": "Cylinder R/t = 200, axial compression",
}
PANELS = ("plate_compression", "cross_ply", "plate_shear", "column", "cylinder_axial")


def main():
    warnings.simplefilter("ignore")
    cross_ply = Laminate.from_angles(create_default_library().get("T300/5208"), [0, 90, 90, 0], 0.25e-3)
    problems = dict(PROBLEMS, cross_ply=lambda e, n: plate_compression(e, n, cross_ply, aspect=2.0))
    results = {}
    print(f"{'problem':<20}{'element':<14}" + "".join(f"{'n=' + str(n):>18}" for n in (4, 8, 16)))
    for problem in TITLES:
        for name, element in ELEMENTS.items():
            if problem == "cylinder_axial" and name == "Tri3":
                continue  # far too slow to converge to be worth the run time
            runs = [problems[problem](element, n) for n in MESHES[problem]]
            results[problem, name] = runs
            print(f"{problem:<20}{name:<14}" + "".join(f"{r.ratio:>12.4f} ({r.n:>2d})" for r in runs))
        print(f"{'':<20}{'reference':<14}{runs[0].reference:>18.6g}")

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.6))
    for ax, problem in zip(axes.flat, PANELS):
        for name in ELEMENTS:
            if (problem, name) not in results:
                continue
            runs = results[problem, name]
            style = dict(ls=(0, (4, 2)), marker="s", mfc="none") if name in ("Tri6", "Curved Quad8") else dict(marker="o")
            ax.plot([r.n_dof for r in runs], [r.ratio for r in runs], color=COLOURS[name], label=name, **style)
        ax.axhline(1.0, color=REFERENCE, ls="--", lw=1.0)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(0.9, 1.3)
        ax.set_title(TITLES[problem])
        ax.set_xlabel("degrees of freedom")
        ax.set_ylabel("computed / closed form")
    axes.flat[0].legend(loc="upper right")

    # first shear buckling mode of the square plate
    b = 0.5
    named = RectangularPlateBuilder(b, b, 16, 16, element="quad8").build_named(
        isotropic_laminate(70e9, 0.3, 0.005), FSDTModel())
    mesh = named.mesh
    x_edges, y_edges = named.nodes("left") + named.nodes("right"), named.nodes("bottom") + named.nodes("top")
    bcs = BoundaryConditions().fix(set(x_edges + y_edges), [DOF.UZ]).fix(x_edges, [DOF.RX]).fix(y_edges, [DOF.RY])
    bcs.fix([mesh.closest_node((0, 0, 0))], [DOF.UX, DOF.UY]).fix([mesh.closest_node((b, 0, 0))], [DOF.UY])
    load = (LoadCase().add_line_load(named.nodes("bottom"), (-1.0, 0, 0)).add_line_load(named.nodes("top"), (1.0, 0, 0))
            .add_line_load(named.nodes("left"), (0, -1.0, 0)).add_line_load(named.nodes("right"), (0, 1.0, 0)))
    mode = LinearBucklingRunner(mesh, bcs, load, eigensolver=SolveBucklingEigenproblem(1, signs="both")) \
        .execute().get_results()[0]
    xy, w = mesh.coordinates_array()[:, :2], mode.displacement_array()[:, 2]
    ax = axes.flat[5]
    levels = np.linspace(-1.0, 1.0, 21)
    tc = ax.tricontourf(xy[:, 0], xy[:, 1], w, levels=levels, cmap=DIVERGING)
    ax.tricontour(xy[:, 0], xy[:, 1], w, levels=levels[::2], colors="white", linewidths=0.4)
    ax.set_aspect("equal")
    ax.set_title(f"First shear buckling mode, N_xy = {abs(mode.load_factor) / 1e3:.1f} kN/m")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(False)
    fig.colorbar(tc, ax=ax, shrink=0.85, label="w (largest translation = 1)", ticks=[-1, -0.5, 0, 0.5, 1])
    fig.tight_layout()
    save(fig, __file__, "linear_buckling.png")


if __name__ == "__main__":
    main()
