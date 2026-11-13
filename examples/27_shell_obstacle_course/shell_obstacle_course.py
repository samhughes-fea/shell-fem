"""The shell obstacle course: every shell element on the standard reference problems.

Scordelis-Lo roof, pinched cylinder, pinched hemisphere and the twisted beam (both
load directions), from MacNeal & Harder (1985) and Belytschko et al. (1985), each
on four meshes. The problems live in ``shell_fem.verification.shell_benchmarks``.

Panels 1-5: computed / reference against degrees of freedom. Panel 6: the twisted
beam on the 12 x 2 mesh with the flat elements, with and without the warping
correction and with a stiffer drilling penalty.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.plotting import REFERENCE, SERIES, TEXT_SECONDARY, plt, save
from shell_fem.verification.shell_benchmarks import PROBLEMS, twisted_beam

ELEMENTS = {"Tri3": "tri3", "Quad4": "quad4", "Quad8": "quad8", "Tri6": "tri6", "Curved Quad8": "quad8_curved"}
COLOURS = {"Tri3": SERIES[0], "Quad4": SERIES[1], "Quad8": SERIES[2], "Tri6": SERIES[3], "Curved Quad8": SERIES[4]}
MESHES = (4, 8, 16, 32)
TITLES = {
    "scordelis_lo": "Scordelis-Lo roof",
    "pinched_cylinder": "Pinched cylinder",
    "pinched_hemisphere": "Pinched hemisphere",
    "twisted_beam_in_plane": "Twisted beam, in-plane load",
    "twisted_beam_out_of_plane": "Twisted beam, out-of-plane load",
}


def main():
    warnings.simplefilter("ignore")  # the coarse flat meshes are warped on purpose
    results = {}
    print(f"{'problem':<28}{'element':<14}" + "".join(f"{n:>9}" for n in MESHES))
    for problem in PROBLEMS:
        for name, element in ELEMENTS.items():
            runs = [PROBLEMS[problem](element, n) for n in MESHES]
            results[problem, name] = runs
            print(f"{problem:<28}{name:<14}" + "".join(f"{r.ratio:9.4f}" for r in runs))

    variants = {
        "no warping correction": dict(warping_correction=False),
        "warping correction (default)": {},
        "and drilling 0.1 A66": dict(drilling_factor=0.1),
    }
    twisted = {(label, name, d): twisted_beam(ELEMENTS[name], 12, d, **opts).ratio
               for label, opts in variants.items() for name in ("Quad4", "Quad8")
               for d in ("in_plane", "out_of_plane")}
    print("\nTwisted beam, 12 x 2 mesh, flat elements")
    for key, value in twisted.items():
        print(f"  {key[1]:<6} {key[2]:<13} {key[0]:<30} {value:.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.6))
    for ax, problem in zip(axes.flat, PROBLEMS):
        for name in ELEMENTS:
            runs = results[problem, name]
            style = dict(ls=(0, (4, 2)), marker="s", mfc="none") if name in ("Tri6", "Curved Quad8") else dict(marker="o")
            ax.plot([r.n_dof for r in runs], [r.ratio for r in runs], color=COLOURS[name], label=name, **style)
        ax.axhline(1.0, color=REFERENCE, ls="--", lw=1.0)
        ax.set_xscale("log")
        ax.set_ylim(0.0, 2.6 if problem.startswith("twisted") else 1.5)
        ax.set_title(TITLES[problem])
        ax.set_xlabel("degrees of freedom")
        ax.set_ylabel("computed / reference")
    axes.flat[0].legend(loc="lower right")

    ax = axes.flat[5]
    width, x = 0.13, np.arange(2)
    for k, (label, colour) in enumerate(zip(variants, (SERIES[7], SERIES[1], SERIES[5]))):
        for j, name in enumerate(("Quad4", "Quad8")):
            values = [twisted[label, name, d] for d in ("in_plane", "out_of_plane")]
            ax.bar(x + (3 * j + k - 2.5) * width, values, width, color=colour, alpha=1.0 if j == 0 else 0.55,
                   label=label if j == 0 else None)
    ax.axhline(1.0, color=REFERENCE, ls="--", lw=1.0)
    ax.set_xticks(x, ["in-plane load", "out-of-plane load"])
    ax.set_ylabel("computed / reference")
    ax.set_title("Twisted beam, flat elements, 12 x 2")
    ax.text(0.02, 0.97, "solid: Quad4   faded: Quad8", transform=ax.transAxes, va="top", fontsize=7.5,
            color=TEXT_SECONDARY)
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, __file__, "shell_obstacle_course.png")


if __name__ == "__main__":
    main()
