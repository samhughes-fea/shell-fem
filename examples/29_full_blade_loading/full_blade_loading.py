"""Full-blade loading: a tapered, twisted blade clamped at the root under spanwise loads.

The blade is built by implicit-geometry-kernel from public NACA 4-digit sections
(no reference-blade data): 10 m long, chord 1.0 m at the root to 0.4 m at the tip,
NACA 0030 to NACA 0015, twisted 10 degrees to 0, with a box spar (two caps, two webs)
and a tip closure. shell-fem meshes nothing itself: it takes the kernel's own shell
mesh (``BladeBuilder.from_shell_mesh``).

Load cases, from ``shell_fem.pre_processing.load_library.blade_loads``:

- aero: a flapwise force per unit span, an edgewise drag of a tenth of it and a nose-down
  pitching moment per unit span (illustrative distributions, not from a rotor code);
- gravity: the blade horizontal, gravity edgewise;
- rotation: 2 rad/s about the rotor axis, 1 m inboard of the root.

``BladeStaticRunner`` solves them and the combination ULS = 1.35 aero + 1.1 gravity +
1.0 rotation, reports root loads, the internal loads at every span position from the shell,
deflection and twist along the span, and the linear
buckling load factor of ULS.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.plotting import DIVERGING, SERIES, TEXT_SECONDARY, plt, save
from implicit_geometry_kernel import Airfoil, BladeGeometry, Station
import implicit_geometry_kernel.mesh as kernel_mesh
from shell_fem import BladeBuilder, FSDTModel, Laminate, LoadCase, LoadCombination, create_default_library
from shell_fem.pre_processing.load_library.blade_loads import (add_sectional_force, add_sectional_torque,
                                                               spanwise)
from shell_fem.simulation_runner.blade import BladeStaticRunner

L = 10.0
CASES = ("aero", "gravity", "rotation", "ULS")
COLOURS = dict(zip(CASES, (SERIES[0], SERIES[1], SERIES[2], SERIES[7])))


def blade(h: float = 0.1, **mesh_options):
    """The kernel's shell mesh of the blade at edge length ``h``, or with the kernel's other mesh options."""
    stations = [Station(0.0, 1.0, Airfoil.naca4("0030"), pitch_axis=0.3, twist=10.0),
                Station(L, 0.4, Airfoil.naca4("0015"), pitch_axis=0.3, twist=0.0)]
    geometry = BladeGeometry(stations, "2D-F-LC-0R", 0.003, pivot_fraction=0.3, spar_cap_thickness=0.008,
                             shear_web_thickness=0.002, web_positions=(0.18, 0.48))
    library = create_default_library()
    glass, carbon = library.get("E-glass/epoxy"), library.get("T300/5208")
    laminates = {"skin": Laminate.from_angles(glass, [0, 45, -45, -45, 45, 0], 0.5e-3),
                 "web": Laminate.from_angles(glass, [45, -45, -45, 45], 0.5e-3),
                 "cap": Laminate.from_angles(carbon, [0] * 16, 0.5e-3)}
    return BladeBuilder.from_shell_mesh(kernel_mesh.shell_mesh(geometry, **(mesh_options or {"h": h})), geometry).build(laminates, FSDTModel())


def aero_distributions():
    """The illustrative aero force (N/m) and pitching moment (N m/m) per unit span."""
    s = np.linspace(0.0, 1.0, 11)
    flap = 2500.0 * np.sin(0.5 * np.pi * s) * (1.0 - 0.6 * s ** 4)        # N/m
    chord = 1.0 - 0.6 * s
    force = spanwise(L * s, np.column_stack([-0.1 * flap, flap, 0.0 * s]))
    torque = spanwise(L * s, -0.05 * chord * flap)                         # N m/m, nose down
    return force, torque


def load_cases(b):
    """aero, gravity and rotation on blade ``b`` (the aero loads are spread over its own span bays)."""
    force, torque = aero_distributions()
    aero = add_sectional_torque(add_sectional_force(LoadCase("aero"), b, force), b, torque)
    gravity = LoadCase("gravity").add_body_load((-9.81, 0.0, 0.0))
    rotation = LoadCase("rotation").add_centrifugal(2.0, origin=(0.0, 0.0, -1.0), axis=(0.0, 1.0, 0.0))
    return [aero, gravity, rotation]


ULS = LoadCombination("ULS", {"aero": 1.35, "gravity": 1.1, "rotation": 1.0})


def main():
    warnings.simplefilter("ignore")
    b = blade()
    print(f"kernel shell mesh: {b.mesh.n_elements} elements, {b.mesh.n_dof} DOFs")
    force, torque = aero_distributions()
    results = BladeStaticRunner(b, load_cases(b), [ULS], buckling_modes=1,
                                buckling_cases=["ULS"]).execute().get_results()

    print(f"\n{'case':<10}{'Fx':>10}{'Fy':>10}{'Fz':>10}{'Mx':>11}{'My':>11}{'Mz':>11}"
          f"{'tip u':>10}{'tip v':>10}{'twist':>9}{'residual':>10}{'cuts':>10}")
    for name in CASES:
        r = results[name]
        print(f"{name:<10}" + "".join(f"{v / 1e3:10.2f}" for v in r.root_load[:3])
              + "".join(f"{v / 1e3:11.2f}" for v in r.root_load[3:])
              + f"{r.tip_translation[0]:10.4f}{r.tip_translation[1]:10.4f}{np.degrees(r.tip_twist):9.3f}"
              + f"{np.abs(r.equilibrium_residual).max() / np.abs(r.applied_load).max():10.1e}"
              + f"{r.section_loads.relative_residual:10.1e}")
    print("(root loads in kN and kN m about the pitch axis at the root; tip in m; twist in degrees;")
    print(" cuts: largest section-cut residual over all span positions)")
    print("\nflapwise bending moment Mx (kN m) at each span position, ULS: shell cut / applied from the tip")
    cut = results["ULS"].section_loads
    for z, mi, ma in list(zip(cut.stations, cut.internal[:, 3], -cut.applied[:, 3]))[::10]:
        print(f"  z = {z:5.2f}  {mi / 1e3:9.3f}  {ma / 1e3:9.3f}")
    buckling = results["ULS"].buckling
    print(f"\nULS linear buckling load factor: {buckling.critical_load_factor:.2f}")

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.6))
    ax = axes[0, 0]
    zz = np.linspace(0.0, L, 101)
    ax.plot(zz, [force(z)[1] / 1e3 for z in zz], color=SERIES[0], label="flapwise force (kN/m)")
    ax.plot(zz, [force(z)[0] / 1e3 for z in zz], color=SERIES[1], label="edgewise force (kN/m)")
    ax.plot(zz, [torque(z) / 1e3 for z in zz], color=SERIES[2], label="pitching moment (kN m/m)")
    ax.axhline(0.0, color=TEXT_SECONDARY, lw=0.8)
    ax.set_title("Aero load per unit span")
    ax.set_xlabel("span z (m)")
    ax.legend(fontsize=8)

    for ax, k, title in ((axes[0, 1], 1, "Flapwise deflection"), (axes[0, 2], 0, "Edgewise deflection")):
        for name in CASES:
            d = results[name].deflection
            ax.plot(d.stations, 1e3 * d.translation[:, k], color=COLOURS[name], label=name,
                    ls="--" if name == "ULS" else "-")
        ax.set_title(title)
        ax.set_xlabel("span z (m)")
        ax.set_ylabel("section translation (mm)")
    axes[0, 1].legend(fontsize=8)

    ax = axes[1, 0]
    for name in CASES:
        d = results[name].deflection
        ax.plot(d.stations, np.degrees(d.twist), color=COLOURS[name], label=name, ls="--" if name == "ULS" else "-")
    ax.set_title("Section twist about z")
    ax.set_xlabel("span z (m)")
    ax.set_ylabel("twist (degrees)")

    ax = axes[1, 1]
    for name in CASES:
        cut = results[name].section_loads
        ax.plot(cut.stations, -cut.applied[:, 3] / 1e3, color=COLOURS[name], ls="--" if name == "ULS" else "-",
                label=name)
        ax.plot(cut.stations[::6], cut.internal[::6, 3] / 1e3, "o", color=COLOURS[name], ms=4, mfc="none")
    ax.set_title("Flapwise bending moment along the span")
    ax.set_xlabel("span z (m)")
    ax.set_ylabel("Mx (kN m)")
    ax.text(0.98, 0.95, "lines: applied loads integrated from the tip\ncircles: shell stresses at the cuts (every 6th shown)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=TEXT_SECONDARY)

    ax = axes[1, 2]
    mode = buckling[0].displacement_array()
    xyz = b.mesh.coordinates_array()
    suction = sorted({n for e in b.elements("skin_suction_surface") for n in b.mesh.element(e).node_ids})
    idx = [b.mesh.node_ids.index(n) for n in suction]
    w = mode[idx, 1]
    sc = ax.scatter(xyz[idx, 2], xyz[idx, 0], c=w, cmap=DIVERGING, vmin=-1.0, vmax=1.0, s=12, linewidths=0)
    ax.set_title(f"ULS buckling mode, suction side (load factor {buckling.critical_load_factor:.2f})")
    ax.set_xlabel("span z (m)")
    ax.set_ylabel("chord x (m)")
    ax.grid(False)
    fig.colorbar(sc, ax=ax, shrink=0.85, label="flapwise mode displacement")
    fig.tight_layout()
    save(fig, __file__, "full_blade_loading.png")


if __name__ == "__main__":
    main()
