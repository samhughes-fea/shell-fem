"""Full-blade loading checks on a cantilever tube, against beam theory.

A thin circular tube clamped at z = 0 stands in for a blade: its section properties
are closed-form, so each spanwise load the blade loads module applies has a known
answer. ``tube_blade`` builds it as a ``BladeMesh`` (node sets ``root``, ``tip``,
element set ``skin``, span bays at the mesh sections), and each check runs
``BladeStaticRunner`` and returns a ``BenchmarkResult`` (computed over reference):

=======================  ============================================  ===========================================
check                    load                                          reference
=======================  ============================================  ===========================================
``tube_bending``         uniform force per span f along y             tip deflection f L^4 / (8 EI) + f L^2 / (2 kGA)
``tube_torsion``         uniform torque per span m about z            tip twist m L^2 / (2 G J)
``tube_centrifugal``     rotation about the y axis through the root    tip extension rho_A P Omega^2 L^3 / (3 EA)
=======================  ============================================  ===========================================

with EI = pi R^3 t E, GJ = 2 pi R^3 t G, EA = 2 pi R t E and the thin-tube shear
factor k = 1/2. R = 0.5, t = 0.01, L = 10, aluminium (E = 70 GPa, nu = 0.3).
"""

from __future__ import annotations

import numpy as np

from laminate_constitutive_model.laminate_kinematics import FSDTModel
from shell_fem.pre_processing.load_library.blade_loads import add_sectional_force, add_sectional_torque
from shell_fem.pre_processing.load_library.load_case import LoadCase
from shell_fem.pre_processing.mesh_library.blade import BladeMesh
from shell_fem.pre_processing.mesh_library.builders import _structured
from shell_fem.simulation_runner.blade.blade_static_simulation import BladeStaticRunner
from shell_fem.verification.shell_benchmarks import BenchmarkResult, isotropic_laminate

R, T, L = 0.5, 0.01, 10.0
E, NU, RHO = 70.0e9, 0.3, 2700.0


def tube_blade(element: str = "quad8_curved", n_around: int = 16, n_span: int = 20, centre=(0.0, 0.0),
               **element_options) -> BladeMesh:
    """Cantilever tube along z as a ``BladeMesh``: ``n_around`` elements round, ``n_span`` along.

    ``centre`` is the (x, y) of the tube axis; off the origin, the pitch axis x = y = 0
    lies outside the skin centroid, as on an aerofoil.
    """
    lam = isotropic_laminate(E, NU, T, density=RHO)
    cx, cy = (float(v) for v in centre)

    def point(s, t):
        a = 2.0 * np.pi * s
        return np.array([cx + R * np.cos(a), cy + R * np.sin(a), L * t])

    named = _structured(n_around, n_span, point, lam, FSDTModel(), element, ("", "", "root", "tip"),
                        element_options, periodic_u=True)
    sets = {"skin": list(named.element_sets["all"]), "all": list(named.element_sets["all"])}
    return BladeMesh(named.mesh, dict(named.node_sets), sets, (), np.linspace(0.0, L, n_span + 1))


def _section():
    G = E / (2.0 * (1.0 + NU))
    return dict(EI=np.pi * R ** 3 * T * E, GJ=2.0 * np.pi * R ** 3 * T * G, EA=2.0 * np.pi * R * T * E,
                kGA=0.5 * 2.0 * np.pi * R * T * G)


def tube_bending(element: str = "quad8_curved", n_around: int = 16, n_span: int = 20) -> BenchmarkResult:
    blade = tube_blade(element, n_around, n_span)
    f = 1000.0
    load = add_sectional_force(LoadCase("bending"), blade, lambda z: (0.0, f, 0.0))
    v = BladeStaticRunner(blade, load).execute().get_results()["bending"].tip_translation[1]
    s = _section()
    return BenchmarkResult("tube_bending", element, n_span, blade.mesh.n_dof, v,
                           f * L ** 4 / (8.0 * s["EI"]) + f * L ** 2 / (2.0 * s["kGA"]))


def tube_torsion(element: str = "quad8_curved", n_around: int = 16, n_span: int = 20) -> BenchmarkResult:
    blade = tube_blade(element, n_around, n_span)
    m = 1000.0
    load = add_sectional_torque(LoadCase("torsion"), blade, lambda z: m)
    twist = BladeStaticRunner(blade, load).execute().get_results()["torsion"].tip_twist
    return BenchmarkResult("tube_torsion", element, n_span, blade.mesh.n_dof, twist, m * L ** 2 / (2.0 * _section()["GJ"]))


def tube_centrifugal(element: str = "quad8_curved", n_around: int = 16, n_span: int = 20,
                     omega: float = 3.0) -> BenchmarkResult:
    blade = tube_blade(element, n_around, n_span)
    load = LoadCase("rotation").add_centrifugal(omega, origin=(0.0, 0.0, 0.0), axis=(0.0, 1.0, 0.0))
    w = BladeStaticRunner(blade, load).execute().get_results()["rotation"].tip_translation[2]
    mass_per_length = RHO * T * 2.0 * np.pi * R
    return BenchmarkResult("tube_centrifugal", element, n_span, blade.mesh.n_dof, w,
                           mass_per_length * omega ** 2 * L ** 3 / (3.0 * _section()["EA"]))


PROBLEMS = {"tube_bending": tube_bending, "tube_torsion": tube_torsion, "tube_centrifugal": tube_centrifugal}

__all__ = ["PROBLEMS", "tube_bending", "tube_blade", "tube_centrifugal", "tube_torsion"]
