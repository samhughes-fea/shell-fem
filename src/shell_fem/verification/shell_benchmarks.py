"""The standard shell obstacle course: reference problems with published answers.

Each function builds one problem for a given element type and mesh density, solves
it with the native solver and returns a ``BenchmarkResult`` whose ``ratio`` is the
computed value over the reference (1.0 is exact). Together they test membrane and
bending behaviour, curved geometry, warped elements and locking:

=====================  ==========================================  ==================
problem                tests                                       reference
=====================  ==========================================  ==================
``scordelis_lo``       membrane-dominated curved shell             MacNeal & Harder (1985)
``pinched_cylinder``   inextensional bending with membrane         Belytschko et al. (1985)
                       locking risk
``pinched_hemisphere`` inextensional bending, rigid rotations,     MacNeal & Harder (1985)
                       doubly curved
``twisted_beam``       warped (non-planar) elements                MacNeal & Harder (1985)
=====================  ==========================================  ==================

References: R. H. MacNeal and R. L. Harder, "A proposed standard set of problems to
test finite element accuracy", Finite Elements in Analysis and Design 1 (1985) 3-20;
T. Belytschko et al., "Stress projection for membrane and shear locking in shell
finite elements", Comput. Methods Appl. Mech. Engrg. 51 (1985) 221-258.

The materials are isotropic single-ply laminates under FSDT; every quantity is in
the problem's own consistent units. Extra keyword arguments (``drilling_factor``,
``warping_correction``, ...) are passed to every element.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np

from laminate_constitutive_model import (ElasticProperties3D, HygrothermalProperties, Laminate, MaterialCard,
                                         StrengthProperties3D)
from laminate_constitutive_model.laminate_kinematics import FSDTModel
from shell_fem.pre_processing.containers.boundary_conditions import BoundaryConditions
from shell_fem.pre_processing.containers.mesh import DOF
from shell_fem.pre_processing.element_library.connectors import add_reference_node, distributing_link
from shell_fem.pre_processing.load_library.load_case import LoadCase
from shell_fem.pre_processing.mesh_library.builders import CylindricalPanelBuilder, NamedMesh, _structured
from shell_fem.simulation_runner.static.linear_static_simulation import StructuralGlobalSolver

ELEMENTS = ("quad4", "tri3", "quad8", "quad8_curved", "tri6")


@dataclass(frozen=True)
class BenchmarkResult:
    problem: str
    element: str
    n: int  # elements along each edge of the mesh
    n_dof: int
    value: float
    reference: float

    @property
    def ratio(self) -> float:
        return self.value / self.reference


def isotropic_laminate(E: float, nu: float, t: float, alpha: float = 0.0) -> Laminate:
    """One isotropic ply of thickness t (thermal expansion ``alpha``, no strength limits)."""
    G = E / (2.0 * (1.0 + nu))
    thermal = HygrothermalProperties(alpha, alpha, alpha) if alpha else None
    card = MaterialCard("isotropic", ElasticProperties3D(E1=E, E2=E, E3=E, v12=nu, v13=nu, v23=nu, G12=G, G13=G, G23=G),
                        StrengthProperties3D(1e30, 1e30, 1e30, 1e30, 1e30), thermal, density=1.0)
    return Laminate.from_angles(card, [0], t)


def _closest(named: NamedMesh, point) -> int:
    return named.mesh.closest_node(point)


def scordelis_lo(element: str, n: int, **element_options) -> BenchmarkResult:
    """Barrel vault under its own weight; vertical deflection at the middle of the free edge.

    R = 25, L = 50, t = 0.25, 80 degree arc, E = 4.32e8, nu = 0, weight 90 per unit
    area. The ends rest on rigid diaphragms (v = w = 0). Full model, n x n mesh.
    Reference 0.3024 (MacNeal & Harder use 0.3086 for the deep-shell limit; 0.3024
    is the shallow value most element papers quote).
    """
    R, L = 25.0, 50.0
    lam = isotropic_laminate(4.32e8, 0.0, 0.25)
    named = CylindricalPanelBuilder(R, L, n, n, 50.0, 130.0, element=element).build_named(lam, FSDTModel(),
                                                                                     **element_options)
    mesh = named.mesh
    bcs = BoundaryConditions().fix(named.nodes("x0") + named.nodes("x1"), [DOF.UY, DOF.UZ])
    bcs.fix([_closest(named, (L / 2, 0.0, R))], [DOF.UX])
    load = LoadCase().add_body_load((0.0, 0.0, -90.0 / 0.25))  # weight per volume = 90 / t
    sol = StructuralGlobalSolver().solve(mesh, bcs, load)
    phi = np.radians(50.0)
    w = sol.node_displacement(_closest(named, (L / 2, R * np.cos(phi), R * np.sin(phi))))[2]
    return BenchmarkResult("scordelis_lo", element, n, mesh.n_dof, -w, 0.3024)


def pinched_cylinder(element: str, n: int, **element_options) -> BenchmarkResult:
    """Cylinder with rigid end diaphragms pinched by two opposite unit loads; deflection under the load.

    R = 300, L = 600, t = 3, E = 3e6, nu = 0.3. One octant is meshed (x from the
    diaphragm to mid-length, 90 degrees of arc) with symmetry on the three cut planes,
    and a quarter of the load. Reference 1.8248e-5.
    """
    R, L = 300.0, 600.0
    lam = isotropic_laminate(3.0e6, 0.3, 3.0)
    named = CylindricalPanelBuilder(R, L / 2, n, n, 0.0, 90.0, element=element).build_named(lam, FSDTModel(),
                                                                                         **element_options)
    mesh = named.mesh
    bcs = BoundaryConditions()
    bcs.fix(named.nodes("x0"), [DOF.UY, DOF.UZ, DOF.RX])          # diaphragm
    bcs.fix(named.nodes("x1"), [DOF.UX, DOF.RY, DOF.RZ])          # plane x = L/2
    bcs.fix(named.nodes("phi0"), [DOF.UZ, DOF.RX, DOF.RY])        # plane z = 0
    bcs.fix(named.nodes("phi1"), [DOF.UY, DOF.RX, DOF.RZ])        # plane y = 0
    top = _closest(named, (L / 2, 0.0, R))
    sol = StructuralGlobalSolver().solve(mesh, bcs, LoadCase().add_nodal_load(top, DOF.UZ, -0.25))
    return BenchmarkResult("pinched_cylinder", element, n, mesh.n_dof, -sol.node_displacement(top)[2], 1.8248e-5)


def pinched_hemisphere(element: str, n: int, **element_options) -> BenchmarkResult:
    """Hemisphere with an 18 degree hole at the pole, pinched at the equator; radial deflection under a load.

    R = 10, t = 0.04, E = 6.825e7, nu = 0.3, loads of 2 alternating in and out every
    90 degrees round the equator. One quadrant is meshed with symmetry on its two
    cut planes (each load point lies on one plane, so it takes half its load).
    Reference 0.094.
    """
    R = 10.0
    lam = isotropic_laminate(6.825e7, 0.3, 0.04)
    theta0 = np.radians(18.0)

    def point(s, t):  # s round the quadrant (azimuth), t from the hole to the equator (polar angle)
        phi = 0.5 * np.pi * s
        theta = theta0 + (0.5 * np.pi - theta0) * t
        return np.array([R * np.sin(theta) * np.cos(phi), R * np.sin(theta) * np.sin(phi), R * np.cos(theta)])

    named = _structured(n, n, point, lam, FSDTModel(), element, ("phi0", "phi90", "hole", "equator"),
                        element_options)
    mesh = named.mesh
    bcs = BoundaryConditions()
    bcs.fix(named.nodes("phi0"), [DOF.UY, DOF.RX, DOF.RZ])        # plane y = 0
    bcs.fix(named.nodes("phi90"), [DOF.UX, DOF.RY, DOF.RZ])       # plane x = 0
    a, b = _closest(named, (R, 0.0, 0.0)), _closest(named, (0.0, R, 0.0))
    bcs.fix([a], [DOF.UZ])                                         # the vertical rigid-body motion
    load = LoadCase().add_nodal_load(a, DOF.UX, 1.0).add_nodal_load(b, DOF.UY, -1.0)
    sol = StructuralGlobalSolver().solve(mesh, bcs, load)
    return BenchmarkResult("pinched_hemisphere", element, n, mesh.n_dof, sol.node_displacement(a)[0], 0.094)


def twisted_beam(element: str, n: int, direction: str = "in_plane", **element_options) -> BenchmarkResult:
    """Cantilever strip twisted through 90 degrees, unit tip load; tip deflection along the load.

    L = 12, width 1.1, t = 0.32, E = 29e6, nu = 0.22. The load acts at the tip
    centre (through a distributing link) in the plane of the tip (its width
    direction, ``"in_plane"``, reference 5.424e-3) or normal to it
    (``"out_of_plane"``, reference 1.754e-3). The mesh has n elements along the
    length and max(1, n // 6) across the width (MacNeal & Harder's 12 x 2).
    """
    L, width = 12.0, 1.1
    lam = isotropic_laminate(29.0e6, 0.22, 0.32)

    def point(s, t):  # s across the width, t along the length
        y = (s - 0.5) * width
        twist = 0.5 * np.pi * t
        return np.array([t * L, y * np.cos(twist), y * np.sin(twist)])

    named = _structured(max(1, n // 6), n, point, lam, FSDTModel(), element, ("e0", "e1", "root", "tip"),
                        element_options)
    mesh = named.mesh
    bcs = BoundaryConditions().clamp(named.nodes("root"))
    ref = add_reference_node(mesh, (L, 0.0, 0.0))
    distributing_link(mesh, bcs, ref, named.nodes("tip"), rotations=False)
    dof, reference = (DOF.UZ, 5.424e-3) if direction == "in_plane" else (DOF.UY, 1.754e-3)
    sol = StructuralGlobalSolver().solve(mesh, bcs, LoadCase().add_nodal_load(ref, dof, 1.0))
    return BenchmarkResult(f"twisted_beam_{direction}", element, n, mesh.n_dof, sol.node_displacement(ref)[int(dof)],
                           reference)


PROBLEMS: Dict[str, Callable[..., BenchmarkResult]] = {
    "scordelis_lo": scordelis_lo,
    "pinched_cylinder": pinched_cylinder,
    "pinched_hemisphere": pinched_hemisphere,
    "twisted_beam_in_plane": lambda e, n, **o: twisted_beam(e, n, "in_plane", **o),
    "twisted_beam_out_of_plane": lambda e, n, **o: twisted_beam(e, n, "out_of_plane", **o),
}


def run_obstacle_course(meshes: Sequence[int] = (4, 8, 16), elements: Sequence[str] = ELEMENTS,
                        problems: Sequence[str] = tuple(PROBLEMS), **element_options) -> List[BenchmarkResult]:
    """Every problem with every element on each mesh density (``element_options`` go to every element)."""
    return [PROBLEMS[p](e, n, **element_options) for p in problems for e in elements for n in meshes]


__all__ = ["BenchmarkResult", "ELEMENTS", "PROBLEMS", "isotropic_laminate", "pinched_cylinder",
           "pinched_hemisphere", "run_obstacle_course", "scordelis_lo", "twisted_beam"]
