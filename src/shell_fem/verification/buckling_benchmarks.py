"""Linear buckling reference problems with closed-form answers.

Each function builds one problem for a given element type and mesh density, runs
``LinearBucklingRunner`` and returns a ``BenchmarkResult`` whose ``value`` is the
critical load (in the problem's own units) and whose ``ratio`` is the computed value
over the closed form (1.0 is exact):

=====================  ==============================================  =========================
problem                load                                            reference
=====================  ==============================================  =========================
``plate_compression``  uniaxial compression, simply supported plate    N = (pi^2/b^2) min_m [D11 (mb/a)^2
                       (isotropic or specially orthotropic)              + 2(D12 + 2 D66) + D22 (a/mb)^2]
``plate_shear``        pure shear, simply supported square plate       N = 9.3245 pi^2 D / b^2
``plate_thermal``      uniform heating, edges held in plane            dT = 2 pi^2 D (1 - nu) / (b^2 E alpha t)
``column``             axial tip load, cantilever strip                P = pi^2 E I / (4 L^2)
``cylinder_axial``     axial compression, simply supported cylinder    N = E t^2 / (R sqrt(3 (1 - nu^2)))
=====================  ==============================================  =========================

The closed forms are classical (Kirchhoff) results; the shell elements are
Reissner-Mindlin, so on the thin plates used here they sit 0.1 % or less below them.
The plates are "hard" simply supported: w = 0 and no rotation about the edge normal
(the soft support, which leaves that rotation free, is about 1 % more flexible at
b/t = 100 because of the Mindlin boundary layer).

References: S. P. Timoshenko and J. M. Gere, Theory of Elastic Stability (1961);
the shear coefficient 9.3245 is the converged value of the series solution
(Timoshenko & Gere quote 9.34); R. M. Jones, Mechanics of Composite Materials (1999)
for the orthotropic plate.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from laminate_constitutive_model import HygrothermalState, Laminate
from laminate_constitutive_model.laminate_kinematics import FSDTModel
from shell_fem.pre_processing.containers.boundary_conditions import BoundaryConditions
from shell_fem.pre_processing.containers.mesh import DOF
from shell_fem.pre_processing.load_library.load_case import LoadCase
from shell_fem.pre_processing.mesh_library.builders import CylindricalPanelBuilder, NamedMesh, RectangularPlateBuilder
from shell_fem.processing.buckling.operations.eigen import SolveBucklingEigenproblem
from shell_fem.simulation_runner.buckling.linear_buckling_simulation import LinearBucklingResults, LinearBucklingRunner
from shell_fem.verification.shell_benchmarks import BenchmarkResult, isotropic_laminate

E_AL, NU_AL, ALPHA_AL = 70.0e9, 0.3, 23.0e-6
SHEAR_COEFFICIENT = 9.3245


def _bending_stiffness(named: NamedMesh) -> np.ndarray:
    return named.mesh.elements[0].section.C_ee[3:6, 3:6]


def _simply_supported(named: NamedMesh, bcs: BoundaryConditions) -> BoundaryConditions:
    """Hard simple support on the four edges of a plate in the xy plane."""
    x_edges, y_edges = named.nodes("left") + named.nodes("right"), named.nodes("bottom") + named.nodes("top")
    bcs.fix(set(x_edges + y_edges), [DOF.UZ])
    bcs.fix(x_edges, [DOF.RX])
    return bcs.fix(y_edges, [DOF.RY])


def _run(named: NamedMesh, bcs, load, signs: str = "positive", n_modes: int = 1) -> LinearBucklingResults:
    eigen = SolveBucklingEigenproblem(n_modes, signs=signs)
    return LinearBucklingRunner(named.mesh, bcs, load, eigensolver=eigen).execute().get_results()


def plate_compression(element: str, n: int, laminate: Optional[Laminate] = None, aspect: float = 1.0,
                      **element_options) -> BenchmarkResult:
    """Simply supported a x b plate (b = 0.5, a = aspect b) under uniaxial compression N_x; critical N_x.

    The default laminate is 5 mm of aluminium (b/t = 100). Any laminate with
    D16 = D26 = 0 and B = 0 (cross-ply symmetric, say) has the closed form above,
    minimised over the number of half-waves m along x.
    """
    b = 0.5
    a = aspect * b
    lam = laminate or isotropic_laminate(E_AL, NU_AL, 0.005)
    named = RectangularPlateBuilder(a, b, max(1, round(n * aspect)), n, element=element).build_named(
        lam, FSDTModel(), **element_options)
    bcs = _simply_supported(named, BoundaryConditions())
    bcs.fix(named.nodes("left"), [DOF.UX]).fix([named.mesh.closest_node((0.0, 0.0, 0.0))], [DOF.UY])
    load = LoadCase().add_line_load(named.nodes("right"), (-1.0, 0.0, 0.0))
    result = _run(named, bcs, load)
    D = _bending_stiffness(named)
    m = np.arange(1, 20)
    reference = np.pi ** 2 / b ** 2 * np.min(D[0, 0] * (m * b / a) ** 2 + 2.0 * (D[0, 1] + 2.0 * D[2, 2])
                                             + D[1, 1] * (a / (m * b)) ** 2)
    return BenchmarkResult("plate_compression", element, n, named.mesh.n_dof, result.critical_load_factor, reference)


def plate_shear(element: str, n: int, **element_options) -> BenchmarkResult:
    """Simply supported square aluminium plate (b = 0.5, t = 5 mm) under pure shear N_xy; critical |N_xy|.

    Shear buckling has the same load factor for either sign of the shear, so both
    signs are searched and the lowest magnitude is taken.
    """
    b, t = 0.5, 0.005
    named = RectangularPlateBuilder(b, b, n, n, element=element).build_named(
        isotropic_laminate(E_AL, NU_AL, t), FSDTModel(), **element_options)
    bcs = _simply_supported(named, BoundaryConditions())
    mesh = named.mesh
    bcs.fix([mesh.closest_node((0.0, 0.0, 0.0))], [DOF.UX, DOF.UY]).fix([mesh.closest_node((b, 0.0, 0.0))], [DOF.UY])
    load = (LoadCase().add_line_load(named.nodes("bottom"), (-1.0, 0.0, 0.0))
            .add_line_load(named.nodes("top"), (1.0, 0.0, 0.0))
            .add_line_load(named.nodes("left"), (0.0, -1.0, 0.0))
            .add_line_load(named.nodes("right"), (0.0, 1.0, 0.0)))
    result = _run(named, bcs, load, signs="both")
    D = E_AL * t ** 3 / (12.0 * (1.0 - NU_AL ** 2))
    return BenchmarkResult("plate_shear", element, n, mesh.n_dof, abs(result.critical_load_factor),
                           SHEAR_COEFFICIENT * np.pi ** 2 * D / b ** 2)


def plate_thermal(element: str, n: int, **element_options) -> BenchmarkResult:
    """Simply supported square aluminium plate, edges held in plane, heated uniformly; critical temperature rise.

    The restrained expansion gives N_x = N_y = -E alpha t dT / (1 - nu), and the plate
    buckles at the biaxial load 2 pi^2 D / b^2. The prestress comes through the
    hygrothermal state of the load case, not through nodal loads.
    """
    b, t = 0.5, 0.005
    named = RectangularPlateBuilder(b, b, n, n, element=element).build_named(
        isotropic_laminate(E_AL, NU_AL, t, alpha=ALPHA_AL), FSDTModel(), **element_options)
    bcs = _simply_supported(named, BoundaryConditions())
    edges = set(named.nodes("left") + named.nodes("right") + named.nodes("bottom") + named.nodes("top"))
    bcs.fix(edges, [DOF.UX, DOF.UY])
    load = LoadCase().set_hygrothermal(HygrothermalState(delta_temperature=1.0))
    result = _run(named, bcs, load)
    D = E_AL * t ** 3 / (12.0 * (1.0 - NU_AL ** 2))
    reference = 2.0 * np.pi ** 2 * D / b ** 2 * (1.0 - NU_AL) / (E_AL * ALPHA_AL * t)
    return BenchmarkResult("plate_thermal", element, n, named.mesh.n_dof, result.critical_load_factor, reference)


def column(element: str, n: int, **element_options) -> BenchmarkResult:
    """Cantilever aluminium strip (L = 1, width 0.05, t = 5 mm, nu = 0) under an axial tip load; critical load.

    n elements along the length and max(1, n // 8) across. nu = 0 keeps the strip a
    beam (no anticlastic plate stiffening), so the reference is Euler's.
    """
    L, width, t = 1.0, 0.05, 0.005
    named = RectangularPlateBuilder(L, width, n, max(1, n // 8), element=element).build_named(
        isotropic_laminate(E_AL, 0.0, t), FSDTModel(), **element_options)
    bcs = BoundaryConditions().clamp(named.nodes("left"))
    load = LoadCase().add_line_load(named.nodes("right"), (-1.0 / width, 0.0, 0.0))
    result = _run(named, bcs, load)
    EI = E_AL * width * t ** 3 / 12.0
    return BenchmarkResult("column", element, n, named.mesh.n_dof, result.critical_load_factor,
                           np.pi ** 2 * EI / (4.0 * L ** 2))


def cylinder_axial(element: str, n: int, **element_options) -> BenchmarkResult:
    """Simply supported aluminium cylinder (R = 0.5, L = 0.5, t = 2.5 mm, R/t = 200) under axial compression.

    One octant is meshed (90 degrees of arc, half the length) with n x n elements and
    symmetry on the cut planes, so only modes symmetric about those planes are found;
    the classical load has many modes of nearly the same load, and some of them have
    that symmetry. The critical N_x is compared with the classical E t^2 / (R sqrt(3 (1 - nu^2))).
    """
    R, L, t = 0.5, 0.5, 0.0025
    named = CylindricalPanelBuilder(R, L / 2, n, n, 0.0, 90.0, element=element).build_named(
        isotropic_laminate(E_AL, NU_AL, t), FSDTModel(), **element_options)
    bcs = BoundaryConditions()
    bcs.fix(named.nodes("x0"), [DOF.UY, DOF.UZ])                  # end diaphragm, loaded axially
    bcs.fix(named.nodes("x1"), [DOF.UX, DOF.RY, DOF.RZ])          # plane x = L/2
    bcs.fix(named.nodes("phi0"), [DOF.UZ, DOF.RX, DOF.RY])        # plane z = 0
    bcs.fix(named.nodes("phi1"), [DOF.UY, DOF.RX, DOF.RZ])        # plane y = 0
    load = LoadCase().add_line_load(named.nodes("x0"), (1.0, 0.0, 0.0))  # compresses towards x = L/2
    result = _run(named, bcs, load)
    reference = E_AL * t ** 2 / (R * np.sqrt(3.0 * (1.0 - NU_AL ** 2)))
    return BenchmarkResult("cylinder_axial", element, n, named.mesh.n_dof, result.critical_load_factor, reference)


PROBLEMS: Dict[str, Callable[..., BenchmarkResult]] = {
    "plate_compression": plate_compression,
    "plate_shear": plate_shear,
    "plate_thermal": plate_thermal,
    "column": column,
    "cylinder_axial": cylinder_axial,
}

__all__ = ["PROBLEMS", "SHEAR_COEFFICIENT", "column", "cylinder_axial", "plate_compression", "plate_shear",
           "plate_thermal"]
