"""Linear buckling: element geometric stiffness, the buckling stages and closed-form benchmarks.

Tolerances are the accuracy each element reaches on the given mesh; the convergence
tables and figures are in examples/28_linear_buckling.
"""

import warnings

import numpy as np
import pytest

from shell_fem import (BoundaryConditions, DOF, FEAMesh, FSDTModel, Laminate, LoadCase, RectangularPlateBuilder,
                       StructuralGlobalSolver)
from shell_fem.processing.buckling.operations import SolveBucklingEigenproblem
from shell_fem.simulation_runner.buckling import LinearBucklingRunner
from shell_fem.verification.buckling_benchmarks import (column, cylinder_axial, plate_compression, plate_shear,
                                                        plate_thermal)
from shell_fem.verification.shell_benchmarks import isotropic_laminate

WARPED_Q4 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.15], [1.0, 1.0, 0.0], [0.0, 1.0, 0.15]])
WARPED_Q8 = np.vstack([WARPED_Q4, 0.5 * (WARPED_Q4 + np.roll(WARPED_Q4, -1, axis=0))])
TRI3 = np.array([[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.2, 0.9, 0.0]])
TRI6 = np.vstack([TRI3, 0.5 * (TRI3 + np.roll(TRI3, -1, axis=0))])
TRI6[3:, 2] = 0.05  # curved: mid-side nodes lifted off the corner plane
ELEMENTS = [("quad4", WARPED_Q4), ("tri3", TRI3), ("quad8", WARPED_Q8), ("quad8_curved", WARPED_Q8),
            ("tri6", TRI6)]
LAM = isotropic_laminate(70e9, 0.3, 0.005)


def single_element(name, X):
    mesh = FEAMesh()
    for i, x in enumerate(X, start=1):
        mesh.add_node(i, x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mesh.add_element(1, list(range(1, len(X) + 1)), LAM, FSDTModel(),
                         element_type="quad8_curved" if name == "quad8_curved" else None)
    return mesh.elements[0]


@pytest.mark.parametrize("name, X", ELEMENTS, ids=[e[0] for e in ELEMENTS])
def test_geometric_stiffness_is_symmetric_and_rigid_translations_cost_nothing(name, X):
    element = single_element(name, X)
    Kg = element.geometric_stiffness_matrix(resultants=(-3.0, 1.0, 0.5))
    assert np.allclose(Kg, Kg.T, atol=1e-12 * np.abs(Kg).max())
    for k in range(3):
        u = np.zeros(element.n_dof)
        u[k::element.dofs_per_node] = 1.0
        assert np.linalg.norm(Kg @ u) < 1e-12 * np.abs(Kg).max()


@pytest.mark.parametrize("name, X", ELEMENTS, ids=[e[0] for e in ELEMENTS])
def test_geometric_stiffness_is_the_work_of_the_prestress_on_the_rotation_of_the_surface(name, X):
    # a translation w = x varying linearly over a flat element tilts it: u^T Kg u = Nx * area
    X = X.copy()
    X[:, 2] = 0.0
    element = single_element(name, X)
    Kg = element.geometric_stiffness_matrix(resultants=(-2.0, 0.0, 0.0))
    u = np.zeros(element.n_dof)
    u[2::element.dofs_per_node] = X[:, 0]
    assert u @ Kg @ u == pytest.approx(-2.0 * element.area, rel=1e-10)


def test_prestress_from_displacements_matches_the_uniform_resultants():
    element = single_element("quad8_curved", np.c_[WARPED_Q8[:, :2], np.zeros(8)])
    u = np.zeros(element.n_dof)
    u[0::element.dofs_per_node] = -1e-4 * element.coordinates[:, 0]  # uniform eps_x = -1e-4, nu = 0.3, free y
    u[1::element.dofs_per_node] = 0.3e-4 * element.coordinates[:, 1]
    N = element.membrane_prestress(u, 0.2, -0.3)
    assert N == pytest.approx([-70e9 * 0.005 * 1e-4, 0.0, 0.0], abs=1e-6 * 35e3)
    assert np.allclose(element.geometric_stiffness_matrix(u), element.geometric_stiffness_matrix(resultants=N))


def small_plate(element="quad8", n=6):
    named = RectangularPlateBuilder(0.5, 0.5, n, n, element=element).build_named(LAM, FSDTModel())
    edges = set(sum((named.nodes(k) for k in ("left", "right", "bottom", "top")), []))
    bcs = BoundaryConditions().fix(edges, [DOF.UZ]).fix(named.nodes("left"), [DOF.UX])
    bcs.fix([named.mesh.closest_node((0.0, 0.0, 0.0))], [DOF.UY])
    return named, bcs


def test_load_factor_scales_inversely_with_the_reference_load_and_modes_are_normalised():
    named, bcs = small_plate()
    one = LinearBucklingRunner(named.mesh, bcs, LoadCase().add_line_load(named.nodes("right"), (-1.0, 0, 0)),
                               n_modes=3).execute().get_results()
    two = LinearBucklingRunner(named.mesh, bcs, LoadCase().add_line_load(named.nodes("right"), (-2.0, 0, 0)),
                               n_modes=3).execute().get_results()
    assert np.allclose(one.load_factors, 2.0 * two.load_factors, rtol=1e-10)
    assert np.all(np.diff(one.load_factors) >= 0.0)
    for mode in one:
        assert np.abs(mode.displacement_array()[:, :3]).max() == pytest.approx(1.0)
        assert mode.displacement_array()[:, :3].max() == pytest.approx(1.0)
    assert {"assemble_geometric", "eigen", "modes"} <= set(one.diagnostics.phase_times)


def test_reference_solution_is_the_linear_static_solution():
    named, bcs = small_plate()
    load = LoadCase().add_line_load(named.nodes("right"), (-1.0, 0, 0))
    results = LinearBucklingRunner(named.mesh, bcs, load).execute().get_results()
    static = StructuralGlobalSolver().solve(named.mesh, bcs, load)
    assert np.allclose(results.reference.displacements, static.displacements)
    mode = results[0].as_solution()
    assert mode.info["load_factor"] == results.critical_load_factor


def test_lanczos_and_dense_eigensolvers_agree():
    named, bcs = small_plate(n=8)
    load = LoadCase().add_line_load(named.nodes("right"), (-1.0, 0, 0))
    runs = {m: LinearBucklingRunner(named.mesh, bcs, load, eigensolver=SolveBucklingEigenproblem(4, method=m))
            .execute().get_results() for m in ("dense", "lanczos")}
    assert runs["dense"].eigen_method == "dense" and runs["lanczos"].eigen_method == "lanczos"
    assert np.allclose(runs["dense"].load_factors, runs["lanczos"].load_factors, rtol=1e-8)
    for a, b in zip(runs["dense"], runs["lanczos"]):
        assert abs(np.dot(a.displacements, b.displacements)) == pytest.approx(
            np.linalg.norm(a.displacements) * np.linalg.norm(b.displacements), rel=1e-6)


def test_tension_finds_no_positive_buckling_load_but_the_reversed_load_does():
    named, bcs = small_plate()
    tension = LoadCase().add_line_load(named.nodes("right"), (1.0, 0, 0))
    none = LinearBucklingRunner(named.mesh, bcs, tension).execute().get_results()
    assert len(none) == 0 and np.isnan(none.critical_load_factor)
    both = LinearBucklingRunner(named.mesh, bcs, tension, eigensolver=SolveBucklingEigenproblem(1, signs="both"))
    compression = LinearBucklingRunner(named.mesh, bcs, LoadCase().add_line_load(named.nodes("right"), (-1.0, 0, 0)))
    assert both.execute().get_results().critical_load_factor == pytest.approx(
        -compression.execute().get_results().critical_load_factor, rel=1e-8)


def test_solid_elements_are_refused():
    mesh = FEAMesh()
    for i, x in enumerate([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, .1), (1, 0, .1), (1, 1, .1), (0, 1, .1)],
                          start=1):
        mesh.add_node(i, x)
    mesh.add_solid(1, range(1, 9), LAM)
    bcs = BoundaryConditions().fix([1, 2, 3, 4], [DOF.UX, DOF.UY, DOF.UZ])
    with pytest.raises(NotImplementedError, match="shell elements only"):
        LinearBucklingRunner(mesh, bcs, LoadCase().add_nodal_load(5, DOF.UZ, -1.0)).execute()


@pytest.mark.parametrize("element, n, tol", [("quad4", 16, 0.01), ("quad8", 8, 0.002), ("quad8_curved", 8, 0.002),
                                             ("tri6", 8, 0.002), ("tri3", 16, 0.04)])
def test_simply_supported_plate_in_compression(element, n, tol):
    assert plate_compression(element, n).ratio == pytest.approx(1.0, abs=tol)


@pytest.mark.parametrize("angles, aspect", [([0, 90, 90, 0], 1.0), ([90, 0, 0, 90], 2.0), ([0, 90, 90, 0], 3.0)])
def test_cross_ply_plate_in_compression(cfrp, angles, aspect):
    lam = Laminate.from_angles(cfrp, angles, 0.25e-3)
    assert plate_compression("quad8", 16, lam, aspect).ratio == pytest.approx(1.0, abs=0.002)


def test_plate_in_shear():
    assert plate_shear("tri6", 8).ratio == pytest.approx(1.0, abs=0.003)
    assert plate_shear("quad8", 16).ratio == pytest.approx(1.0, abs=0.003)


def test_heated_plate_buckles_from_the_hygrothermal_prestress():
    assert plate_thermal("quad8", 8).ratio == pytest.approx(1.0, abs=0.002)


@pytest.mark.parametrize("element, n, tol", [("quad4", 16, 0.003), ("quad8", 8, 1e-3), ("tri6", 8, 1e-3)])
def test_euler_column(element, n, tol):
    assert column(element, n).ratio == pytest.approx(1.0, abs=tol)


def test_cylinder_in_axial_compression():
    assert cylinder_axial("quad8_curved", 16).ratio == pytest.approx(1.0, abs=0.02)
