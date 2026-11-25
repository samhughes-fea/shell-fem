"""Per-ply failure field, its envelope over load cases, the blade runner option and the fringe plots."""

import warnings

import numpy as np
import pytest

from shell_fem import (DOF, BoundaryConditions, FEAMesh, FSDTModel, HashinCriterion, Laminate, LoadCase,
                       MaximumStressCriterion, ResultsExtractor, StructuralGlobalSolver, TsaiWuCriterion)
from shell_fem.pre_processing.load_library.blade_loads import add_sectional_force
from shell_fem.processing.static.results.ply_failure import failure_envelope, ply_failure_field
from shell_fem.simulation_runner.blade import BladeStaticRunner
from shell_fem.verification.blade_benchmarks import tube_blade

ANGLES = [0, 45, -45, 90]
T_PLY = 0.125e-3
L, B = 0.1, 0.04


def _tension(cfrp, Nx):
    lam = Laminate.from_angles(cfrp, ANGLES, T_PLY)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mesh = FEAMesh.rectangular_plate(L, B, 5, 2, lam, FSDTModel())
    bcs = BoundaryConditions().fix(mesh.node_ids, [DOF.UZ, DOF.RX, DOF.RY])
    bcs.fix(mesh.nodes_on_line(0, 0.0), [DOF.UX]).fix([mesh.closest_node((0, 0))], [DOF.UY])
    loads = LoadCase("tension")
    right = sorted(mesh.nodes_on_line(0, L), key=lambda n: mesh.node_coordinates(n)[1])
    for i, node in enumerate(right):
        loads.add_nodal_load(node, DOF.UX, Nx * B / 2 * (0.5 if i in (0, len(right) - 1) else 1.0))
    return lam, mesh, StructuralGlobalSolver().solve(mesh, bcs, loads)


def test_membrane_ply_failure_matches_classical_laminate_theory(cfrp):
    """Uniform Nx on a [0/45/-45/90] laminate: each ply's index is the criterion on Q̄ A^-1 N."""
    Nx = 1.0e5
    lam, mesh, solution = _tension(cfrp, Nx)
    A = sum(p.Q_bar * p.thickness for p in lam.plies)
    eps0 = np.linalg.solve(A, [Nx, 0.0, 0.0])
    criterion = MaximumStressCriterion()
    field = ply_failure_field(solution, criterion)
    assert field.index.shape == (mesh.n_elements, len(ANGLES))
    for k, ply in enumerate(lam.plies):
        sigma = ply.to_material_stress(ply.Q_bar @ eps0)
        expected = criterion.evaluate(sigma, ply.material.strength)
        np.testing.assert_allclose(field.index[:, k], expected.index, rtol=1e-6)
        np.testing.assert_allclose(field.strength_ratio[:, k], expected.strength_ratio, rtol=1e-6)
        assert set(field.mode[:, k]) == {expected.mode.value}
    # the 90° ply fails first in transverse tension
    assert set(field.governing_ply) == {3} and field.governing_mode[0] == "matrix_tension"
    worst = field.worst()
    assert worst["ply"] == 3 and worst["index"] == pytest.approx(field.max_index.max())
    assert worst["strength_ratio"] == pytest.approx(1.0 / worst["index"], rel=1e-9)   # max stress is linear
    assert not field.failed.any()


@pytest.mark.parametrize("criterion", [TsaiWuCriterion(), HashinCriterion()])
def test_ply_failure_agrees_with_the_results_extractor(plate_bending, criterion):
    solution = plate_bending
    field = ply_failure_field(solution, criterion)
    elements = ResultsExtractor(solution, criterion=criterion).elements()
    rows = field.for_elements(elements.element_ids)
    np.testing.assert_allclose(field.max_index[rows], elements.max_failure_index, rtol=1e-12)
    np.testing.assert_array_equal(field.governing_ply[rows], elements.critical_ply)


@pytest.fixture(scope="module")
def plate_bending(request):
    from shell_fem import create_default_library

    cfrp = create_default_library().get("T300/5208")
    lam = Laminate.from_angles(cfrp, [0, 90, 90, 0], 0.5e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mesh = FEAMesh.rectangular_plate(0.2, 0.2, 6, 6, lam, FSDTModel())
    bcs = BoundaryConditions().fix(mesh.nodes_on_line(0, 0.0), [DOF.UX, DOF.UY, DOF.UZ, DOF.RX, DOF.RY, DOF.RZ])
    return StructuralGlobalSolver().solve(mesh, bcs, LoadCase("p").add_pressure(2.0e4))


def test_envelope_takes_the_worst_case_per_ply(cfrp):
    _, _, light = _tension(cfrp, 5.0e4)
    _, _, heavy = _tension(cfrp, 1.0e5)
    criterion = MaximumStressCriterion()
    a, b = ply_failure_field(light, criterion), ply_failure_field(heavy, criterion)
    env = failure_envelope({"light": a, "heavy": b})
    np.testing.assert_allclose(env.field.index, b.index)
    assert set(env.case.ravel()) == {"heavy"} and set(env.governing_case) == {"heavy"}
    np.testing.assert_allclose(a.index * 2.0, b.index, rtol=1e-9)
    with pytest.raises(ValueError):
        failure_envelope({})


def test_blade_runner_reports_ply_failure_and_its_envelope():
    blade = tube_blade("quad4", 8, 4, laminate=Laminate.from_angles(_cfrp(), [0, 90, 0], 1.0e-3))
    up = add_sectional_force(LoadCase("up"), blade, lambda z: (0.0, 50.0, 0.0))
    side = add_sectional_force(LoadCase("side"), blade, lambda z: (80.0, 0.0, 0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = BladeStaticRunner(blade, [up, side], failure_criterion=TsaiWuCriterion()).execute().get_results()
    for name in ("up", "side"):
        assert results[name].failure.index.shape == (blade.mesh.n_elements, 3)
    env = results.failure_envelope()
    assert env.field.max_index.max() == pytest.approx(max(results["up"].failure.max_index.max(),
                                                          results["side"].failure.max_index.max()))
    assert set(env.governing_case) <= {"up", "side"}
    worst = env.field.worst(blade.elements("skin"))
    assert worst["location"][2] < 0.5 * blade.span_positions[-1]   # root bending governs


def _cfrp():
    from shell_fem import create_default_library

    return create_default_library().get("T300/5208")


def test_fringe_plots_draw_one_shape_per_element():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from shell_fem.post_processing.plots import section_fringe, surface_fringe

    blade = tube_blade("quad4", 8, 4)
    ids = np.array(blade.elements("skin"))
    values = np.arange(ids.size, dtype=float)
    fig, (a, b) = plt.subplots(1, 2)
    polys = surface_fringe(a, blade.mesh, ids, values)
    assert len(polys.get_paths()) == ids.size and polys.get_clim() == (0.0, ids.size - 1.0)
    lines = section_fringe(b, blade.mesh, ids, values, z=0.3 * blade.span_positions[-1])
    assert len(lines.get_segments()) == 8                    # one per element round the tube
    plt.close(fig)
    with pytest.raises(ValueError):
        surface_fringe(a, blade.mesh, ids, values[:-1])
