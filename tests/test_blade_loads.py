"""Full-blade loading: tractions, centrifugal loads, spanwise distributions and BladeStaticRunner.

The tube checks compare against beam theory; examples/29_full_blade_loading has the
figures and a kernel-built blade.
"""

import warnings

import numpy as np
import pytest

from shell_fem import FEAMesh, FSDTModel, LoadCase, LoadCombination
from shell_fem.pre_processing.load_library.blade_loads import (add_sectional_force, add_sectional_torque,
                                                               add_spanwise_pressure, clamp_root, span_bays, spanwise)
from shell_fem.simulation_runner.blade import BladeStaticRunner, resultant_about
from shell_fem.simulation_runner.backends.decks import equivalent_nodal_forces
from shell_fem.processing.static.operations.assembly import AssembleLoadVector, GlobalAssembler
from shell_fem.verification.blade_benchmarks import (L, R, tube_bending, tube_blade, tube_centrifugal,
                                                     tube_torsion)
from shell_fem.verification.shell_benchmarks import isotropic_laminate

Q4 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.15], [1.0, 1.0, 0.0], [0.0, 1.0, 0.15]])
Q8 = np.vstack([Q4, 0.5 * (Q4 + np.roll(Q4, -1, axis=0))])
Q8[4:, 2] += 0.05
T3 = np.array([[0.0, 0.0, 0.0], [1.0, 0.1, 0.2], [0.2, 0.9, 0.0]])
T6 = np.vstack([T3, 0.5 * (T3 + np.roll(T3, -1, axis=0))])
T6[3:, 2] += 0.05
ELEMENTS = [("quad4", Q4), ("quad8", Q8), ("quad8_curved", Q8), ("tri3", T3), ("tri6", T6)]
LAM = isotropic_laminate(70e9, 0.3, 0.005, density=2700.0)


def single(name, X):
    mesh = FEAMesh()
    for i, x in enumerate(X, start=1):
        mesh.add_node(i, x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mesh.add_element(1, list(range(1, len(X) + 1)), LAM, FSDTModel(),
                         element_type="quad8_curved" if name == "quad8_curved" else None)
    return mesh, mesh.elements[0]


def integrate(element, fn):
    return sum(fn(element.global_coordinates(xi, eta)) * element.jacobian(xi, eta)[1] * w
               for xi, eta, w in element.load_rule())


@pytest.mark.parametrize("name, X", ELEMENTS, ids=[e[0] for e in ELEMENTS])
def test_traction_and_centrifugal_loads_integrate_to_their_resultants(name, X):
    mesh, element = single(name, X)
    t = lambda x: np.array([1.0 + x[0], -2.0, 3.0 * x[1]])  # noqa: E731
    F = element.traction_load_vector(t)
    exact = integrate(element, t)
    assert np.allclose(resultant_about(mesh, F, (0, 0, 0))[:3], exact, rtol=1e-12)
    # the moment is exact where the element surface is the mesh surface: curved elements, or flat ones on a plane
    planar = X.copy()
    planar[:, 2] = 0.0
    mesh_m, element_m = (mesh, element) if name in ("quad8_curved", "tri6") else single(name, planar)
    moment = integrate(element_m, lambda x: np.cross(x, t(x)))
    F_m = element_m.traction_load_vector(t)
    assert np.allclose(resultant_about(mesh_m, F_m, (0, 0, 0))[3:], moment, rtol=1e-9, atol=1e-9 * np.abs(moment).max())
    omega, axis = 3.0, np.array([0.0, 1.0, 0.0])
    C = element.centrifugal_load_vector(omega, (0.2, 0.0, 0.0), axis)
    rhoA = LAM.areal_density()
    r_perp = lambda x: (x - [0.2, 0, 0]) - np.dot(x - [0.2, 0, 0], axis) * axis  # noqa: E731
    assert np.allclose(resultant_about(mesh, C, (0, 0, 0))[:3], rhoA * omega ** 2 * integrate(element, r_perp),
                       rtol=1e-12)


def test_tractions_and_centrifugal_loads_scale_and_combine():
    blade = tube_blade("quad4", 8, 4)
    a = LoadCase("a").add_traction((0.0, 10.0, 0.0)).add_centrifugal(2.0, axis=(0, 1, 0))
    b = LoadCase("b").add_traction(lambda x: (x[2], 0.0, 0.0), blade.elements("skin")[:5])
    loads = AssembleLoadVector(blade.mesh, GlobalAssembler())
    Fa, Fb = loads.run(a), loads.run(b)
    assert np.allclose(loads.run(LoadCase.combine([(a, 1.5), (b, -2.0)])), 1.5 * Fa - 2.0 * Fb)
    assert np.allclose(loads.run(a.scaled(-1.0)), -Fa)
    forces = equivalent_nodal_forces(blade.mesh, a)
    total = sum(v[:3] for v in forces.values())
    assert np.allclose(total, resultant_about(blade.mesh, Fa, (0, 0, 0))[:3])


def test_spanwise_table():
    f = spanwise([0.0, 1.0, 3.0], [[0.0, 2.0], [1.0, 2.0], [5.0, 0.0]])
    assert np.allclose(f(2.0), [3.0, 1.0])
    assert np.allclose(f(-1.0), [0.0, 2.0]) and np.allclose(f(9.0), [5.0, 0.0])
    assert spanwise([0.0, 2.0], [1.0, 3.0])(0.5) == pytest.approx(1.5)
    with pytest.raises(ValueError):
        spanwise([0.0, 0.0], [1.0, 2.0])


def test_sectional_force_and_torque_carry_their_span_integrals():
    blade = tube_blade("quad8_curved", 12, 10)
    bays = span_bays(blade)
    assert len(bays) == 10 and sum(len(b.element_ids) for b in bays) == len(blade.elements("skin"))
    assert sum(b.area for b in bays) == pytest.approx(2 * np.pi * R * L, rel=1e-3)
    loads = AssembleLoadVector(blade.mesh, GlobalAssembler())
    force = add_sectional_force(LoadCase(), blade, spanwise([0.0, L], [[100.0, 0.0, 0.0], [300.0, 50.0, 0.0]]))
    total = resultant_about(blade.mesh, loads.run(force), (0, 0, 0))
    assert np.allclose(total[:3], [200.0 * L, 25.0 * L, 0.0], rtol=1e-9)
    # the moment about the root of a linearly varying force: integral of z f(z) dz
    assert total[4] == pytest.approx(100.0 * L ** 2 / 2 + 200.0 * L ** 2 / 3, rel=1e-3)
    torque = add_sectional_torque(LoadCase(), blade, spanwise([0.0, L], [400.0, 100.0]))
    total = resultant_about(blade.mesh, loads.run(torque), (0, 0, 0))
    assert np.allclose(total[:3], 0.0, atol=1e-9 * 250.0 * L)
    assert total[5] == pytest.approx(250.0 * L, rel=1e-9)
    pressure = add_spanwise_pressure(LoadCase(), blade, lambda z: 1.0 + z)
    assert np.any(loads.run(pressure))


@pytest.mark.parametrize("point", [(0.0, 0.0), (0.3, -0.2), lambda z: (0.1 * z, 0.05)])
def test_sectional_force_acts_at_its_application_point(point):
    """Off the skin centroid, the force keeps its resultant and takes the moment of its point."""
    blade = tube_blade("quad8_curved", 12, 10, centre=(0.4, 0.15))
    loads = AssembleLoadVector(blade.mesh, GlobalAssembler())
    f = np.array([120.0, -300.0, 80.0])
    total = resultant_about(blade.mesh, loads.run(add_sectional_force(LoadCase(), blade, lambda z: f,
                                                                      point=point)), (0, 0, 0))
    at = point if callable(point) else (lambda z, p=point: p)
    zs, ws = np.polynomial.legendre.leggauss(8)
    zs, ws = 0.5 * L * (zs + 1.0), 0.5 * L * ws
    moment = sum(w * np.cross([*at(z), z], f) for z, w in zip(zs, ws))
    np.testing.assert_allclose(total[:3], f * L, rtol=1e-9)
    np.testing.assert_allclose(total[3:], moment, rtol=1e-6, atol=1e-9 * np.abs(f).max() * L ** 2)


def test_sectional_force_at_the_skin_centroid_with_no_point():
    blade = tube_blade("quad4", 12, 6, centre=(0.4, 0.0))
    loads = AssembleLoadVector(blade.mesh, GlobalAssembler())
    total = resultant_about(blade.mesh, loads.run(add_sectional_force(LoadCase(), blade, lambda z: (0.0, 100.0, 0.0),
                                                                      point=None)), (0, 0, 0))
    assert total[5] == pytest.approx(0.4 * 100.0 * L, rel=1e-6)   # at the tube axis, not the pitch axis


def test_blade_runner_balances_root_loads_against_every_applied_load():
    blade = tube_blade("quad8_curved", 12, 8)
    gravity = LoadCase("gravity").add_body_load((0.0, -9.81, 0.0))
    aero = add_sectional_force(LoadCase("aero"), blade, lambda z: (50.0 * z, 200.0, 0.0))
    aero = add_sectional_torque(aero, blade, lambda z: 30.0)
    aero = add_spanwise_pressure(aero, blade, lambda z: 10.0 * z)
    spin = LoadCase("spin").add_centrifugal(2.0, origin=(0.0, 0.0, -1.0), axis=(0.0, 1.0, 0.0))
    combo = LoadCombination("uls", {"gravity": 1.35, "aero": 1.5, "spin": 1.0})
    results = BladeStaticRunner(blade, [gravity, aero, spin], [combo]).execute().get_results()
    assert set(results) == {"gravity", "aero", "spin", "uls"}
    for name in results:
        case = results[name]
        scale = np.abs(case.applied_load).max()
        assert np.abs(case.equilibrium_residual).max() < 1e-8 * scale, name
    uls = results["uls"].root_load
    assert np.allclose(uls, 1.35 * results["gravity"].root_load + 1.5 * results["aero"].root_load
                       + results["spin"].root_load, rtol=1e-9, atol=1e-9 * np.abs(uls).max())
    assert results["aero"].tip_translation[1] > 0.0 and results["aero"].tip_twist > 0.0
    assert results["aero"].deflection.stations.size == 9


def test_blade_runner_buckling_option():
    blade = tube_blade("quad4", 12, 8)
    load = add_sectional_force(LoadCase("compression"), blade, lambda z: (0.0, 0.0, -1.0e3))
    results = BladeStaticRunner(blade, load, buckling_modes=2).execute().get_results()
    buckling = results["compression"].buckling
    assert len(buckling) == 2 and buckling.critical_load_factor > 0.0
    assert len(clamp_root(blade).prescribed) >= 6 * len(blade.nodes("root"))  # the default supports


def test_tube_against_beam_theory():
    assert tube_bending("quad8_curved", 16, 20).ratio == pytest.approx(1.0, abs=0.003)
    assert tube_torsion("quad8_curved", 16, 20).ratio == pytest.approx(1.0, abs=0.001)
    assert tube_centrifugal("quad8_curved", 16, 20).ratio == pytest.approx(1.0, abs=0.004)


def test_section_loads_from_the_shell_match_beam_integrals_along_the_span():
    """Shear, bending moment, torque and axial force at every cut, against ∫ from the tip."""
    blade = tube_blade("quad4", n_around=12, n_span=8)
    f, m, omega = 1000.0, 300.0, 3.0
    case = add_sectional_torque(add_sectional_force(LoadCase("c"), blade, lambda z: (0.0, f, 0.0)), blade,
                                lambda z: m)
    case.add_centrifugal(omega, origin=(0.0, 0.0, 0.0), axis=(0.0, 1.0, 0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = BladeStaticRunner(blade, case).execute().get_results()["c"]
    s = result.section_loads
    z = s.stations[:-1]
    np.testing.assert_allclose(s.stations, blade.span_positions)
    np.testing.assert_allclose(s.internal[0], result.root_load, atol=1e-6 * np.abs(result.root_load).max())
    assert s.relative_residual < 1e-9
    internal = s.internal[:-1]
    np.testing.assert_allclose(internal[:, 1], -f * (L - z), rtol=1e-6)
    np.testing.assert_allclose(internal[:, 3], 0.5 * f * (L - z) ** 2, rtol=1e-6)
    np.testing.assert_allclose(internal[:, 5], -m * (L - z), rtol=1e-6)
    from shell_fem.verification.blade_benchmarks import RHO, T
    mass_per_length = RHO * T * 12 * 2.0 * R * np.sin(np.pi / 12)   # the flat elements' 12-sided perimeter
    np.testing.assert_allclose(internal[:, 2], -0.5 * mass_per_length * omega ** 2 * (L ** 2 - z ** 2), rtol=1e-6)


def test_section_cuts_need_nodes_on_the_cut():
    from shell_fem.processing.static.results.section_loads import SectionLoadIntegrator

    blade = tube_blade("quad4", n_around=8, n_span=4)
    with pytest.raises(ValueError, match="no nodes at the cut"):
        SectionLoadIntegrator(blade.mesh, [0.123])


def test_section_cuts_must_not_cross_elements():
    from shell_fem.processing.static.results.section_loads import SectionLoadIntegrator

    blade = tube_blade("quad8", n_around=8, n_span=4)
    with pytest.raises(ValueError, match="crosses"):
        SectionLoadIntegrator(blade.mesh, [0.5 * L / 4])    # through the midside nodes of the first bay
    SectionLoadIntegrator(blade.mesh, blade.span_positions)
