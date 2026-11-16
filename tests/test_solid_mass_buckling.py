"""Layered solids assemble mass and geometric stiffness for modal and buckling solves."""

import numpy as np
import pytest

from shell_fem import (
    BoundaryConditions,
    DOF,
    FEAMesh,
    Laminate,
    LinearBucklingRunner,
    LoadCase,
    solid_block,
    vibration_modes,
)


def _brick(aluminium, element, nx=1, ny=1):
    L, b, h = 0.04, 0.01, 0.008
    lam = Laminate.from_angles(aluminium, [0.0], h)
    mesh = FEAMesh()
    named = solid_block(mesh, lam, L, b, nx, ny, element=element, layers="laminate")
    return named, lam, L, b, h


@pytest.mark.parametrize("element", ["hex8", "hex20"])
def test_solid_mass_of_a_rigid_translation_is_the_block_mass(aluminium, element):
    named, lam, L, b, h = _brick(aluminium, element)
    element_ = named.mesh.elements[0]
    mass = aluminium.density * L * b * h
    assert named.mesh.solid_elements[0].volume == pytest.approx(L * b * h, rel=1e-12)
    u = np.zeros(element_.n_dof)
    u[0::6] = 1.0
    assert float(u @ element_.mass_matrix() @ u) == pytest.approx(mass, rel=1e-9)
    assembled = float(u @ (named.mesh.elements[0].mass_matrix()) @ u)
    assert assembled == pytest.approx(mass, rel=1e-9)
    M = named.mesh.elements[0].mass_matrix()
    assert np.allclose(M, M.T, atol=1e-14 * np.abs(M).max())
    rot = np.zeros(element_.n_dof)
    rot[3::6] = 1.0
    assert float(rot @ M @ rot) == pytest.approx(0.0, abs=1e-18)


@pytest.mark.parametrize("element", ["hex8", "hex20"])
def test_solid_geometric_stiffness_is_the_work_of_the_prestress_on_the_gradient(aluminium, element):
    named, _, L, b, h = _brick(aluminium, element)
    e = named.mesh.elements[0]
    Kg = e.geometric_stiffness_matrix(stress=(-2.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert np.allclose(Kg, Kg.T, atol=1e-12 * max(np.abs(Kg).max(), 1.0))
    for k in range(3):
        u = np.zeros(e.n_dof)
        u[k::6] = 1.0
        assert np.linalg.norm(Kg @ u) < 1e-12 * max(np.abs(Kg).max(), 1.0)
    u = np.zeros(e.n_dof)
    xyz = e.coordinates
    u[2::6] = xyz[:, 0]
    assert u @ Kg @ u == pytest.approx(-2.0 * e.volume, rel=1e-9)


def test_solid_geometric_stiffness_from_displacements_matches_uniform_stress(aluminium):
    named, _, L, b, h = _brick(aluminium, "hex8")
    e = named.mesh.elements[0]
    u = np.zeros(e.n_dof)
    u[0::6] = -1.0e-4 * e.coordinates[:, 0]
    point = e.stress_at(u, 0.0, 0.0, 0.0)
    Kg_u = e.geometric_stiffness_matrix(u)
    Kg_s = e.geometric_stiffness_matrix(stress=point.stress_global)
    assert np.allclose(Kg_u, Kg_s, rtol=1e-8, atol=1e-8 * np.abs(Kg_s).max())


def _cantilever_solid(aluminium, element="hex8", n=6):
    L, b, h = 0.2, 0.02, 0.01
    lam = Laminate.from_angles(aluminium, [0.0], h)
    mesh = FEAMesh()
    named = solid_block(mesh, lam, L, b, n, 1, element=element, layers="laminate")
    bcs = BoundaryConditions().fix(named.nodes("x0"), [DOF.UX, DOF.UY, DOF.UZ])
    return named, bcs, L, b, h


def test_solid_block_vibration_matches_the_cantilever_beam(aluminium):
    named, bcs, L, b, h = _cantilever_solid(aluminium, n=8)
    E = aluminium.elastic.E1
    I = b * h**3 / 12.0
    mu = aluminium.density * b * h
    omega = 1.8751040684**2 * np.sqrt(E * I / (mu * L**4))
    modes = vibration_modes(named.mesh, bcs, n_modes=2)
    assert modes.angular_frequencies_rad_s[0] == pytest.approx(omega, rel=0.08)
    assert modes.angular_frequencies_rad_s[1] > modes.angular_frequencies_rad_s[0]
    assert modes.generalised_mass[0] == pytest.approx(1.0, rel=1e-8)


def test_solid_block_buckling_matches_the_euler_cantilever(aluminium):
    named, bcs, L, b, h = _cantilever_solid(aluminium, n=8)
    E = aluminium.elastic.E1
    I = b * h**3 / 12.0
    P_euler = np.pi**2 * E * I / (4.0 * L**2)
    load = LoadCase()
    tip = named.nodes("x1")
    share = -1.0 / len(tip)
    for node in tip:
        load.add_nodal_load(node, DOF.UX, share)
    results = LinearBucklingRunner(named.mesh, bcs, load, n_modes=1).execute().get_results()
    assert results.critical_load_factor == pytest.approx(P_euler, rel=0.12)
