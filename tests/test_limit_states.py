"""Deflection profile, CLPT outer-ply stress and strain, linear buckling and blade vibration."""

from types import SimpleNamespace

import numpy as np
import pytest

from conftest import simply_supported
from shell_fem import (
    DOF,
    BoundaryConditions,
    CLPTModel,
    ExactStressRecoverer,
    FEAMesh,
    Laminate,
    LoadCase,
    StructuralGlobalSolver,
    deflection_profile,
    integrate_abd,
    linear_buckling,
    outer_ply_stress_strain,
    vibration_modes,
)
from shell_fem.processing import eigenpairs


def _plate_modulus(material, h):
    E, nu = material.elastic.E1, material.elastic.v12
    D = E * h**3 / (12.0 * (1.0 - nu**2))
    return E, nu, D


def cylindrical_strip(material, n=8):
    """Narrow plate in cylindrical bending: ε_y = 0 and κ_y = 0, clamped at x = 0.

    The flexural rigidity is then D * b, with D the plate modulus, so the strip is an
    Euler-Bernoulli beam without retuning the mesh.
    """
    L, b, h = 0.5, 0.05, 0.005
    mesh = FEAMesh.rectangular_plate(L, b, n, 1, Laminate.from_angles(material, [0.0], h), CLPTModel())
    bcs = BoundaryConditions().fix(mesh.node_ids, [DOF.UY, DOF.RX])
    bcs.clamp(mesh.nodes_on_line(0, 0.0))
    return mesh, bcs, L, b, h


def test_element_mass_of_a_rigid_translation_is_the_laminate_mass(aluminium):
    mesh, _, _, _, h = cylindrical_strip(aluminium, n=1)
    element = mesh.elements[0]
    direction = element.frame[2]
    u = np.zeros(element.n_dof)
    d = element.dofs_per_node
    for i in range(element.N_NODES):
        u[d * i : d * i + 3] = direction
    mass = element.laminate.areal_density() * element.area
    assert mass == pytest.approx(aluminium.density * h * element.area)
    assert float(u @ element.mass_matrix() @ u) == pytest.approx(mass, rel=1e-12)


def test_deflection_profile_matches_the_cantilever_beam(aluminium):
    mesh, bcs, L, b, h = cylindrical_strip(aluminium)
    P = 1.0
    load = LoadCase("tip").add_line_load(mesh.nodes_on_line(0, L), (0.0, 0.0, P / b))
    result = StructuralGlobalSolver().solve(mesh, bcs, load)
    _, _, D = _plate_modulus(aluminium, h)
    EI = D * b
    profile = deflection_profile(result, axis=0)
    w = P * profile.stations**2 * (3.0 * L - profile.stations) / (6.0 * EI)
    assert profile.stations[0] == pytest.approx(0.0)
    assert profile.stations[-1] == pytest.approx(L)
    # The tip is the beam deflection. Stations near the root are a small fraction of
    # that tip value, so the profile is compared with an absolute band of 2% of the tip.
    assert profile.translation[-1, 2] == pytest.approx(w[-1], rel=0.02)
    assert profile.translation[:, 2] == pytest.approx(w, abs=0.02 * w[-1])
    # The default axis is the blade span. This plate lies in the xy-plane, so that
    # profile is the single station z = 0.
    span = deflection_profile(result)
    assert span.axis == 2 and span.stations.shape == (1,)
    assert span.stations[0] == pytest.approx(0.0)


def test_outer_ply_matches_classical_laminate_theory_and_keeps_the_offset(cfrp, aluminium):
    lam = Laminate.from_angles(cfrp, [0, 90, 90, 0], 0.25e-3)
    L, b = 0.1, 0.04
    mesh = FEAMesh.rectangular_plate(L, b, 4, 2, lam, CLPTModel())
    bcs = BoundaryConditions().fix(mesh.node_ids, [DOF.UZ, DOF.RX, DOF.RY])
    bcs.fix(mesh.nodes_on_line(0, 0.0), [DOF.UX])
    bcs.fix([mesh.closest_node((0.0, 0.0))], [DOF.UY])
    force = 1.0e5
    right = sorted(mesh.nodes_on_line(0, L), key=lambda nid: mesh.node_coordinates(nid)[1])
    loads = LoadCase()
    edge = b / 2.0
    for i, node in enumerate(right):
        share = 0.5 if i in (0, len(right) - 1) else 1.0
        loads.add_nodal_load(node, DOF.UX, force * edge * share)
    result = StructuralGlobalSolver().solve(mesh, bcs, loads)
    eps0 = np.linalg.solve(integrate_abd(lam)[:3, :3], [force, 0.0, 0.0])
    field = outer_ply_stress_strain(result)
    assert set(field.face.tolist()) == {"bottom", "top"}
    for row in range(field.element_ids.size):
        ply = lam.plies[int(field.ply_index[row])]
        assert field.strain_global[row] == pytest.approx(eps0, rel=1e-8, abs=1e-12)
        assert field.strain_mechanical[row] == pytest.approx(eps0, rel=1e-8, abs=1e-12)
        assert field.stress_global[row] == pytest.approx(ply.Q_bar @ eps0, rel=1e-6, abs=1.0)
        assert field.stress_material[row] == pytest.approx(ply.to_material_stress(ply.Q_bar @ eps0), rel=1e-6, abs=1.0)
        assert field.strain_material[row] == pytest.approx(ply.to_material_strain(eps0), rel=1e-8, abs=1e-12)

    # The same recovery on a solved plate agrees with the existing CLPT recoverer,
    # and an offset stack reports z from the reference surface while the nodes stay put.
    h = 1.0e-3
    offset = 0.5 * h
    eccentric = Laminate.from_angles(aluminium, [45.0, -45.0], h / 2.0, offset=offset)
    plate = FEAMesh.rectangular_plate(0.2, 0.2, 4, 4, eccentric, CLPTModel())
    solved = StructuralGlobalSolver().solve(plate, simply_supported(plate, 0.2, 0.2), LoadCase().add_pressure(1.0e3))
    recovered = outer_ply_stress_strain(solved)
    assert np.max(np.abs(plate.coordinates_array()[:, 2])) == pytest.approx(0.0)
    bottoms = recovered.face == "bottom"
    tops = recovered.face == "top"
    assert recovered.z[bottoms] == pytest.approx(eccentric.z_interfaces[0])
    assert recovered.z[tops] == pytest.approx(eccentric.z_interfaces[-1])
    assert eccentric.z_interfaces[0] == pytest.approx(0.0)
    assert eccentric.z_interfaces[-1] == pytest.approx(h)
    recoverer = ExactStressRecoverer()
    for element in plate.shell_elements:
        u = solved.element_displacements(element.element_id)
        xi, eta = element.NATURAL_CENTROID
        for face, z, side in (("bottom", eccentric.z_interfaces[0], "above"), ("top", eccentric.z_interfaces[-1], "below")):
            point = recoverer.recover(element, u, float(z), xi, eta, side=side)
            row = np.flatnonzero((recovered.element_ids == element.element_id) & (recovered.face == face))[0]
            assert recovered.strain_mechanical[row] == pytest.approx(point.strain_mechanical, rel=1e-10, abs=1e-15)
            assert recovered.stress_global[row] == pytest.approx(point.stress_global, rel=1e-10, abs=1e-8)
            assert recovered.stress_material[row] == pytest.approx(point.stress_material, rel=1e-10, abs=1e-8)


def _supported_plate(material, n=8):
    a, h = 0.2, 0.002
    mesh = FEAMesh.rectangular_plate(a, a, n, n, Laminate.from_angles(material, [0.0], h), CLPTModel())
    return mesh, simply_supported(mesh, a, a), a, h


def test_vibration_modes_match_the_kirchhoff_plate(aluminium):
    mesh, bcs, a, h = _supported_plate(aluminium)
    _, _, D = _plate_modulus(aluminium, h)
    omega = 2.0 * np.pi**2 / a**2 * np.sqrt(D / (aluminium.density * h))
    modes = vibration_modes(mesh, bcs, n_modes=3)
    assert len(modes) == 3
    assert modes.angular_frequencies_rad_s[0] == pytest.approx(omega, rel=0.03)
    assert modes.frequencies_hz[0] == pytest.approx(omega / (2.0 * np.pi), rel=0.03)
    assert np.all(np.diff(modes.angular_frequencies_rad_s) > 0.0)
    centre = mesh.closest_node((a / 2.0, a / 2.0))
    centre_w = abs(modes.modes[mesh.dof_index(centre, DOF.UZ), 0])
    assert centre_w > 0.0


def test_cantilever_vibration_matches_the_beam(aluminium):
    mesh, bcs, L, b, h = cylindrical_strip(aluminium)
    _, _, D = _plate_modulus(aluminium, h)
    EI = D * b
    mu = aluminium.density * h * b
    beta = 1.8751040684
    omega = beta**2 * np.sqrt(EI / (mu * L**4))
    modes = vibration_modes(SimpleNamespace(mesh=mesh), bcs, n_modes=2)
    assert modes.angular_frequencies_rad_s[0] == pytest.approx(omega, rel=0.03)
    assert modes.angular_frequencies_rad_s[1] > modes.angular_frequencies_rad_s[0]


def test_sparse_vibration_matches_the_dense_plate(aluminium, monkeypatch):
    mesh, bcs, a, h = _supported_plate(aluminium, n=4)
    dense = vibration_modes(mesh, bcs, n_modes=2)
    monkeypatch.setattr(eigenpairs, "DENSE_LIMIT", 0)
    sparse = vibration_modes(mesh, bcs, n_modes=2)
    assert sparse.angular_frequencies_rad_s == pytest.approx(dense.angular_frequencies_rad_s, rel=1e-6)


def test_linear_buckling_matches_the_classical_plate(aluminium):
    mesh, bcs, a, h = _supported_plate(aluminium)
    bcs.fix(mesh.nodes_on_line(0, 0.0), [DOF.UX])
    load = LoadCase("compression").add_line_load(mesh.nodes_on_line(0, a), (-1.0, 0.0, 0.0))
    _, _, D = _plate_modulus(aluminium, h)
    Ncr = 4.0 * np.pi**2 * D / a**2
    # Fixed 8 by 8 mesh. It reads a little high of the classical load.
    result = linear_buckling(SimpleNamespace(mesh=mesh), bcs, load, n_modes=2)
    assert len(result) == 2
    assert result.load_case == "compression"
    assert result.load_factors[0] == pytest.approx(Ncr, rel=0.03)
    assert result.load_factors[1] > result.load_factors[0]
    assert np.max(np.abs(result.modes[:, 0])) == pytest.approx(1.0)


def test_sparse_buckling_matches_the_dense_plate(aluminium, monkeypatch):
    mesh, bcs, a, _ = _supported_plate(aluminium, n=4)
    bcs.fix(mesh.nodes_on_line(0, 0.0), [DOF.UX])
    load = LoadCase("compression").add_line_load(mesh.nodes_on_line(0, a), (-1.0, 0.0, 0.0))
    dense = linear_buckling(mesh, bcs, load, n_modes=1)
    monkeypatch.setattr(eigenpairs, "DENSE_LIMIT", 0)
    sparse = linear_buckling(mesh, bcs, load, n_modes=1)
    assert sparse.load_factors == pytest.approx(dense.load_factors, rel=1e-5)


def test_vibration_refuses_a_zero_density_and_a_prescribed_displacement(aluminium):
    from dataclasses import replace

    mesh, bcs, *_ = cylindrical_strip(aluminium, n=2)
    weightless = replace(aluminium, density=0.0)
    empty = FEAMesh.rectangular_plate(0.2, 0.05, 2, 1, Laminate.from_angles(weightless, [0.0], 0.005), CLPTModel())
    held = BoundaryConditions().clamp(empty.nodes_on_line(0, 0.0))
    with pytest.raises(ValueError, match="density"):
        vibration_modes(empty, held, n_modes=1)
    bcs.fix(mesh.nodes_on_line(0, 0.5), [DOF.UZ], value=1.0e-4)
    with pytest.raises(ValueError, match="homogeneous"):
        vibration_modes(mesh, bcs, n_modes=1)
