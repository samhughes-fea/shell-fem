"""Labelled strip-heave added mass on shells (Newman 1977 / Theodorsen 1935).

Default vibration is unchanged. The hook attaches only when a StripHeaveAddedMass
is passed; it uses the same per-span formula as 1d-fem, spread over the chord.
"""

import numpy as np
import pytest

from conftest import simply_supported
from shell_fem import (
    AssembleAddedMassMatrix,
    FEAMesh,
    FSDTModel,
    Laminate,
    STRIP_HEAVE_FORMULA,
    STRIP_HEAVE_SOURCE,
    StripHeaveAddedMass,
    element_added_mass_matrix,
    solid_block,
    strip_heave_added_mass_per_length,
    strip_heave_areal_density,
    vibration_modes,
)
from shell_fem.processing.vibration import AssembleMassMatrix


def _dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def test_strip_heave_formula_is_newman_theodorsen():
    rho, chord = 1025.0, 0.8
    ma = strip_heave_added_mass_per_length(rho, chord)
    assert ma == pytest.approx(rho * np.pi * (chord / 2.0) ** 2)
    assert strip_heave_areal_density(rho, chord) == pytest.approx(ma / chord)
    assert STRIP_HEAVE_FORMULA == "rho_fluid * pi * (chord/2)**2"
    assert "Newman" in STRIP_HEAVE_SOURCE and "Theodorsen" in STRIP_HEAVE_SOURCE
    labelled = StripHeaveAddedMass(rho_fluid=rho, chord=chord)
    assert labelled.model == "strip_heave"
    assert labelled.mass_per_length == pytest.approx(ma)
    assert labelled.areal_density == pytest.approx(rho * np.pi * chord / 4.0)
    with pytest.raises(ValueError, match="strip_heave"):
        StripHeaveAddedMass(rho, chord, model="panel_3d")
    with pytest.raises(ValueError, match="rho_fluid"):
        StripHeaveAddedMass(-1.0, chord)
    with pytest.raises(ValueError, match="chord"):
        strip_heave_added_mass_per_length(rho, -0.1)


def test_element_added_mass_of_a_unit_normal_is_the_strip_mass(aluminium):
    chord, span, h = 0.5, 1.2, 0.005
    mesh = FEAMesh.rectangular_plate(
        chord, span, 1, 1, Laminate.from_angles(aluminium, [0.0], h), FSDTModel()
    )
    element = mesh.elements[0]
    model = StripHeaveAddedMass(rho_fluid=1000.0, chord=chord)
    Ma = element_added_mass_matrix(element, model)
    assert np.allclose(Ma, Ma.T, atol=1e-12 * np.abs(Ma).max())
    u = np.zeros(element.n_dof)
    u[2::element.dofs_per_node] = 1.0
    expected = strip_heave_added_mass_per_length(1000.0, chord) * span
    assert float(u @ Ma @ u) == pytest.approx(expected, rel=1e-10)
    in_plane = np.zeros(element.n_dof)
    in_plane[0::element.dofs_per_node] = 1.0
    assert abs(float(in_plane @ Ma @ in_plane)) < 1e-12 * expected


def test_default_vibration_is_unchanged_and_labelled_mass_lowers_heave(aluminium):
    a, h = 0.2, 0.002
    mesh = FEAMesh.rectangular_plate(a, a, 4, 4, Laminate.from_angles(aluminium, [0.0], h), FSDTModel())
    bcs = simply_supported(mesh, a, a)
    air = vibration_modes(mesh, bcs, n_modes=2)
    also = vibration_modes(mesh, bcs, n_modes=2, added_mass=None)
    assert np.array_equal(air.angular_frequencies_rad_s, also.angular_frequencies_rad_s)
    wet = vibration_modes(mesh, bcs, n_modes=2, added_mass=StripHeaveAddedMass(1000.0, a))
    assert np.all(wet.angular_frequencies_rad_s <= air.angular_frequencies_rad_s + 1e-12)
    assert np.any(wet.angular_frequencies_rad_s < air.angular_frequencies_rad_s)


def test_added_mass_skips_layered_solids(aluminium):
    L, b, h = 0.04, 0.01, 0.008
    lam = Laminate.from_angles(aluminium, [0.0], h)
    named = solid_block(FEAMesh(), lam, L, b, 1, 1, element="hex8", layers="laminate")
    Ma = _dense(AssembleAddedMassMatrix(named.mesh, StripHeaveAddedMass(1000.0, b)).run())
    assert Ma.shape == (named.mesh.n_dof, named.mesh.n_dof)
    assert np.allclose(Ma, 0.0)
    M = _dense(AssembleMassMatrix(named.mesh).run())
    assert float(np.linalg.norm(M)) > 0.0
