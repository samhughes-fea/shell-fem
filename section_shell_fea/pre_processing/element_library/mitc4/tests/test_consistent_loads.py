"""Virtual-work and total-resultant tests for MITC4 consistent loads."""

from __future__ import annotations

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.mitc4.utilities import (
    ElementLoadOperator,
    ShapeFunctionOperator,
)


def test_area_pressure_reproduces_total_force_and_constant_field_work() -> None:
    # Arrange
    shape = ShapeFunctionOperator(L_s=2.0, L_x=3.0)
    operator = ElementLoadOperator(shape)
    pressure = 17.0

    # Act
    force = operator.area_pressure(pressure)
    w_dofs = np.zeros(24)
    w_dofs[2::6] = 0.4

    # Assert
    assert np.isclose(np.sum(force[2::6]), pressure * 6.0)
    assert np.isclose(force @ w_dofs, pressure * 6.0 * 0.4)


def test_edge_loads_reproduce_force_moment_and_virtual_work() -> None:
    # Arrange
    shape = ShapeFunctionOperator(L_s=2.0, L_x=3.0)
    operator = ElementLoadOperator(shape)

    # Act
    force = operator.edge_resultants(
        "eta_plus",
        membrane_traction=(5.0, -7.0),
        edge_moment=(11.0, 13.0),
        transverse_shear=17.0,
    )
    constant_field = np.zeros(24)
    for node in range(4):
        constant_field[node * 6 : node * 6 + 5] = [0.2, -0.3, 0.4, 0.5, -0.6]

    # Assert
    assert np.isclose(np.sum(force[0::6]), 10.0)
    assert np.isclose(np.sum(force[1::6]), -14.0)
    assert np.isclose(np.sum(force[2::6]), 34.0)
    assert np.isclose(np.sum(force[3::6]), 22.0)
    assert np.isclose(np.sum(force[4::6]), -26.0)
    expected_work_density = 5.0 * 0.2 + (-7.0) * (-0.3) + 17.0 * 0.4
    expected_work_density += 11.0 * 0.5 + 13.0 * 0.6
    assert np.isclose(force @ constant_field, 2.0 * expected_work_density)


def test_legacy_strip_adapter_matches_existing_constant_nodal_loads() -> None:
    # Arrange
    shape = ShapeFunctionOperator(L_s=0.8, L_x=1.5)
    operator = ElementLoadOperator(shape)

    # Act
    force = operator.legacy_strip_load(nx=100.0, nxy=-40.0, kappa=0.2)

    # Assert
    expected = np.zeros(24)
    for node in (2, 3):
        expected[node * 6 + 0] = 40.0
        expected[node * 6 + 1] = -16.0
    expected[2::6] = 100.0 * 0.2 * 0.8 * 1.5 / 4.0
    assert np.allclose(force, expected)


def test_constant_initial_resultant_matches_boundary_flux_balance() -> None:
    # Arrange
    shape = ShapeFunctionOperator(L_s=2.0, L_x=3.0)
    operator = ElementLoadOperator(shape)
    resultants = np.array([5.0, 7.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    affine_field = np.zeros(24)
    coordinates = shape.geometry.local_coordinates
    for node, (x1, x2) in enumerate(coordinates):
        base = node * 6
        affine_field[base + 0] = 0.2 * x1 + 0.3 * x2
        affine_field[base + 1] = -0.1 * x2 + 0.3 * x1

    # Act
    force = operator.initial_resultants(resultants)

    # Assert
    strain = np.array([0.2, -0.1, 0.6])
    assert np.isclose(force @ affine_field, 6.0 * (resultants[:3] @ strain))
