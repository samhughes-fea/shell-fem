"""Equation-level tests for the flat-Q4 Reissner--Mindlin operators."""

from __future__ import annotations

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.mitc4.utilities import (
    FlatQuad4Geometry,
    ShapeFunctionOperator,
    StrainDisplacementOperator,
)


def _distorted_shape() -> ShapeFunctionOperator:
    return ShapeFunctionOperator.from_local_coordinates(
        np.array(
            [
                [-0.8, -0.5],
                [-0.9, 0.6],
                [1.1, 0.8],
                [0.7, -0.7],
            ]
        )
    )


def test_basis_partition_and_physical_linear_completeness() -> None:
    # Arrange
    shape = _distorted_shape()
    xi = np.array([-0.7, 0.0, 0.5])
    eta = np.array([0.4, -0.2, 0.8])

    # Act
    phi, gradients, determinant, _ = shape.physical_coordinate_form(xi, eta)
    coordinates = shape.geometry.local_coordinates

    # Assert
    assert np.allclose(np.sum(phi, axis=1), 1.0)
    assert np.allclose(np.sum(gradients, axis=1), 0.0)
    reproduced = np.einsum("gia,ib->gab", gradients, coordinates)
    assert np.allclose(reproduced, np.eye(2)[None, :, :])
    assert np.all(determinant > 0.0)


def test_displacement_interpolation_uses_canonical_rotation_adapter() -> None:
    # Arrange
    shape = ShapeFunctionOperator(L_s=2.0, L_x=4.0)
    nodal = np.zeros(24)
    for node in range(4):
        nodal[node * 6 : node * 6 + 6] = [2.0, -3.0, 4.0, 5.0, 7.0, 11.0]

    # Act
    field = shape.displacement_interpolation(
        np.array([0.23]), np.array([-0.41]), ndof=6
    )[0] @ nodal

    # Assert
    assert np.allclose(field, [2.0, -3.0, 4.0, 5.0, -7.0, 11.0])


def test_compatible_shear_vanishes_when_theta_equals_gradient_w() -> None:
    # Arrange
    shape = _distorted_shape()
    operator = StrainDisplacementOperator(shape)
    coordinates = shape.geometry.local_coordinates
    gradient_w = np.array([0.31, -0.27])
    nodal = np.zeros(24)
    for node, (x1, x2) in enumerate(coordinates):
        base = node * 6
        nodal[base + 2] = gradient_w @ np.array([x1, x2])
        nodal[base + 3] = gradient_w[0]
        nodal[base + 4] = -gradient_w[1]

    # Act
    points = np.array([-0.77, -0.1, 0.63])
    compatible = operator.compatible_shear(points, points[::-1]) @ nodal
    tied = operator.mitc_shear(points, points[::-1]) @ nodal

    # Assert
    assert np.allclose(compatible, 0.0, atol=1.0e-13)
    assert np.allclose(tied, 0.0, atol=1.0e-13)


def test_constant_membrane_curvature_and_shear_patch_states() -> None:
    # Arrange
    shape = _distorted_shape()
    operator = StrainDisplacementOperator(shape)
    coordinates = shape.geometry.local_coordinates
    epsilon = np.array([0.012, -0.021, 0.017])
    kappa = np.array([0.031, -0.019, 0.011])
    gamma = np.array([0.23, -0.14])
    membrane_dofs = np.zeros(24)
    bending_dofs = np.zeros(24)
    shear_dofs = np.zeros(24)
    for node, (x1, x2) in enumerate(coordinates):
        base = node * 6
        membrane_dofs[base + 0] = epsilon[0] * x1 + 0.5 * epsilon[2] * x2
        membrane_dofs[base + 1] = epsilon[1] * x2 + 0.5 * epsilon[2] * x1
        theta1 = -kappa[0] * x1 - 0.5 * kappa[2] * x2
        theta2 = -kappa[1] * x2 - 0.5 * kappa[2] * x1
        bending_dofs[base + 3] = theta1
        bending_dofs[base + 4] = -theta2
        shear_dofs[base + 2] = gamma @ np.array([x1, x2])
    xi = np.array([-0.73, -0.11, 0.57])
    eta = np.array([0.66, -0.28, 0.13])

    # Act
    bm, bb, bs, _ = operator.natural_coordinate_form(xi, eta)

    # Assert
    assert np.allclose(bm @ membrane_dofs, epsilon)
    assert np.allclose(bb @ bending_dofs, kappa)
    assert np.allclose(bs @ shear_dofs, gamma)


def test_folded_geometry_is_rejected() -> None:
    # Arrange / Act / Assert
    with np.testing.assert_raises(ValueError):
        FlatQuad4Geometry(
            np.array(
                [
                    [-1.0, -1.0],
                    [1.0, 1.0],
                    [1.0, -1.0],
                    [-1.0, 1.0],
                ]
            )
        )
