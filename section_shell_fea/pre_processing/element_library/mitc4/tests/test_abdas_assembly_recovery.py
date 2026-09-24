"""Assembly/recovery parity tests for the canonical ABDAs element path."""

from __future__ import annotations

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.mitc4.element import (
    assemble_mitc4_stiffness,
    mitc4_generalized_resultants_at,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.mitc4.utilities import (
    MaterialStiffnessOperator,
    ShapeFunctionOperator,
    StrainDisplacementOperator,
)
from blade_structure_utilities.material_utilities.laminate import (
    FSDTLaminateStiffness,
)


def _constitutive() -> FSDTLaminateStiffness:
    a = np.diag([120.0, 90.0, 40.0])
    b = np.array([[1.0, 0.2, 0.0], [0.2, 0.7, 0.0], [0.0, 0.0, 0.3]])
    d = np.diag([12.0, 9.0, 4.0])
    abd = np.block([[a, b], [b.T, d]])
    a_s = np.array([[8.0, 1.5], [1.5, 5.0]])
    c = np.zeros((8, 8))
    c[:6, :6] = abd
    c[6:, 6:] = a_s
    return FSDTLaminateStiffness(a, b, d, a_s, abd, c, "explicit")


def test_named_stiffness_contributions_sum_to_total_and_are_symmetric() -> None:
    # Arrange
    constitutive = _constitutive()
    shape = ShapeFunctionOperator(L_s=0.7, L_x=1.3)
    material = MaterialStiffnessOperator(
        constitutive, thickness=0.02, alpha_drill=0.1
    )

    # Act
    contributions = assemble_mitc4_stiffness(shape, material)

    # Assert
    total_from_names = sum(
        (
            contributions.K_membrane,
            contributions.K_membrane_bending,
            contributions.K_bending,
            contributions.K_shear,
            contributions.K_drilling,
        )
    )
    assert np.allclose(contributions.K_e, total_from_names)
    assert np.allclose(contributions.K_e, contributions.K_e.T)
    assert np.min(np.linalg.eigvalsh(contributions.K_e)) > -1.0e-10


def test_recovery_uses_same_dg_and_q_equals_as_gamma() -> None:
    # Arrange
    constitutive = _constitutive()
    shape = ShapeFunctionOperator(L_s=0.7, L_x=1.3)
    displacement = np.linspace(-0.02, 0.03, 24)
    xi, eta = 0.17, -0.23

    # Act
    result = mitc4_generalized_resultants_at(
        displacement,
        0.7,
        1.3,
        constitutive,
        xi=xi,
        eta=eta,
    )
    generalized_strain = StrainDisplacementOperator(shape).generalized(
        np.array([xi]), np.array([eta])
    )[0] @ displacement
    expected = constitutive.C_ABDAs @ generalized_strain

    # Assert
    assert np.allclose(list(result.values()), expected)
    assert np.allclose([result["Q1"], result["Q2"]], constitutive.A_s @ generalized_strain[6:])


def test_drilling_stabilization_removes_only_four_nodal_drilling_null_modes() -> None:
    # Arrange
    constitutive = _constitutive()
    shape = ShapeFunctionOperator(L_s=0.7, L_x=1.3)
    unstabilized = MaterialStiffnessOperator(
        constitutive, thickness=0.02, alpha_drill=0.0
    )
    stabilized = MaterialStiffnessOperator(
        constitutive, thickness=0.02, alpha_drill=0.1
    )

    # Act
    eigen_unstabilized = np.linalg.eigvalsh(
        assemble_mitc4_stiffness(shape, unstabilized).K_e
    )
    eigen_stabilized = np.linalg.eigvalsh(
        assemble_mitc4_stiffness(shape, stabilized).K_e
    )
    tolerance = 1.0e-9 * np.max(eigen_stabilized)

    # Assert
    null_unstabilized = int(np.count_nonzero(eigen_unstabilized < tolerance))
    null_stabilized = int(np.count_nonzero(eigen_stabilized < tolerance))
    assert null_unstabilized - null_stabilized == 4
    assert null_stabilized == 6


def test_orientation_preserving_cyclic_node_numbering_preserves_energy() -> None:
    # Arrange
    coordinates = np.array(
        [[-0.8, -0.5], [-0.9, 0.6], [1.1, 0.8], [0.7, -0.7]]
    )
    base_shape = ShapeFunctionOperator.from_local_coordinates(coordinates)
    shifted_shape = ShapeFunctionOperator.from_local_coordinates(
        np.roll(coordinates, -1, axis=0)
    )
    material = MaterialStiffnessOperator(_constitutive(), thickness=0.02)
    base = assemble_mitc4_stiffness(base_shape, material).K_e
    shifted = assemble_mitc4_stiffness(shifted_shape, material).K_e
    map_new_to_old = np.r_[np.arange(6, 24), np.arange(0, 6)]
    permutation = np.eye(24)[map_new_to_old]

    # Act
    shifted_in_base_order = permutation.T @ shifted @ permutation

    # Assert
    np.testing.assert_allclose(base, shifted_in_base_order, rtol=1e-11, atol=1e-11)
