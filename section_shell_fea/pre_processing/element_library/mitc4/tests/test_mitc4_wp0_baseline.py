"""WP0 baselines for the legacy MITC4 public API and sign convention.

``contract-invariant`` baselines should survive the canonical-sign migration.
``legacy-sign-sensitive`` baselines intentionally freeze values that may change
only when the documented curvature/coupling migration is implemented.
"""

from __future__ import annotations

import inspect

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.mitc4 import (
    element as mitc4,
)


def _symmetric_abd() -> np.ndarray:
    """Return a deterministic symmetric-laminate stiffness fixture."""
    a_mat = np.array(
        [[120.0, 20.0, 5.0], [20.0, 90.0, 3.0], [5.0, 3.0, 40.0]]
    )
    d_mat = np.array(
        [[12.0, 2.0, 0.5], [2.0, 9.0, 0.3], [0.5, 0.3, 4.0]]
    )
    zeros = np.zeros((3, 3))
    return np.block([[a_mat, zeros], [zeros, d_mat]])


def _representative_dofs() -> np.ndarray:
    """Return four legacy five-DOF nodal rows in public order."""
    return np.array(
        [
            0.01,
            -0.02,
            0.03,
            0.04,
            -0.05,
            0.02,
            0.01,
            -0.01,
            0.02,
            0.03,
            -0.01,
            0.04,
            0.02,
            -0.03,
            0.01,
            0.00,
            -0.01,
            0.05,
            0.01,
            -0.02,
        ]
    )


def test_scalar_public_api_signatures_match_wp0_baseline() -> None:
    """Classification: contract-invariant API baseline."""
    # Arrange
    expected = {
        "mitc4_stiffness": (
            "(L_s: 'float', L_x: 'float', ABD: 'NDArray', thickness: 'float', "
            "ks: 'float' = 0.8333333333333334, G_eff: 'float | None' = None, "
            "include_drilling: 'bool' = True, alpha_drill: 'float' = 0.25) "
            "-> 'NDArray'"
        ),
        "mitc4_resultants": (
            "(d_elem: 'NDArray', L_s: 'float', L_x: 'float', ABD: 'NDArray', "
            "ndof: 'int' = 6) -> 'dict[str, float]'"
        ),
        "mitc4_resultants_at": (
            "(d_elem: 'NDArray', L_s: 'float', L_x: 'float', ABD: 'NDArray', "
            "*, xi: 'float', eta: 'float', ndof: 'int' = 6) -> 'dict[str, float]'"
        ),
        "mitc4_geometric_stiffness": (
            "(L_s: 'float', L_x: 'float', N11: 'float', N22: 'float', "
            "N12: 'float', *, n_gauss: 'int' = 2, ndof: 'int' = 6) -> 'NDArray'"
        ),
        "mitc4_shear_resultants": (
            "(d_elem: 'NDArray', L_s: 'float', L_x: 'float', G_eff: 'float', "
            "thickness: 'float', *, ks: 'float' = 0.8333333333333334, "
            "ndof: 'int' = 6) -> 'tuple[float, float]'"
        ),
        "mitc4_edge_resultants": (
            "(d_elem: 'NDArray', L_s: 'float', L_x: 'float', ABD: 'NDArray', "
            "ndof: 'int' = 6) -> 'dict[str, dict[str, float]]'"
        ),
        "mitc4_edge_shear_traction_integrated": (
            "(d_elem: 'NDArray', L_s: 'float', L_x: 'float', ABD: 'NDArray', "
            "*, edge: 'str', gauss_n: 'int' = 4, ndof: 'int' = 6) "
            "-> 'dict[str, object]'"
        ),
    }

    # Act
    actual = {
        name: str(inspect.signature(getattr(mitc4, name))) for name in expected
    }

    # Assert
    assert actual == expected


def test_symmetric_laminate_stiffness_matches_wp0_baseline() -> None:
    """Classification: contract-invariant symmetric stiffness baseline."""
    # Arrange
    abd = _symmetric_abd()

    # Act
    stiffness = mitc4.mitc4_stiffness(
        0.4,
        1.2,
        abd,
        0.02,
        ks=0.8,
        G_eff=30.0,
        include_drilling=False,
    )
    observed = np.array(
        [
            stiffness[0, 0],
            stiffness[0, 5],
            stiffness[2, 2],
            stiffness[3, 3],
            stiffness[4, 4],
            stiffness[7, 7],
            stiffness[10, 10],
            stiffness[19, 19],
            np.trace(stiffness),
            np.linalg.norm(stiffness),
        ]
    )
    expected = np.array(
        [
            55.833333333333336,
            -33.333333333333336,
            0.5333333333333333,
            5.602533333333333,
            9.613644444444443,
            0.5333333333333333,
            55.833333333333336,
            9.313644444444442,
            652.5091555555555,
            325.81735800937213,
        ]
    )

    # Assert
    assert stiffness.shape == (20, 20)
    np.testing.assert_allclose(stiffness, stiffness.T, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(observed, expected, rtol=1e-13, atol=1e-13)


def test_symmetric_laminate_nmq_matches_normative_sign_migration() -> None:
    """Intentional migration: canonical kappa reverses the legacy bending sign."""
    # Arrange
    abd = _symmetric_abd()
    dofs = _representative_dofs()

    # Act
    nm = mitc4.mitc4_resultants(dofs, 0.4, 1.2, abd, ndof=5)
    q = mitc4.mitc4_shear_resultants(
        dofs, 0.4, 1.2, 30.0, 0.02, ks=0.8, ndof=5
    )

    # Assert
    assert tuple(nm) == ("N11", "N22", "N12", "M11", "M22", "M12")
    np.testing.assert_allclose(
        list(nm.values()),
        [
            0.08333333333333337,
            8.716666666666667,
            0.8833333333333335,
            0.7145833333333333,
            1.3279166666666666,
            0.3745833333333333,
        ],
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(q, [0.0052, -0.04560000000000001], rtol=1e-13)


def test_unsymmetric_laminate_nmq_matches_normative_coupling_sign() -> None:
    """Intentional migration: B coupling follows the canonical curvature sign."""
    # Arrange
    abd = _symmetric_abd()
    b_mat = np.array([[7.0, -2.0, 1.0], [-2.0, 5.0, 0.5], [1.0, 0.5, 3.0]])
    abd[:3, 3:] = b_mat
    abd[3:, :3] = b_mat.T
    dofs = _representative_dofs()

    # Act
    nm = mitc4.mitc4_resultants(dofs, 0.4, 1.2, abd, ndof=5)
    q = mitc4.mitc4_shear_resultants(
        dofs, 0.4, 1.2, 30.0, 0.02, ks=0.8, ndof=5
    )

    # Assert
    np.testing.assert_allclose(
        list(nm.values()),
        [
            0.12083333333333335,
            9.377083333333333,
            1.2229166666666669,
            0.41458333333333336,
            1.8695833333333334,
            0.4579166666666667,
        ],
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(q, [0.0052, -0.04560000000000001], rtol=1e-13)
