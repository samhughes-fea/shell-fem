"""Tests for the non-Hashin FSDT interface-shear screening criterion."""

from __future__ import annotations

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_quaternary.compute_hashin_fi import (
    interface_shear_screening_from_tau,
)
from blade_structure_utilities.material_utilities.laminate import LaminateDefinition
from blade_structure_utilities.material_utilities.orthotropic import OrthotropicPly


def _ply(*, angle_strength: float, s13: float | None, s23: float | None) -> OrthotropicPly:
    return OrthotropicPly(
        name=f"ply-{angle_strength}",
        E1=30e9,
        E2=8e9,
        G12=3e9,
        nu12=0.3,
        rho=1500.0,
        t_ply=1e-3,
        Xt=500e6,
        Xc=300e6,
        Yt=50e6,
        Yc=200e6,
        S12=40e6,
        Zt=30e6,
        S13=s13,
        S23=s23,
    )


def test_distinct_strengths_and_adjacent_material_frames_are_used() -> None:
    # Arrange
    lower = _ply(angle_strength=0.0, s13=10.0, s23=20.0)
    upper = _ply(angle_strength=90.0, s13=40.0, s23=5.0)
    lam = LaminateDefinition(plies=[(lower, 0.0), (upper, 90.0)])
    tau = np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 0.0]])

    # Act
    result = interface_shear_screening_from_tau(tau, lam)

    # Assert
    # Lower: (10/10)^2 + (10/20)^2 = 1.25.
    # Upper 90deg: (10/40)^2 + (-10/5)^2 = 4.0625 and governs.
    assert np.isclose(result.failure_index, 4.0625)
    assert result.governing_interface == 1
    assert result.governing_side == "upper"
    assert np.isclose(result.tau13_ratio, 0.25)
    assert np.isclose(result.tau23_ratio, -2.0)
    assert not result.needs_review
    assert result.strength_provenance == "authoritative_s13_s23"


def test_missing_authoritative_strength_cannot_silently_pass() -> None:
    # Arrange
    lam = LaminateDefinition(
        plies=[
            (_ply(angle_strength=0.0, s13=None, s23=20.0), 0.0),
            (_ply(angle_strength=0.0, s13=10.0, s23=20.0), 0.0),
        ]
    )
    tau = np.array([[0.0, 0.0], [1.0, 2.0], [0.0, 0.0]])

    # Act
    result = interface_shear_screening_from_tau(tau, lam)
    compatibility = interface_shear_screening_from_tau(
        tau, lam, allow_s12_fallback=True
    )

    # Assert
    assert np.isnan(result.failure_index)
    assert result.needs_review
    assert result.strength_provenance == "missing_s13_s23"
    assert np.isfinite(compatibility.failure_index)
    assert compatibility.needs_review
    assert compatibility.strength_provenance == "compatibility_s12_fallback"


def test_untrusted_recovery_is_unavailable() -> None:
    # Arrange
    p = _ply(angle_strength=0.0, s13=10.0, s23=20.0)
    lam = LaminateDefinition(plies=[(p, 0.0), (p, 0.0)])

    # Act
    result = interface_shear_screening_from_tau(
        np.zeros((3, 2)), lam, trusted=False
    )

    # Assert
    assert np.isnan(result.failure_index)
    assert result.needs_review
    assert result.strength_provenance == "unavailable"
