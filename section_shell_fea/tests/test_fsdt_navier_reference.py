"""Independent continuum-limit tests for the FSDT Navier oracle."""

from __future__ import annotations

import pytest

from blade_structure_utilities.fem_utilities.section_shell_fea.validation.fsdt_navier import (
    simply_supported_uniform_pressure_fsdt,
    simply_supported_uniform_pressure_kirchhoff,
)


def test_fsdt_converges_to_kirchhoff_as_shear_stiffness_increases() -> None:
    # Arrange
    kwargs = dict(
        length_x=1.0,
        length_y=0.8,
        bending_stiffness=1200.0,
        pressure=1000.0,
        modes=3,
    )

    # Act
    kirchhoff = simply_supported_uniform_pressure_kirchhoff(**kwargs)
    fsdt = simply_supported_uniform_pressure_fsdt(
        **kwargs, poisson=0.3, shear_stiffness=1.0e15
    )

    # Assert
    assert fsdt == pytest.approx(kirchhoff, rel=1.0e-5)


def test_thick_plate_includes_positive_shear_compliance() -> None:
    # Arrange
    kwargs = dict(
        length_x=1.0,
        length_y=1.0,
        bending_stiffness=1200.0,
        pressure=1000.0,
        modes=3,
    )

    # Act
    kirchhoff = simply_supported_uniform_pressure_kirchhoff(**kwargs)
    thick = simply_supported_uniform_pressure_fsdt(
        **kwargs, poisson=0.3, shear_stiffness=2.0e4
    )

    # Assert
    assert thick > kirchhoff
