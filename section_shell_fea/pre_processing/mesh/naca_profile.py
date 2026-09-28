"""Compat shim — import from ``visualisation.demos.thin_wall_naca_demo_profile``."""

from __future__ import annotations

import warnings

warnings.warn(
    "Import naca_profile from "
    "blade_structure_utilities.fem_utilities.section_shell_fea.visualisation.demos.thin_wall_naca_demo_profile.",
    DeprecationWarning,
    stacklevel=2,
)

from blade_structure_utilities.fem_utilities.section_shell_fea.visualisation.demos.thin_wall_naca_demo_profile import (  # noqa: E402,F401
    chordwise_stations,
    naca_four_digit,
)

__all__ = ["chordwise_stations", "naca_four_digit"]
