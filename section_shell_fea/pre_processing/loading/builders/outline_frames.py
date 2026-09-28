"""Compat re-export — outline B↔NACA hop lives on the loads worker.

Canonical home:
``blade_structure_utilities.load_utilities.section_loads.surface_traction.outline_frames``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Import outline-frame helpers from "
    "blade_structure_utilities.load_utilities.section_loads.surface_traction.outline_frames "
    "(or the section_loads barrel). FE builders.outline_frames is a temporary re-export.",
    DeprecationWarning,
    stacklevel=2,
)

from blade_structure_utilities.load_utilities.section_loads.surface_traction import (  # noqa: E402,F401
    blade_outline_to_unit_chord,
    blade_to_naca_rotation,
    rotate_traction_field,
    xfoil_te_arc_offset,
)

__all__ = [
    "blade_outline_to_unit_chord",
    "blade_to_naca_rotation",
    "rotate_traction_field",
    "xfoil_te_arc_offset",
]
