"""Compatibility shim — moved to ``reference_thin_wall.sectorial_warping``.

Canonical home:
``blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.reference_thin_wall.sectorial_warping``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Import from blade_structure_utilities.fem_utilities.section_shell_fea."
    "pre_processing.reference_thin_wall.sectorial_warping. "
    "loading.thin_wall.sectorial_warping is a temporary re-export.",
    DeprecationWarning,
    stacklevel=2,
)

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.reference_thin_wall.sectorial_warping import (  # noqa: E402,F401
    modulus_weighted_open_line,
    omega_increment,
    omega_vertices_open_chain,
    open_outline_from_airfoil,
    polygon_area_signed,
    sectorial_static_moments_closed,
    sectorial_static_moments_open,
    shear_center_from_sectorial,
    shear_center_outer_skin_loop,
)

__all__ = [
    "omega_increment",
    "open_outline_from_airfoil",
    "omega_vertices_open_chain",
    "polygon_area_signed",
    "sectorial_static_moments_open",
    "sectorial_static_moments_closed",
    "shear_center_from_sectorial",
    "modulus_weighted_open_line",
    "shear_center_outer_skin_loop",
]
