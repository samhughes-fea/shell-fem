"""Optional shear-centre ``about_yz`` for beam→shell resultant transport.

Production FI defaults to the pitching / load-stacking origin
(:data:`~blade_structure_utilities.section_contract.BEAM_LOAD_AXIS_YZ`) and transports to the
elastic centroid inside ``run_section_linear_static_from_resultants``. Use this helper to
reference moments about the sectorial shear centre for diagnostics.
"""

from __future__ import annotations

from typing import Any

from blade_structure_utilities.section_shell_model.pre_processing.loading.thin_wall.sectorial_warping import (
    shear_center_outer_skin_loop,
)


def shear_centre_about_yz_from_outer_skin(
    verts: Any,
    thickness: float,
    E_mean: float,
) -> tuple[float, float]:
    """Return ``(y_sc, z_sc)`` from the outer-skin sectorial estimate."""
    y_sc, z_sc, _, _ = shear_center_outer_skin_loop(verts, thickness, E_mean)
    return float(y_sc), float(z_sc)


__all__ = ["shear_centre_about_yz_from_outer_skin"]
