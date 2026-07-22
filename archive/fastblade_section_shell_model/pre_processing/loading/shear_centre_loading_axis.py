"""Optional shear-centre ``about_yz`` for beam→shell resultant transport.

Production FI defaults to the pitching / load-stacking origin
(:data:`~blade_precompute.contract.BEAM_LOAD_AXIS_YZ`) and transports to the
elastic centroid inside ``run_section_periodic_bc``. Use this helper to
reference moments about the sectorial shear centre for diagnostics.
"""

from __future__ import annotations

from typing import Any

from blade_precompute.section_shell_model.engine.sectorial_warping import (
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
