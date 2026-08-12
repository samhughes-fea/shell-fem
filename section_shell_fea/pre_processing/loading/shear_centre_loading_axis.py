"""Optional shear-centre ``about_yz`` for beam→shell resultant transport.

Production FI defaults to the pitching / load-stacking origin
(:data:`~blade_structure_utilities.geometry_utilities.section_contract.BEAM_LOAD_AXIS_YZ`) and transports to the
elastic centroid inside ``run_section_linear_static_from_resultants``.

Opt-in Saint-Venant torsion about the sectorial shear centre uses
:func:`shear_centre_about_yz_from_panel_topology` (or the lower-level outer-skin
helper) so pure ``T`` maps as twist about SC rather than the pitching origin.
This is **not** a Channel-A periodic ``K6`` revive.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.loading.thin_wall.sectorial_warping import (
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


def shear_centre_about_yz_from_panel_topology(
    panel_topology: Any,
    *,
    E_mean: float = 70e9,
) -> tuple[float, float]:
    """Sectorial shear centre from the the panel topology outer-skin midline.

    Uses the first ``kind=="skin"`` midline polyline and its thickness. Suitable
    for opt-in ``torsion_about_shear_centre`` on the linear-static resultant path.
    """
    shell_inputs = getattr(panel_topology, "shell_inputs", None)
    midlines = getattr(shell_inputs, "midlines", None) or ()
    skins = [m for m in midlines if str(getattr(m, "kind", "")).lower() == "skin"]
    if not skins:
        raise ValueError(
            "shear_centre_about_yz_from_panel_topology: no skin midline on ShellPanelTopology"
        )
    m0 = skins[0]
    verts = np.asarray(
        getattr(m0, "midline_b", None)
        if getattr(m0, "midline_b", None) is not None
        else getattr(m0, "points", None),
        dtype=float,
    )
    if verts.ndim != 2 or verts.shape[0] < 3 or verts.shape[1] < 2:
        raise ValueError(
            f"skin midline must be (N,2) with N>=3; got shape {getattr(verts, 'shape', None)!r}"
        )
    t = float(getattr(m0, "thickness_m", 0.01) or 0.01)
    return shear_centre_about_yz_from_outer_skin(verts[:, :2], t, float(E_mean))


__all__ = [
    "shear_centre_about_yz_from_outer_skin",
    "shear_centre_about_yz_from_panel_topology",
]
