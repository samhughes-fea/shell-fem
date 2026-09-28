"""Optional shear-centre ``about_xy`` for beam→shell resultant transport.

Sectorial shear-centre estimates used when
``SectionLinearStaticRunner(..., torsion_about_shear_centre=True)`` retags the
moment reference. Prefer wiring the returned ``(X_*, Y_*)`` into
``BeamResultantVector.about_xy`` / runner ``about_xy=``.

Production FI defaults to the pitching / load-stacking origin
(:data:`~blade_structure_utilities.geometry_utilities.section_contract.BEAM_LOAD_AXIS_XY`)
and transports to the elastic centroid inside
``run_section_linear_static_from_resultants``.

Opt-in Saint-Venant torsion about the sectorial shear centre uses
:func:`shear_centre_about_yz_from_panel_topology` (or the lower-level outer-skin
helper) so pure ``T`` maps as twist about SC rather than the pitching origin.
This is **not** a Channel-A periodic ``K6`` revive.

When ``ShellStripMeshingInputs.cell_wall_loops`` are present, SC comes from
``shear_centre_from_midlines`` (thin_wall_section / FEA-SC-002). Otherwise the
outer-skin sectorial estimate is retained (FEA-SC-001 thickness refuse).
Sectorial ω̂ / Batho / ``I_ω`` live on ``thin_wall_section.sectorial_warping``
(TW-002); FEA ``compute_section_vlasov`` consumes that kernel when midlines and
loops are present.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.reference_thin_wall.sectorial_warping import (
    shear_center_outer_skin_loop,
)


def _kind_of_label(label: str) -> str:
    """Panel kind from label prefix (``skin:`` / ``cap:`` / ``web:``).

    Local classifier — avoids importing processing constraint geometry from
    pre-processing loading adapters.
    """
    return label.split(":", 1)[0] if ":" in label else "panel"


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
    """Sectorial shear centre from panel topology midlines.

    Prefer ``cell_wall_loops`` → ``shear_centre_from_midlines`` (FEA-SC-002).
    Fallback: first skin midline + outer-skin sectorial estimate (FEA-SC-001
    thickness refuse still applies).
    """
    shell_inputs = getattr(panel_topology, "shell_inputs", None)
    midlines = getattr(shell_inputs, "midlines", None) or ()
    loops = getattr(shell_inputs, "cell_wall_loops", None) or ()
    if midlines and loops:
        from blade_structure_utilities.geometry_utilities.section_properties.reference_axes.shear_centre import (
            shear_centre_from_midlines,
        )

        sc = shear_centre_from_midlines(midlines, cell_wall_loops=loops)
        return float(sc.xs), float(sc.ys)

    skins = [
        m
        for m in midlines
        if str(getattr(m, "kind", "")).lower() == "skin"
        or _kind_of_label(str(getattr(m, "label", ""))) == "skin"
    ]
    if not skins:
        raise ValueError(
            "shear_centre_about_yz_from_panel_topology: no skin midline on ShellPanelTopology"
        )
    m0 = skins[0]
    verts = np.asarray(
        getattr(m0, "midline_twisted_pitch_axis_blade_frame", None)
        if getattr(m0, "midline_twisted_pitch_axis_blade_frame", None) is not None
        else getattr(m0, "points", None),
        dtype=float,
    )
    if verts.ndim != 2 or verts.shape[0] < 3 or verts.shape[1] < 2:
        raise ValueError(
            f"skin midline must be (N,2) with N>=3; got shape {getattr(verts, 'shape', None)!r}"
        )
    t = float(getattr(m0, "thickness_m", 0.0) or 0.0)
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError(
            "shear_centre_about_yz_from_panel_topology: skin midline "
            f"{getattr(m0, 'label', '<unknown>')!r} has no positive thickness_m "
            f"(got {t!r}; refusing silent 0.01 m default — FEA-SC-001)"
        )
    return shear_centre_about_yz_from_outer_skin(verts[:, :2], t, float(E_mean))


__all__ = [
    "shear_centre_about_yz_from_outer_skin",
    "shear_centre_about_yz_from_panel_topology",
]
