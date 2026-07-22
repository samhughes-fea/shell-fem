"""Beam + face-traction solving on one periodic constrained free set.

Face loads are reduced with the same ``T_pbc`` used by K6 homogenisation.
Legacy contour-clamped overlay is retired: there is no separate
``build_clamp_supports`` solve and no ``d_beam + d_tr_contour`` path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .periodic_reduced_bvp import PeriodicReducedBVP, build_periodic_reduced_bvp

# Provenance tag for recovery when face traction is active on the shared BVP.
PERIODIC_FACE_TRACTION_PROVENANCE = "periodic BC + face traction (single BVP)"


def traction_face_force_vector(
    node_meta: Any,
    *,
    traction_field: Any,
    traction_points_yz: np.ndarray,
    ndof: int,
    L_x: float = 1.0,
) -> NDArray[np.float64]:
    """Assemble Channel-B nodal forces from a ``TractionField``."""
    from ..interface.traction_face_load import traction_face_loads

    return np.asarray(
        traction_face_loads(
            node_meta,
            np.asarray(traction_field.total_forces, dtype=float),
            np.asarray(traction_points_yz, dtype=float),
            ndof=ndof,
            L_x=L_x,
        ),
        dtype=np.float64,
    )


def _resolve_periodic_bvp(
    K_global: Any,
    node_meta: Any,
    panel_maps: list[Any],
    *,
    ndof: int,
    L_x: float,
    periodic_bvp: PeriodicReducedBVP | None,
    y_e: float | None,
    z_e: float | None,
) -> PeriodicReducedBVP:
    if periodic_bvp is not None:
        return periodic_bvp
    ye = 0.0 if y_e is None else float(y_e)
    ze = 0.0 if z_e is None else float(z_e)
    return build_periodic_reduced_bvp(
        K_global,
        node_meta,
        panel_maps,
        y_e=ye,
        z_e=ze,
        L_x=L_x,
        ndof=ndof,
    )


def traction_displacement_periodic(
    bvp: PeriodicReducedBVP,
    f_tr: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve face traction on the periodic free set (zero macro jump)."""
    return bvp.solve_external_load(f_tr)


def traction_displacement(
    K_global: Any,
    node_meta: Any,
    panel_maps: list[Any],
    *,
    traction_field: Any,
    traction_points_yz: np.ndarray,
    ndof: int,
    bc_mode: str = "contour_support",
    L_x: float = 1.0,
    clamp_dofs_extra: tuple[int, ...] = (),
    periodic_bvp: PeriodicReducedBVP | None = None,
    y_e: float | None = None,
    z_e: float | None = None,
) -> np.ndarray:
    """Solve face traction on the periodic free set.

    ``bc_mode`` / ``clamp_dofs_extra`` are ignored (API compatibility); contour
    clamps are not applied.
    """
    del bc_mode, clamp_dofs_extra
    bvp = _resolve_periodic_bvp(
        K_global,
        node_meta,
        panel_maps,
        ndof=ndof,
        L_x=L_x,
        periodic_bvp=periodic_bvp,
        y_e=y_e,
        z_e=z_e,
    )
    f_tr = traction_face_force_vector(
        node_meta,
        traction_field=traction_field,
        traction_points_yz=traction_points_yz,
        ndof=ndof,
        L_x=L_x,
    )
    return traction_displacement_periodic(bvp, f_tr)


def superpose_traction_on_beam_displacement(
    d_beam: np.ndarray,
    K_global: Any,
    node_meta: Any,
    panel_maps: list[Any],
    *,
    traction_field: Any | None,
    traction_points_yz: np.ndarray | None,
    ndof: int,
    bc_mode: str = "contour_support",
    L_x: float = 1.0,
    clamp_dofs_extra: tuple[int, ...] = (),
    periodic_bvp: PeriodicReducedBVP | None = None,
    y_e: float | None = None,
    z_e: float | None = None,
) -> np.ndarray:
    """Return ``d_beam + d_tr`` with both fields on the periodic free set."""
    del bc_mode, clamp_dofs_extra
    d_total = np.asarray(d_beam, dtype=float)
    if traction_field is None or traction_points_yz is None:
        return d_total
    d_tr = traction_displacement(
        K_global,
        node_meta,
        panel_maps,
        traction_field=traction_field,
        traction_points_yz=traction_points_yz,
        ndof=ndof,
        L_x=L_x,
        periodic_bvp=periodic_bvp,
        y_e=y_e,
        z_e=z_e,
    )
    return d_total + d_tr


__all__ = [
    "PERIODIC_FACE_TRACTION_PROVENANCE",
    "superpose_traction_on_beam_displacement",
    "traction_displacement",
    "traction_displacement_periodic",
    "traction_face_force_vector",
]
