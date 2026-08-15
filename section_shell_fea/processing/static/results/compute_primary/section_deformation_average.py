"""Primary/assembled kinematics add-on 1a: thickness×arc section deformation average.

Weighted average of nodal curvilinear-strip (**C**) DOFs about the pitch /
load-stacking origin (:data:`BEAM_LOAD_AXIS_XY` by default). Lives under
Primary/assembled — does not replace secondary→quaternary for FI.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.geometry_utilities.section_contract.beam_resultant_vector import BEAM_LOAD_AXIS_XY
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.loading.traction_face_load import (
    _skin_node_pairs,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.shell_mesh_map import normal_2d
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.connected_assembly_core import (
    BETA_S,
    BETA_X,
    BETA_ZET,
    U_S,
    U_X,
    W,
    gdof,
)


@dataclass(frozen=True)
class SectionDeformationAverage1a:
    """Thickness×arc weighted section-average kinematics (Primary/Global 1a)."""

    about_xy: tuple[float, float]
    """Moment / pitch reference used for the average metadata."""
    mean_local_dof: NDArray[np.float64]
    """Length-6 average ``[u_x, u_s, w, β_s, β_x, β_z]`` in curvilinear **C**."""
    mean_uyz: NDArray[np.float64]
    """Length-2 average **B**-plane displacement ``(u_y, u_z)`` from ``u_s, w``."""
    weight_sum: float
    n_pairs: int

    @property
    def about_yz(self) -> tuple[float, float]:
        """Deprecated alias of :attr:`about_xy`."""
        warnings.warn(
            "SectionDeformationAverage1a.about_yz is deprecated; use about_xy",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.about_xy


def _panel_thickness(panel: Any) -> float:
    lam = getattr(panel, "lam", None)
    if lam is None:
        return 1.0
    t = getattr(lam, "t", None)
    if t is not None:
        try:
            return max(float(t), 1e-12)
        except (TypeError, ValueError):
            pass
    return 1.0


def _pair_thicknesses(
    gn_pairs: list[tuple[int, int]],
    *,
    zones: list[Any] | None,
    panel_maps: list[Any] | None,
    thickness: float,
) -> NDArray[np.float64]:
    n = len(gn_pairs)
    t_arr = np.full(n, max(float(thickness), 1e-12), dtype=float)
    if not zones or not panel_maps:
        return t_arr
    gn_to_panel: dict[int, int] = {}
    for pm in panel_maps:
        pi = int(getattr(pm, "panel_index", -1))
        for gn in getattr(pm, "global_nodes", []) or []:
            if int(gn) >= 0:
                gn_to_panel[int(gn)] = pi
    for i, (gb, _gt) in enumerate(gn_pairs):
        pi = gn_to_panel.get(int(gb))
        if pi is not None and 0 <= pi < len(zones):
            t_arr[i] = _panel_thickness(zones[pi])
    return t_arr


def compute_section_deformation_average_1a(
    *,
    d_global: NDArray[np.float64],
    node_meta: Any,
    ndof: int = 6,
    about_xy: tuple[float, float] = BEAM_LOAD_AXIS_XY,
    zones: list[Any] | None = None,
    panel_maps: list[Any] | None = None,
    thickness: float = 1.0,
) -> SectionDeformationAverage1a:
    """Thickness×arc weighted average of nodal DOFs (Primary/Global 1a).

    Skin node-pairs (bottom+top) share the midline. Pair weight is ``t·ds``
    where ``ds`` is the local arc spacing along the ordered pair sequence.
    """
    origin = (float(about_xy[0]), float(about_xy[1]))
    locs, tangents, gn_pairs = _skin_node_pairs(node_meta)
    n = len(gn_pairs)
    if n == 0:
        return SectionDeformationAverage1a(
            about_xy=origin,
            mean_local_dof=np.zeros(6, dtype=float),
            mean_uyz=np.zeros(2, dtype=float),
            weight_sum=0.0,
            n_pairs=0,
        )

    t_pair = _pair_thicknesses(
        gn_pairs, zones=zones, panel_maps=panel_maps, thickness=thickness,
    )
    ds = np.zeros(n, dtype=float)
    if n == 1:
        ds[0] = 1.0
    else:
        for i in range(n):
            if i == 0:
                ds[i] = float(np.hypot(*(locs[1] - locs[0])))
            elif i == n - 1:
                ds[i] = float(np.hypot(*(locs[-1] - locs[-2])))
            else:
                ds[i] = 0.5 * (
                    float(np.hypot(*(locs[i] - locs[i - 1])))
                    + float(np.hypot(*(locs[i + 1] - locs[i])))
                )
            ds[i] = max(ds[i], 1e-12)
    w = t_pair * ds
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        w = np.ones(n, dtype=float)
        w_sum = float(n)

    d = np.asarray(d_global, dtype=float).ravel()
    acc = np.zeros(6, dtype=float)
    acc_uyz = np.zeros(2, dtype=float)
    local_dofs = (U_X, U_S, W, BETA_S, BETA_X, BETA_ZET)
    for i, (gb, gt) in enumerate(gn_pairs):
        wi = float(w[i])
        pair_local = np.zeros(6, dtype=float)
        for k, dof in enumerate(local_dofs):
            pair_local[k] = 0.5 * (
                float(d[gdof(gb, dof, ndof)]) + float(d[gdof(gt, dof, ndof)])
            )
        acc += wi * pair_local
        t = tangents[i]
        nvec = normal_2d(t)
        uy = pair_local[U_S] * t[0] + pair_local[W] * nvec[0]
        uz = pair_local[U_S] * t[1] + pair_local[W] * nvec[1]
        acc_uyz += wi * np.array([uy, uz], dtype=float)

    return SectionDeformationAverage1a(
        about_xy=origin,
        mean_local_dof=acc / w_sum,
        mean_uyz=acc_uyz / w_sum,
        weight_sum=w_sum,
        n_pairs=n,
    )


__all__ = [
    "SectionDeformationAverage1a",
    "compute_section_deformation_average_1a",
]
