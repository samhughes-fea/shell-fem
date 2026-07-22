"""Map station BeamResultantVector to an equivalent MITC4 nodal force vector.

Statically equivalent cut loads: transport R to the skin weighted centre,
distribute like the CCX outboard-ring algorithm in the strip local DOF frame
``[u_x, u_s, w, ...]``, so net force and moments about the requested ``about_yz``
are recovered by equipollence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.section_contract.beam_resultant_vector import (
    BEAM_LOAD_AXIS_YZ,
    FRAME_SECTION_YZ,
    BeamResultantVector,
)
from blade_structure_utilities.section_shell_model.pre_processing.loading.traction_face_load import (
    _skin_node_pairs,
)
from blade_structure_utilities.section_shell_model.pre_processing.mesh.shell_mesh_map import normal_2d
from blade_structure_utilities.section_shell_model.processing.static.operations.connected_assembly_core import (
    U_S,
    U_X,
    W,
    gdof,
)


def _pair_arc_weights(
    locs: NDArray[np.float64],
    *,
    thickness: float = 1.0,
) -> NDArray[np.float64]:
    """Approximate ``t·ds`` weights along the skin midline node-pair sequence."""
    n = int(locs.shape[0])
    if n == 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.asarray([max(float(thickness), 1e-12)], dtype=float)
    ds = np.zeros(n, dtype=float)
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
    return ds * max(float(thickness), 1e-12)


def beam_resultants_to_nodal_load(
    node_meta: Any,
    resultants: BeamResultantVector | NDArray[np.float64],
    *,
    about_yz: tuple[float, float] | None = None,
    ndof: int = 6,
    L_x: float = 1.0,
    thickness: float = 1.0,
    f: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Build / accumulate a statically equivalent nodal load vector from ``R``.

    Forces and couples are applied about the skin ``t·ds`` weighted centre so
    net force produces no parasitic moment there; ``resultants`` are first
    transported from ``about_yz`` to that centre (equipollent). Reconstructing
    about the original ``about_yz`` recovers ``R``.
    """
    if isinstance(resultants, BeamResultantVector):
        origin = about_yz if about_yz is not None else resultants.about_yz
        vec0 = (
            resultants
            if resultants.about_yz == (float(origin[0]), float(origin[1]))
            else resultants.transported_to(float(origin[0]), float(origin[1]))
        )
    else:
        origin = about_yz if about_yz is not None else BEAM_LOAD_AXIS_YZ
        vec0 = BeamResultantVector(
            R=np.asarray(resultants, dtype=float).ravel(),
            frame=FRAME_SECTION_YZ,
            about_yz=(float(origin[0]), float(origin[1])),
        )

    n_gdof = int(node_meta.n_nodes) * int(ndof)
    if f is None:
        f = np.zeros(n_gdof, dtype=float)
    else:
        f = np.asarray(f, dtype=float)
        if f.size != n_gdof:
            raise ValueError(f"f length {f.size} != n_gdof {n_gdof}")

    locs, tangents, gn_pairs = _skin_node_pairs(node_meta)
    if locs.shape[0] == 0:
        return f

    w = _pair_arc_weights(locs, thickness=thickness)
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        w = np.ones(len(gn_pairs), dtype=float)
        w_sum = float(len(gn_pairs))
    wn = w / w_sum
    y = locs[:, 0]
    z = locs[:, 1]
    yc = float(np.sum(wn * y))
    zc = float(np.sum(wn * z))

    # Equipollent at the distribution centre (CCX outboard-ring pattern).
    vec = vec0.transported_to(yc, zc)
    y0 = y - yc
    z0 = z - zc

    scale = float(L_x)
    fx = float(vec.N) * scale * wn
    fy = float(vec.Vy) * scale * wn
    fz = float(vec.Vz) * scale * wn

    # Axial couples for (My, Mz) with full planar inertia (incl. Iyz) so
    # cambered sections do not cross-talk My↔Mz.
    iy = float(np.sum(w * y0 * y0))
    iz = float(np.sum(w * z0 * z0))
    iyz = float(np.sum(w * y0 * z0))
    my = float(vec.My) * scale
    mz = float(vec.Mz) * scale
    if abs(my) > 0.0 or abs(mz) > 0.0:
        # [ Iz  Iyz ] [λy] = [ My ]
        # [ Iyz Iy  ] [λz]   [-Mz ]
        det = iz * iy - iyz * iyz
        if abs(det) > 1e-30:
            lam_y = (iy * my - iyz * (-mz)) / det
            lam_z = (iz * (-mz) - iyz * my) / det
            fx = fx + lam_y * (w * z0) + lam_z * (w * y0)

    if abs(float(vec.T)) > 0.0:
        r2 = y0 * y0 + z0 * z0
        denom = float(np.sum(w * r2))
        if denom > 1e-30:
            c = float(vec.T) * scale / denom
            fy = fy + c * (-z0) * w
            fz = fz + c * y0 * w

    for i, (gb, gt) in enumerate(gn_pairs):
        t = tangents[i]
        n = normal_2d(t)
        f_s = float(fy[i] * t[0] + fz[i] * t[1])
        f_w = float(fy[i] * n[0] + fz[i] * n[1])
        half_fx = 0.5 * float(fx[i])
        half_fs = 0.5 * f_s
        half_fw = 0.5 * f_w
        for gn in (gb, gt):
            f[gdof(gn, U_X, ndof)] += half_fx
            f[gdof(gn, U_S, ndof)] += half_fs
            f[gdof(gn, W, ndof)] += half_fw
    return f


def integrate_nodal_load_resultants(
    node_meta: Any,
    f_global: NDArray[np.float64],
    *,
    about_yz: tuple[float, float] = BEAM_LOAD_AXIS_YZ,
    ndof: int = 6,
) -> BeamResultantVector:
    """Reconstruct net ``(N, Vy, Vz, My, Mz, T)`` from a nodal load vector.

    Uses skin node-pairs only (each pair's net translational load once) so
    bottom/top layers are not double-counted.
    """
    locs, tangents, gn_pairs = _skin_node_pairs(node_meta)
    y0, z0 = float(about_yz[0]), float(about_yz[1])
    N = Vy = Vz = My = Mz = T = 0.0
    for i, (gb, gt) in enumerate(gn_pairs):
        t = tangents[i]
        n = normal_2d(t)
        fx = 0.0
        fs = 0.0
        fw = 0.0
        for gn in (gb, gt):
            fx += float(f_global[gdof(gn, U_X, ndof)])
            fs += float(f_global[gdof(gn, U_S, ndof)])
            fw += float(f_global[gdof(gn, W, ndof)])
        fy = fs * t[0] + fw * n[0]
        fz = fs * t[1] + fw * n[1]
        y, z = float(locs[i, 0]), float(locs[i, 1])
        dy, dz = y - y0, z - z0
        N += fx
        Vy += fy
        Vz += fz
        My += fx * dz
        Mz += -fx * dy
        T += dy * fz - dz * fy
    return BeamResultantVector.from_named(
        N=N, Vy=Vy, Vz=Vz, My=My, Mz=Mz, T=T,
        frame=FRAME_SECTION_YZ,
        about_yz=(y0, z0),
    )


__all__ = [
    "beam_resultants_to_nodal_load",
    "integrate_nodal_load_resultants",
]
