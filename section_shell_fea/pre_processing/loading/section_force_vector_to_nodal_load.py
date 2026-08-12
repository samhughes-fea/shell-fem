"""Map ``SectionForceVector`` contour densities onto MITC4 skin node-pairs."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.load_utilities.section_loads import (
    SectionForceVector,
    sample_arc_weights,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.loading.traction_face_load import (
    _skin_node_pairs,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.connected_assembly_core import (
    U_S,
    U_X,
    W,
    gdof,
)


def section_force_vector_to_nodal_load(
    node_meta: Any,
    sfv: SectionForceVector,
    *,
    ndof: int = 6,
    L_x: float = 1.0,
    f: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Project Channel-C contour force densities onto skin through-thickness pairs.

    Each sample contributes absolute force ``density * ds * L_x`` lumped to the
    nearest skin node-pair (split 50/50 bottom/top) in curvilinear **C** DOFs
    ``[u_x, u_s, w]``.
    """
    n_gdof = int(node_meta.n_nodes) * int(ndof)
    if f is None:
        f = np.zeros(n_gdof, dtype=float)
    else:
        f = np.asarray(f, dtype=float)
        if f.size != n_gdof:
            raise ValueError(f"f length {f.size} != n_gdof {n_gdof}")

    locs, _tangents, gn_pairs = _skin_node_pairs(node_meta)
    if locs.shape[0] == 0 or not sfv.samples:
        return f

    ds = sample_arc_weights(sfv.samples)
    scale = float(L_x)
    # Nearest pair for each sample
    for sm, dsi in zip(sfv.samples, ds):
        d2 = (locs[:, 0] - sm.y) ** 2 + (locs[:, 1] - sm.z) ** 2
        i = int(np.argmin(d2))
        gb, gt = gn_pairs[i]
        fx = float(sm.f_x) * float(dsi) * scale
        fs = float(sm.f_s) * float(dsi) * scale
        fw = float(sm.f_w) * float(dsi) * scale
        for gn in (gb, gt):
            f[gdof(gn, U_X, ndof)] += 0.5 * fx
            f[gdof(gn, U_S, ndof)] += 0.5 * fs
            f[gdof(gn, W, ndof)] += 0.5 * fw
    return f


__all__ = ["section_force_vector_to_nodal_load"]
