"""Secondary-tier nodal resolution: bilinear extrapolation from 2×2 Gauss points."""

from __future__ import annotations

from typing import Any

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.containers.formulation_objects import (
    ShellFormulationCache,
)
from ..containers.gaussian_results import ShellGaussianResults
from ..containers.nodal_results import ShellNodalResults

# GP order must match compute_gaussian_fields._XI_ETA:
# (-a,-a), (-a,a), (a,-a), (a,a) with a = 1/√3
_A = 1.0 / np.sqrt(3.0)
_GP_XI = np.array([-_A, -_A, _A, _A], dtype=float)
_GP_ETA = np.array([-_A, _A, -_A, _A], dtype=float)


def _bilinear_N(xi: float, eta: float) -> np.ndarray:
    """Bilinear shape functions at parent (xi, eta); nodes at (±1, ±1) strip order.

    Node order: (-1,-1), (1,-1), (1,1), (-1,1) matching MITC4 local corners
    ``i0, i1, i2, i3``.
    """
    return 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ],
        dtype=float,
    )


def _gp_to_node_matrix() -> np.ndarray:
    """Return ``M`` such that ``nodal_vals = M @ gp_vals`` for a bilinear field.

    Built by inverting ``A`` where ``gp_vals = A @ nodal_vals`` and
    ``A[g, n] = N_n(xi_g, eta_g)``.
    """
    A = np.stack([_bilinear_N(float(_GP_XI[g]), float(_GP_ETA[g])) for g in range(4)], axis=0)
    return np.linalg.inv(A)


_EXTRAP = _gp_to_node_matrix()


def compute_nodal_resultants(
    gaussian: ShellGaussianResults,
    cache: ShellFormulationCache,
    *,
    panel_maps: list[Any] | None = None,
) -> ShellNodalResults:
    """Extrapolate Gauss-point resultants to the four element corners, then average.

    ``fields`` keys
    ---------------
    ``element_<key>`` : ``(n_elem, 4)`` — corner values before averaging
    ``<key>``         : ``(n_nodes,)`` — averaged over contributing elements when
                        ``panel_maps`` provide ``global_nodes``; otherwise unset
    """
    keys = ("Nx", "Ny", "Nxy", "Mx", "My", "Mxy")
    fields: dict[str, Any] = {}
    n_elem = 0
    for k in keys:
        gp = np.asarray(gaussian.fields.get(k, []), dtype=float)
        if gp.ndim != 2 or gp.shape[1] != 4:
            continue
        n_elem = gp.shape[0]
        # nodal[e,:] = _EXTRAP @ gp[e,:]
        fields[f"element_{k}"] = (gp @ _EXTRAP.T)

    if n_elem == 0:
        return ShellNodalResults(fields=fields)

    if panel_maps is not None and cache.stiffness_objects:
        max_gn = -1
        for pm in panel_maps:
            gns = getattr(pm, "global_nodes", None) or []
            if gns:
                max_gn = max(max_gn, int(max(gns)))
        if max_gn >= 0:
            n_nodes = max_gn + 1
            for k in keys:
                ek = fields.get(f"element_{k}")
                if ek is None:
                    continue
                acc = np.zeros(n_nodes, dtype=float)
                w = np.zeros(n_nodes, dtype=float)
                for ei, stiff in enumerate(cache.stiffness_objects):
                    if ei >= n_elem:
                        break
                    pi = int(stiff.panel_index)
                    lei = int(stiff.local_element_index)
                    if not (0 <= pi < len(panel_maps)):
                        continue
                    pm = panel_maps[pi]
                    elems = getattr(pm, "elements", None) or []
                    gns = getattr(pm, "global_nodes", None) or []
                    if not (0 <= lei < len(elems)):
                        continue
                    corners = elems[lei]
                    for local_c, ln in enumerate(corners):
                        gn = int(gns[ln])
                        acc[gn] += float(ek[ei, local_c])
                        w[gn] += 1.0
                avg = np.full(n_nodes, np.nan, dtype=float)
                mask = w > 0
                avg[mask] = acc[mask] / w[mask]
                fields[k] = avg
                fields[f"{k}_weight"] = w

    fields["n_elem"] = n_elem
    return ShellNodalResults(fields=fields)


__all__ = ["compute_nodal_resultants"]
