"""Secondary-tier Gaussian resolution: membrane/bending resultants at 2×2 Gauss points."""

from __future__ import annotations

from typing import Any

import numpy as np

from blade_precompute.section_shell_model.pre_processing.containers.formulation_objects import (
    ShellFormulationCache,
)
from blade_precompute.section_shell_model.pre_processing.element_library.kernel import (
    MITC4_KERNEL,
    resolve_kernel,
)
from ..containers.gaussian_results import ShellGaussianResults

# 2×2 Gauss-Legendre abscissae (order matching MITC4 shear integration).
_GP = 1.0 / np.sqrt(3.0)
_XI_ETA = (
    (-_GP, -_GP),
    (-_GP, _GP),
    (_GP, -_GP),
    (_GP, _GP),
)


def _element_L_s(
    *,
    pi: int,
    local_element_index: int,
    panel_maps: list[Any] | None,
) -> float:
    if panel_maps is None or not (0 <= pi < len(panel_maps)):
        return 1.0
    pm = panel_maps[pi]
    elems = getattr(pm, "elements", None) or []
    if not (0 <= local_element_index < len(elems)):
        return 1.0
    i0, i1, _i2, _i3 = elems[local_element_index]
    s_nodes = np.asarray(pm.s_nodes, dtype=float)
    return float(abs(s_nodes[i1] - s_nodes[i0]))


def compute_gaussian_resultants(
    cache: ShellFormulationCache,
    U_e: list,
    *,
    L_x: float = 1.0,
    panel_maps: list[Any] | None = None,
    shell_theory: Any | None = None,
) -> ShellGaussianResults:
    """Evaluate kernel ``recover_resultants_at`` at 2×2 Gauss points for each element.

    Populates ``fields`` with stacked arrays of shape ``(n_elem, 4)`` for
    ``Nx, Ny, Nxy, Mx, My, Mxy`` plus ``xi``, ``eta``, and element metadata.
    """
    try:
        kernel = resolve_kernel(shell_theory) if shell_theory is not None else MITC4_KERNEL
    except (ValueError, KeyError, TypeError):
        kernel = MITC4_KERNEL
    recover_at = getattr(kernel, "recover_resultants_at", None)
    if recover_at is None:
        raise NotImplementedError(
            f"kernel {kernel.name!r} does not support recover_resultants_at (R10 Gaussian)"
        )
    n_elem = min(len(cache.stiffness_objects), len(U_e))
    n_gp = len(_XI_ETA)
    keys = ("Nx", "Ny", "Nxy", "Mx", "My", "Mxy")
    arrays = {k: np.full((n_elem, n_gp), np.nan, dtype=float) for k in keys}
    xi_a = np.full((n_elem, n_gp), np.nan, dtype=float)
    eta_a = np.full((n_elem, n_gp), np.nan, dtype=float)
    elem_ids = np.full(n_elem, -1, dtype=np.int64)
    panel_idx = np.full(n_elem, -1, dtype=np.int64)

    for ei, (stiff, u_elem) in enumerate(zip(cache.stiffness_objects, U_e)):
        if ei >= n_elem:
            break
        pi = int(stiff.panel_index)
        local_ei = int(stiff.local_element_index)
        L_s = _element_L_s(pi=pi, local_element_index=local_ei, panel_maps=panel_maps)
        if L_s < 1e-30 or pi < 0 or pi >= len(cache.panel_abd_stacks):
            continue
        ABD = cache.panel_abd_stacks[pi][0]
        u = np.asarray(u_elem, dtype=float)
        elem_ids[ei] = int(stiff.element_id)
        panel_idx[ei] = pi
        for g, (xi, eta) in enumerate(_XI_ETA):
            xi_a[ei, g] = xi
            eta_a[ei, g] = eta
            res = recover_at(u, L_s, float(L_x), ABD, xi=xi, eta=eta)
            for k in keys:
                arrays[k][ei, g] = float(res[k])

    fields: dict[str, Any] = {
        "xi": xi_a,
        "eta": eta_a,
        "element_ids": elem_ids,
        "panel_index": panel_idx,
        "n_gauss": n_gp,
        **arrays,
    }
    return ShellGaussianResults(fields=fields)


__all__ = ["compute_gaussian_resultants"]
