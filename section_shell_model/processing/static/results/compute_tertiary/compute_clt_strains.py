"""Tier 3: batched ABD solve for mid-surface strains/curvatures from trusted resultant rows.

Extracted from the CLT-solve half of ``api/fi_recovery.hashin_fi_per_ply_from_rows_batch``
(same ``np.linalg.solve`` batching/ordering, preserved bit-for-bit), built on
``materials/clpt_recovery.laminate_midstrains_curvatures``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.containers.types import ShellPanelResultants
from .provenance import clpt_inputs_trusted


def batched_clt_strains(
    rows: list[ShellPanelResultants | None],
    lam: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Solve ``ABD @ [eps0; kappa] = [N; M]`` for every provenance-trusted row.

    Rows that are ``None`` or fail :func:`clpt_inputs_trusted` are skipped (left as
    zeros with ``trusted_mask=False``); the batched solve only ever runs over the
    trusted subset, matching the fused path's ``trusted_idx`` bookkeeping exactly.

    Returns
    -------
    eps0, kappa
        ``(n_rows, 3)`` each; zero rows for untrusted/skipped elements.
    trusted_mask
        ``(n_rows,)`` bool — rows that entered the batched solve.
    """
    from blade_structure_utilities.material_utilities.laminate import LaminateDefinition

    n_rows = len(rows)
    eps0 = np.zeros((n_rows, 3), dtype=np.float64)
    kappa = np.zeros((n_rows, 3), dtype=np.float64)
    trusted_mask = np.zeros(n_rows, dtype=bool)
    if not isinstance(lam, LaminateDefinition) or not lam.plies:
        return eps0, kappa, trusted_mask

    trusted_idx: list[int] = []
    nm_stack: list[np.ndarray] = []
    for ri, row in enumerate(rows):
        if row is None or not clpt_inputs_trusted(row):
            continue
        trusted_idx.append(ri)
        nm_stack.append(
            np.concatenate(
                [
                    np.asarray(row.to_N_vec(), dtype=float).ravel()[:3],
                    np.asarray(row.to_M_vec(), dtype=float).ravel()[:3],
                ]
            )
        )
    if not nm_stack:
        return eps0, kappa, trusted_mask

    abd = lam.stiffness_bundle().ABD
    nm_mat = np.stack(nm_stack, axis=0)
    try:
        sol = np.linalg.solve(abd, nm_mat.T).T
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(abd, nm_mat.T, rcond=None)[0].T

    for ri, row_sol in zip(trusted_idx, sol):
        eps0[ri] = row_sol[:3]
        kappa[ri] = row_sol[3:6]
        trusted_mask[ri] = True
    return eps0, kappa, trusted_mask


__all__ = ["batched_clt_strains"]
