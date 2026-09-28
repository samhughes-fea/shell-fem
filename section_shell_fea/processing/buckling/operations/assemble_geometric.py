"""Geometric stiffness helpers for panel and full-section linear buckling."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.formulation_objects import (
    ShellFormulationCache,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import ShellTheory
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.kernel import (
    resolve_kernel,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.topology.shell_mesh_map import (
    PanelGlobalMap,
)


def assemble_panel_geometric_stiffness(
    *,
    L_s: float,
    L_x: float,
    N11: float,
    N22: float,
    N12: float,
    n_gauss: int = 2,
    shell_theory: ShellTheory = ShellTheory.FSDT,
) -> NDArray:
    """Return element geometric stiffness via the registered strip kernel."""
    kernel = resolve_kernel(shell_theory)
    return kernel.geometric_stiffness(
        float(L_s),
        float(L_x),
        float(N11),
        float(N22),
        float(N12),
        n_gauss=int(n_gauss),
    )


def _element_L_s(pm: PanelGlobalMap, local_element_index: int) -> float:
    elems = pm.elements or []
    if not (0 <= local_element_index < len(elems)):
        return 0.0
    i0, i1, _i2, _i3 = elems[local_element_index]
    s_nodes = np.asarray(pm.s_nodes, dtype=float)
    return float(abs(s_nodes[i1] - s_nodes[i0]))


def assemble_section_geometric_stiffness(
    cache: ShellFormulationCache,
    panel_maps: Sequence[PanelGlobalMap],
    prestress_by_element: Mapping[tuple[int, int], tuple[float, float, float]],
    n_gdof: int,
    *,
    L_x: float = 1.0,
    shell_theory: ShellTheory = ShellTheory.FSDT,
    n_gauss: int = 2,
) -> sp.csr_matrix:
    """Scatter-add element ``Kg`` over the connected section mesh.

    ``prestress_by_element`` maps ``(panel_index, local_element_index)`` →
    ``(N11, N22, N12)`` membrane prestress [N/m] from static secondary recovery.
    """
    row_chunks: list[np.ndarray] = []
    col_chunks: list[np.ndarray] = []
    val_chunks: list[np.ndarray] = []
    L_x_f = float(L_x)
    maps = list(panel_maps)

    for stiff in cache.stiffness_objects:
        pi = int(stiff.panel_index)
        lei = int(stiff.local_element_index)
        g_dofs = stiff.global_dof_indices
        if g_dofs is None:
            raise ValueError(f"element {stiff.element_id}: global_dof_indices not set")
        key = (pi, lei)
        if key not in prestress_by_element:
            continue
        N11, N22, N12 = prestress_by_element[key]
        if not (np.isfinite(N11) and np.isfinite(N22) and np.isfinite(N12)):
            continue
        if abs(N11) + abs(N22) + abs(N12) < 1e-30:
            continue
        if not (0 <= pi < len(maps)):
            continue
        # WP2: prefer the persisted assembly length (blended outboard L_s) so the
        # geometric stiffness uses the same L_s as the material stiffness; fall
        # back to recomputed plain contour length for pre-WP2 caches.
        L_s = float(getattr(stiff, "L_s", float("nan")))
        if not np.isfinite(L_s) or L_s < 1e-30:
            L_s = _element_L_s(maps[pi], lei)
        if L_s < 1e-30:
            continue
        Kg_e = assemble_panel_geometric_stiffness(
            L_s=L_s,
            L_x=L_x_f,
            N11=float(N11),
            N22=float(N22),
            N12=float(N12),
            n_gauss=int(n_gauss),
            shell_theory=shell_theory,
        )
        n = int(g_dofs.size)
        if Kg_e.shape != (n, n):
            raise ValueError(
                f"element {stiff.element_id}: Kg shape {Kg_e.shape} != ({n}, {n})"
            )
        row_chunks.append(np.repeat(g_dofs, n))
        col_chunks.append(np.tile(g_dofs, n))
        val_chunks.append(np.asarray(Kg_e, dtype=float).ravel())

    if not row_chunks:
        return sp.csr_matrix((int(n_gdof), int(n_gdof)), dtype=float)
    rows = np.concatenate(row_chunks)
    cols = np.concatenate(col_chunks)
    vals = np.concatenate(val_chunks)
    return sp.coo_matrix(
        (vals, (rows, cols)), shape=(int(n_gdof), int(n_gdof)), dtype=float
    ).tocsr()


__all__ = [
    "assemble_panel_geometric_stiffness",
    "assemble_section_geometric_stiffness",
]
