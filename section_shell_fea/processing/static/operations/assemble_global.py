"""AssembleGlobalShellSystem — scatter-add local Ke/Fe to global K, f.

Canonical home for :func:`assemble_section_system` (1d-fem alignment / R8).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.formulation_objects import (
    AssemblyDiagnostics,
    ShellFormulationCache,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.solve_states import GlobalShellSystem
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.containers.map_results import ShellMapEntry


def _validate_element_contribution(
    ke: np.ndarray,
    fe: np.ndarray,
    *,
    element_id: int,
    n_dof: int,
) -> None:
    if ke.ndim != 2 or ke.shape[0] != ke.shape[1]:
        raise ValueError(f"element {element_id}: Ke must be square, got {ke.shape}")
    if not np.all(np.isfinite(ke)):
        raise ValueError(f"element {element_id}: Ke contains non-finite values")
    expected = 4 * n_dof
    if fe.shape != (expected,):
        raise ValueError(
            f"element {element_id}: fe shape {fe.shape} != ({expected},)"
        )
    if not np.all(np.isfinite(fe)):
        raise ValueError(f"element {element_id}: fe contains non-finite values")


def scatter_element_fe(
    fe: np.ndarray,
    global_dof_indices: np.ndarray,
    f_global: np.ndarray,
) -> None:
    for i, gi in enumerate(global_dof_indices):
        f_global[int(gi)] += float(fe[i])


def assemble_section_system(
    cache: ShellFormulationCache,
    n_gdof: int,
    *,
    validate: bool = True,
) -> tuple[sp.csr_matrix, np.ndarray, AssemblyDiagnostics]:
    """Scatter-add element Ke/Fe into global K and f."""
    n_dof = cache.n_dof_node
    row_chunks: list[np.ndarray] = []
    col_chunks: list[np.ndarray] = []
    val_chunks: list[np.ndarray] = []
    f_global = np.zeros(n_gdof, dtype=float)

    for stiff, force in zip(cache.stiffness_objects, cache.force_objects):
        if validate:
            _validate_element_contribution(
                stiff.K_e, force.F_e, element_id=stiff.element_id, n_dof=n_dof
            )
        g_dofs = stiff.global_dof_indices
        if g_dofs is None:
            raise ValueError(f"element {stiff.element_id}: global_dof_indices not set")
        n = g_dofs.size
        row_chunks.append(np.repeat(g_dofs, n))
        col_chunks.append(np.tile(g_dofs, n))
        val_chunks.append(np.asarray(stiff.K_e, dtype=float).ravel())
        scatter_element_fe(force.F_e, g_dofs, f_global)

    if row_chunks:
        rows = np.concatenate(row_chunks)
        cols = np.concatenate(col_chunks)
        vals = np.concatenate(val_chunks)
    else:
        rows = cols = np.array([], dtype=np.int64)
        vals = np.array([], dtype=float)
    k_global = sp.coo_matrix((vals, (rows, cols)), shape=(n_gdof, n_gdof)).tocsr()

    sym_diff = 0.0
    if k_global.nnz > 0:
        diff = k_global - k_global.T
        sym_diff = float(np.max(np.abs(diff.data))) if diff.nnz else 0.0

    diag = AssemblyDiagnostics(
        n_global_dof=n_gdof,
        k_norm=float(spla.norm(k_global)),
        f_norm=float(np.linalg.norm(f_global)),
        nnz=int(k_global.nnz),
        symmetry_max_abs_diff=sym_diff,
    )
    return k_global, f_global, diag


class AssembleGlobalShellSystem:
    def run(self, cache: ShellFormulationCache, n_gdof: int) -> tuple[GlobalShellSystem, list[ShellMapEntry]]:
        k, f, diag = assemble_section_system(cache, n_gdof)
        global_sys = GlobalShellSystem(K=k, f=f, diagnostics=diag)
        assembly_map = self.build_assembly_map(cache, n_gdof)
        return global_sys, assembly_map

    @staticmethod
    def build_assembly_map(cache: ShellFormulationCache, n_gdof: int) -> list[ShellMapEntry]:
        seen: dict[int, ShellMapEntry] = {}
        for stiff in cache.stiffness_objects:
            g_dofs = stiff.global_dof_indices
            if g_dofs is None:
                continue
            for gi in g_dofs:
                gi = int(gi)
                if gi not in seen:
                    seen[gi] = ShellMapEntry(global_dof=gi, independent=True)
        for dof in range(n_gdof):
            if dof not in seen:
                seen[dof] = ShellMapEntry(global_dof=dof, independent=True)
        return [seen[d] for d in sorted(seen)]


__all__ = [
    "AssembleGlobalShellSystem",
    "assemble_section_system",
    "scatter_element_fe",
]
