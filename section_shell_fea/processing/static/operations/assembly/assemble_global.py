"""AssembleGlobalShellSystem — scatter-add local Ke/Fe to global K, f.

Canonical home for :func:`assemble_section_system` (1d-fem alignment / R8).

Validation remains enabled by default. Callers with a trusted formulation cache
(already shape-/finite-checked upstream) may pass ``validate=False`` to skip
per-element Exactness checks; that path is intentional and documented, not a
global weakening of assembly guarantees.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.formulation_objects import (
    AssemblyDiagnostics,
    ShellFormulationCache,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.contracts.solve_states import GlobalShellSystem
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.containers.map_results import ShellMapEntry


def _validate_element_contribution(
    ke: np.ndarray,
    fe: np.ndarray,
    *,
    element_id: int,
    n_elem_dof: int,
) -> None:
    if ke.ndim != 2 or ke.shape[0] != ke.shape[1]:
        raise ValueError(f"element {element_id}: Ke must be square, got {ke.shape}")
    if not np.all(np.isfinite(ke)):
        raise ValueError(f"element {element_id}: Ke contains non-finite values")
    if ke.shape != (n_elem_dof, n_elem_dof):
        raise ValueError(
            f"element {element_id}: Ke shape {ke.shape} != ({n_elem_dof}, {n_elem_dof})"
        )
    if fe.shape != (n_elem_dof,):
        raise ValueError(
            f"element {element_id}: fe shape {fe.shape} != ({n_elem_dof},)"
        )
    if not np.all(np.isfinite(fe)):
        raise ValueError(f"element {element_id}: fe contains non-finite values")


def scatter_element_fe(
    fe: np.ndarray,
    global_dof_indices: np.ndarray,
    f_global: np.ndarray,
) -> None:
    """Duplicate-safe indexed force scatter (``np.add.at``)."""
    np.add.at(f_global, np.asarray(global_dof_indices, dtype=np.intp), np.asarray(fe, dtype=float))


def assemble_section_system(
    cache: ShellFormulationCache,
    n_gdof: int,
    *,
    validate: bool = True,
) -> tuple[sp.csr_matrix, np.ndarray, AssemblyDiagnostics]:
    """Scatter-add element Ke/Fe into global K and f.

    COO row/column/value buffers are preallocated from the known total element
    contribution count (sum of ``n_e**2``) so assembly avoids per-element chunk
    lists and three concatenations. Force scatter uses duplicate-safe indexed
    accumulation.

    Parameters
    ----------
    validate:
        When ``True`` (default), each element Ke/Fe is shape- and finite-checked.
        Pass ``False`` only for a trusted cache path that already validated
        contributions upstream.
    """
    stiffness_objects = cache.stiffness_objects
    force_objects = cache.force_objects
    n_elem = len(stiffness_objects)
    if n_elem != len(force_objects):
        raise ValueError(
            f"stiffness/force object count mismatch: {n_elem} vs {len(force_objects)}"
        )

    elem_n = np.empty(n_elem, dtype=np.int64)
    total_nnz_contrib = 0
    for i, stiff in enumerate(stiffness_objects):
        g_dofs = stiff.global_dof_indices
        if g_dofs is None:
            raise ValueError(f"element {stiff.element_id}: global_dof_indices not set")
        n = int(g_dofs.size)
        elem_n[i] = n
        total_nnz_contrib += n * n

    rows = np.empty(total_nnz_contrib, dtype=np.int64)
    cols = np.empty(total_nnz_contrib, dtype=np.int64)
    vals = np.empty(total_nnz_contrib, dtype=float)
    f_global = np.zeros(n_gdof, dtype=float)

    offset = 0
    for stiff, force, n in zip(stiffness_objects, force_objects, elem_n, strict=True):
        g_dofs = np.asarray(stiff.global_dof_indices, dtype=np.int64)
        n = int(n)
        if validate:
            _validate_element_contribution(
                stiff.K_e, force.F_e, element_id=stiff.element_id, n_elem_dof=n
            )
        nn = n * n
        sl = slice(offset, offset + nn)
        rows[sl] = np.repeat(g_dofs, n)
        cols[sl] = np.tile(g_dofs, n)
        vals[sl] = np.asarray(stiff.K_e, dtype=float).ravel()
        scatter_element_fe(force.F_e, g_dofs, f_global)
        offset += nn

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
    def run(
        self,
        cache: ShellFormulationCache,
        n_gdof: int,
        *,
        collect_maps: bool = True,
    ) -> tuple[GlobalShellSystem, list[ShellMapEntry]]:
        k, f, diag = assemble_section_system(cache, n_gdof)
        global_sys = GlobalShellSystem(K=k, f=f, diagnostics=diag)
        if not collect_maps:
            return global_sys, []
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
