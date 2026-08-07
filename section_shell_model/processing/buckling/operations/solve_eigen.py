"""Panel linear buckling eigenproduct: K φ = λ (−Kg) φ."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from numpy.typing import NDArray

from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.containers.buckling_types import (
    ShellPanelBucklingInput,
)
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.containers.types import ShellTheory
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.element_library.kernel import (
    StripElementKernel,
    resolve_kernel,
)
from blade_structure_utilities.fem_utilities.section_shell_model.processing.buckling.operations.assemble_geometric import (
    assemble_panel_geometric_stiffness,
)
from blade_structure_utilities.fem_utilities.section_shell_model.processing.buckling.operations.eigen_states import (
    BucklingEigenSolveState,
    SectionBucklingSolveState,
)

# Contour-local DOF indices shared by registered strip kernels:
# [u_x, u_s, w, β_s, β_x, (optional drilling)]
_U_X = 0
_U_S = 1
_W = 2
_BETA_S = 3
_BETA_X = 4


def _node_dof(node_idx: int, local_dof: int, *, ndof: int) -> int:
    return int(node_idx) * int(ndof) + int(local_dof)


def _build_strip_mesh(n_elements: int) -> tuple[NDArray, list[list[int]]]:
    """Unit-s strip: bottom row x=0, top row x=1; n_elements along contour."""
    n_nodes_s = int(n_elements) + 1
    s_nodes = np.linspace(0.0, 1.0, n_nodes_s)
    coords = np.zeros((2 * n_nodes_s, 2), dtype=float)
    coords[:n_nodes_s, 0] = s_nodes
    coords[n_nodes_s:, 0] = s_nodes
    coords[n_nodes_s:, 1] = 1.0
    elems: list[list[int]] = []
    for i in range(n_nodes_s - 1):
        elems.append([i, i + 1, i + 1 + n_nodes_s, i + n_nodes_s])
    return coords, elems


def _fixed_dofs_simply_supported(n_nodes_s: int, *, ndof: int) -> list[int]:
    """Minimal in-plane RBM + simple support (w, β) at both contour ends."""
    fixed: list[int] = [
        _node_dof(0, _U_X, ndof=ndof),
        _node_dof(0, _U_S, ndof=ndof),
        _node_dof(max(n_nodes_s - 1, 0), _U_X, ndof=ndof),
    ]
    for idx in (0, max(n_nodes_s - 1, 0)):
        for row in (idx, idx + n_nodes_s):
            fixed.extend(
                [
                    _node_dof(row, _W, ndof=ndof),
                    _node_dof(row, _BETA_S, ndof=ndof),
                    _node_dof(row, _BETA_X, ndof=ndof),
                ]
            )
    return list(set(fixed))


def _g_eff_from_abd(abd: NDArray, thickness_m: float) -> float:
    """Estimate transverse shear modulus from membrane A66 / thickness."""
    a66 = float(np.asarray(abd, dtype=float)[2, 2])
    return max(a66 / max(float(thickness_m), 1e-30), 1.0)


def _assemble_elastic(
    *,
    kernel: StripElementKernel,
    width_m: float,
    length_m: float,
    abd: NDArray,
    thickness_m: float,
    n_elements: int,
) -> tuple[sp.csr_matrix, list[list[int]]]:
    """Assemble panel strip elastic stiffness (uniform L_s = width / n_elements)."""
    _coords, elements = _build_strip_mesh(n_elements)
    ndof = int(kernel.ndof)
    n_nodes_s = n_elements + 1
    n_dof = 2 * n_nodes_s * ndof
    L_s = float(width_m) / float(n_elements)
    L_x = float(length_m)
    g_eff = _g_eff_from_abd(abd, thickness_m)
    Ke = kernel.stiffness(
        L_s,
        L_x,
        np.asarray(abd, dtype=float),
        thickness=float(thickness_m),
        G_eff=g_eff,
    )
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for elem_nodes in elements:
        glob = [_node_dof(ni, d, ndof=ndof) for ni in elem_nodes for d in range(ndof)]
        for a, ga in enumerate(glob):
            for b, gb in enumerate(glob):
                rows.append(ga)
                cols.append(gb)
                vals.append(float(Ke[a, b]))
    K = sp.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()
    return K, elements


def _assemble_geometric(
    *,
    shell_theory: ShellTheory,
    ndof: int,
    width_m: float,
    length_m: float,
    Nx: float,
    Ny: float,
    Nxy: float,
    n_elements: int,
    elements: list[list[int]],
) -> sp.csr_matrix:
    n_nodes_s = n_elements + 1
    n_dof = 2 * n_nodes_s * int(ndof)
    L_s = float(width_m) / float(n_elements)
    L_x = float(length_m)
    Kg_e = assemble_panel_geometric_stiffness(
        L_s=L_s,
        L_x=L_x,
        Nx=float(Nx),
        Ny=float(Ny),
        Nxy=float(Nxy),
        shell_theory=shell_theory,
    )
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for elem_nodes in elements:
        glob = [_node_dof(ni, d, ndof=ndof) for ni in elem_nodes for d in range(ndof)]
        for a, ga in enumerate(glob):
            for b, gb in enumerate(glob):
                rows.append(ga)
                cols.append(gb)
                vals.append(float(Kg_e[a, b]))
    return sp.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()


def _empty_state(
    inp: ShellPanelBucklingInput,
    *,
    is_compressive: bool,
    status: str,
    reason: str,
    n_modes: int,
    extra: dict[str, Any] | None = None,
) -> BucklingEigenSolveState:
    diag: dict[str, Any] = {"reason": reason}
    if extra:
        diag.update(extra)
    return BucklingEigenSolveState(
        panel_index=int(inp.panel_index),
        panel_label=str(inp.panel_label),
        lambdas=np.zeros(0, dtype=float),
        modeshapes=np.zeros((0, 0), dtype=float),
        n_modes_found=0,
        is_compressive=bool(is_compressive),
        status=status,
        diagnostics=diag,
    )


def solve_panel_linear_buckling_eigen(
    inp: ShellPanelBucklingInput,
    *,
    n_modes: int = 3,
    n_elements: int = 4,
) -> BucklingEigenSolveState:
    """Solve K φ = λ (−Kg) φ for one panel under uniform membrane prestress.

    Elastic and geometric stiffness come from the registered strip kernel for
    ``inp.provenance.shell_theory``. Non-compressive panels (Nx ≥ 0) and panels
    missing ABD return ``skipped`` with empty eigenarrays — never fabricated
    eigenvalues.
    """
    mem = np.asarray(inp.membrane_resultants, dtype=float).reshape(3)
    Nx, Ny, Nxy = float(mem[0]), float(mem[1]), float(mem[2])
    is_compressive = bool(Nx < 0.0)
    shell_theory = getattr(inp.provenance, "shell_theory", None) or ShellTheory.FSDT
    base_diag: dict[str, Any] = {
        "Nx": Nx,
        "Ny": Ny,
        "Nxy": Nxy,
        "shell_theory": str(shell_theory),
    }

    if not is_compressive:
        return _empty_state(
            inp,
            is_compressive=False,
            status="skipped",
            reason="not_compressive",
            n_modes=n_modes,
            extra=base_diag,
        )
    if inp.abd is None:
        return _empty_state(
            inp,
            is_compressive=True,
            status="skipped",
            reason="missing_abd",
            n_modes=n_modes,
            extra=base_diag,
        )

    try:
        kernel = resolve_kernel(shell_theory)
    except Exception as exc:
        return _empty_state(
            inp,
            is_compressive=True,
            status="skipped",
            reason="no_native_strip_kernel",
            n_modes=n_modes,
            extra={**base_diag, "error": str(exc)},
        )

    n_el = max(int(n_elements), 1)
    n_req = max(int(n_modes), 1)
    width = max(float(inp.width_m), 1e-12)
    length = max(float(inp.length_m), 1e-12)
    thickness = max(float(inp.thickness_m), 1e-12)
    abd = np.asarray(inp.abd, dtype=float)
    ndof = int(kernel.ndof)

    K, elements = _assemble_elastic(
        kernel=kernel,
        width_m=width,
        length_m=length,
        abd=abd,
        thickness_m=thickness,
        n_elements=n_el,
    )
    Kg = _assemble_geometric(
        shell_theory=shell_theory,
        ndof=ndof,
        width_m=width,
        length_m=length,
        Nx=Nx,
        Ny=Ny,
        Nxy=Nxy,
        n_elements=n_el,
        elements=elements,
    )

    n_nodes_s = n_el + 1
    n_dof = 2 * n_nodes_s * ndof
    fixed = set(_fixed_dofs_simply_supported(n_nodes_s, ndof=ndof))
    free = np.array([d for d in range(n_dof) if d not in fixed], dtype=int)
    if free.size < 2:
        return _empty_state(
            inp,
            is_compressive=True,
            status="skipped",
            reason="insufficient_free_dofs",
            n_modes=n_modes,
            extra=base_diag,
        )

    Ke_ff = K[np.ix_(free, free)].toarray()
    Kg_ff = Kg[np.ix_(free, free)].toarray()
    # Compressive Nx < 0 → Kg entries negative; use −Kg so λ > 0 for buckling.
    Kg_neg = -Kg_ff
    if float(np.max(np.abs(Kg_neg))) < 1e-30:
        return _empty_state(
            inp,
            is_compressive=True,
            status="skipped",
            reason="vanishing_geometric_stiffness",
            n_modes=n_modes,
            extra=base_diag,
        )

    try:
        eigvals, eigvecs = la.eig(Ke_ff, Kg_neg)
    except la.LinAlgError as exc:
        return _empty_state(
            inp,
            is_compressive=True,
            status="failed",
            reason="eigen_solve_failed",
            n_modes=n_modes,
            extra={**base_diag, "error": str(exc)},
        )

    real_mask = np.abs(eigvals.imag) < 1e-6 * (np.abs(eigvals.real) + 1.0)
    eigvals_real = eigvals[real_mask].real
    eigvecs_real = eigvecs[:, real_mask].real
    pos_mask = eigvals_real > 1e-6
    eigvals_pos = eigvals_real[pos_mask]
    eigvecs_pos = eigvecs_real[:, pos_mask]
    if eigvals_pos.size == 0:
        return _empty_state(
            inp,
            is_compressive=True,
            status="skipped",
            reason="no_positive_eigenvalues",
            n_modes=n_modes,
            extra=base_diag,
        )

    order = np.argsort(eigvals_pos)
    n_found = int(min(n_req, order.size))
    lambdas = np.asarray(eigvals_pos[order[:n_found]], dtype=float)
    modes = np.asarray(eigvecs_pos[:, order[:n_found]].T, dtype=float)

    return BucklingEigenSolveState(
        panel_index=int(inp.panel_index),
        panel_label=str(inp.panel_label),
        lambdas=lambdas,
        modeshapes=modes,
        n_modes_found=n_found,
        is_compressive=True,
        status="ok",
        diagnostics={
            **base_diag,
            "kernel": str(kernel.name),
            "ndof": ndof,
            "n_elements": n_el,
            "n_free_dof": int(free.size),
            "lambda_min": float(lambdas[0]) if n_found else None,
        },
        K=K,
        Kg=Kg,
    )


def prestress_map_from_nested_resultants(
    nested_rows: Sequence[Sequence[Any]],
) -> dict[tuple[int, int], tuple[float, float, float]]:
    """Build ``(panel_index, local_element_index) → (Nx, Ny, Nxy)`` from nested secondary rows.

    Secondary rows store the local element index in ``station_index`` (see
    ``compute_secondary``); panels are keyed by ``panel_index``.
    """
    out: dict[tuple[int, int], tuple[float, float, float]] = {}
    for pi, panel_rows in enumerate(nested_rows):
        for row in panel_rows or []:
            lei = int(getattr(row, "station_index", 0) or 0)
            Nx = float(getattr(row, "Nx", 0.0) or 0.0)
            Ny = float(getattr(row, "Ny", 0.0) or 0.0)
            Nxy = float(getattr(row, "Nxy", 0.0) or 0.0)
            out[(int(pi), lei)] = (Nx, Ny, Nxy)
    return out


def solve_section_linear_buckling_eigen(
    *,
    K_global: sp.spmatrix,
    Kg_global: sp.spmatrix,
    T: sp.spmatrix,
    n_modes: int = 3,
    extra_diagnostics: dict[str, Any] | None = None,
) -> SectionBucklingSolveState:
    """Solve the connected-section pencil ``K_r φ = λ (−Kg_r) φ`` with shell ``T``.

    ``K_r = Tᵀ K T`` and ``Kg_r = Tᵀ Kg T`` reuse the static reduction transform so
    supports and junction MPCs match the prestress solve.
    """
    n_req = max(int(n_modes), 1)
    base_diag: dict[str, Any] = dict(extra_diagnostics or {})
    K = K_global.tocsr()
    Kg = Kg_global.tocsr()
    T_csr = T.tocsr()
    n_gdof = int(K.shape[0])
    if Kg.shape != K.shape:
        raise ValueError(f"Kg shape {Kg.shape} != K shape {K.shape}")
    if T_csr.shape[0] != n_gdof:
        raise ValueError(f"T rows {T_csr.shape[0]} != n_gdof {n_gdof}")

    compressive = bool(float(np.max(np.abs(Kg.data))) > 1e-30) if Kg.nnz else False
    if not compressive:
        return SectionBucklingSolveState(
            lambdas=np.zeros(0, dtype=float),
            modeshapes_global=np.zeros((0, n_gdof), dtype=float),
            n_modes_found=0,
            is_compressive=False,
            status="skipped",
            diagnostics={**base_diag, "reason": "vanishing_geometric_stiffness"},
            K=K,
            Kg=Kg,
            T=T_csr,
        )

    K_r = (T_csr.T @ K @ T_csr).toarray()
    Kg_r = (T_csr.T @ Kg @ T_csr).toarray()
    K_r = 0.5 * (K_r + K_r.T)
    Kg_neg = -0.5 * (Kg_r + Kg_r.T)
    if float(np.max(np.abs(Kg_neg))) < 1e-30:
        return SectionBucklingSolveState(
            lambdas=np.zeros(0, dtype=float),
            modeshapes_global=np.zeros((0, n_gdof), dtype=float),
            n_modes_found=0,
            is_compressive=False,
            status="skipped",
            diagnostics={**base_diag, "reason": "vanishing_reduced_geometric_stiffness"},
            K=K,
            Kg=Kg,
            T=T_csr,
        )

    try:
        eigvals, eigvecs = la.eig(K_r, Kg_neg)
    except la.LinAlgError as exc:
        return SectionBucklingSolveState(
            lambdas=np.zeros(0, dtype=float),
            modeshapes_global=np.zeros((0, n_gdof), dtype=float),
            n_modes_found=0,
            is_compressive=True,
            status="failed",
            diagnostics={**base_diag, "reason": "eigen_solve_failed", "error": str(exc)},
            K=K,
            Kg=Kg,
            T=T_csr,
        )

    real_mask = np.abs(eigvals.imag) < 1e-6 * (np.abs(eigvals.real) + 1.0)
    eigvals_real = eigvals[real_mask].real
    eigvecs_real = eigvecs[:, real_mask].real
    pos_mask = eigvals_real > 1e-6
    eigvals_pos = eigvals_real[pos_mask]
    eigvecs_pos = eigvecs_real[:, pos_mask]
    if eigvals_pos.size == 0:
        return SectionBucklingSolveState(
            lambdas=np.zeros(0, dtype=float),
            modeshapes_global=np.zeros((0, n_gdof), dtype=float),
            n_modes_found=0,
            is_compressive=True,
            status="skipped",
            diagnostics={**base_diag, "reason": "no_positive_eigenvalues"},
            K=K,
            Kg=Kg,
            T=T_csr,
        )

    order = np.argsort(eigvals_pos)
    n_found = int(min(n_req, order.size))
    lambdas = np.asarray(eigvals_pos[order[:n_found]], dtype=float)
    modes_r = np.asarray(eigvecs_pos[:, order[:n_found]], dtype=float)
    modes_g = np.asarray(T_csr @ modes_r, dtype=float).T  # (n_modes, n_gdof)

    return SectionBucklingSolveState(
        lambdas=lambdas,
        modeshapes_global=modes_g,
        n_modes_found=n_found,
        is_compressive=True,
        status="ok",
        diagnostics={
            **base_diag,
            "n_gdof": n_gdof,
            "n_reduced_dof": int(K_r.shape[0]),
            "lambda_min": float(lambdas[0]) if n_found else None,
            "eigen_track": "section",
        },
        K=K,
        Kg=Kg,
        T=T_csr,
    )


__all__ = [
    "prestress_map_from_nested_resultants",
    "solve_panel_linear_buckling_eigen",
    "solve_section_linear_buckling_eigen",
]
