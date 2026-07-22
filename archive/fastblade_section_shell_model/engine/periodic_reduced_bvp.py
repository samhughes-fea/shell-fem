"""Periodic unit-cell BVP reduction — shared free set for beam modes and face loads.

Element-agnostic: consumes assembled ``K_global``, node/constraint metadata, and
DOF count only. This is the generalised section BVP seam: one constraint graph
and reconstruction map for all load channels (beam resultants and face traction).
Production solves are linear static on this free set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from blade_precompute.section_shell_model.pre_processing.containers.fem_config import MITC4_DEFAULTS

_NDOF_NODE = MITC4_DEFAULTS.n_dof_node


@dataclass
class PeriodicReducedBVP:
    """Reduced periodic unit-cell system on a shared free-DOF set.

    ``d_macro`` holds the six unit-mode particular jump fields used by K6
    homogenisation. Face-load solves use zero particular solution
    (``reconstruct(u_reduced)`` with no ``d_macro`` column).
    """

    T_pbc: sp.csr_matrix
    d_macro: NDArray[np.float64]  # (n_gdof, 6)
    K_rr: sp.csc_matrix
    K_rr_factor: Any  # SuperLU
    n_gdof: int
    ndof: int
    L_x: float = 1.0
    y_e: float = 0.0
    z_e: float = 0.0
    cluster_mode: bool = False
    pinned_dofs: frozenset[int] = field(default_factory=frozenset)

    def reduce_load(self, f_global: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map a global external load onto the periodic free set."""
        f = np.asarray(f_global, dtype=np.float64).ravel()
        if f.size != self.n_gdof:
            raise ValueError(
                f"f_global length {f.size} must match n_gdof={self.n_gdof}"
            )
        return np.asarray(self.T_pbc.T @ f, dtype=np.float64).ravel()

    def reconstruct(
        self,
        u_reduced: NDArray[np.float64],
        *,
        d_particular: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Lift a reduced displacement to global DOFs."""
        d = np.asarray(self.T_pbc @ np.asarray(u_reduced, dtype=np.float64).ravel(), dtype=np.float64)
        if d_particular is not None:
            d = d + np.asarray(d_particular, dtype=np.float64).ravel()
        return d

    def solve_external_load(self, f_global: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve ``K_rr u = Tᵀ f`` with zero particular jump (face loads)."""
        f_r = self.reduce_load(f_global)
        u_r = self.K_rr_factor.solve(f_r)
        return self.reconstruct(u_r)

    def compute_k6_modes(
        self,
        K_global: sp.spmatrix,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(K6, u_modes)`` via the energy bilinear form on unit modes."""
        n_gdof = self.n_gdof
        u_modes = np.zeros((n_gdof, 6), dtype=np.float64)
        Kd0 = K_global @ self.d_macro
        for j in range(6):
            f_j = -(self.T_pbc.T @ Kd0[:, j])
            q_j = self.K_rr_factor.solve(np.asarray(f_j, dtype=np.float64).ravel())
            u_modes[:, j] = self.reconstruct(q_j, d_particular=self.d_macro[:, j])

        K6 = np.zeros((6, 6), dtype=np.float64)
        Ku = K_global @ u_modes
        for i in range(6):
            for j in range(i, 6):
                K6[i, j] = float(u_modes[:, i] @ Ku[:, j])
                K6[j, i] = K6[i, j]
        return K6, u_modes


def build_periodic_reduced_bvp(
    K_global: sp.spmatrix,
    node_meta: Any,
    panel_maps: list[Any],
    *,
    y_e: float,
    z_e: float,
    L_x: float = 1.0,
    topological: bool = False,
    y_e_out: float | None = None,
    z_e_out: float | None = None,
    ndof: int = _NDOF_NODE,
    dof_frame: str = "local",
) -> PeriodicReducedBVP:
    """Build the periodic transform, reduced stiffness, and factor once."""
    from .homogenisation import (
        _build_node_pairs,
        _build_node_pairs_by_group,
        _build_node_pairs_topological,
        _build_periodic_transform_clustered,
        _build_periodic_transform_simple,
    )

    constraints = getattr(node_meta, "constraints", None)
    cluster_mode = bool(constraints)

    if cluster_mode or topological:
        pairs = _build_node_pairs_by_group(node_meta)
        if not pairs:
            pairs = _build_node_pairs_topological(panel_maps, node_meta)
    else:
        pairs = _build_node_pairs(node_meta)
    if not pairs:
        raise ValueError("No periodic node pairs were found for BVP reduction")

    n_gdof = int(K_global.shape[0])
    y_e_use = float(y_e if y_e_out is None else y_e_out)
    z_e_use = float(z_e if z_e_out is None else z_e_out)
    yz = node_meta.yz
    tangents = node_meta.tangent_yz
    pinned: set[int] = set()

    if cluster_mode:
        T_pbc, d_0, col_to_gdof = _build_periodic_transform_clustered(
            pairs, constraints, n_gdof, ndof, yz, tangents,
            y_e_use, z_e_use, L_x, clamp_dofs=set(), dof_frame=dof_frame,
        )
        K_rr = (T_pbc.T @ K_global @ T_pbc).tocsc()
        diag = K_rr.diagonal()
        if diag.size:
            tol = 1e-9 * max(float(np.max(np.abs(diag))), 1.0)
            zero_cols = np.where(np.abs(diag) <= tol)[0]
            if zero_cols.size:
                pinned = {int(col_to_gdof[c]) for c in zero_cols.tolist()}
                T_pbc, d_0, col_to_gdof = _build_periodic_transform_clustered(
                    pairs, constraints, n_gdof, ndof, yz, tangents,
                    y_e_use, z_e_use, L_x, clamp_dofs=pinned, dof_frame=dof_frame,
                )
                K_rr = (T_pbc.T @ K_global @ T_pbc).tocsc()
    else:
        T_pbc, d_0 = _build_periodic_transform_simple(
            pairs, n_gdof, ndof, yz, tangents, y_e_use, z_e_use, L_x, dof_frame=dof_frame,
        )
        K_rr = (T_pbc.T @ K_global @ T_pbc).tocsc()

    lu = spla.splu(K_rr)
    return PeriodicReducedBVP(
        T_pbc=T_pbc,
        d_macro=np.asarray(d_0, dtype=np.float64),
        K_rr=K_rr,
        K_rr_factor=lu,
        n_gdof=n_gdof,
        ndof=int(ndof),
        L_x=float(L_x),
        y_e=y_e_use,
        z_e=z_e_use,
        cluster_mode=cluster_mode,
        pinned_dofs=frozenset(pinned),
    )


__all__ = ["PeriodicReducedBVP", "build_periodic_reduced_bvp"]
