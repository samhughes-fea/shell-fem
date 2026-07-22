"""Tier 3 orchestrator: CLT recovery at the material frame (no failure-criterion logic).

Consumes only ``ShellSecondaryResultSet`` + ``ShellFormulationCache``; laminates
resolve via ``cache.panel_laminates[row.panel_index]`` (no laminate side argument).
Composes :class:`InPlaneLaminateRecoveryOperator` and
:class:`FSDTInterfaceShearRecoveryOperator`. See PROCESSING_ARCHITECTURE_PROPOSAL.md
tier 3.
"""

from __future__ import annotations

import numpy as np

from blade_structure_utilities.section_shell_model.pre_processing.containers.formulation_objects import ShellFormulationCache
from blade_structure_utilities.section_shell_model.pre_processing.containers.types import ShellPanelResultants
from ..containers.container_hopper import ShellSecondaryResultSet, ShellTertiaryResultSet
from ..containers.ply_results import ShellPlyResults
from .provenance import interlaminar_inputs_trusted
from .recovery_operators import (
    FSDTInterfaceShearRecoveryOperator,
    InPlaneLaminateRecoveryOperator,
    SHEAR_RECOVERY_MODE_FSDT_PARABOLIC,
)


class ShellTertiaryResultsOrchestrator:
    """CLT ply-stress recovery from secondary resultants + the formulation cache.

    Resultant rows are grouped by ``panel_index`` because the batched ABD solve is
    per-laminate: each panel may carry a different laminate (different ABD, ply
    count). Rows failing provenance gating (see ``compute_tertiary.provenance``)
    are left untrusted (zeros) rather than raising, matching the fused path's
    ``needs_review``-style bookkeeping one tier up.
    """

    def __init__(
        self,
        *,
        secondary: ShellSecondaryResultSet,
        cache: ShellFormulationCache,
        in_plane_operator: InPlaneLaminateRecoveryOperator | None = None,
        shear_operator: FSDTInterfaceShearRecoveryOperator | None = None,
    ) -> None:
        self.secondary = secondary
        self.cache = cache
        self.in_plane_operator = in_plane_operator or InPlaneLaminateRecoveryOperator()
        self.shear_operator = shear_operator or FSDTInterfaceShearRecoveryOperator()

    def compute(self) -> ShellTertiaryResultSet:
        rows: list[ShellPanelResultants] = []
        if self.secondary.elemental_results is not None:
            rows = list(self.secondary.elemental_results.resultant_rows)
        n_rows = len(rows)

        eps0 = np.zeros((n_rows, 3), dtype=np.float64)
        kappa = np.zeros((n_rows, 3), dtype=np.float64)
        trusted_mask = np.zeros(n_rows, dtype=bool)
        shear_trusted_mask = np.zeros(n_rows, dtype=bool)
        transverse_shear = np.full((n_rows, 2), np.nan, dtype=np.float64)
        thickness_m = np.zeros(n_rows, dtype=np.float64)
        element_index = np.zeros(n_rows, dtype=np.int64)
        panel_index_arr = np.zeros(n_rows, dtype=np.int64)
        eps_ply_per_row: list[np.ndarray | None] = [None] * n_rows
        sig_ply_per_row: list[np.ndarray | None] = [None] * n_rows
        tau_iface_per_row: list[np.ndarray | None] = [None] * n_rows

        by_panel: dict[int, list[int]] = {}
        for i, row in enumerate(rows):
            by_panel.setdefault(row.panel_index, []).append(i)
            thickness_m[i] = row.thickness_m
            element_index[i] = row.station_index
            panel_index_arr[i] = row.panel_index
            # Pass-through Qx/Qy whenever both present (legacy contract for tier 4
            # scalar 1.5|Q|/t fallback). Interlaminar provenance is tracked separately.
            if row.Qx is not None and row.Qy is not None:
                transverse_shear[i, 0] = row.Qx
                transverse_shear[i, 1] = row.Qy

        n_ply_max = 0
        n_iface_max = 0
        panel_laminates = self.cache.panel_laminates
        for panel_index, idxs in by_panel.items():
            lam = panel_laminates[panel_index] if panel_index < len(panel_laminates) else None
            if lam is None or not getattr(lam, "plies", None):
                continue
            panel_rows = [rows[i] for i in idxs]
            in_plane = self.in_plane_operator.recover_panel(panel_rows, lam)
            n_ply_max = max(n_ply_max, in_plane.n_ply)
            n_iface_max = max(n_iface_max, in_plane.n_ply + 1)
            for local_i, global_i in enumerate(idxs):
                if in_plane.trusted_mask[local_i]:
                    trusted_mask[global_i] = True
                    eps0[global_i] = in_plane.eps0[local_i]
                    kappa[global_i] = in_plane.kappa[local_i]
                    eps_ply_per_row[global_i] = in_plane.eps_ply_mat[local_i]
                    sig_ply_per_row[global_i] = in_plane.sig_ply_mat[local_i]

                row = rows[global_i]
                if not interlaminar_inputs_trusted(row):
                    continue
                shear_trusted_mask[global_i] = True
                tau_iface_per_row[global_i] = self.shear_operator.recover_interfaces(
                    qx=float(row.Qx),
                    qy=float(row.Qy),
                    lam=lam,
                    thickness_m=float(row.thickness_m),
                )

        eps_ply_mat = np.zeros((n_rows, n_ply_max, 3), dtype=np.float64)
        sig_ply_mat = np.zeros((n_rows, n_ply_max, 3), dtype=np.float64)
        tau_iface = np.zeros((n_rows, max(n_iface_max, 0), 2), dtype=np.float64)
        for i in range(n_rows):
            if eps_ply_per_row[i] is not None:
                n_ply_i = eps_ply_per_row[i].shape[0]
                eps_ply_mat[i, :n_ply_i] = eps_ply_per_row[i]
                sig_ply_mat[i, :n_ply_i] = sig_ply_per_row[i]
            if tau_iface_per_row[i] is not None:
                n_if = tau_iface_per_row[i].shape[0]
                tau_iface[i, :n_if] = tau_iface_per_row[i]

        ply_results = ShellPlyResults(
            eps0=eps0,
            kappa=kappa,
            eps_ply_mat=eps_ply_mat,
            sig_ply_mat=sig_ply_mat,
            trusted_mask=trusted_mask,
            tau_iface=tau_iface if n_iface_max > 0 else None,
            shear_trusted_mask=shear_trusted_mask,
        )
        return ShellTertiaryResultSet(
            ply_results=ply_results,
            transverse_shear=transverse_shear,
            thickness_m=thickness_m,
            element_index=element_index,
            panel_index=panel_index_arr,
            meta={
                "n_rows": n_rows,
                "n_ply": n_ply_max,
                "n_iface": n_iface_max,
                "shear_recovery_mode": SHEAR_RECOVERY_MODE_FSDT_PARABOLIC,
            },
        )


__all__ = ["ShellTertiaryResultsOrchestrator"]
